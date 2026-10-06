"""Tests for YCode-LM v3: the 40M preset and the Muon optimizer."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from ycode.lm.model import GPT, PRESETS, GPTConfig  # noqa: E402
from ycode.lm.optim import Muon, build_optimizer, orthogonalize  # noqa: E402
from ycode.lm.train import TrainConfig, load_checkpoint, train  # noqa: E402


def test_v3_preset_is_about_40m_params():
    cfg = GPTConfig(vocab_size=8192, **PRESETS["v3-40m"])
    params = GPT(cfg).num_params()
    assert 39e6 <= params <= 42e6
    assert cfg.arch_version == 3 and cfg.block_size == 1024 and cfg.kv_heads == 2


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_orthogonalize_gives_near_unit_singular_values(shape):
    torch.manual_seed(0)
    s = torch.linalg.svdvals(orthogonalize(torch.randn(*shape)).float())
    # Muon's 5-step quintic iteration maps singular values into roughly [0.7, 1.2] by design.
    assert s.max() < 1.3 and s.min() > 0.5


def test_orthogonalize_improves_conditioning():
    torch.manual_seed(0)
    u, _, v = torch.linalg.svd(torch.randn(48, 48))
    g = u @ torch.diag(torch.logspace(-1, 1, 48)) @ v  # condition number 100
    s = torch.linalg.svdvals(orthogonalize(g).float())
    assert s.max() / s.min() < 5


def test_muon_rejects_non_matrix_params():
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.ones(4))])


def test_build_optimizer_splits_parameters():
    model = GPT(GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=16))
    opt = build_optimizer(model, kind="muon", lr=1e-3, muon_lr=0.02, weight_decay=0.1)
    muon, adam = opt.optimizers
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    assert id(model.embed.weight) not in muon_ids  # embeddings stay on AdamW
    assert id(model.blocks[0].attn.qkv.weight) in muon_ids
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == model.num_params()
    opt.set_lr_scale(0.5)
    assert muon.param_groups[0]["lr"] == pytest.approx(0.01) and adam.param_groups[0]["lr"] == pytest.approx(5e-4)
    with pytest.raises(ValueError):
        build_optimizer(model, kind="sgd", lr=1e-3, muon_lr=0.02, weight_decay=0.0)


def test_muon_step_reduces_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    opt = build_optimizer(model, kind="muon", lr=3e-3, muon_lr=0.02, weight_decay=0.0)
    x = torch.randint(0, 50, (8, 16))
    first = None
    for _ in range(30):
        _, loss = model(x, x)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first * 0.7


def _cfg(**kw):
    base = dict(preset="v3-tiny", model_overrides={"block_size": 32}, batch_size=4, max_steps=20, warmup_steps=2,
                eval_interval=20, eval_iters=2, log_interval=1000, device="cpu", precision="fp32",
                optimizer="muon", lr=3e-3)
    base.update(kw)
    return TrainConfig(**base)


def test_v3_training_with_muon_and_resume(data_dir, tmp_path):
    logs = []
    train(data_dir, tmp_path / "m", _cfg(max_steps=10), log=logs.append)
    assert any("YCode-LM v3" in line and "optimizer muon" in line for line in logs)
    summary = train(data_dir, tmp_path / "m", _cfg(max_steps=14, resume=True), log=logs.append)
    assert summary["steps"] == 14
    assert not any("starting it fresh" in line for line in logs)  # Muon state resumed
    model, _, payload = load_checkpoint(tmp_path / "m")
    assert payload["version"] == 3


def test_resume_with_different_optimizer_is_handled(data_dir, tmp_path):
    logs = []
    train(data_dir, tmp_path / "m", _cfg(max_steps=4, optimizer="adamw"), log=logs.append)
    train(data_dir, tmp_path / "m", _cfg(max_steps=6, resume=True), log=logs.append)
    assert any("starting it fresh" in line for line in logs)


def test_cli_defaults_to_muon_for_v3(data_dir, tmp_path, capsys):
    from ycode.lm import cli

    assert cli.main(["train", "--data", str(data_dir), "--out", str(tmp_path / "m"), "--preset", "v3-tiny",
                     "--context", "32", "--steps", "2", "--batch-size", "2", "--device", "cpu"]) == 0
    assert "optimizer muon" in capsys.readouterr().out


def test_segmented_training_matches_schedule(data_dir, tmp_path):
    from ycode.lm.train import lr_at

    logs = []
    first = train(data_dir, tmp_path / "m", _cfg(max_steps=12, until_step=5), log=logs.append)
    assert first["steps"] == 5
    second = train(data_dir, tmp_path / "m", _cfg(max_steps=12, resume=True), log=logs.append)
    assert second["steps"] == 12
    cfg = _cfg(max_steps=12)
    assert lr_at(5, cfg) == pytest.approx(cfg.lr)  # segment 1 ended mid-run at full LR, not decayed


def test_sft_resume_continues_in_out_dir(data_dir, tmp_path):
    train(data_dir, tmp_path / "base", _cfg(max_steps=4), log=lambda *_: None)
    sft = dict(stage="sft", init_from=str(tmp_path / "base"), optimizer="adamw")
    train(data_dir, tmp_path / "chat", _cfg(max_steps=8, until_step=3, **sft), log=lambda *_: None)
    logs = []
    summary = train(data_dir, tmp_path / "chat", _cfg(max_steps=8, resume=True, **sft), log=logs.append)
    assert summary["steps"] == 8
    assert any("chat" in line and "step 3" in line for line in logs)  # resumed the SFT run, not the base


def test_save_interval_checkpoints_between_evals(data_dir, tmp_path):
    import json

    logs = []
    train(data_dir, tmp_path / "m", _cfg(max_steps=20, until_step=7, eval_interval=20, save_interval=3),
          log=logs.append)
    assert [line for line in logs if "checkpoint saved" in line] == [
        "checkpoint saved at step 3", "checkpoint saved at step 6"]
    assert json.loads((tmp_path / "m" / "info.json").read_text())["step"] == 7  # final save at the segment end


def test_resume_without_checkpoint_starts_fresh(data_dir, tmp_path):
    logs = []
    summary = train(data_dir, tmp_path / "new", _cfg(max_steps=3, resume=True), log=logs.append)
    assert summary["steps"] == 3 and any("starting a new run" in line for line in logs)


def test_export_bf16_is_smaller_and_loads(data_dir, tmp_path):
    from ycode.lm import cli
    from ycode.lm.generate import LocalLM

    train(data_dir, tmp_path / "m", _cfg(max_steps=3), log=lambda *_: None)
    assert cli.main(["export", "--model", str(tmp_path / "m"), "--out", str(tmp_path / "slim")]) == 0
    full, slim = tmp_path / "m" / "model.pt", tmp_path / "slim" / "model.pt"
    assert slim.stat().st_size < full.stat().st_size * 0.6
    original, _, _ = load_checkpoint(tmp_path / "m")
    exported, _, payload = load_checkpoint(tmp_path / "slim")
    assert payload["exported_dtype"] == "bf16"
    state = payload["model"]
    assert state["embed.weight"].data_ptr() == state["head.weight"].data_ptr()  # tied weights stored once
    import json
    assert json.loads((tmp_path / "slim" / "info.json").read_text())["params"] == original.num_params()
    assert next(exported.parameters()).dtype == torch.float32  # converted back on load
    x = torch.randint(0, original.cfg.vocab_size, (1, 8))
    assert torch.allclose(original(x, x)[0], exported(x, x)[0], atol=0.1)
    assert isinstance(LocalLM(tmp_path / "slim", device="cpu").chat([("user", "hi")], max_new_tokens=3), str)
    assert cli.main(["export", "--model", str(tmp_path / "nope"), "--out", str(tmp_path / "x")]) == 1


def test_no_repeat_ngram_blocks_loops():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=128, n_layer=1, n_head=2, n_embd=16))
    plain = model.generate(torch.tensor([[1, 2, 3]]), 60, temperature=0)[0, 3:].tolist()
    blocked = model.generate(torch.tensor([[1, 2, 3]]), 60, temperature=0, no_repeat_ngram=4)[0, 3:].tolist()

    def repeats(seq):
        grams = [tuple(seq[i:i + 4]) for i in range(len(seq) - 3)]
        return len(grams) - len(set(grams))

    assert repeats(plain) > 0  # an untrained model loops under greedy decoding
    assert repeats(blocked) == 0


def test_write_examples_are_compact_standalone_functions():
    import random

    from ycode.lm.data import extract_python_examples

    source = '''
def area(width, height):
    """Return the area of a rectangle.

    A much longer explanation that should not appear in answers.

    >>> area(2, 3)
    6
    """
    result = width * height
    return result


class Shape:
    def scale(self, factor):
        """Scale the shape by the given factor."""
        self.size = self.size * factor
        return self.size
'''
    examples = []
    for seed in range(5):
        examples += extract_python_examples(source, random.Random(seed))
    writes = [e for e in examples if e.response.startswith("```python") and "bug" not in e.response.lower()]
    assert writes and all("area" in e.response for e in writes)  # no method `scale` in write answers
    for e in writes:
        assert '"""Return the area of a rectangle."""' in e.response
        assert ">>>" not in e.response and "longer explanation" not in e.response


def test_prepare_sft_only_reuses_tokenizer(data_dir, tmp_path):
    from ycode.lm import cli
    from ycode.lm.tokenizer import BPETokenizer

    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text('def area(w, h):\n    """Return the area of a rectangle."""\n'
                              '    if w < 0:\n        return 0\n    return w * h\n')
    out = tmp_path / "sft"
    assert cli.main(["prepare", "--source", str(src), "--out", str(out), "--sft-only"]) == 2  # needs --tokenizer
    assert cli.main(["prepare", "--source", str(src), "--out", str(out), "--sft-only",
                     "--tokenizer", str(data_dir / "tokenizer.json")]) == 0
    assert BPETokenizer.load(out / "tokenizer.json").merges == BPETokenizer.load(data_dir / "tokenizer.json").merges
    assert (out / "sft_index.npy").is_file() and not (out / "train.bin").exists()


def test_insertion_plan():
    from ycode.lm.grow import insertion_plan

    assert sum(insertion_plan(13, 34)) == 21 and max(insertion_plan(13, 34)) - min(insertion_plan(13, 34)) <= 1
    assert insertion_plan(4, 4) == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        insertion_plan(5, 3)


def test_grown_model_computes_the_same_function_and_trains(data_dir, tmp_path):
    from ycode.lm import cli

    torch.manual_seed(0)
    small = GPT(GPTConfig(vocab_size=120, block_size=32, n_layer=3, n_head=4, n_kv_head=2, n_embd=32,
                          qk_norm=True, arch_version=3)).eval()
    from ycode.lm.grow import grow_depth

    big = grow_depth(small, 8, arch_version=4).eval()
    assert big.cfg.n_layer == 8 and big.cfg.arch_version == 4 and big.num_params() > small.num_params()
    x = torch.randint(0, 120, (2, 16))
    assert torch.allclose(small(x, x)[0], big(x, x)[0], atol=1e-5)  # identical at step 0
    assert torch.equal(small.generate(x[:1], 6, temperature=0), big.generate(x[:1], 6, temperature=0))

    # CLI round trip, then continued pretraining from the grown checkpoint.
    train(data_dir, tmp_path / "v3", _cfg(max_steps=3), log=lambda *_: None)
    assert cli.main(["grow", "--model", str(tmp_path / "v3"), "--out", str(tmp_path / "v4"), "--layers", "5",
                     "--version", "4"]) == 0
    grown, _, payload = load_checkpoint(tmp_path / "v4")
    assert grown.cfg.n_layer == 5 and payload["stage"] == "grown"
    summary = train(data_dir, tmp_path / "v4b", _cfg(max_steps=3, init_from=str(tmp_path / "v4")),
                    log=lambda *_: None)
    assert summary["steps"] == 3


def test_v4_preset_is_about_100m_params():
    cfg = GPTConfig(vocab_size=8192, **PRESETS["v4-100m"])
    assert 99e6 <= GPT(cfg).num_params() <= 101e6 and cfg.arch_version == 4


def test_train_cli_accepts_init_from(data_dir, tmp_path):
    from ycode.lm import cli

    train(data_dir, tmp_path / "base", _cfg(max_steps=2), log=lambda *_: None)
    assert cli.main(["train", "--data", str(data_dir), "--init-from", str(tmp_path / "base"), "--out",
                     str(tmp_path / "more"), "--steps", "2", "--batch-size", "2", "--device", "cpu"]) == 0
    assert load_checkpoint(tmp_path / "more")[2]["step"] == 2


def test_sharded_export_round_trip(data_dir, tmp_path):
    from ycode.lm import cli

    train(data_dir, tmp_path / "m", _cfg(max_steps=2), log=lambda *_: None)
    assert cli.main(["export", "--model", str(tmp_path / "m"), "--out", str(tmp_path / "s"),
                     "--max-shard-mb", "0.05"]) == 0
    shards = sorted((tmp_path / "s").glob("model-*-of-*.pt"))
    assert len(shards) >= 3 and all(f.stat().st_size < 0.2e6 for f in shards)
    original, _, _ = load_checkpoint(tmp_path / "m")
    loaded, _, payload = load_checkpoint(tmp_path / "s")
    assert "head.weight" in payload["tied"]  # tied weights stored once
    x = torch.randint(0, original.cfg.vocab_size, (1, 8))
    assert torch.allclose(original(x, x)[0], loaded(x, x)[0], atol=0.1)
    # Re-exporting unsharded removes stale shards; a missing shard is a clear error.
    assert cli.main(["export", "--model", str(tmp_path / "m"), "--out", str(tmp_path / "s")]) == 0
    assert not list((tmp_path / "s").glob("model-*-of-*.pt"))
    assert cli.main(["export", "--model", str(tmp_path / "m"), "--out", str(tmp_path / "t"),
                     "--max-shard-mb", "0.05"]) == 0
    sorted((tmp_path / "t").glob("model-*-of-*.pt"))[0].unlink()
    with pytest.raises(FileNotFoundError, match="Missing model shard"):
        load_checkpoint(tmp_path / "t")
