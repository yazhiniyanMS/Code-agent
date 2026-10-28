"""v5.1 audit fixes: masking and alignment checks, held-out leakage, data hygiene, skills opt-in."""

import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from ycode.lm.data import InstructionExample  # noqa: E402
from ycode.lm.model import GPT, GPTConfig  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
                         qk_norm=True, arch_version=5)).eval()


def test_attention_never_sees_future_tokens():
    model = _tiny()
    x = torch.randint(0, 300, (2, 40))
    later = x.clone()
    later[:, 25:] = torch.randint(0, 300, (2, 15))
    with torch.no_grad():
        a, _ = model(x, x)
        b, _ = model(later, later)
    assert torch.equal(a[:, :25], b[:, :25])  # positions before the change are bit-identical
    assert not torch.equal(a[:, 25:], b[:, 25:])


def test_sft_targets_are_shifted_and_only_answers_are_weighted(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.tokenizer import BPETokenizer
    from ycode.lm.train import SFTData

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    short = InstructionExample("Write a Python function `neg(x)` that returns minus x.", "```python\ndef neg(x):\n    return -x\n```")
    long = InstructionExample("Write a Python function `add_three(x)` that returns x plus three, as a number.",
                              "```python\ndef add_three(x):\n    \"\"\"Return x plus three.\"\"\"\n    return x + 3\n```")
    write_basics_dataset(tmp_path, data_dir / "tokenizer.json", examples=[short, long] * 20, seed=1, log=lambda *_: None)
    for pack in (True, False):
        data = SFTData(tmp_path, 128, 4, "cpu", pad_id=tok.eot_id, pack=pack)
        x, y, m = data.batch("train", torch.Generator().manual_seed(0), ordered=True)
        if pack:  # one continuous token stream per row: each target is the next input
            assert torch.equal(x[:, 1:], y[:, :-1])
        for row in range(x.size(0)):
            n = int((m[row] > 0).nonzero().max()) + 1 if (m[row] > 0).any() else 0
            targets = tok.decode([int(t) for t, w in zip(y[row], m[row]) if w > 0])
            assert "Write a Python function" not in targets  # no prompt text is ever a weighted target
            assert "def " in targets and n > 0
        if not pack:  # rows of different lengths: the padding is never weighted
            padded = (x == tok.eot_id) & (m > 0)
            assert not padded.any()


def test_skill_families_are_opt_in(monkeypatch):
    import ycode.lm.compose as compose
    from ycode.lm.erroranalysis import _body_key, training_bodies

    assert compose.INCLUDE_SKILLS is False and compose._skills() == []
    skill_names = {e.name for e in compose.skill_exercises()}
    assert not skill_names & {e.name for e in compose.all_exercises(3)}
    monkeypatch.setattr(compose, "INCLUDE_SKILLS", True)
    assert skill_names <= {e.name for e in compose.all_exercises(3)}
    # the error analysis still recognises a skill's body (earlier checkpoints were trained on them)
    one = compose.skill_exercises()[0]
    assert _body_key(one.source()) in training_bodies()


def test_leak_check_separates_equivalents_from_weak_test_passes():
    from ycode.lm.leakage import behaviour_leaks, held_out_items

    items = held_out_items()
    functions = {
        "def sum_to_n(n):\n    return n * (n + 1) // 2": ("sum_to_n", 1),  # passes fibonacci's 3 tests
        "def fib(k):\n    a, b = 0, 1\n    for _ in range(k):\n        a, b = b, a + b\n    return a": ("fib", 1),
        "def count_occurrences(text, ch):\n    return text.count(ch)": ("count_occurrences", 2),
        "def count_true(flags):\n    return sum(1 for x in flags if x)": ("count_true", 1),
    }
    found = {(lk.function, lk.item): lk.agreement for lk in behaviour_leaks(functions, items, near=0.0)}
    assert found[("fib", "fibonacci")] == 1.0
    assert found[("count_occurrences", "char_count")] == 1.0
    assert found[("sum_to_n", "fibonacci")] < 0.5  # passes the weak tests, disagrees with the reference
    assert found.get(("count_true", "dict_size"), 0.0) < 1.0  # dict keys may be falsy


def test_no_training_function_computes_a_held_out_item(monkeypatch):
    """Every concept the generators can emit (basics, compose with loop renderings, skills, novel
    constants for two seeds) is run against every benchmark and dev item; only the reviewed shared
    building blocks (leakage.SHARED_OPERATIONS) may match."""
    import ycode.lm.compose as compose
    from ycode.lm.basics import training_concepts
    from ycode.lm.leakage import behaviour_leaks, is_shared_operation

    monkeypatch.setattr(compose, "INCLUDE_SKILLS", True)
    functions = {}
    for c in training_concepts():
        arity = len([p for p in c.params.split(",") if p.strip()])
        for n in (c.name, *c.aliases):
            functions[c.source(n)] = (n, arity)
    for seed in (7, 1101):
        for e in compose.all_exercises(seed) + compose.novel_exercises(seed):
            for x in (e, compose.loop_rendering(e)):
                if x is not None:
                    functions[x.source()] = (x.name, len([p for p in x.params.split(",") if p.strip()]))
    leaks = [lk for lk in behaviour_leaks(functions, near=1.0) if not is_shared_operation(lk.source, lk.item)]
    assert not leaks, [(lk.function, lk.split, lk.item) for lk in leaks]


def test_novel_condition_pairs_skip_degenerate_tasks():
    from ycode.lm.compose import _DEGENERATE_PAIRS, novel_exercises

    for seed in range(40):
        keys = {e.key[2:] for e in novel_exercises(seed) if e.key[1] in ("count2", "keep2")}
        assert not keys & _DEGENERATE_PAIRS


def test_hygiene_drops_duplicates_and_contradictions_but_keeps_renderings():
    from ycode.lm.datacheck import clean_examples

    task = "Write a Python function `{n}(numbers)` that returns the largest of the numbers greater than 2."
    one_liner = "```python\ndef {n}(numbers):\n    return max(x for x in numbers if x > 2)\n```"
    loop = ("```python\ndef {n}(numbers):\n    best = None\n    for x in numbers:\n        if x > 2:\n"
            "            if best is None or x > best:\n                best = x\n    return best\n```")
    wrong = "```python\ndef {n}(numbers):\n    return min(x for x in numbers if x > 2)\n```"
    examples = [InstructionExample(task.format(n="f"), one_liner.format(n="f")),
                InstructionExample(task.format(n="f"), one_liner.format(n="f")),  # exact duplicate
                InstructionExample(task.format(n="g"), loop.format(n="g")),       # same behaviour: kept
                InstructionExample(task.format(n="h"), wrong.format(n="h"))]      # contradicts row 0
    kept, removed = clean_examples(examples, held_out=False)
    assert [r.reason for r in removed] == ["duplicate", "contradiction"]
    assert [r.index for r in removed] == [1, 3]
    assert len(kept) == 2


def test_hygiene_removes_held_out_equivalents_and_keeps_shared_building_blocks():
    from ycode.lm.datacheck import clean_examples

    examples = [
        InstructionExample("Write a Python function `f(numbers)` that returns the sum of the non-zero numbers.",
                           "```python\ndef f(numbers):\n    return sum(x for x in numbers if x != 0)\n```"),  # = total
        InstructionExample("Write a Python function `size(items)` that returns the number of items in the list.",
                           "```python\ndef size(items):\n    return len(items)\n```"),  # len(): reviewed, kept
    ]
    kept, removed = clean_examples(examples, dedup=False, contradictions=False)
    assert [(r.index, r.reason) for r in removed] == [(0, "held-out equivalent")]
    assert "total" in removed[0].detail and len(kept) == 1


def test_round_dataset_writes_manifest_and_hygiene(data_dir, tmp_path, monkeypatch):
    import ycode.lm.rounddata as rounddata

    small = [InstructionExample("Write a Python function `f(numbers)` that returns the sum of the non-zero numbers.",
                                "```python\ndef f(numbers):\n    return sum(x for x in numbers if x != 0)\n```"),
             InstructionExample("Write a Python function `neg(x)` that returns minus x.",
                                "```python\ndef neg(x):\n    return -x\n```")]
    monkeypatch.setattr(rounddata, "round_examples_tagged", lambda seed, options: (small + small[1:], ["legacy:write"] * 3))
    manifest = rounddata.write_round_dataset(tmp_path, 5, data_dir / "tokenizer.json", options={"dedup": True, "replay": 0},
                                             log=lambda *_: None)
    assert manifest["removed"] == {"held-out equivalent": 1, "duplicate": 1} and manifest["kept"] == 1
    assert json.loads((tmp_path / "manifest.json").read_text())["files"]["sft_tokens.bin"] == manifest["files"]["sft_tokens.bin"]
    assert {r["reason"] for r in json.loads((tmp_path / "hygiene.json").read_text())} == {"held-out equivalent", "duplicate"}
    with pytest.raises(ValueError):
        rounddata.write_round_dataset(tmp_path, 5, data_dir / "tokenizer.json", options={"replay_size": 3})


# ------------------------------------------------------------ training health (phase 4)


def _cfg(**kw):
    from ycode.lm.train import TrainConfig

    base = dict(preset="v2-tiny", model_overrides={"block_size": 64}, batch_size=4, max_steps=4, warmup_steps=1,
                eval_interval=1000, eval_iters=0, log_interval=1000, device="cpu", precision="fp32")
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture
def base_model(data_dir, tmp_path):
    from ycode.lm.train import train

    train(data_dir, tmp_path / "base", _cfg(max_steps=2), log=lambda *_: None)
    return tmp_path / "base"


def test_tiny_model_overfits_a_tiny_dataset(data_dir, base_model, tmp_path):
    """The training path can learn known examples: a tiny model fine-tuned on three examples must
    drive their loss near zero and answer them verbatim."""
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.generate import LocalLM
    from ycode.lm.train import train

    examples = [InstructionExample("Write `neg(x)`.", "```python\ndef neg(x):\n    return -x\n```"),
                InstructionExample("Write `twice(x)`.", "```python\ndef twice(x):\n    return 2 * x\n```"),
                InstructionExample("Write `head(s)`.", "```python\ndef head(s):\n    return s[0]\n```")]
    sft = tmp_path / "sft"
    write_basics_dataset(sft, data_dir / "tokenizer.json", examples=examples * 8, seed=0, log=lambda *_: None)
    out = tmp_path / "chat"
    train(sft, out, _cfg(stage="sft", init_from=str(base_model), max_steps=150, lr=3e-3, optimizer="adamw",
                         weight_decay=0.0), log=lambda *_: None)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(r["loss"] for r in rows[-10:]) / 10 < 0.05, rows[-10:]
    lm = LocalLM(out, device="cpu")
    for ex in examples:
        assert lm.chat([("user", ex.prompt)], max_new_tokens=40, temperature=0) == ex.response


def test_metrics_and_run_record(data_dir, base_model, tmp_path):
    from ycode.lm.train import file_sha256, train, verify_checkpoint

    out = tmp_path / "chat"
    summary = train(data_dir, out, _cfg(stage="sft", init_from=str(base_model), optimizer="lion", lr=1e-4),
                    log=lambda *_: None)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["grad_norm"] > 0 and r["tokens"] > 0 and r["answer_tokens"] > 0 for r in rows)
    record = json.loads((out / "train_log.json").read_text())["summary"]
    assert record["checkpoint_sha256"] == file_sha256(out / "model.pt") == verify_checkpoint(out)
    assert record["init_sha256"] == json.loads((base_model / "info.json").read_text())["sha256"]
    assert record["config"]["optimizer"] == "lion" and record["peak_rss_mb"] > 0
    assert summary["empty_batches"] == 0
    with (out / "model.pt").open("r+b") as f:  # flip one byte: the recorded hash catches it
        f.seek(100)
        byte = f.read(1)
        f.seek(100)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError):
        verify_checkpoint(out)


def test_nan_loss_stops_before_any_update(data_dir, base_model, tmp_path):
    from ycode.lm.train import assert_finite_weights, load_checkpoint, save_checkpoint, train

    model, tok, _ = load_checkpoint(base_model)
    with torch.no_grad():
        model.blocks[0].mlp.down.weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        assert_finite_weights(model)
    save_checkpoint(tmp_path / "broken", model, tok, step=2, stage="pretrain", val_loss=None)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(data_dir, tmp_path / "chat", _cfg(stage="sft", init_from=str(tmp_path / "broken"), optimizer="lion"),
              log=lambda *_: None)
    assert not (tmp_path / "chat" / "model.pt").exists()  # nothing was saved


def test_lion_never_applies_a_nonfinite_gradient():
    from ycode.lm.optim import build_optimizer

    lin = torch.nn.Linear(4, 4, bias=False)
    before = lin.weight.detach().clone()
    opt = build_optimizer(lin, kind="lion", lr=0.1, muon_lr=0.0, weight_decay=0.0)
    lion = opt.optimizers[0]
    lion.attach()
    (lin(torch.full((1, 4), float("inf"))).sum()).backward()
    assert lion.nonfinite == 1 and torch.equal(lin.weight, before)
    lion.reset_stats()
    (lin(torch.ones(1, 4)).sum()).backward()
    assert lion.nonfinite == 0 and lion.grad_sq > 0 and not torch.equal(lin.weight, before)


def test_check_batch_rejects_malformed_batches():
    from ycode.lm.train import check_batch

    x = torch.tensor([[1, 2, 3]])
    m = torch.tensor([[0, 1, 1]])
    assert check_batch(x, x, m, 10) is True
    assert check_batch(x, x, torch.zeros_like(m), 10) is False  # well-formed, nothing to learn
    for bad in ((x, x[:, :2], m), (x, torch.tensor([[1, 2, 10]]), m), (x, x, torch.tensor([[0, -1, 1]]))):
        with pytest.raises(ValueError):
            check_batch(*bad, 10)


def test_generalisation_check_reports_three_splits(data_dir, base_model, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.diagnose import split_losses

    examples = [InstructionExample(f"Write `add_{k}(x)`.", f"```python\ndef add_{k}(x):\n    return x + {k}\n```")
                for k in range(40)]
    write_basics_dataset(tmp_path / "d", data_dir / "tokenizer.json", examples=examples, seed=0, log=lambda *_: None)
    out = split_losses(base_model, tmp_path / "d", n=8)
    for split in ("train", "heldout_rows", "unseen_tasks"):
        assert out[split]["examples"] > 0 and out[split]["answer_tokens"] > 0 and out[split]["token_loss"] > 0
    assert out["unseen_tasks"]["examples"] >= 40  # every dev problem with its reference


def test_experiment_runner_end_to_end_with_a_tiny_model(data_dir, base_model, tmp_path, monkeypatch):
    """Two arms x two seeds through the real subprocess path (sft, eval, diagnose) on a tiny model,
    with a fast stand-in for the round data; then the paired summary."""
    import ycode.lm.rounddata as rounddata
    from ycode.lm import experiment
    from ycode.lm.basics import write_basics_dataset

    def fake_round(out_dir, seed, tokenizer_path, *, replay_dir=None, options=None, log=print, **_):
        examples = [InstructionExample(f"Write `add_{k}(x)`.", f"```python\ndef add_{k}(x):\n    return x + {k}\n```")
                    for k in range(seed, seed + 30)]
        info = write_basics_dataset(out_dir, tokenizer_path, examples=examples, seed=seed, log=log)
        manifest = {"seed": seed, "options": options or {}, "removed": {}, "kept": len(examples), **info}
        (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest))
        (Path(out_dir) / "hygiene.json").write_text("[]")
        return manifest

    from pathlib import Path

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    spec = {"name": "tiny", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "out": str(tmp_path / "exp"), "seeds": [1, 2], "metric": "dev_clean",
            "train": {"steps": 2, "precision": "fp32", "train_layers": 2, "init_optim": False},
            "eval": {"limit": 2, "diagnose": 4},
            "arms": [{"name": "A"}, {"name": "B", "data": {"skills": True}}], "comparisons": [["B", "A"]]}
    out = tmp_path / "exp"
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            record = experiment.run_one(spec, arm, seed, out, log=lambda *_: None)
            assert record["training"]["exit"] == 0 and record["training"]["steps"] == 2
            assert record["dev"]["of"] == 4 and record["bench"]["of"] == 4  # limit 2 writes + 2 fixes
            assert record["unseen"]["of"] == 2 and "failure_classes" in record["unseen"]  # limit 2 writes, no fixes
            assert record["training"]["peak_rss_mb"] > 0 and record["diagnose"]["unseen_new"]["examples"] > 0
            assert not (out / arm["name"] / str(seed) / "model").exists()  # weights deleted after scoring
    summary = experiment.summarize(spec, out)
    assert set(summary["arms"]) == {"A", "B"} and summary["comparisons"][0]["seeds"] == [1, 2]
    assert "Verdict" in experiment.render(summary)


@pytest.mark.parametrize("final_bf16", [False, True])
def test_v6_runner_path_scores_unseen2_and_keeps_a_bf16_model(data_dir, base_model, tmp_path, monkeypatch, final_bf16):
    """V6 options on a tiny model: unseen-v2 evaluation, the all_clean metric and a kept bf16 copy
    (V7: with final_bf16 the final checkpoint itself is kept)."""
    import json
    from pathlib import Path

    from ycode.lm import experiment, rounddata
    from ycode.lm.train import file_sha256

    def fake_round(out_dir, seed, tokenizer_path, **kwargs):
        import shutil

        shutil.copytree(data_dir, out_dir, dirs_exist_ok=True)
        manifest = {"seed": seed, "options": kwargs.get("options") or {}}
        (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest))
        (Path(out_dir) / "hygiene.json").write_text("[]")
        return manifest

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    spec = {"name": "tiny6", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "out": str(tmp_path / "exp"), "seeds": [1], "metric": "all_clean", "category_rule": "items_lost",
            "keep_models": "compact", "train": {"steps": 1, "precision": "fp32", "train_layers": 2, "init_optim": False,
                                                "compact": True, "final_bf16": final_bf16},
            "eval": {"limit": 2, "diagnose": 0, "unseen2": True}, "arms": [{"name": "C"}], "comparisons": []}
    record = experiment.run_one(spec, spec["arms"][0], 1, tmp_path / "exp", log=lambda *_: None)
    assert record["unseen2"]["of"] == 2 and "failure_classes" in record["unseen2"]
    assert "categories" in record["dev"] and "contaminated" in record["dev"]
    kept = Path(record["kept_model"]["path"])
    assert kept.is_file() and file_sha256(kept) == record["kept_model"]["sha256"]
    import torch

    from ycode.lm.train import load_checkpoint

    assert torch.load(kept, weights_only=True)["base"]["path"] == str(Path(base_model).resolve())  # compact copy
    load_checkpoint(kept.parent)  # loads with the anchor's frozen tensors
    payload = torch.load(kept, weights_only=True)
    assert {v.dtype for v in payload["model"].values() if v.is_floating_point()} == {torch.bfloat16}
    assert not (tmp_path / "exp" / "C" / "1" / "model").exists()  # the full checkpoint is still deleted
    summary = experiment.summarize(spec, tmp_path / "exp")
    assert summary["arms"]["C"]["all_clean"]["values"] == [record["dev"]["clean"] + record["unseen"]["clean"]
                                                           + record["unseen2"]["clean"]]


def test_paired_comparison_needs_more_than_one_lucky_seed():
    from ycode.lm.experiment import compare

    def run(dev_clean, cats=None):
        return {"dev": {"points": dev_clean, "clean": dev_clean,
                        "lost_by_category": cats or {"reasoning": 5, "hallucination": 1}}}

    lucky = compare({1: run(30), 2: run(25), 3: run(26)}, {1: run(25), 2: run(26), 3: run(25)}, "dev_clean")
    assert lucky["verdict"] == "inconclusive"  # +5, -1, +1
    steady = compare({1: run(28), 2: run(29), 3: run(28)}, {1: run(25), 2: run(26), 3: run(25)}, "dev_clean")
    assert steady["verdict"] == "retained"
    damaged = compare({1: run(28, {"reasoning": 5, "hallucination": 4})}, {1: run(25)}, "dev_clean")
    assert damaged["verdict"].startswith("inconclusive")  # one seed never decides
    damaged2 = compare({s: run(28, {"reasoning": 5, "hallucination": 4}) for s in (1, 2)}, {s: run(25) for s in (1, 2)},
                       "dev_clean")
    assert damaged2["verdict"] == "rejected"  # hallucinations +3 on average


def test_interrupted_experiment_run_resumes_training(data_dir, base_model, tmp_path, monkeypatch):
    """A run killed after its mid-run checkpoint picks up from it (same data, same step count)."""
    import ycode.lm.rounddata as rounddata
    from ycode.lm import experiment
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.train import TrainConfig, train

    def fake_round(out_dir, seed, tokenizer_path, *, options=None, log=print, **_):
        examples = [InstructionExample(f"Write `add_{k}(x)`.", f"```python\ndef add_{k}(x):\n    return x + {k}\n```")
                    for k in range(30)]
        info = write_basics_dataset(out_dir, tokenizer_path, examples=examples, seed=seed, log=log)
        (out_dir / "manifest.json").write_text(json.dumps({"seed": seed, "removed": {}, **info}))
        (out_dir / "hygiene.json").write_text("[]")
        return info

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    spec = {"name": "t", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "seeds": [1], "train": {"steps": 4, "precision": "fp32", "train_layers": 2, "init_optim": False},
            "eval": {"limit": 1, "diagnose": 0, "bench": False}, "arms": [{"name": "A"}]}
    run_dir = tmp_path / "exp" / "A" / "1"
    fake_round(run_dir / "data", 1, data_dir / "tokenizer.json", log=lambda *_: None)
    # the first attempt got halfway (2 of 4 steps) before the container went away
    train(run_dir / "data", run_dir / "model", TrainConfig(stage="sft", init_from=str(base_model), max_steps=4,
          until_step=2, batch_size=4, warmup_steps=2, eval_iters=0, eval_interval=100000, device="cpu",
          precision="fp32", optimizer="lion", lr=1.5e-5, train_layers=2, seed=1, save_interval=2,
          save_in_place=True), log=lambda *_: None)
    record = experiment.run_one(spec, spec["arms"][0], 1, tmp_path / "exp", log=lambda *_: None)
    steps = [json.loads(line)["step"] for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4] and record["training"]["steps"] == 4


def _sft(data, out, base_model, **kw):
    from ycode.lm.train import TrainConfig, train

    cfg = dict(stage="sft", init_from=str(base_model), max_steps=4, batch_size=4, warmup_steps=2, eval_iters=0,
               eval_interval=100000, device="cpu", precision="fp32", optimizer="lion", lr=1e-3, train_layers=2,
               seed=1, save_interval=2, save_in_place=True)
    train(data, out, TrainConfig(**{**cfg, **kw}), log=lambda *_: None)
    return out


def _weights(path):
    from ycode.lm.train import load_checkpoint

    model, _, _ = load_checkpoint(path)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _bf16_base(base_model, tmp_path):
    """A base that stores its weights in bf16, like round 54's frozen layers: full checkpoints round
    frozen weights to bf16, base-referenced ones keep the base's, so the two agree exactly here."""
    from ycode.lm.train import export_checkpoint

    return export_checkpoint(base_model, tmp_path / "base16", dtype="bf16").parent


def test_compact_checkpoints_train_and_resume_exactly_like_full_ones(data_dir, base_model, tmp_path):
    """Base-referenced training checkpoints (only the trained tensors; the frozen ones are the
    init_from checkpoint's) give the same weights as full ones, also across an interrupted run."""
    import torch

    base_model = _bf16_base(base_model, tmp_path)

    full = _sft(data_dir, tmp_path / "full", base_model)
    _sft(data_dir, tmp_path / "compact", base_model, compact_checkpoints=True, until_step=2)  # interrupted at 2
    compact = _sft(data_dir, tmp_path / "compact", base_model, compact_checkpoints=True, resume=True)
    payload = torch.load(compact / "model.pt", weights_only=True)
    assert payload["base"]["path"] == str(base_model.resolve()) and payload["step"] == 4
    assert (compact / "model.pt").stat().st_size < (full / "model.pt").stat().st_size
    a, b = _weights(full), _weights(compact)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_final_bf16_checkpoint_is_the_bf16_load_of_a_full_one(data_dir, base_model, tmp_path):
    """--final-bf16: the last save holds the trained tensors in bf16 and no optimizer state, and the
    weights it loads are exactly those evaluation loads (bf16) from an ordinary compact checkpoint."""
    import torch

    from ycode.lm.train import load_checkpoint

    base_model = _bf16_base(base_model, tmp_path)
    full = _sft(data_dir, tmp_path / "full", base_model, compact_checkpoints=True)
    final = _sft(data_dir, tmp_path / "final", base_model, compact_checkpoints=True, final_bf16=True, save_interval=4)
    assert (full / "optim.pt").is_file() and not (final / "optim.pt").exists()
    payload = torch.load(final / "model.pt", weights_only=True)
    assert payload["base"] and payload["step"] == 4
    assert {v.dtype for v in payload["model"].values() if v.is_floating_point()} == {torch.bfloat16}
    assert (final / "model.pt").stat().st_size < (full / "model.pt").stat().st_size
    a = load_checkpoint(full, dtype=torch.bfloat16)[0].state_dict()
    b = load_checkpoint(final, dtype=torch.bfloat16)[0].state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_base_referenced_export_and_soup_match_full_ones(data_dir, base_model, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    import torch

    from ycode.lm.train import export_checkpoint

    base_model = _bf16_base(base_model, tmp_path)
    one = _sft(data_dir, tmp_path / "one", base_model)
    two = _sft(data_dir, tmp_path / "two", base_model, seed=2, lr=3e-3)
    for run in (one, two):
        export_checkpoint(run, run / "full", dtype="bf16")
        export_checkpoint(run, run / "compact", dtype="bf16", base=base_model)
        assert (run / "compact" / "model.pt").stat().st_size < (run / "full" / "model.pt").stat().st_size
        a, b = _weights(run / "full"), _weights(run / "compact")
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    soup = Path(__file__).resolve().parents[1] / "experiments" / "v6" / "soup.py"
    for kind in ("full", "compact"):
        subprocess.run([sys.executable, str(soup), str(tmp_path / f"soup_{kind}"), str(one / kind), str(two / kind)],
                       check=True, capture_output=True)
    a, b = _weights(tmp_path / "soup_full"), _weights(tmp_path / "soup_compact")
    assert all(torch.equal(a[k], b[k]) for k in a)
    w1, w2 = _weights(one / "full"), _weights(two / "full")
    k = next(k for k in a if not torch.equal(w1[k], w2[k]))  # a trained tensor: the soup is their mean
    assert torch.equal(a[k], ((w1[k] + w2[k]) / 2).to(torch.bfloat16).float())  # stored in the inputs' bf16


# ------------------------------------------------------- evaluation reliability (v5.1 mission)

KNOWN_WRONG = [  # answers the official tests accepted in v5 runs although they are wrong
    ("bench", "write", "fibonacci", "def fibonacci(n):\n    return n * (n + 1) // 2"),
    ("bench", "write", "first_word", "def first_word(text):\n    return text.split()[-2]"),
    ("bench", "bugfix", "maximum", "def maximum(items):\n    best = items[1]\n    for x in items:\n        if x > best:\n"
                                   "            best = x\n    return best"),
    ("bench", "bugfix", "minimum", "def minimum(items):\n    best = items[0]\n    for x in items:\n        if x % 2 == 0:\n"
                                   "            return x\n    return best"),
    ("dev", "write", "first_negative", "def first_negative(numbers):\n    best = None\n    for x in numbers:\n"
                                       "        if x < 0 and (best is None or x > best):\n            best = x\n    return best"),
    ("dev", "bugfix", "is_between", "def is_between(x, low, high):\n    return x < high"),
    ("dev", "bugfix", "longest_word", "def longest_word(text):\n    return text.split()[-2]"),
]
KNOWN_RIGHT = [  # different but correct implementations: the robust suite must not fail them
    ("bench", "bugfix", "maximum", "def maximum(items):\n    best = items[0]\n    for x in items:\n        if x > best:\n"
                                   "            best = x\n    return best"),
    ("dev", "write", "longest_word", "def longest_word(text):\n    best = ''\n    for word in text.split():\n"
                                     "        if len(word) > len(best):\n            best = word\n    return best"),
    ("dev", "write", "longest_word", "def longest_word(text):\n    return max(text.split(), key=len)"),
    ("bench", "write", "fibonacci", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"),
    ("bench", "write", "factorial", "def factorial(n):\n    import math\n    return math.factorial(n)"),
]


def _item(split, suite, name):
    from ycode.lm.evalsets import eval_items

    return next(i for i in eval_items(split) if i.suite == suite and i.name == name)


@pytest.mark.parametrize("split,suite,name,code", KNOWN_WRONG)
def test_robust_tests_reject_known_wrong_answers_the_official_tests_accept(split, suite, name, code):
    from ycode.lm.evalsets import robust_pass
    from ycode.lm.evaluate import run_tests

    item = _item(split, suite, name)
    assert run_tests(code, item.tests)  # the official verdict stays as it was
    assert not robust_pass(item, code)


@pytest.mark.parametrize("split,suite,name,code", KNOWN_RIGHT)
def test_robust_tests_accept_other_correct_implementations(split, suite, name, code):
    from ycode.lm.evalsets import robust_pass

    assert robust_pass(_item(split, suite, name), code)


def test_every_reference_passes_official_and_robust_tests():
    from ycode.lm.evalsets import eval_items
    from ycode.lm.evaluate import run_tests
    from ycode.lm.leakage import held_out_items

    refs = {(i.split, i.name): i.reference for i in held_out_items()}
    for split in ("bench", "dev", "unseen"):
        for item in eval_items(split):
            if item.suite == "write":
                ref = refs[(split, item.name)]
                assert run_tests(ref, item.tests + "\n" + item.robust_tests), (split, item.name)


def test_eval_sets_have_explicit_denominators_and_stable_fingerprints():
    from ycode.lm.evalsets import eval_items, fingerprint, manifest
    from ycode.lm.unseen import unseen_problems

    m = manifest()
    assert (m["bench"]["items"], m["bench"]["clean_items"]) == (65, 61)
    assert (m["dev"]["items"], m["dev"]["clean_items"]) == (62, 44)
    assert m["unseen"]["items"] == m["unseen"]["clean_items"] == len(unseen_problems()) >= 40
    assert fingerprint(eval_items("dev")) == m["dev"]["fingerprint"] == manifest()["dev"]["fingerprint"]
    with pytest.raises(ValueError):
        eval_items("train")


def test_robust_test_module_is_reproducible(tmp_path, monkeypatch):
    """The committed robust tests are exactly what the generator writes (no hand edits, no drift)."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    committed = (root / "ycode" / "lm" / "robust_tests_v1.py").read_text()
    spec = importlib.util.spec_from_file_location("make_robust_tests", root / "experiments" / "v51" / "make_robust_tests.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    (tmp_path / "ycode" / "lm").mkdir(parents=True)
    gen.main()
    assert (tmp_path / "ycode" / "lm" / "robust_tests_v1.py").read_text() == committed


def test_scores_report_official_clean_and_robust_with_denominators():
    from ycode.lm.evalsets import eval_items, score

    items = eval_items("bench")
    answers = [{"suite": i.suite, "name": i.name, "pass": False, "code": ""} for i in items]
    for a in answers:
        if (a["suite"], a["name"]) == ("write", "first_word"):
            a.update({"pass": True, "code": "def first_word(text):\n    return text.split()[-2]"})
        if (a["suite"], a["name"]) == ("write", "total"):  # contaminated: counts officially, not clean
            a.update({"pass": True, "code": "def total(numbers):\n    return sum(numbers)"})
    s = score(answers, "bench")
    assert (s["official"], s["official_of"]) == (2, 65)
    assert (s["clean"], s["clean_of"]) == (1, 61)
    assert (s["robust"], s["robust_of"]) == (1, 65) and s["weak_passes"] == ["write:first_word"]


def test_unseen_set_is_held_out_of_every_generator():
    from ycode.lm.basics import _passes, held_out_names
    from ycode.lm.unseen import unseen_names, unseen_problems, unseen_references

    refs = unseen_references()
    for p in unseen_problems():
        assert _passes(refs[p.name], p.tests), p.name
        assert p.tests.count("assert") >= 3, p.name  # strong tests from the start
    assert unseen_names() <= held_out_names()


def test_failure_classes_follow_the_rules():
    from ycode.lm.failures import _candidates_by_arity, classify_failure
    from ycode.lm.leakage import held_out_items

    items = {(i.split, i.name): i for i in held_out_items()}
    cands = _candidates_by_arity()
    fourth = items[("dev", "fourth_item")]
    cases = [
        ("def fourth_item(items)\n    return items[3]", "formatting_or_parsing"),
        ("def fourth_item(items):\n    return helper(items)", "incorrect_code_generation"),
        ("def fourth_item(items):\n    return items[4]", "incorrect_task_interpretation"),  # = fifth_item
        ("def fourth_item(items):\n    return items", "algorithmic_or_logic_error"),  # returns its input
    ]
    for code, expected in cases:
        category = "hallucination" if "helper" in code else "reasoning"
        reason = "NameError: name 'helper' is not defined" if "helper" in code else "wrong value"
        got, _ = classify_failure({"code": code}, fourth, category, reason, cands)
        assert got == expected, (code, got)
    between = items[("dev", "is_between")]  # exclusive bounds: right on most inputs, wrong at the edges
    got, _ = classify_failure({"code": "def is_between(x, low, high):\n    return low < x < high"}, between, "reasoning",
                              "is_between(1, 1, 3) gave False", cands)
    assert got == "edge_case_failure"


def test_diagnose_labels_whether_rows_were_trained_on(data_dir, base_model, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.diagnose import split_losses

    examples = [InstructionExample(f"Write `add_{k}(x)`.", f"```python\ndef add_{k}(x):\n    return x + {k}\n```")
                for k in range(40)]
    write_basics_dataset(tmp_path / "d", data_dir / "tokenizer.json", examples=examples, seed=0, log=lambda *_: None)
    out = split_losses(base_model, tmp_path / "d", n=6)
    assert out["trained_on"] is False and out["note"]
    assert out["train"]["token_loss_ci95"][0] <= out["train"]["token_loss"] <= out["train"]["token_loss_ci95"][1]
    assert out["train"]["by_kind"] and out["unseen_new"]["examples"] >= 40
    assert out["unseen_dev_clean"]["examples"] < out["unseen_tasks"]["examples"]
    for split in ("unseen_dev_clean", "unseen_new"):  # code-only loss in both answer formats
        assert out[split]["code_token_loss_plain"] > 0 and out[split]["code_token_loss_spec_first"] > 0


def test_restated_references_add_only_the_task_docstring():
    from ycode.lm.diagnose import _restated

    ref = "def tens_digit(n):\n    return (n // 10) % 10"
    out = _restated("Write a Python function `tens_digit(n)` that returns the tens digit of n.", ref)
    assert out == 'def tens_digit(n):\n    """Return the tens digit of n."""\n    return (n // 10) % 10'


def test_spec_first_answers_restate_the_task_and_keep_the_code(monkeypatch):
    """The treatment changes only the answer format of written functions: a docstring restating the
    prompt's task, then the same, still-correct code. Bug fixes and the default data are untouched."""
    import re

    import ycode.lm.basics as basics
    from ycode.lm.basics import _passes, _rename_tests, training_concepts
    from ycode.lm.rounddata import round_examples

    plain = round_examples(11, {})
    spec = round_examples(11, {"spec_first": True})
    assert basics.SPEC_FIRST is False  # restored after the build
    assert len(plain) == len(spec)
    tests = {c.name: c.tests for c in training_concepts()}
    checked = 0
    for a, b in zip(plain, spec):
        assert a.prompt == b.prompt
        if a.prompt.startswith("This function has a bug"):
            assert a.response == b.response
            continue
        doc = re.search(r'"""(.+?)"""', b.response)
        assert '"""' not in a.response and doc, b.response
        # the docstring is the prompt's task, as an imperative sentence
        task = doc.group(1).rstrip(".").lower()
        assert task.split(" ", 1)[1] in a.prompt.lower(), (task, a.prompt)
        assert re.sub(r'\n\s*""".*?"""', "", b.response) == a.response  # same code
        name = re.search(r"def (\w+)\(", b.response).group(1)
        if name in tests:
            assert _passes(b.response.strip("`").removeprefix("python\n"), _rename_tests(tests[name], name))
            checked += 1
    assert checked > 50
