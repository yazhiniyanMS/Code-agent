"""Tests for the from-scratch model stack (tokenizer, transformer, data, training, inference)."""

import random
import textwrap

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from ycode.lm.data import (  # noqa: E402
    InstructionExample,
    encode_example,
    extract_python_examples,
    format_chat,
    prepare_dataset,
)
from ycode.lm.tokenizer import END, EOT, USER, BPETokenizer, split_chunks  # noqa: E402
from ycode.lm.model import GPT, GPTConfig  # noqa: E402
from ycode.lm.train import TrainConfig, load_checkpoint, save_checkpoint, train  # noqa: E402

CODE = textwrap.dedent('''
    def add(a, b):
        """Return the sum of a and b."""
        return a + b


    def greet(name):
        """Build a friendly greeting for the given name."""
        message = "Hello, " + name
        return message


    class Stack:
        def push(self, item):
            """Push an item onto the top of the stack."""
            self.items.append(item)
            return item
''')


# ------------------------------------------------------------- tokenizer


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train([CODE * 20], vocab_size=400)


def test_tokenizer_round_trip(tok):
    for text in [CODE, "naïve café → 你好 🚀", "x\t= 1\r\n", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_merges_compress(tok):
    assert len(tok.encode(CODE)) < len(CODE.encode("utf-8")) / 2


def test_special_tokens(tok):
    ids = tok.encode(f"{USER}\nhi\n{END}")
    assert ids[0] == tok.token_id(USER) and ids[-1] == tok.token_id(END)
    literal = tok.encode(END, allow_special=False)
    assert tok.token_id(END) not in literal and tok.decode(literal) == END
    assert tok.decode(ids, skip_special=True) == "\nhi\n"


def test_tokenizer_save_load(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    again = BPETokenizer.load(tmp_path / "t.json")
    assert again.encode(CODE) == tok.encode(CODE) and again.vocab_size == tok.vocab_size


def test_indentation_is_one_chunk():
    assert "\n        " in split_chunks("def f():\n        return 1")


def test_vocab_size_validation():
    with pytest.raises(ValueError):
        BPETokenizer.train(["abc"], vocab_size=100)


# ----------------------------------------------------------------- model


def tiny_model(vocab=400, block=64):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32))


def test_forward_and_masked_loss():
    model = tiny_model()
    x = torch.randint(0, 400, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 400) and loss.item() > 0
    mask = torch.zeros_like(x)
    mask[:, -4:] = 1
    _, masked = model(x, x, loss_mask=mask)
    assert masked.item() != pytest.approx(loss.item())


def test_kv_cache_matches_full_recompute():
    model = tiny_model().eval()
    prompt = torch.randint(0, 400, (1, 10))
    cached = model.generate(prompt, 12, temperature=0)
    seq = prompt
    with torch.no_grad():
        for _ in range(12):
            logits = model.head(model.norm(_hidden(model, seq)))[:, -1]
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, seq)


def _hidden(model, idx):
    T = idx.size(1)
    x = model.embed(idx)
    for block in model.blocks:
        x = block(x, model.rope_cos[:T], model.rope_sin[:T])
    return x


def test_generate_respects_stop_and_context():
    model = tiny_model(block=32)
    out = model.generate(torch.randint(0, 400, (1, 40)), 8, temperature=0.9)
    assert out.size(1) <= 32 + 8


# ------------------------------------------------------------------- data


def test_extract_python_examples():
    examples = extract_python_examples(CODE, random.Random(0))
    prompts = " ".join(e.prompt for e in examples)
    assert "add" in prompts and "greet" in prompts and "push" in prompts
    write = [e for e in examples if e.response.startswith("```python")]
    assert any("return a + b" in e.response for e in write)
    explain = [e for e in examples if "returns the sum of a and b" in e.response]
    assert explain and '"""' not in explain[0].prompt  # docstring removed from the question


def test_encode_example_masks_only_the_answer(tok):
    ids, mask = encode_example(tok, InstructionExample("What is 1+1?", "2"))
    assert len(ids) == len(mask)
    answer = tok.decode([i for i, m in zip(ids, mask) if m])
    assert answer == f"2\n{END}"
    assert tok.decode(ids).startswith(format_chat("What is 1+1?"))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    for i in range(6):
        (src / f"mod{i}.py").write_text(CODE.replace("add", f"add{i}") * 3)
    (src / "notes.bin").write_bytes(b"\x00\x01")
    out = tmp_path_factory.mktemp("data")
    meta = prepare_dataset([src], out, vocab_size=400, log=lambda *_: None)
    return out, meta


def test_prepare_dataset(dataset):
    out, meta = dataset
    for name in ("tokenizer.json", "train.bin", "val.bin", "sft_tokens.bin", "sft_mask.bin", "sft_index.npy",
                 "meta.json"):
        assert (out / name).is_file(), name
    assert meta["files"] == 6 and meta["sft_examples"] >= 12
    assert np.fromfile(out / "train.bin", dtype=np.uint16).max() < meta["vocab_size"]


# --------------------------------------------------------------- training


def _cfg(**kw):
    base = dict(preset="tiny", model_overrides={"block_size": 32}, batch_size=4, max_steps=40, warmup_steps=5,
                eval_interval=40, eval_iters=2, log_interval=1000, device="cpu", precision="fp32")
    base.update(kw)
    return TrainConfig(**base)


def test_pretrain_reduces_loss_and_checkpoints(dataset, tmp_path):
    data_dir, _ = dataset
    logs = []
    summary = train(data_dir, tmp_path / "base", _cfg(lr=3e-3), log=logs.append)
    history = __import__("json").loads((tmp_path / "base" / "train_log.json").read_text())
    assert summary["steps"] == 40
    model, tok, payload = load_checkpoint(tmp_path / "base")
    assert payload["stage"] == "pretrain" and payload["step"] == 40
    # Untrained model loss is ~ln(vocab) = ln(400) ~ 6.0; training must beat it clearly.
    assert history["history"][-1]["train"] < 5.0


def test_sft_from_pretrained(dataset, tmp_path):
    data_dir, _ = dataset
    train(data_dir, tmp_path / "base", _cfg(max_steps=10), log=lambda *_: None)
    with pytest.raises(ValueError):
        train(data_dir, tmp_path / "x", _cfg(stage="sft"), log=lambda *_: None)
    summary = train(data_dir, tmp_path / "chat", _cfg(stage="sft", max_steps=10, init_from=str(tmp_path / "base")),
                    log=lambda *_: None)
    assert summary["steps"] == 10
    assert load_checkpoint(tmp_path / "chat")[2]["stage"] == "sft"


def test_resume_continues_step_count(dataset, tmp_path):
    data_dir, _ = dataset
    train(data_dir, tmp_path / "m", _cfg(max_steps=5), log=lambda *_: None)
    summary = train(data_dir, tmp_path / "m", _cfg(max_steps=8, resume=True), log=lambda *_: None)
    assert summary["steps"] == 8


def test_checkpoint_round_trip_preserves_outputs(tok, tmp_path):
    model = tiny_model().eval()
    save_checkpoint(tmp_path, model, tok, step=3, stage="pretrain", val_loss=1.0)
    loaded, tok2, _ = load_checkpoint(tmp_path)
    x = torch.randint(0, 400, (1, 8))
    assert torch.allclose(model(x, x)[0], loaded(x, x)[0])
    assert tok2.encode(CODE) == tok.encode(CODE)


def test_load_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope")


# -------------------------------------------------------------- inference


def test_local_lm_chat_and_complete(dataset, tmp_path):
    from ycode.lm.generate import LocalLM

    data_dir, _ = dataset
    train(data_dir, tmp_path / "m", _cfg(max_steps=5), log=lambda *_: None)
    lm = LocalLM(tmp_path / "m", device="cpu")
    chunks = []
    answer = lm.chat([("user", "Write add")], max_new_tokens=10, on_text=chunks.append)
    assert isinstance(answer, str) and answer == "".join(chunks).strip()
    assert EOT not in answer and END not in answer
    assert isinstance(lm.complete("def add(", max_new_tokens=5), str)


def test_ycode_lm_cli_end_to_end(tmp_path, capsys):
    from ycode.lm import cli

    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        (src / f"m{i}.py").write_text(CODE * 3)
    data, base, chat = tmp_path / "data", tmp_path / "base", tmp_path / "chat"
    assert cli.main(["prepare", "--source", str(src), "--out", str(data), "--vocab-size", "400"]) == 0
    assert cli.main(["train", "--data", str(data), "--out", str(base), "--preset", "tiny", "--context", "32",
                     "--steps", "4", "--batch-size", "2", "--device", "cpu"]) == 0
    assert cli.main(["sft", "--data", str(data), "--init-from", str(base), "--out", str(chat), "--steps", "2",
                     "--batch-size", "2", "--device", "cpu"]) == 0
    assert cli.main(["info", "--model", str(chat)]) == 0
    assert "Stage:      sft" in capsys.readouterr().out
    assert cli.main(["sample", "--model", str(base), "--prompt", "def", "--max-tokens", "3"]) == 0
    assert cli.main(["info", "--model", str(tmp_path / "missing")]) == 1
