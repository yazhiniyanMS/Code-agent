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
    monkeypatch.setattr(rounddata, "round_examples", lambda seed, options: small + small[1:])
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
