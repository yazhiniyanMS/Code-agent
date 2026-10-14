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
