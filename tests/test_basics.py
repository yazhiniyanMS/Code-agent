"""Basic programming exercises for fine-tuning, and training only the top layers."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from ycode.lm.basics import (  # noqa: E402
    BUGFIX_PROMPT,
    CONCEPTS,
    FRESH_PROBLEMS,
    _parses,
    _passes,
    _rename_tests,
    assert_no_overlap,
    basics_examples,
    bugfix_examples,
    verify_concepts,
)
from ycode.lm.evaluate import BUGFIX_PROMPT as EVAL_BUGFIX_PROMPT  # noqa: E402
from ycode.lm.evaluate import BUGGY, PROBLEMS, extract_code, fresh_problems  # noqa: E402


def test_every_reference_solution_passes_its_tests():
    verify_concepts()
    assert len(CONCEPTS) >= 100


def test_training_concepts_never_reuse_an_evaluation_problem():
    assert_no_overlap()
    names = {c.name for c in CONCEPTS} | {a for c in CONCEPTS for a in c.aliases}
    assert not names & {p.name for p in PROBLEMS}
    assert not names & {sig.split("(")[0] for sig, _ in BUGGY}
    assert not names & {name for name, *_ in FRESH_PROBLEMS}


def test_fresh_problems_are_valid():
    problems = fresh_problems()
    assert len(problems) == 20
    assert all(p.prompt.startswith("Write a Python function `") for p in problems)


def test_bugfix_examples_are_realistic_and_correct(monkeypatch):
    monkeypatch.setattr("ycode.lm.basics.NEUTRAL_P", 0.0)  # keep the concept names, to look up the tests
    examples = bugfix_examples(random.Random(0))
    assert len(examples) > 150
    assert BUGFIX_PROMPT == EVAL_BUGFIX_PROMPT  # same task format as the benchmark (different functions)
    by_name = {}
    for c in CONCEPTS:
        for name in (c.name, *c.aliases):
            by_name[name] = c
    for ex in examples:
        buggy = ex.prompt.split("```python\n", 1)[1].rsplit("```", 1)[0]
        fixed = extract_code(ex.response)
        name = buggy.split("def ", 1)[1].split("(", 1)[0]
        tests = _rename_tests(by_name[name].tests, name)
        assert _parses(buggy), buggy  # valid Python...
        assert not _passes(buggy, tests), buggy  # ...that really is broken...
        assert _passes(fixed, tests), fixed  # ...and the answer fixes it


def test_infinite_loop_bugs_time_out():
    assert not _passes("def f(n):\n    while True:\n        n += 1", "f(1)", timeout=0.2)


def test_basics_examples_mix_writing_and_fixing():
    examples = basics_examples()
    fixes = [e for e in examples if e.prompt.startswith("This function has a bug")]
    writes = [e for e in examples if not e.prompt.startswith("This function has a bug")]
    assert len(writes) > 600 and len(fixes) > 150
    assert all(e.response.startswith("```python\ndef ") for e in writes)


def test_training_only_the_top_layers(data_dir, tmp_path):
    from ycode.lm.model import GPT, PRESETS, GPTConfig
    from ycode.lm.tokenizer import BPETokenizer
    from ycode.lm.train import TrainConfig, load_checkpoint, save_checkpoint, train

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    base = GPT(GPTConfig(vocab_size=tok.vocab_size, **{**PRESETS["v5-tiny"], "block_size": 32}))
    with torch.no_grad():  # like the real v5, start from bf16 weights (frozen layers are stored in bf16)
        for p in base.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    save_checkpoint(tmp_path / "base", base, tok, step=0, stage="pretrain", val_loss=None)
    cfg = TrainConfig(stage="sft", batch_size=2, max_steps=4, warmup_steps=1, eval_interval=100, eval_iters=1,
                      log_interval=1000, device="cpu", precision="fp32", optimizer="lion", lr=1e-3,
                      init_from=str(tmp_path / "base"), train_layers=2, grad_checkpoint=True)
    train(data_dir, tmp_path / "tuned", cfg, log=lambda *_: None)
    tuned, _, _ = load_checkpoint(tmp_path / "tuned")
    n = len(base.blocks)
    for i in range(n):
        same = all(torch.equal(a, b) for a, b in zip(base.blocks[i].parameters(), tuned.blocks[i].parameters()))
        assert same == (i < n - 2), f"block {i}"  # only the top 2 blocks changed
    assert torch.equal(base.embed.weight, tuned.embed.weight)
    stored = torch.load(tmp_path / "tuned" / "model.pt", weights_only=True)["model"]
    assert stored["embed.weight"].dtype == torch.bfloat16  # frozen: stored compactly
    assert stored[f"blocks.{n - 1}.mlp.down.weight"].dtype == torch.float32  # trained: full precision


def test_compositional_exercises_are_verified_and_hold_out_the_benchmark():
    from ycode.lm.compose import HELD_OUT_KEYS, all_exercises, compose_bugfix_examples, compose_examples

    exercises = all_exercises()
    assert len(exercises) > 500
    assert not {e.key for e in exercises} & HELD_OUT_KEYS
    held_names = {p.name for p in PROBLEMS} | {name for name, *_ in FRESH_PROBLEMS}
    assert not {e.name for e in exercises} & held_names
    for ex in exercises[::25]:
        tests = ex.tests()
        assert tests and _passes(ex.source(), tests)
    writes = compose_examples()
    assert all(w.response.startswith("```python\ndef ") for w in writes)
    fixes = compose_bugfix_examples()
    assert len(fixes) > 400 and all(f.response.startswith("```python\ndef ") for f in fixes[:50])


def test_bugfix_answers_are_code_only():
    for ex in bugfix_examples(random.Random(1))[:20]:
        assert ex.response.startswith("```python\ndef ") and "The bug is" not in ex.response
