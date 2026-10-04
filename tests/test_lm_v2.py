"""Tests for YCode-LM v2: GQA/QK-norm, v1 compatibility, schedule, packing, annealing,
new synthetic data, parallel encoding and the evaluation harness."""

import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from ycode.lm import evaluate  # noqa: E402
from ycode.lm.data import (  # noqa: E402
    extract_python_examples,
    inject_bug,
    parallel_encode,
    strip_license_header,
)
from ycode.lm.model import GPT, PRESETS, GPTConfig  # noqa: E402
from ycode.lm.tokenizer import BPETokenizer  # noqa: E402
from ycode.lm.train import (  # noqa: E402
    PretrainData,
    SFTData,
    TrainConfig,
    in_decay_phase,
    load_checkpoint,
    lr_at,
    save_checkpoint,
    train,
)

from tests.test_lm import CODE  # noqa: E402


def v2_model(block=64):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=block, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                         qk_norm=True, arch_version=2))


# ------------------------------------------------------------------ model


def test_gqa_kv_cache_matches_full_recompute():
    model = v2_model().eval()
    prompt = torch.randint(0, 300, (1, 9))
    cached = model.generate(prompt, 10, temperature=0)
    seq = prompt
    with torch.no_grad():
        for _ in range(10):
            logits, _ = model(seq, seq)
            seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, seq)


def test_gqa_cache_stores_only_kv_heads():
    model = v2_model().eval()
    caches = [dict() for _ in model.blocks]
    model(torch.randint(0, 300, (1, 5)), caches=caches)
    assert caches[0]["k"].shape == (1, 2, 5, 8)  # 2 KV heads, not 4


def test_v2_has_fewer_attention_params_than_v1():
    v1 = GPT(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=4, n_embd=32))
    assert v2_model().num_params() < v1.num_params()


def test_v1_checkpoints_still_load(tmp_path):
    tok = BPETokenizer.train([CODE * 5], vocab_size=300)
    v1 = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=16))
    save_checkpoint(tmp_path, v1, tok, step=1, stage="pretrain", val_loss=None)
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    for key in ("n_kv_head", "qk_norm", "arch_version"):  # simulate a checkpoint written by v1 code
        payload["config"].pop(key)
    torch.save(payload, tmp_path / "model.pt")
    loaded, _, _ = load_checkpoint(tmp_path)
    assert loaded.cfg.arch_version == 1 and loaded.blocks[0].attn.q_norm is None
    x = torch.randint(0, tok.vocab_size, (1, 6))
    assert torch.allclose(loaded(x, x)[0], v1(x, x)[0])


def test_v2_presets_are_valid():
    for name, preset in PRESETS.items():
        if name.startswith("v2"):
            cfg = GPTConfig(vocab_size=512, **preset)
            assert cfg.arch_version == 2 and cfg.n_head % cfg.kv_heads == 0


# --------------------------------------------------------------- schedule


def test_wsd_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100, decay_frac=0.2)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(50, cfg) == pytest.approx(1e-3)  # stable phase
    assert lr_at(79, cfg) == pytest.approx(1e-3)
    assert lr_at(90, cfg) < 1e-3
    assert lr_at(100, cfg) == pytest.approx(1e-4)
    assert not in_decay_phase(cfg, 0.5) and in_decay_phase(cfg, 0.85)


def test_time_based_progress_forces_decay():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=0, max_steps=10_000)
    # Only 1% of steps done, but 99% of the time budget used: LR must already be near min.
    assert lr_at(100, cfg, progress=0.99) < 2e-4


def test_cosine_schedule_still_available():
    cfg = TrainConfig(schedule="cosine", lr=1e-3, min_lr=0.0, warmup_steps=0, max_steps=100)
    assert lr_at(50, cfg) == pytest.approx(5e-4, rel=1e-3)


# ------------------------------------------------------------------- data


def test_inject_bug_is_a_single_line_change():
    src = "def area(w, h):\n    if w < 0:\n        return 0\n    return w * h + 0\n"
    for seed in range(10):
        bug = inject_bug(src, random.Random(seed))
        assert bug is not None
        buggy, bad, good = bug
        assert buggy != src and bad != good
        assert good in __import__("ast").unparse(__import__("ast").parse(src))


def test_inject_bug_without_candidates():
    assert inject_bug("def f(x):\n    return x\n", random.Random(0)) is None


def test_new_instruction_tasks_present():
    rng = random.Random(0)
    examples = []
    for _ in range(5):
        examples += extract_python_examples(CODE + "\n\nclass Box:\n    \"\"\"Store one value for later use.\"\"\"\n"
                                            "    def get(self):\n        return self.v\n", rng)
    prompts = " ".join(e.prompt for e in examples)
    assert "fix" in prompts.lower()
    assert "Complete this" in prompts or "Fill in the body" in prompts
    assert "docstring" in prompts
    assert "class" in prompts


def test_strip_license_header():
    text = "# Copyright 2020 X\n# Licensed under MIT\n# more\n# lines\n\nimport os\n"
    assert strip_license_header(text).strip() == "import os"
    keep = "# helper module\nimport os\n"
    assert strip_license_header(keep) == keep


def test_parallel_encode_matches_serial():
    tok = BPETokenizer.train([CODE * 5], vocab_size=300)
    texts = [CODE.replace("add", f"add{i}") for i in range(12)]
    assert parallel_encode(tok, texts, workers=2) == [tok.encode(t, allow_special=False) for t in texts]


def test_sft_packing_fills_rows(data_dir):
    packed = SFTData(data_dir, 64, 4, "cpu", pad_id=0, pack=True)
    x, y, m = packed.batch("train", torch.Generator().manual_seed(0))
    assert x.shape == (4, 64) and m.sum() > 0
    padded = SFTData(data_dir, 64, 4, "cpu", pad_id=0, pack=False)
    x2, _, _ = padded.batch("train", torch.Generator().manual_seed(0))
    assert x2.shape[1] <= 64


def test_anneal_mix_uses_instruction_stream(data_dir):
    data = PretrainData(data_dir, 32, 4, "cpu")
    x, y, _ = data.batch("train", torch.Generator().manual_seed(0), mix=0.5)
    assert x.shape == (4, 32) and torch.equal(x[:, 1:], y[:, :-1])


def test_v2_training_end_to_end(data_dir, tmp_path):
    cfg = TrainConfig(preset="v2-tiny", model_overrides={"block_size": 32}, batch_size=4, max_steps=30,
                      warmup_steps=3, eval_interval=30, eval_iters=2, log_interval=1000, device="cpu",
                      precision="fp32", lr=3e-3)
    logs = []
    train(data_dir, tmp_path / "base", cfg, log=logs.append)
    assert any("YCode-LM v2" in line for line in logs)
    assert any("annealing" in line for line in logs)
    model, _, payload = load_checkpoint(tmp_path / "base")
    assert payload["version"] == 2 and model.cfg.n_kv_head == 2


# ------------------------------------------------------------- evaluation

REFERENCE = {
    "add": "def add(a, b):\n    return a + b",
    "subtract": "def subtract(a, b):\n    return a - b",
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "square": "def square(x):\n    return x * x",
    "maximum": "def maximum(items):\n    return max(items)",
    "minimum": "def minimum(items):\n    return min(items)",
    "total": "def total(numbers):\n    return sum(numbers)",
    "average": "def average(numbers):\n    return sum(numbers) / len(numbers)",
    "reverse_string": "def reverse_string(s):\n    return s[::-1]",
    "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
    "factorial": "def factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)",
    "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "count_vowels": "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s)",
    "is_prime": "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))",
    "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "celsius_to_fahrenheit": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "filter_even": "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]",
    "unique": "def unique(items):\n    return list(dict.fromkeys(items))",
    "count_words": "def count_words(text):\n    return len(text.split())",
    "flatten": "def flatten(lists):\n    return [x for l in lists for x in l]",
    "clamp": "def clamp(x, low, high):\n    return max(low, min(x, high))",
    "char_count": "def char_count(s, ch):\n    return s.count(ch)",
    "merge_dicts": "def merge_dicts(a, b):\n    return {**a, **b}",
    "first_word": "def first_word(text):\n    return text.split()[0]",
    "to_upper": "def to_upper(s):\n    return s.upper()",
    "is_sorted": "def is_sorted(items):\n    return items == sorted(items)",
    "second_largest": "def second_largest(numbers):\n    return sorted(set(numbers))[-2]",
    "binary_search": ("def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
                      "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
                      "        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
                      "    return -1"),
    "is_anagram": "def is_anagram(a, b):\n    return sorted(a) == sorted(b)",
    "digit_sum": "def digit_sum(n):\n    return sum(int(d) for d in str(n))",
}


def test_every_problem_has_a_working_reference():
    assert set(REFERENCE) == {p.name for p in evaluate.PROBLEMS}
    for problem in evaluate.PROBLEMS:
        assert evaluate.run_tests(REFERENCE[problem.name], problem.tests), problem.name


def test_wrong_and_hanging_solutions_fail():
    problem = evaluate.PROBLEMS[0]
    assert not evaluate.run_tests("def add(a, b):\n    return a - b", problem.tests)
    assert not evaluate.run_tests("def add(a, b):\n    while True: pass", problem.tests, timeout=1)
    assert not evaluate.run_tests("this is not python", problem.tests)


def test_extract_code():
    assert evaluate.extract_code("Sure:\n```python\ndef f():\n    return 1\n```\nDone") == "def f():\n    return 1"
    assert evaluate.extract_code("Here: def f(): return 2") == "def f(): return 2"
    assert evaluate.extract_code("```python\ndef g():\n    pass") == "def g():\n    pass"  # unterminated


def test_pass_at_k():
    assert evaluate.pass_at_k(5, 0, 1) == 0
    assert evaluate.pass_at_k(5, 5, 1) == 1
    assert evaluate.pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert evaluate.pass_at_k(4, 1, 4) == 1


def test_functional_eval_with_perfect_model():
    class Oracle:
        def chat(self, turns, *, max_new_tokens, temperature, on_text=None):
            name = turns[-1][1].split("`")[1].split("(")[0]
            return f"```python\n{REFERENCE[name]}\n```"

    result = evaluate.functional_eval(Oracle(), samples=2)
    assert result.pass_at_1 == 1.0 and result.pass_at_k == 1.0 and len(result.solved) == len(evaluate.PROBLEMS)


def test_bits_per_byte_untrained_vs_trained(data_dir, tmp_path):
    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    random_model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=16))
    bpb_random = evaluate.bits_per_byte(random_model, tok, [CODE])
    cfg = TrainConfig(preset="v2-tiny", model_overrides={"block_size": 32}, batch_size=8, max_steps=60,
                      warmup_steps=5, eval_interval=60, eval_iters=1, log_interval=1000, device="cpu",
                      precision="fp32", lr=3e-3)
    train(data_dir, tmp_path / "m", cfg, log=lambda *_: None)
    trained, tok2, _ = load_checkpoint(tmp_path / "m")
    assert evaluate.bits_per_byte(trained, tok2, [CODE]) < bpb_random


def test_eval_cli_compares_models(data_dir, tmp_path, capsys):
    from ycode.lm import cli

    cfg = TrainConfig(preset="v2-tiny", model_overrides={"block_size": 32}, batch_size=2, max_steps=2,
                      warmup_steps=1, eval_interval=2, eval_iters=1, log_interval=1000, device="cpu",
                      precision="fp32")
    train(data_dir, tmp_path / "a", cfg, log=lambda *_: None)
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "h.py").write_text(CODE)
    small = evaluate.PROBLEMS[:2]
    original = evaluate.PROBLEMS
    evaluate.PROBLEMS = small
    try:
        assert cli.main(["eval", "--model", str(tmp_path / "a"), "--heldout", str(heldout), "--device", "cpu"]) == 0
    finally:
        evaluate.PROBLEMS = original
    out = capsys.readouterr().out
    assert "bits_per_byte" in out and "pass@1" in out
