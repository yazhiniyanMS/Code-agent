"""The fine-tuning recipe: held-out dev set, cue-free bugs, loss weights, ordered sampling, answer logs."""

import json
import random
import re

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from ycode.lm import evaluate  # noqa: E402
from ycode.lm.basics import NEUTRAL_NAMES, _passes, held_out_names, neutralize  # noqa: E402
from ycode.lm.data import DIFF_WEIGHT, InstructionExample, encode_example, inject_bug  # noqa: E402
from ycode.lm.devset import DEV_CONCEPT_NAMES, DEV_KEYS, dev_bugs, dev_problems, verify_devset  # noqa: E402


def test_dev_set_is_verified_and_never_trained_on():
    from ycode.lm.basics import basics_examples
    from ycode.lm.compose import HELD_OUT_KEYS, all_exercises, base_key

    verify_devset()
    assert len(dev_problems()) >= 40 and len(dev_bugs()) == 15
    assert not DEV_KEYS & HELD_OUT_KEYS
    exercises = all_exercises(3)
    assert not {base_key(e.key) for e in exercises} & DEV_KEYS
    dev = {p.name for p in dev_problems()}
    text = " ".join(e.prompt + e.response for e in basics_examples(3, write_per_concept=1, bugfix_per_concept=2))
    assert not {n for n in DEV_CONCEPT_NAMES if re.search(rf"(?:def |`){n}\(", text)}
    assert not {e.name for e in exercises} & (held_out_names() | dev)


def test_spliced_bugs_change_only_the_bug():
    src = ('def f(xs, k):\n    """Doc."""\n    total = 0\n    for x in xs:\n'
           '        if x > k and (x % 2 == 0):\n            total += sum(1 for _ in range(x)) * 2\n    return total')
    kinds = set()
    for seed in range(200):
        bug = inject_bug(src, random.Random(seed))
        if bug is None:
            continue
        buggy, wrong, right = bug
        changed = [(b, o) for b, o in zip(buggy.splitlines(), src.splitlines()) if b != o]
        assert len(changed) == 1 and (changed[0][0].strip(), changed[0][1].strip()) == (wrong, right)
        assert buggy.count("(") == src.count("(")  # no reformatting gives the bug away
        kinds.add(wrong)
    assert any("x < k" in k for k in kinds)  # comparison direction flips
    assert any("-=" in k for k in kinds)  # augmented assignment swaps
    assert any(k.startswith("return ") and k != "return total" for k in kinds)  # wrong variable returned


def test_new_mutations_make_real_bugs():
    src = 'def g(items):\n    best = items[0]\n    for x in items:\n        if x > best:\n            best = x\n    return best'
    tests = "assert g([3, 9, 2]) == 9\nassert g([-1, -5]) == -1"
    assert _passes(src, tests)
    flips = {inject_bug(src, random.Random(s))[0] for s in range(100)} - {None}
    assert any("x < best" in b and not _passes(b, tests) for b in flips)


def test_bug_fix_loss_weights_target_the_change(data_dir):
    from ycode.lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    fixed = 'def h(x):\n    """Return x squared."""\n    return x * x'
    buggy = fixed.replace("x * x", "x * 2")
    ex = InstructionExample(evaluate.BUGFIX_PROMPT.format(code=buggy), f"```python\n{fixed}\n```")
    ids, mask = encode_example(tok, ex, weighted=True)
    plain_ids, plain = encode_example(tok, ex)
    assert ids == plain_ids and set(plain) == {0, 1}
    prompt_len = plain.index(1)
    assert set(mask[:prompt_len]) == {0}
    heavy = tok.decode([t for t, w in zip(ids, mask) if w == DIFF_WEIGHT])
    assert "x" in heavy and "2" not in heavy and "squared" not in heavy


def test_neutral_names_rename_everywhere_and_never_clash():
    ex = InstructionExample("Write a Python function `twice(n)` that returns twice n.",
                            "```python\ndef twice(n):\n    return 2 * n if n else twice(1)\n```")
    out = neutralize(ex, "twice", random.Random(0), p=1.0)
    new = re.search(r"`(\w+)\(", out.prompt).group(1)
    assert new.split("_")[0] in NEUTRAL_NAMES and "twice(" not in out.prompt + out.response
    assert "returns twice n" in out.prompt  # the word in the task text stays
    assert out.response.count(f"{new}(") == 2
    assert not set(NEUTRAL_NAMES) & held_out_names()


def test_number_words_reach_the_task_text():
    from ycode.lm.compose import all_exercises

    tasks = " ".join(e.task for e in all_exercises(5))
    assert re.search(r"\b(three|four|seven) times x\b|\bx plus (two|five|ten)\b", tasks)
    assert "has_1_items" not in " ".join(e.name for e in all_exercises(5))


def test_replay_filter_drops_benchmark_rows(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.data import write_sft_files
    from ycode.lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    old = tmp_path / "old"
    old.mkdir()
    rows = [InstructionExample("Write a Python function `is_prime(n)` that returns True if n is prime.",
                               "```python\ndef is_prime(n):\n    return n > 1\n```"),
            InstructionExample(evaluate.BUGFIX_PROMPT.format(code="def q(a):\n    return a - 1"),
                               "The bug is in `return a - 1`. It should be `return a + 1`."),
            InstructionExample("Write a function `tidy(s)` that strips s.", "```python\ndef tidy(s):\n    return s.strip()\n```")]
    write_sft_files(tok, rows, old)
    examples = [InstructionExample("Write a Python function `ok(x)` that returns x.", "```python\ndef ok(x):\n    return x\n```")]
    info = write_basics_dataset(tmp_path / "new", data_dir / "tokenizer.json", replay_dir=old, replay=3, seed=1,
                                examples=examples, log=lambda *_: None)
    assert info["replay"] == 1  # only `tidy` survives


def test_sampler_sees_every_example_once_per_epoch(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.train import SFTData

    examples = [InstructionExample(f"Write `f{i}(x)`.", f"```python\ndef f{i}(x):\n    return {i}\n```") for i in range(40)]
    write_basics_dataset(tmp_path / "d", data_dir / "tokenizer.json", examples=examples, log=lambda *_: None)
    data = SFTData(tmp_path / "d", 16, 1, "cpu", pad_id=0, pack=False, seed=3)
    order = list(data.epoch_order())
    seen = [data._next_train() for _ in range(len(order))]
    assert sorted(seen) == sorted(order) and data.epoch == 0
    state = data.state_dict()
    again = SFTData(tmp_path / "d", 16, 1, "cpu", pad_id=0, pack=False, seed=3)
    again.load_state_dict(state)
    assert again._next_train() == data._next_train()  # resumes into the next epoch identically


def test_mask_weights_scale_the_loss():
    from ycode.lm.model import GPT, PRESETS, GPTConfig

    model = GPT(GPTConfig(vocab_size=64, **{**PRESETS["v5-tiny"], "block_size": 16})).eval()
    x = torch.randint(0, 64, (1, 8))
    y = torch.randint(0, 64, (1, 8))
    ones = torch.ones(1, 8, dtype=torch.long)
    heavy = ones.clone()
    heavy[0, 3] = 8
    _, a = model(x, y, loss_mask=ones)
    _, b = model(x, y, loss_mask=heavy)
    logits, _ = model(x, y)
    per = torch.nn.functional.cross_entropy(logits.view(-1, 64), y.view(-1), reduction="none")
    assert torch.allclose(b, (per * heavy.view(-1)).sum() / heavy.sum())
    assert not torch.allclose(a, b)


def test_lion_rejects_mismatched_state_and_keeps_the_new_lr():
    from ycode.lm.optim import Lion

    p = torch.nn.Parameter(torch.zeros(3))
    opt = Lion([p], lr=0.1)
    p.grad = torch.ones(3)
    opt.step()
    other = Lion([torch.nn.Parameter(torch.zeros(4))], lr=0.5)
    with pytest.raises(ValueError):
        other.load_state_dict(opt.state_dict())


def test_training_without_eval_passes(data_dir, tmp_path):
    from ycode.lm.train import TrainConfig, train

    cfg = TrainConfig(stage="pretrain", preset="v2-tiny", model_overrides={"block_size": 32}, batch_size=2,
                      max_steps=2, warmup_steps=1, eval_interval=1, eval_iters=0, log_interval=1000, device="cpu",
                      precision="fp32")
    summary = train(data_dir, tmp_path / "m", cfg, log=lambda *_: None)
    assert summary["final"] == {"step": 2} and (tmp_path / "m" / "model.pt").is_file()


def test_bugfix_eval_logs_answers_and_copies():
    class Echo:
        def chat(self, turns, *, max_new_tokens, temperature, on_text=None):
            return "```python\n" + turns[0][1].split("```python\n", 1)[1].rsplit("```", 1)[0] + "```"

    answers = []
    rate, fixed = evaluate.bugfix_eval(Echo(), answers=answers)
    assert rate == 0 and fixed == [] and len(answers) == len(evaluate.BUGGY)
    assert all(a["copied"] and not a["pass"] for a in answers)


def test_eval_cli_dev_problems_and_answers_file(data_dir, tmp_path, monkeypatch, capsys):
    from ycode.lm import cli, devset
    from ycode.lm.train import TrainConfig, train

    cfg = TrainConfig(preset="v2-tiny", model_overrides={"block_size": 32}, batch_size=2, max_steps=1,
                      warmup_steps=1, eval_interval=1, eval_iters=1, log_interval=1000, device="cpu", precision="fp32")
    train(data_dir, tmp_path / "a", cfg, log=lambda *_: None)
    monkeypatch.setattr(devset, "dev_problems", lambda: dev_problems()[:2])
    monkeypatch.setattr(devset, "dev_bugs", lambda: dev_bugs()[:2])
    out_file = tmp_path / "answers.jsonl"
    assert cli.main(["eval", "--model", str(tmp_path / "a"), "--problems", "dev", "--device", "cpu",
                     "--answers-out", str(out_file)]) == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert [r["suite"] for r in rows] == ["write", "write", "bugfix", "bugfix"]
    last = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[{")][-1]
    result = json.loads(last)[0]
    assert "solved" in result and "fixed" in result and result["n_bugs"] == 2
