"""V12: curriculum examples are executed before training; a wrong target stops the dataset build."""
import dataclasses

import pytest

from ycode.lm import rounddata

R1_OPTIONS = {"idioms_families": "v3", "idioms_select": "v9_grammar_train", "idioms_write": 2, "basics_write": 0,
              "basics_bugfix": 0, "compose_write": 0, "compose_bugfix": 0, "contrast": 0, "novel_write": 0,
              "replay": 0}


@pytest.fixture(scope="module")
def curriculum():
    o = {**rounddata.DEFAULTS, **R1_OPTIONS}
    examples, tags = rounddata.round_examples_tagged(1302, o)
    return examples, tags, rounddata._curriculum_programs(o)


def test_every_generated_curriculum_answer_computes_its_program(curriculum):
    examples, tags, programs = curriculum
    v = rounddata.validate_curriculum(examples, tags, programs)
    assert v["checked"] == 700 and v["passed"] == 700 and not v["failures"]  # 350 programs x 2 examples


def test_a_wrong_or_broken_target_is_caught(curriculum):
    from ycode.lm.idioms import _run

    examples, tags, programs = curriculum
    by_name = {p.name: p for p in programs}
    i, j = [k for k, t in enumerate(tags) if t.startswith("curriculum:")][:2]
    p = by_name[tags[i].rsplit(":", 1)[-1]]
    want = repr(_run(p.source(0), p.name, list(p.inputs)))
    # a sibling program with the same parameters that computes something else on p's inputs
    q = next(q for q in programs if q.params == p.params and q.name != p.name
             and repr(_run(q.source(0, p.name), p.name, list(p.inputs))) != want)
    wrong = list(examples)
    wrong[i] = dataclasses.replace(examples[i], response=f"```python\n{q.source(0, p.name)}\n```")
    wrong[j] = dataclasses.replace(examples[j], response="```python\ndef broken(:\n    pass\n```")
    v = rounddata.validate_curriculum(wrong, tags, programs)
    reasons = {f["index"]: f["reason"] for f in v["failures"]}
    assert reasons == {i: "computes something else", j: "raises or times out"}
    assert v["passed"] == v["checked"] - 2


def test_dataset_build_refuses_a_wrong_target(curriculum, data_dir, tmp_path, monkeypatch):
    examples, tags, _ = curriculum
    i = next(k for k, t in enumerate(tags) if t.startswith("curriculum:"))
    bad = list(examples)
    bad[i] = dataclasses.replace(examples[i], response="```python\ndef f(x):\n    raise ValueError\n```")
    monkeypatch.setattr(rounddata, "round_examples_tagged", lambda seed, o: (bad, tags))
    with pytest.raises(ValueError, match="do not compute their program"):
        rounddata.write_round_dataset(tmp_path / "round", 1302, data_dir / "tokenizer.json", options=R1_OPTIONS,
                                      log=lambda *_: None)
    assert not (tmp_path / "round" / "sft_tokens.bin").exists()
