"""One fine-tuning round's dataset: the production mix, cleaned, with a manifest.

``write_round_dataset`` builds the same mix as the production loop (basics, compose, contrast and
novel exercises plus replay from the original SFT data), then:

* always drops rows whose answer computes a benchmark or dev item (``datacheck``,
  ``ycode.lm.leakage``), generated and replayed alike;
* optionally drops exact duplicates and contradictory answers (``dedup``, ``contradictions``);
* writes ``hygiene.json`` (every removed row and why) and ``manifest.json`` (seed, options, code
  version, counts, SHA-256 of every data file), so a run can be traced to the exact data it saw.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

DEFAULTS = {
    "skills": False,          # the skill families (compose.INCLUDE_SKILLS)
    "basics_write": 6, "basics_bugfix": 4, "compose_write": 3, "compose_bugfix": 1, "contrast": 1,
    "novel_write": 1, "novel_draws": 6, "novel_ordinals": 4,
    "replay": 200,            # rows replayed from the original instruction data
    "dedup": False,           # drop exact (prompt, answer) duplicates
    "contradictions": False,  # drop answers that compute something else than another answer to the same task
    "spec_first": False,      # written functions restate the task in a docstring before the code (basics.SPEC_FIRST)
    "idioms_write": 0,        # write examples per idiom-curriculum program (ycode.lm.idioms); 0 = none
    "idioms_bugfix": 0,       # bug-fix examples per idiom-curriculum program
    "idioms_contrast": 0,     # plausible-but-wrong repairs per program (a sibling program's body under its docstring)
    "idioms_io": 0.0,         # share of idiom write prompts with worked input-output examples (one a boundary case)
    "idioms_families": "v3",  # curriculum family set (ycode.lm.idioms.FAMILY_SETS); runs pin the set they used
    "idioms_pairs": 0,        # V7: write examples per curriculum program that has a sibling (idioms.pair_examples)
    "pair_weight": 0,         # V7: loss weight of the answer tokens that differ from the sibling's code; 0 = usual weights
    "idioms_select": None,    # V7: None = every family of idioms_families; "k_families" = only those parameterised by a constant;
                              # V8: "v8_transfer" = the training programs of ycode/lm/v8_transfer_split.json;
                              # "v8_comp_train" / "v8_comp_more" = the composition arms (ycode/lm/v8_comp_split.json);
                              # V9: "v9_grammar_train" = grammar v2's training programs (ycode/lm/v9_grammar_split.json);
                              # V10: "v10_grammar3_train" = grammar v3's (ycode/lm/v10_grammar3_split.json);
                              # "v10_grammars_2_3" = both
    # V8: replay from round 54's own training distribution (one production round: the legacy counts above at
    # their defaults and 200 replay rows), sampled to this share of the dataset's tokens and added to the
    # examples the other options build; 0 = off. The share reached is in the manifest (tokens_by_source).
    "legacy_share": 0.0,
}

PRODUCTION = {k: DEFAULTS[k] for k in ("basics_write", "basics_bugfix", "compose_write", "compose_bugfix", "contrast",
                                       "novel_write", "replay")}


def round_examples(seed: int, options: dict):
    """The generated examples of one round (no replay), built exactly like the production mix."""
    return round_examples_tagged(seed, options)[0]


def round_examples_tagged(seed: int, options: dict) -> tuple[list, list[str]]:
    """``round_examples`` and a source tag per example: "legacy:write" / "legacy:bugfix" for the
    production mix's examples, "curriculum:<family>:<program>" for curriculum examples (by behaviour)."""
    o = {**DEFAULTS, **options}
    legacy = _legacy_examples(seed, o)
    curriculum, ctags = _idiom_examples_tagged(seed, o)
    return legacy + curriculum, [_legacy_tag(e) for e in legacy] + ctags


def _legacy_tag(ex) -> str:
    return "legacy:bugfix" if ex.prompt.startswith("This function has a bug") else "legacy:write"


def _legacy_examples(seed: int, o: dict) -> list:
    import ycode.lm.basics as basics
    import ycode.lm.compose as compose
    from ycode.lm.basics import basics_examples

    saved = compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS, basics.SPEC_FIRST
    compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS = o["skills"], o["novel_draws"], o["novel_ordinals"]
    basics.SPEC_FIRST = o["spec_first"]
    try:
        return (basics_examples(seed, write_per_concept=o["basics_write"], bugfix_per_concept=o["basics_bugfix"])
                + compose.compose_examples(seed, per_exercise=o["compose_write"])
                + compose.compose_bugfix_examples(seed, per_exercise=o["compose_bugfix"])
                + compose.contrast_examples(seed, per_exercise=o["contrast"])
                + compose.novel_examples(seed, per_exercise=o["novel_write"]))
    finally:
        compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS, basics.SPEC_FIRST = saved


def _idiom_examples(seed: int, o: dict):
    return _idiom_examples_tagged(seed, o)[0]


def _curriculum_programs(o: dict):
    from ycode.lm import idioms

    if o["idioms_select"] == "k_families":
        return idioms.k_family_programs(o["idioms_families"])
    if o["idioms_select"] == "v8_transfer":  # the training programs of V8's transfer split, with their aliases
        split = json.loads((Path(__file__).with_name("v8_transfer_split.json")).read_text())
        keys = {(family, behaviour) for family, _name, behaviour in split["trained"]}
        clean = idioms.held_out_check(idioms.candidate_programs(family_list=idioms.families("v3")))[0]
        return [p for p in clean if (p.family, p.source(0, "f")) in keys]
    if o["idioms_select"] in ("v8_comp_train", "v8_comp_more"):  # V8 stage 3: the composition arms' programs
        from ycode.lm.compositions import composition_programs

        split = json.loads((Path(__file__).with_name("v8_comp_split.json")).read_text())
        wanted = set(split["train" if o["idioms_select"] == "v8_comp_train" else "more"])
        return [p for p in composition_programs()[0] if p.name in wanted]
    if o["idioms_select"] == "v9_grammar_train":  # V9-1: the split holds the programs that passed the hygiene checks
        from ycode.lm.compositions2 import all_combinations

        split = json.loads((Path(__file__).with_name("v9_grammar_split.json")).read_text())
        wanted = set(split["train"])
        return [p for p in all_combinations() if p.name in wanted]
    if o["idioms_select"] == "v10_grammars_2_3":  # V10 R6: grammar v2's and grammar v3's training programs together
        return (_curriculum_programs({**o, "idioms_select": "v9_grammar_train"})
                + _curriculum_programs({**o, "idioms_select": "v10_grammar3_train"}))
    if o["idioms_select"] == "v10_grammar3_train":  # V10 R2: grammar v3's training programs
        from ycode.lm.compositions3 import all_combinations as all3

        split = json.loads((Path(__file__).with_name("v10_grammar3_split.json")).read_text())
        wanted = set(split["train"])
        return [p for p in all3() if p.name in wanted]
    if o["idioms_select"] is not None:
        raise ValueError(f"unknown idioms_select {o['idioms_select']!r}")
    return idioms.held_out_check(idioms.candidate_programs(family_list=idioms.families(o["idioms_families"])))[0]


def _idiom_examples_tagged(seed: int, o: dict):
    import re

    from ycode.lm.leakage import normalized_body

    out = []
    if not (o["idioms_write"] or o["idioms_bugfix"] or o["idioms_contrast"] or o["idioms_pairs"]):
        return [], []
    programs = _curriculum_programs(o)
    if o["idioms_write"] or o["idioms_bugfix"] or o["idioms_contrast"]:
        from ycode.lm.idioms import idiom_examples

        out += idiom_examples(seed, write_per_program=o["idioms_write"], bugfix_per_program=o["idioms_bugfix"],
                              contrast_per_program=o["idioms_contrast"], io_fraction=o["idioms_io"],
                              programs=programs, family_set=o["idioms_families"])
    if o["idioms_pairs"] and o["idioms_select"] == "v9_grammar_train":  # V10 R3: minimal pairs within grammar v2
        from ycode.lm.compositions2 import minimal_pair_examples

        out += minimal_pair_examples(seed, programs, per_program=o["idioms_pairs"], weight=o["pair_weight"])
    elif o["idioms_pairs"]:
        from ycode.lm.idioms import pair_examples

        out += pair_examples(seed, per_program=o["idioms_pairs"], weight=o["pair_weight"], family_set=o["idioms_families"])
    behaviour: dict[str, str] = {}
    for p in programs:  # every implementation of a program maps to its behaviour's first name
        for i in range(len(p.bodies)):
            behaviour.setdefault(normalized_body(p.source(i)), f"curriculum:{p.family}:{p.name}")
    tags = []
    for e in out:
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S)
        tags.append(behaviour.get(normalized_body(code.group(1)) if code else "", "curriculum:other"))
    return out, tags


def _legacy_sample(seed: int, o: dict, own_tokens: int, tokenizer_path, replay_dir, log) -> tuple[list, list[str], int]:
    """Examples and replay rows sampled from one production round (PRODUCTION) so that they make up
    about ``o["legacy_share"]`` of the dataset's tokens: (examples, tags, replay rows)."""
    import random

    import numpy as np

    from ycode.lm.data import encode_example
    from ycode.lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(tokenizer_path)
    production = _legacy_examples(seed, {**o, **PRODUCTION})
    lengths = [len(encode_example(tok, e, weighted=True)[0]) for e in production]
    replay_mean = 0.0
    if replay_dir:
        index = np.load(Path(replay_dir) / "sft_index.npy")
        short = index[index[:, 1] <= 400][:, 1]  # write_basics_dataset replays rows of at most 400 tokens
        replay_mean = float(short.mean()) if len(short) else 0.0
    round_tokens = sum(lengths) + PRODUCTION["replay"] * replay_mean
    share = o["legacy_share"]
    target = share / (1 - share) * own_tokens
    fraction = min(1.0, target / round_tokens)
    rng = random.Random(seed * 7919 + 17)
    keep = sorted(rng.sample(range(len(production)), round(fraction * len(production))))
    replay_rows = round(fraction * PRODUCTION["replay"])
    log(f"Legacy sample: {len(keep)} of {len(production)} production examples and {replay_rows} replay rows "
        f"({fraction:.1%} of a production round) for a {share:.0%} token share")
    picked = [production[i] for i in keep]
    return picked, [_legacy_tag(e) for e in picked], replay_rows


def replay_held_out_filter():
    """A ``write_basics_dataset`` replay filter dropping replayed functions that compute a held-out
    item (checked in a subprocess: replayed code is library code, not ours)."""
    from ycode.lm.leakage import answer_functions, behaviour_leaks, is_shared_operation

    def check(prompt: str, answer: str) -> str | None:
        functions = answer_functions([answer])
        if not functions:
            return None
        for lk in behaviour_leaks(functions, isolate=True, near=1.0):
            if not is_shared_operation(lk.source, lk.item):
                return "held-out equivalent"
        return None

    return check


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ":/ycode"], cwd=Path(__file__).resolve().parent,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def source_group(tag: str) -> str:
    """curriculum / legacy / replay (the generated examples of an untagged dataset: generated)."""
    return tag.split(":", 1)[0]


def tokens_by_source(data_dir) -> dict:
    """Examples, tokens and answer tokens of a written dataset by source (sft_tags.json)."""
    import numpy as np

    data_dir = Path(data_dir)
    tags = json.loads((data_dir / "sft_tags.json").read_text())
    index = np.load(data_dir / "sft_index.npy")
    mask = np.fromfile(data_dir / "sft_mask.bin", dtype=np.uint8)
    out: dict[str, dict] = {}
    for tag, (start, length) in zip(tags, index):
        g = out.setdefault(source_group(tag), {"examples": 0, "tokens": 0, "answer_tokens": 0})
        g["examples"] += 1
        g["tokens"] += int(length)
        g["answer_tokens"] += int((mask[start + 1:start + length] > 0).sum())
    total = sum(g["tokens"] for g in out.values()) or 1
    for g in out.values():
        g["token_share"] = round(g["tokens"] / total, 4)
    return out


def exposure_summary(data_dir, exposures_path) -> dict:
    """What training actually consumed (the trainer's exposures.json) by source, and how often each
    curriculum behaviour was seen: example-equivalents = answer tokens trained / answer tokens of the
    example, summed over the behaviour's examples (a packed row cuts its last example short)."""
    import numpy as np

    data_dir = Path(data_dir)
    tags = json.loads((data_dir / "sft_tags.json").read_text())
    index = np.load(data_dir / "sft_index.npy")
    mask = np.fromfile(data_dir / "sft_mask.bin", dtype=np.uint8)
    seen = {int(i): v for i, v in json.loads(Path(exposures_path).read_text())["rows"]}
    by_source: dict[str, dict] = {}
    behaviours: dict[str, float] = {}
    for row, tag in enumerate(tags):
        start, length = index[row]
        answer = int((mask[start + 1:start + length] > 0).sum())
        tokens, answer_seen = seen.get(row, [0, 0])
        g = by_source.setdefault(source_group(tag), {"tokens": 0, "answer_tokens": 0, "examples_touched": 0})
        g["tokens"] += tokens
        g["answer_tokens"] += answer_seen
        g["examples_touched"] += int(tokens > 0)
        if tag.startswith("curriculum:"):
            behaviours[tag] = behaviours.get(tag, 0.0) + (answer_seen / answer if answer else 0.0)
    total = sum(g["tokens"] for g in by_source.values()) or 1
    for g in by_source.values():
        g["token_share"] = round(g["tokens"] / total, 4)
    vals = sorted(behaviours.values())
    return {"trained_tokens": total, "by_source": by_source,
            "curriculum_behaviours": len(vals),
            "exposures_per_behaviour": ({"mean": round(sum(vals) / len(vals), 2), "min": round(vals[0], 2),
                                         "median": round(vals[len(vals) // 2], 2), "max": round(vals[-1], 2),
                                         "never_seen": sum(v == 0 for v in vals)} if vals else None)}


def write_round_dataset(out_dir, seed: int, tokenizer_path, *, replay_dir=None, options: dict | None = None,
                        check_replay: bool = True, log=print) -> dict:
    """Write one round's SFT data to ``out_dir``; return the manifest (also written to manifest.json)."""
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.datacheck import clean_examples

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    o = {**DEFAULTS, **(options or {})}
    unknown = set(options or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown data options: {sorted(unknown)}")
    examples, tags = round_examples_tagged(seed, o)
    replay_rows = o["replay"]
    if o["legacy_share"]:
        from ycode.lm.data import encode_example
        from ycode.lm.tokenizer import BPETokenizer

        tok = BPETokenizer.load(tokenizer_path)
        own = sum(len(encode_example(tok, e, weighted=True)[0]) for e in examples)
        extra, extra_tags, extra_replay = _legacy_sample(seed, o, own, tokenizer_path, replay_dir, log)
        examples, tags, replay_rows = examples + extra, tags + extra_tags, replay_rows + extra_replay
    kept, removed = clean_examples(examples, dedup=o["dedup"], contradictions=o["contradictions"], held_out=True)
    dropped_index = {r.index for r in removed}
    kept_tags = [t for i, t in enumerate(tags) if i not in dropped_index]
    (out_dir / "hygiene.json").write_text(json.dumps(
        [{"index": r.index, "reason": r.reason, "detail": r.detail, "prompt": examples[r.index].prompt[:160]}
         for r in removed], indent=1))
    reasons: dict[str, int] = {}
    for r in removed:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    log(f"Generated {len(examples)} examples; removed {reasons or 'none'}; kept {len(kept)}")
    info = write_basics_dataset(out_dir, tokenizer_path, replay_dir=replay_dir if replay_rows else None,
                                replay=replay_rows, seed=seed, examples=kept, log=log,
                                replay_filter=replay_held_out_filter() if (check_replay and replay_rows) else None,
                                tags=kept_tags)
    manifest = {
        "seed": seed, "options": o, "code_version": _git_commit(),
        "generated": len(examples), "removed": reasons, "kept": len(kept), **info,
        "tokens_by_source": tokens_by_source(out_dir),
        "files": {name: _sha256(out_dir / name) for name in ("sft_tokens.bin", "sft_mask.bin", "sft_index.npy",
                                                              "tokenizer.json")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
