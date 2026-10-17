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
    "idioms_select": None,    # V7: None = every family of idioms_families; "k_families" = only those parameterised by a constant
}


def round_examples(seed: int, options: dict):
    """The generated examples of one round (no replay), built exactly like the production mix."""
    import ycode.lm.basics as basics
    import ycode.lm.compose as compose
    from ycode.lm.basics import basics_examples

    o = {**DEFAULTS, **options}
    saved = compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS, basics.SPEC_FIRST
    compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS = o["skills"], o["novel_draws"], o["novel_ordinals"]
    basics.SPEC_FIRST = o["spec_first"]
    try:
        return (basics_examples(seed, write_per_concept=o["basics_write"], bugfix_per_concept=o["basics_bugfix"])
                + compose.compose_examples(seed, per_exercise=o["compose_write"])
                + compose.compose_bugfix_examples(seed, per_exercise=o["compose_bugfix"])
                + compose.contrast_examples(seed, per_exercise=o["contrast"])
                + compose.novel_examples(seed, per_exercise=o["novel_write"])
                + _idiom_examples(seed, o))
    finally:
        compose.INCLUDE_SKILLS, compose.NOVEL_DRAWS, compose.NOVEL_ORDINALS, basics.SPEC_FIRST = saved


def _idiom_examples(seed: int, o: dict):
    out = []
    if o["idioms_write"] or o["idioms_bugfix"] or o["idioms_contrast"]:
        from ycode.lm.idioms import idiom_examples

        programs = None
        if o["idioms_select"] == "k_families":
            from ycode.lm.idioms import k_family_programs

            programs = k_family_programs(o["idioms_families"])
        elif o["idioms_select"] is not None:
            raise ValueError(f"unknown idioms_select {o['idioms_select']!r}")
        out += idiom_examples(seed, write_per_program=o["idioms_write"], bugfix_per_program=o["idioms_bugfix"],
                              contrast_per_program=o["idioms_contrast"], io_fraction=o["idioms_io"],
                              programs=programs, family_set=o["idioms_families"])
    if o["idioms_pairs"]:
        from ycode.lm.idioms import pair_examples

        out += pair_examples(seed, per_program=o["idioms_pairs"], weight=o["pair_weight"], family_set=o["idioms_families"])
    return out


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
    examples = round_examples(seed, o)
    kept, removed = clean_examples(examples, dedup=o["dedup"], contradictions=o["contradictions"], held_out=True)
    (out_dir / "hygiene.json").write_text(json.dumps(
        [{"index": r.index, "reason": r.reason, "detail": r.detail, "prompt": examples[r.index].prompt[:160]}
         for r in removed], indent=1))
    reasons: dict[str, int] = {}
    for r in removed:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    log(f"Generated {len(examples)} examples; removed {reasons or 'none'}; kept {len(kept)}")
    info = write_basics_dataset(out_dir, tokenizer_path, replay_dir=replay_dir if o["replay"] else None,
                                replay=o["replay"], seed=seed, examples=kept, log=log,
                                replay_filter=replay_held_out_filter() if (check_replay and o["replay"]) else None)
    manifest = {
        "seed": seed, "options": o, "code_version": _git_commit(),
        "generated": len(examples), "removed": reasons, "kept": len(kept), **info,
        "files": {name: _sha256(out_dir / name) for name in ("sft_tokens.bin", "sft_mask.bin", "sft_index.npy",
                                                              "tokenizer.json")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
