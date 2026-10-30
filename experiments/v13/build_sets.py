"""V13 evaluation sets for string operations, in the V8 task-set format (experiments/v8/build_sets.py), scored by
experiments/v7/prompt_diagnostic.py: ``strict`` = the answer's function gives the reference's output on all 40
probe inputs; ``wrong`` = it computes the trained neighbour's function instead.

usage: build_sets.py            (writes experiments/v13/sets/{strings_trained,strings_heldout}.json)

* strings_trained: the 47 concepts of the V13 curriculum (ycode.lm.strings). Training reproduces them, so this
  set shows whether training took; it is reported, never used to promote a candidate.
* strings_heldout: 32 string tasks in the curriculum's groups that no training concept computes (HELD_OUT below):
  the measure of string-operation correctness. Each task's neighbour, and ``wrong``, is the trained string
  concept with as many parameters and the most shared description words whose outputs differ on the probes;
  the tests include the two shortest inputs where they differ, so ``wrong`` counts the trained function
  written in place of the asked one.

Probe inputs come from a fixed pool of 40 strings with the boundary cases string code gets wrong (empty and blank
strings, leading and trailing spaces, repeated characters, punctuation, every separator the tasks use, digits),
each paired with a character, word or position for tasks with more parameters. Inputs and outputs are stored as Python repr strings.
"""
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v7"))

from neighbors import run  # noqa: E402

from ycode.lm import strings  # noqa: E402
from ycode.lm.basics import _c  # noqa: E402

PROBES = 40
VERSION = "v13-strings-1"

POOL = ("", " ", "a", "aaa", "Ab", "  lead", "trail  ", "  both ends  ", "Hello, World!", "a,b,,c", " , ", "x.y.z",
        "...", "end.", ".start", "snake_case_name", "camelCaseName", "kebab-case-name", "--dash--", "MiXeD cAsE",
        "UPPER", "Title Case Words", "12abc34", "007", "a b  c", "one  two   three", "key = value", "name: Ada",
        "a:b:c", "report.final.pdf", "README", "archive.", "path/to/file.txt", "tab\there", "line1\nline2",
        "[tag] rest", "a|b| c |", "x; y;;z", "!?;:", "Mississippi")  # exactly 40, every separator in play
CHARS = (",", ".", " ", "a", "-", ":", "s")
WORDS = ("a", "ss", ",", " ", "-")

HELD_OUT = (
    # replace
    _c("periods_to_commas", "text", "returns text with every period replaced by a comma.",
       "return text.replace('.', ',')", ""),
    _c("spaces_to_plus", "text", "returns text with every space replaced by a plus sign.",
       "return text.replace(' ', '+')", ""),
    _c("collapse_double_dashes", "text", "returns text with each pair of consecutive dashes, read from the left, "
       "replaced by a single dash.", "return text.replace('--', '-')", ""),
    _c("replace_first_two", "text, old, new", "returns text with the first two occurrences of old replaced by new.",
       "return text.replace(old, new, 2)", ""),
    # split and join
    _c("join_words_with_underscores", "text", "returns the words of text joined with underscores.",
       "return '_'.join(text.split())", ""),
    _c("last_comma_part", "text", "returns the part of text after its last comma, without surrounding spaces "
       "(all of text without surrounding spaces if it has no comma).", "return text.split(',')[-1].strip()", ""),
    _c("count_dot_parts", "text", "returns how many non-empty parts text has when split at its dots.",
       "return len([part for part in text.split('.') if part])", ""),
    _c("split_on_pipes", "text", "returns the parts of text separated by '|' with surrounding spaces removed, "
       "leaving out empty parts.", "return [part.strip() for part in text.split('|') if part.strip()]", ""),
    # strip
    _c("strip_commas", "text", "returns text without the commas at its start and end.",
       "return text.strip(',')", ""),
    _c("strip_leading_dashes", "text", "returns text without the dashes at its start.",
       "return text.lstrip('-')", ""),
    _c("strip_leading_dots", "text", "returns text without the dots at its start.",
       "return text.lstrip('.')", ""),
    _c("strip_and_upper", "text", "returns text without the spaces at its start and end, in uppercase.",
       "return text.strip().upper()", ""),
    # character filtering and transformation
    _c("keep_uppercase", "text", "returns only the uppercase letters in text, in order.",
       "return ''.join(c for c in text if c.isupper())", ""),
    _c("vowels_to_stars", "text", "returns text with every vowel (a, e, i, o or u, either case) replaced by '*'.",
       "return ''.join('*' if c in 'aeiouAEIOU' else c for c in text)", ""),
    _c("remove_letters", "text", "returns text without its letters.",
       "return ''.join(c for c in text if not c.isalpha())", ""),
    _c("keep_punctuation", "text", "returns only the characters . , ! ? ; and : in text, in order.",
       "return ''.join(c for c in text if c in '.,!?;:')", ""),
    # substrings and indexing
    _c("first_half", "text", "returns the first half of text (the shorter half if its length is odd).",
       "return text[:len(text) // 2]", ""),
    _c("after_last_slash", "path", "returns the part of path after its last slash (all of path if it has none).",
       "return path.rsplit('/', 1)[-1]", ""),
    _c("find_last_char", "text, ch", "returns the position of the last occurrence of the character ch in text, "
       "or -1 if it does not occur.", "return text.rfind(ch)", ""),
    _c("text_in_brackets", "text", "returns the text between the first '[' in text and the next ']', or an empty "
       "string if there is no such pair.",
       "start = text.find('[')\nend = text.find(']', start + 1)\nreturn text[start + 1:end] if start >= 0 and end >= 0 "
       "else ''", ""),
    # case
    _c("upper_first_and_last", "text", "returns text with its first and last characters in uppercase.",
       "return text[:1].upper() + text[1:-1] + text[-1:].upper() if len(text) > 1 else text.upper()", ""),
    _c("uppercase_vowels", "text", "returns text with every lowercase vowel (a, e, i, o, u) in uppercase.",
       "return ''.join(c.upper() if c in 'aeiou' else c for c in text)", ""),
    _c("kebab_to_title", "name", "returns the dash-separated words of name, each capitalized, joined with spaces.",
       "return ' '.join(word.capitalize() for word in name.split('-') if word)", ""),
    _c("swap_case_of_first", "text", "returns text with the case of its first character swapped.",
       "return text[:1].swapcase() + text[1:]", ""),
    # compositions
    _c("hashtag", "text", "returns '#' followed by the words of text, each capitalized, with no spaces.",
       "return '#' + ''.join(word.capitalize() for word in text.split())", ""),
    _c("capitalized_words_with_underscores", "text", "returns the words of text, each capitalized, joined with "
       "underscores.", "return '_'.join(word.capitalize() for word in text.split())", ""),
    _c("first_letters_lower", "text", "returns the first letter of each word in text, in lowercase, joined "
       "together.", "return ''.join(word[0].lower() for word in text.split())", ""),
    _c("reverse_comma_parts", "text", "returns the comma-separated parts of text in reverse order, without "
       "surrounding spaces, joined with commas.", "return ','.join(part.strip() for part in text.split(',')[::-1])", ""),
    # near a trained concept (the minimal pairs of the curriculum, one step further)
    _c("text_after_last_colon", "text", "returns the text after the last colon in text with surrounding spaces "
       "removed, or an empty string if there is no colon.",
       "return text.rsplit(':', 1)[1].strip() if ':' in text else ''", ""),
    _c("file_stem_first", "filename", "returns the part of filename before its first dot, or filename itself if "
       "it has no dot.", "return filename.split('.', 1)[0]", ""),
    _c("lowercase_last_two", "text", "returns text with its last two characters in lowercase.",
       "return text[:-2] + text[-2:].lower()", ""),
    _c("split_on_spaces_and_commas", "text", "returns the words of text, treating commas as spaces.",
       "return text.replace(',', ' ').split()", ""),
)


def _arguments(params: str) -> list[tuple]:
    """40 argument tuples: every string of POOL once, with the i-th character, word or position beside it."""
    names = [p.strip() for p in params.split(",")]
    if names == ["parts"]:  # lists of strings: empty, one item, empty items, and runs of the pool
        return ([([],), ([""],), (["", ""],)] + [(list(POOL[i:i + k]),) for k in (1, 2, 3) for i in range(0, 39, 3)])[:40]
    if len(names) == 1:
        return [(s,) for s in POOL]
    if names[1:] == ["ch"]:
        return [(s, CHARS[i % len(CHARS)]) for i, s in enumerate(POOL)]
    if names[1:] == ["old", "new"]:
        return [(s, WORDS[i % len(WORDS)], ("+", "")[i % 2]) for i, s in enumerate(POOL)]
    if names[1:] == ["start", "end"]:
        return [(s, (0, 1, 2, -2)[i % 4], (3, 0, 10, -1)[(i // 4) % 4]) for i, s in enumerate(POOL)]
    raise ValueError(f"no probe inputs for parameters {params!r}")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _arity(params: str) -> int:
    return len([p for p in params.split(",") if p.strip()])


def _neighbour(concept, args: list, want: list):
    """(trained concept, its outputs on ``args``): the trained string concept with as many parameters and the
    most shared description words whose outputs differ from ``want`` on at least one of ``args``."""
    best = None
    for other in strings.STRING_CONCEPTS:
        if _arity(other.params) != _arity(concept.params) or other.name == concept.name:
            continue
        got = [repr(v) if kind == "ok" else f"<{v}>" for kind, v in run(other.source(concept.name), concept.name, args)]
        if got == [repr(v) for v in want]:
            continue
        score = len(_words(concept.task) & _words(other.task)) / len(_words(concept.task) | _words(other.task))
        if best is None or score > best[0]:
            best = (score, other, got)
    return None if best is None else best[1:]


def task(concept, group: str, neighbour: bool) -> dict:
    name, params = concept.name, concept.params
    probes = _arguments(params)
    outs = run(concept.source(), name, probes)
    ok = [(a, v) for a, (kind, v) in zip(probes, outs) if kind == "ok"]
    if len(ok) < PROBES:
        raise ValueError(f"{name}: only {len(ok)} probe inputs run")
    ok = ok[:PROBES]
    by_size = sorted(ok, key=lambda av: len(repr(av[0])))
    tests = by_size[:3] + by_size[len(by_size) // 2: len(by_size) // 2 + 3] + by_size[-2:]
    near = _neighbour(concept, [a for a, _ in ok], [v for _, v in ok]) if neighbour else None
    if near:  # the tests include the two shortest inputs on which the neighbour's function differs
        other, got = near
        differ = [(a, v) for (a, v), g in zip(ok, got) if g != repr(v)]
        tests += [av for av in sorted(differ, key=lambda av: len(repr(av[0])))[:2] if av not in tests]
    args, want = [a for a, _ in tests], [v for _, v in tests]
    wrong = near[0].source(name) if near else None
    wrong_outputs = None
    if near:
        wrong_outputs = [repr(v) if kind == "ok" else f"<{v}>" for kind, v in run(wrong, name, args)]
    return {"group": group, "family": group, "name": name, "params": params,
            "neighbour": near[0].name if near else None, "same_as": None,
            "prompts": {"original": f"Write a Python function `{name}({params})` that {concept.task}"},
            "tests": "\n".join(f"assert {name}({', '.join(repr(x) for x in a)}) == {v!r}" for a, v in zip(args, want)),
            "test_inputs": [list(a) for a in args], "reference": concept.source(),
            "wrong": wrong, "expected": [repr(v) for v in want], "wrong_outputs": wrong_outputs,
            "probe_args": [repr(tuple(a)) for a, _ in ok], "probe_expected": [repr(v) for _, v in ok]}


def build() -> dict[str, dict]:
    trained = [task(c, "strings_trained", False) for c in strings.STRING_CONCEPTS]
    heldout = [task(c, "strings_heldout", True) for c in HELD_OUT]
    return {name: {"version": VERSION, "variants": ["original"], "probes": PROBES, "tasks": tasks}
            for name, tasks in (("strings_trained", trained), ("strings_heldout", heldout))}


def main() -> None:
    out = ROOT / "experiments" / "v13" / "sets"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in build().items():
        (out / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        print(name, len(data["tasks"]), "tasks,", sum(t["wrong"] is not None for t in data["tasks"]), "with a neighbour")


if __name__ == "__main__":
    main()
