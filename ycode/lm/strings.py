"""V13: a verified curriculum of string operations, for fine-tuning past S1 (V12) on text processing.

Eight groups (``GROUPS``), 47 concepts:

1. replacement with ``replace()``;
2. splitting and joining;
3. whitespace removal with ``strip()`` and its relatives;
4. character filtering and transformation;
5. substrings and indexing;
6. case conversion;
7. compositions of several operations;
8. minimal pairs: concepts a word or two from one in another group.

Each concept is a ``basics.Concept``: a task description, a reference body, and executable tests whose
inputs include the boundary cases a string task gets wrong (the empty string, repeated characters,
leading and trailing spaces, punctuation). ``MINIMAL_PAIRS`` lists concepts whose descriptions differ by
a word or two but whose implementations differ; each one's reference fails the other's tests.

Nothing here may coincide with an evaluation item: no name is a benchmark, dev, unseen-v1 or unseen-v2
name or an existing training concept's, and no reference computes any evaluation item's function
(``tests/test_strings.py`` checks both, the latter with ``leakage.behaviour_leaks`` at agreement 0.8,
stricter than the 1.0 of the dataset's hygiene check). Every generated example is executed against its
concept's tests under the name it ends up with (``string_examples``).
"""
from __future__ import annotations

import random
import re

from ycode.lm import basics
from ycode.lm.basics import _c
from ycode.lm.data import InstructionExample
from ycode.lm.evaluate import extract_code

GROUPS: dict[str, tuple[basics.Concept, ...]] = {
    "replace": (
        _c("commas_to_semicolons", "text", "returns text with every comma replaced by a semicolon.",
           "return text.replace(',', ';')",
           "assert f('a,b,,c') == 'a;b;;c'\nassert f('') == ''\nassert f('no commas.') == 'no commas.'"),
        _c("underscores_to_spaces", "name", "returns name with every underscore replaced by a space.",
           "return name.replace('_', ' ')",
           "assert f('snake_case_name') == 'snake case name'\nassert f('__x') == '  x'\nassert f('') == ''"),
        _c("replace_first", "text, old, new", "returns text with only the first occurrence of old replaced by new.",
           "return text.replace(old, new, 1)",
           "assert f('a-b-c', '-', '+') == 'a+b-c'\nassert f('aaa', 'a', 'b') == 'baa'\n"
           "assert f('', 'a', 'b') == ''\nassert f('xyz', 'q', 'r') == 'xyz'"),
        _c("replace_last", "text, old, new", "returns text with only the last occurrence of old replaced by new.",
           "return new.join(text.rsplit(old, 1))",
           "assert f('a-b-c', '-', '+') == 'a-b+c'\nassert f('aaa', 'a', 'b') == 'aab'\n"
           "assert f('', 'a', 'b') == ''\nassert f('xyz', 'q', 'r') == 'xyz'"),
        _c("delete_char", "text, ch", "returns text with every occurrence of the character ch removed.",
           "return text.replace(ch, '')",
           "assert f('banana', 'a') == 'bnn'\nassert f('a.b.c.', '.') == 'abc'\nassert f('', 'x') == ''\n"
           "assert f('zzz', 'z') == ''"),
        _c("remove_hyphens", "word", "returns word with all hyphens removed.",
           "return word.replace('-', '')",
           "assert f('well-known') == 'wellknown'\nassert f('--a--') == 'a'\nassert f('') == ''\n"
           "assert f('a b') == 'a b'"),
    ),
    "split_join": (
        _c("split_on_commas", "text", "returns the comma-separated parts of text with surrounding spaces "
           "removed, leaving out empty parts.",
           "return [part.strip() for part in text.split(',') if part.strip()]",
           "assert f('a, b ,c') == ['a', 'b', 'c']\nassert f('') == []\nassert f(' , ,x,,') == ['x']\n"
           "assert f('one') == ['one']"),
        _c("join_with_slashes", "parts", "returns the strings in parts joined with slashes.",
           "return '/'.join(parts)",
           "assert f(['usr', 'local', 'bin']) == 'usr/local/bin'\nassert f([]) == ''\nassert f(['', 'a']) == '/a'"),
        _c("words_on_lines", "text", "returns the words of text, one per line.",
           "return '\\n'.join(text.split())",
           "assert f('to be  or') == 'to\\nbe\\nor'\nassert f('') == ''\nassert f('  solo  ') == 'solo'"),
        _c("file_extension", "filename", "returns the part of filename after its last dot, or an empty "
           "string if it has no dot.",
           "return filename.rsplit('.', 1)[1] if '.' in filename else ''",
           "assert f('report.final.pdf') == 'pdf'\nassert f('README') == ''\nassert f('') == ''\n"
           "assert f('archive.') == ''"),
        _c("split_key_value", "pair", "returns the text before and after the first '=' in pair, as a tuple "
           "of two strings with surrounding spaces removed.",
           "key, _, value = pair.partition('=')\nreturn key.strip(), value.strip()",
           "assert f('name = Ada') == ('name', 'Ada')\nassert f('a=b=c') == ('a', 'b=c')\n"
           "assert f('flag') == ('flag', '')\nassert f('') == ('', '')"),
        _c("join_nonempty_with_commas", "parts", "returns the non-empty strings in parts joined with a comma "
           "and a space.",
           "return ', '.join(part for part in parts if part)",
           "assert f(['a', '', 'b']) == 'a, b'\nassert f([]) == ''\nassert f(['', '']) == ''\n"
           "assert f(['x']) == 'x'"),
    ),
    "strip": (
        _c("strip_both_ends", "text", "returns text without the spaces at its start and end.",
           "return text.strip()",
           "assert f('  a b  ') == 'a b'\nassert f('') == ''\nassert f('   ') == ''\nassert f('x') == 'x'"),
        _c("strip_start", "text", "returns text without the spaces at its start.",
           "return text.lstrip()",
           "assert f('  a b  ') == 'a b  '\nassert f('') == ''\nassert f('   ') == ''\nassert f('x ') == 'x '"),
        _c("strip_end", "text", "returns text without the spaces at its end.",
           "return text.rstrip()",
           "assert f('  a b  ') == '  a b'\nassert f('') == ''\nassert f('   ') == ''\nassert f(' x') == ' x'"),
        _c("strip_dots", "text", "returns text without the dots at its start and end.",
           "return text.strip('.')",
           "assert f('..a.b..') == 'a.b'\nassert f('') == ''\nassert f('...') == ''\nassert f(' a. ') == ' a. '"),
        _c("strip_leading_zeros", "digits", "returns the digit string digits without its leading zeros.",
           "return digits.lstrip('0')",
           "assert f('00420') == '420'\nassert f('000') == ''\nassert f('') == ''\nassert f('7') == '7'"),
        _c("strip_each_line", "text", "returns text with the spaces at the start and end of each line removed.",
           "return '\\n'.join(line.strip() for line in text.split('\\n'))",
           "assert f('  a \\n b  ') == 'a\\nb'\nassert f('') == ''\nassert f(' x ') == 'x'\n"
           "assert f('a\\n\\n b') == 'a\\n\\nb'"),
    ),
    "filter_transform": (
        _c("keep_digits", "text", "returns only the digits in text, in order.",
           "return ''.join(c for c in text if c.isdigit())",
           "assert f('a1b22c3') == '1223'\nassert f('') == ''\nassert f('no digits!') == ''\nassert f('007') == '007'"),
        _c("remove_punctuation", "text", "returns text without the characters . , ! ? ; and :.",
           "return ''.join(c for c in text if c not in '.,!?;:')",
           "assert f('Hi, there! Ok?') == 'Hi there Ok'\nassert f('') == ''\nassert f('...') == ''\n"
           "assert f('a-b') == 'a-b'"),
        _c("letters_and_spaces", "text", "returns only the letters and spaces in text.",
           "return ''.join(c for c in text if c.isalpha() or c == ' ')",
           "assert f('R2 D2, go!') == 'R D go'\nassert f('') == ''\nassert f('  ') == '  '\nassert f('123') == ''"),
        _c("digits_to_hashes", "text", "returns text with every digit replaced by '#'.",
           "return ''.join('#' if c.isdigit() else c for c in text)",
           "assert f('pin 1234') == 'pin ####'\nassert f('') == ''\nassert f('a1a1') == 'a#a#'"),
        _c("double_the_vowels", "text", "returns text with every vowel (a, e, i, o or u, either case) written twice.",
           "return ''.join(c * 2 if c in 'aeiouAEIOU' else c for c in text)",
           "assert f('Moon') == 'Moooon'\nassert f('') == ''\nassert f('xyz') == 'xyz'\nassert f('Ai!') == 'AAii!'"),
        _c("first_occurrences_only", "text", "returns text with every repeated character after its first "
           "occurrence removed.",
           "seen = set()\nout = []\nfor c in text:\n    if c not in seen:\n        seen.add(c)\n        out.append(c)\n"
           "return ''.join(out)",
           "assert f('banana') == 'ban'\nassert f('') == ''\nassert f('aaaa') == 'a'\nassert f('a b a') == 'a b'"),
    ),
    "substring": (
        _c("prefix_before_digit", "text", "returns the characters of text before its first digit, or all of "
           "text if it has no digit.",
           "for i, c in enumerate(text):\n    if c.isdigit():\n        return text[:i]\nreturn text",
           "assert f('abc12d') == 'abc'\nassert f('7up') == ''\nassert f('') == ''\nassert f('no digits') == 'no digits'"),
        _c("odd_position_chars", "text", "returns the characters of text at odd positions (1, 3, 5, ...).",
           "return text[1::2]",
           "assert f('abcdef') == 'bdf'\nassert f('') == ''\nassert f('a') == ''\nassert f('aabb') == 'ab'"),
        _c("text_after_colon", "text", "returns the text after the first colon in text with surrounding "
           "spaces removed, or an empty string if there is no colon.",
           "return text.split(':', 1)[1].strip() if ':' in text else ''",
           "assert f('Name: Ada') == 'Ada'\nassert f('a:b:c') == 'b:c'\nassert f('none') == ''\nassert f('') == ''"),
        _c("drop_first_and_last_char", "text", "returns text without its first and last characters.",
           "return text[1:-1]",
           "assert f('[x]') == 'x'\nassert f('ab') == ''\nassert f('') == ''\nassert f('a') == ''"),
        _c("slice_between", "text, start, end", "returns the characters of text from position start up to "
           "but not including position end.",
           "return text[start:end]",
           "assert f('python', 1, 4) == 'yth'\nassert f('', 0, 3) == ''\nassert f('abc', 2, 10) == 'c'"),
        _c("find_char", "text, ch", "returns the position of the first occurrence of the character ch in "
           "text, or -1 if it does not occur.",
           "return text.find(ch)",
           "assert f('hello', 'l') == 2\nassert f('hello', 'z') == -1\nassert f('', 'a') == -1\nassert f('aa', 'a') == 0"),
    ),
    "case": (
        _c("lowercase_first_letter", "text", "returns text with its first character in lowercase.",
           "return text[:1].lower() + text[1:]",
           "assert f('Hello World') == 'hello World'\nassert f('') == ''\nassert f('ABC') == 'aBC'"),
        _c("uppercase_last_letter", "text", "returns text with its last character in uppercase.",
           "return text[:-1] + text[-1:].upper()",
           "assert f('hello') == 'hellO'\nassert f('') == ''\nassert f('a!') == 'a!'\nassert f('x') == 'X'"),
        _c("sentence_case", "text", "returns text with its first character in uppercase and every other "
           "character in lowercase.",
           "return text.capitalize()",
           "assert f('hELLO there') == 'Hello there'\nassert f('') == ''\nassert f('1ST') == '1st'"),
        _c("count_lowercase", "text", "returns the number of lowercase letters in text.",
           "return sum(1 for c in text if c.islower())",
           "assert f('Hello World') == 8\nassert f('') == 0\nassert f('ABC 123') == 0\nassert f('aaa') == 3"),
        _c("camel_to_snake", "name", "returns the camelCase name in snake_case.",
           "return ''.join('_' + c.lower() if c.isupper() else c for c in name).lstrip('_')",
           "assert f('camelCaseName') == 'camel_case_name'\nassert f('') == ''\nassert f('Already') == 'already'\n"
           "assert f('x') == 'x'"),
        _c("shout", "text", "returns text in uppercase followed by an exclamation mark.",
           "return text.upper() + '!'",
           "assert f('stop') == 'STOP!'\nassert f('') == '!'\nassert f('Hi there') == 'HI THERE!'"),
    ),
    "composition": (
        _c("make_username", "text", "returns text without surrounding spaces, in lowercase, with every "
           "remaining space replaced by an underscore.",
           "return text.strip().lower().replace(' ', '_')",
           "assert f('  Ada Lovelace ') == 'ada_lovelace'\nassert f('') == ''\nassert f('A  B') == 'a__b'"),
        _c("capitalized_words_with_dashes", "text", "returns the words of text, each capitalized, joined "
           "with dashes.",
           "return '-'.join(word.capitalize() for word in text.split())",
           "assert f('the  quick fox') == 'The-Quick-Fox'\nassert f('') == ''\nassert f(' a ') == 'A'"),
        _c("dotted_initials", "name", "returns the uppercase first letter of each word in name, each "
           "followed by a dot.",
           "return ''.join(word[0].upper() + '.' for word in name.split())",
           "assert f('ada byron lovelace') == 'A.B.L.'\nassert f('') == ''\nassert f('  x  ') == 'X.'"),
        _c("reversed_words_uppercase", "text", "returns the words of text in reverse order, in uppercase, "
           "separated by single spaces.",
           "return ' '.join(text.split()[::-1]).upper()",
           "assert f('one two  three') == 'THREE TWO ONE'\nassert f('') == ''\nassert f('a') == 'A'"),
        _c("stripped_length", "text", "returns the number of characters in text once the spaces at its "
           "start and end are removed.",
           "return len(text.strip())",
           "assert f('  ab c  ') == 4\nassert f('') == 0\nassert f('    ') == 0\nassert f('x') == 1"),
        _c("count_comma_fields", "line", "returns how many non-empty fields the comma-separated line has, "
           "ignoring spaces around them.",
           "return len([field for field in line.split(',') if field.strip()])",
           "assert f('a, b,,c') == 3\nassert f('') == 0\nassert f(' , ') == 0\nassert f('x') == 1"),
    ),
    # each is a word or two from a concept above (MINIMAL_PAIRS): the description alone decides the code
    "minimal_pairs": (
        _c("strip_trailing_dots", "text", "returns text without the dots at its end.",
           "return text.rstrip('.')",
           "assert f('..a.b..') == '..a.b'\nassert f('') == ''\nassert f('...') == ''\nassert f('a. ') == 'a. '"),
        _c("split_on_semicolons", "text", "returns the semicolon-separated parts of text with surrounding spaces "
           "removed, leaving out empty parts.",
           "return [part.strip() for part in text.split(';') if part.strip()]",
           "assert f('a; b ;c') == ['a', 'b', 'c']\nassert f('') == []\nassert f('a,b;c') == ['a,b', 'c']"),
        _c("file_stem", "filename", "returns the part of filename before its last dot, or filename itself if "
           "it has no dot.",
           "return filename.rsplit('.', 1)[0]",
           "assert f('report.final.pdf') == 'report.final'\nassert f('README') == 'README'\nassert f('') == ''\n"
           "assert f('archive.') == 'archive'"),
        _c("text_before_colon", "text", "returns the text before the first colon in text with surrounding "
           "spaces removed, or all of text without surrounding spaces if there is no colon.",
           "return text.split(':', 1)[0].strip()",
           "assert f('Name: Ada') == 'Name'\nassert f('a:b:c') == 'a'\nassert f(' none ') == 'none'\nassert f('') == ''"),
        _c("lowercase_last_letter", "text", "returns text with its last character in lowercase.",
           "return text[:-1] + text[-1:].lower()",
           "assert f('HELLO') == 'HELLo'\nassert f('') == ''\nassert f('A!') == 'A!'\nassert f('X') == 'x'"),
    ),
}

STRING_CONCEPTS: tuple[basics.Concept, ...] = tuple(c for group in GROUPS.values() for c in group)
BY_NAME = {c.name: c for c in STRING_CONCEPTS}

# Descriptions a few words apart (at least half their words shared), implementations that differ: each
# member's reference fails the other's tests (both checked in tests/test_strings.py).
MINIMAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("replace_first", "replace_last"),
    ("strip_both_ends", "strip_start"),
    ("strip_both_ends", "strip_end"),
    ("strip_start", "strip_end"),
    ("strip_dots", "strip_trailing_dots"),
    ("split_on_commas", "split_on_semicolons"),
    ("file_extension", "file_stem"),
    ("text_after_colon", "text_before_colon"),
    ("uppercase_last_letter", "lowercase_last_letter"),
    ("lowercase_first_letter", "sentence_case"),
)


def _name_of(code: str) -> str:
    match = re.search(r"def (\w+)\(", code)
    if match is None:
        raise ValueError(f"no function in {code[:60]!r}")
    return match.group(1)


def check_example(ex: InstructionExample, concept: basics.Concept) -> bool:
    """The example's answer, run under the name it uses, passes the concept's tests."""
    code = extract_code(ex.response)
    return basics._passes(code, basics._rename_tests(concept.tests, _name_of(code)))


def string_examples(rng: random.Random, write_per_concept: int = 4,
                    bugfix_per_concept: int = 2) -> list[tuple[InstructionExample, str]]:
    """(example, tag) for every concept: written functions and bug fixes, from the generators of
    ``ycode.lm.basics``, one concept at a time. Every answer is executed against its concept's tests under
    the name it ends up with (neutralized or not); a failure raises. An exact repeat of an earlier example
    (the same template and name drawn twice) is left out."""
    out, seen = [], set()
    for concept in STRING_CONCEPTS:
        for kind, examples in (("write", basics.write_examples(rng, write_per_concept, concepts=(concept,))),
                               ("bugfix", basics.bugfix_examples(rng, bugfix_per_concept, concepts=(concept,)))):
            for ex in examples:
                if (ex.prompt, ex.response) in seen:
                    continue
                if not check_example(ex, concept):
                    raise ValueError(f"string {kind} example for {concept.name} fails its tests: "
                                     f"{ex.response[:120]!r}")
                seen.add((ex.prompt, ex.response))
                out.append((ex, f"strings:{kind}:{concept.name}"))
    rng.shuffle(out)
    return out
