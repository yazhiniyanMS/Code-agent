"""The evaluation splits stay what their definitions say: no item, by name or prompt, in two splits, and no
repeated item within a split (a name may appear once per suite: a write task and a bug fix of the same
function). Item names and prompts are read; no model is evaluated."""
from collections import Counter
from itertools import combinations

import pytest

from ycode.lm import evalsets

SPLITS = ("bench", "dev", "unseen", "unseen2")


@pytest.fixture(scope="module")
def items():
    return {split: evalsets.eval_items(split) for split in SPLITS}


def test_no_item_is_in_two_splits(items):
    for a, b in combinations(SPLITS, 2):
        assert not {i.name for i in items[a]} & {i.name for i in items[b]}, (a, b)
        assert not {i.prompt.strip() for i in items[a]} & {i.prompt.strip() for i in items[b]}, (a, b)


def test_no_item_repeats_within_a_split(items):
    for split, rows in items.items():
        assert not [k for k, n in Counter((i.suite, i.name) for i in rows).items() if n > 1], split
        assert not [p for p, n in Counter(i.prompt.strip() for i in rows).items() if n > 1], split
