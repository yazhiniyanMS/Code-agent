"""V13's gates (experiments/v13/PLAN.md) judge a run against S1 of the same seed, on S1's own records."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v13_gates", ROOT / "experiments" / "v13" / "gates.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def _records(seed: str):
    s1 = json.loads((G.S1 / seed / "result.json").read_text())
    strings = json.loads((G.BASELINE / f"S1_{seed}" / "result.json").read_text())
    return s1, strings


def _as_run(s1: dict, strings: dict) -> dict:
    run = copy.deepcopy(s1)
    run["sets"] = {**run["sets"], **copy.deepcopy(strings["sets"])}
    return run


@pytest.mark.parametrize("seed", ["1301", "1302"])
def test_s1_against_itself_keeps_everything_and_gains_no_strings(seed):
    s1, strings = _records(seed)
    out = G.gates(_as_run(s1, strings), s1, strings)
    assert [out[g]["pass"] for g in ("G2", "G3", "G4", "G5")] == [True] * 4
    assert out["G1"] == {"pass": False, "run": 0, "s1": 0} and out["verdict"] == "fails"


def test_each_gate_fails_on_the_change_it_guards_against():
    s1, strings = _records("1301")
    run = _as_run(s1, strings)
    run["sets"]["strings_heldout"]["scores"]["original"]["strict"] = 3
    assert G.gates(run, s1, strings)["verdict"] == "passes"  # +3 strings, everything else kept
    lose = copy.deepcopy(run)  # three of S1's clean dev passes now fail, all as reasoning errors
    gone = sorted(G._clean_passes(s1, "dev"))[:3]
    lose["dev"]["passed"] = [k for k in lose["dev"]["passed"] if k not in gone]
    lose["dev"]["clean"] -= 3
    lose["dev"]["categories"].update({k: "reasoning" for k in gone})
    out = G.gates(lose, s1, strings)
    assert out["G3"]["pass"] is False and out["G3"]["by_category"] == {"reasoning": 3}
    assert out["G2"]["pass"] is False and out["G2"]["run"] == out["G2"]["s1"] - 3
    weak = copy.deepcopy(run)
    weak["dev"]["robust_clean"] -= 2
    assert G.gates(weak, s1, strings)["G4"]["pass"] is False
    family = copy.deepcopy(run)
    family["sets"]["c2_test"]["scores"]["original"]["strict"] -= 3
    assert G.gates(family, s1, strings)["G5"]["pass"] is False


def test_a_screen_without_the_task_sets_is_not_a_pass():
    s1, strings = _records("1301")
    screen = _as_run(s1, strings)
    screen["sets"] = {"strings_heldout": copy.deepcopy(strings["sets"]["strings_heldout"])}
    screen["sets"]["strings_heldout"]["scores"]["original"]["strict"] = 5
    out = G.gates(screen, s1, strings)
    assert out["G5"] is None and out["verdict"] == "passes so far (not all gates measured)"
