"""V11: restore a packed checkpoint in clean processes and check that it reproduces its recorded evaluation.

usage: verify_restore.py NAME PACK RECORD_DIR OUT.json [--suites dev,unseen]

1. `python -m ycode.lm.packckpt unpack PACK DIR` in a new process (every tensor checked against the pack's
   content sha256 as it is restored); the restored file's sha256 is recorded.
2. `python -m ycode.lm.cli eval` in a new process for each suite, greedy, exactly as the runner evaluates.
3. Every answer compared with RECORD_DIR/<suite>.jsonl: identical extracted code and pass/fail, item by item.
The restored copy lives on the RAM disk and is removed afterwards (the pack is the durable copy).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    name, pack, record, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    suites = sys.argv[sys.argv.index("--suites") + 1].split(",") if "--suites" in sys.argv else ["dev", "unseen"]
    work = Path("/dev/shm/ycode-verify") / name
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    res = {"name": name, "pack": str(pack), "pack_sha256": sha256(pack), "record": str(record)}
    unpacked = subprocess.run([sys.executable, "-m", "ycode.lm.packckpt", "unpack", str(pack), str(work / "model")],
                              cwd=ROOT, capture_output=True, text=True)
    res["unpack"] = {"exit": unpacked.returncode, "tail": unpacked.stdout[-300:] + unpacked.stderr[-300:]}
    if unpacked.returncode != 0:
        raise SystemExit(f"unpack failed: {res['unpack']}")
    res["restored_model_sha256"] = sha256(work / "model" / "model.pt")
    ok = Path(str(pack) + ".ok")
    res["content_sha256"] = ok.read_text().strip() if ok.is_file() else None
    for suite in suites:
        answers = work / f"{suite}.jsonl"
        r = subprocess.run([sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(work / "model"),
                            "--problems", suite, "--device", "cpu", "--answers-out", str(answers)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{suite} evaluation failed: {r.stderr[-500:]}")
        key = lambda row: f"{row['suite']}:{row['name']}"
        new = {key(x): x for x in map(json.loads, answers.read_text().splitlines())}
        old = {key(x): x for x in map(json.loads, (record / f"{suite}.jsonl").read_text().splitlines())}
        differ = [k for k in old if k not in new or new[k].get("code") != old[k].get("code")
                  or bool(new[k]["pass"]) != bool(old[k]["pass"])]
        res[suite] = {"items": len(old), "identical": len(old) - len(differ), "differ": differ,
                      "differ_detail": {k: {"recorded": old[k].get("code"), "recorded_pass": bool(old[k]["pass"]),
                                            "restored": new.get(k, {}).get("code"),
                                            "restored_pass": bool(new.get(k, {}).get("pass"))} for k in differ},
                      "passes_restored": sum(bool(x["pass"]) for x in new.values()),
                      "passes_recorded": sum(bool(x["pass"]) for x in old.values())}
        print(name, suite, res[suite], flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["verified"] = all(not res[s]["differ"] for s in suites)
    shutil.rmtree(work, ignore_errors=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("name", "content_sha256", "restored_model_sha256", "verified", "minutes")}))


if __name__ == "__main__":
    main()
