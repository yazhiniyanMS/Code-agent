"""V7 step 2: does the prompt decide whether the model writes the requested function or its neighbour?

usage: prompt_diagnostic.py MODEL_DIR OUT.jsonl [--set ycode/lm/neighbours_v1.json] [--variants original,distractor] [--no-generate]

For every task of the frozen neighbour set and each of its four prompt variants (original, io,
constraint, distractor), under the evaluation's own settings (greedy, at most 200 new tokens,
no_repeat_ngram 12, bf16 on this host):

* ``pass``: the answer's code passes the task's tests (exact functional correctness);
* ``wrong``: the answer computes the neighbour's behaviour on every test input (and fails);
* ``margin``: the model's preference for the reference over the neighbour's code under this prompt:
  summed code-token loss of the neighbour's code minus that of the reference (nats; > 0 prefers the
  reference). Code tokens only (BODY_WEIGHT), like the learning probe.

Rows are appended as they finish; a rerun skips (task, variant) pairs already in OUT.jsonl.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def outputs(code: str, name: str, inputs: list) -> list[str] | None:
    """repr of each output (or <ErrorType>) of the answer's ``name`` on the inputs; None if the
    code does not define it. Subprocess with a timeout, like the tests."""
    prog = f"""
import copy, json
_ns = {{}}
try:
    exec({code!r}, _ns)
    _f = _ns[{name!r}]
except Exception:
    print(json.dumps(None)); raise SystemExit
_out = []
for _a in {inputs!r}:
    try:
        _out.append(repr(_f(*copy.deepcopy(_a))))
    except Exception as _e:
        _out.append('<' + type(_e).__name__ + '>')
print(json.dumps(_out))
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "out.py"
        script.write_text(prog)
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp, capture_output=True, text=True,
                                  timeout=10, stdin=subprocess.DEVNULL)
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None


def code_loss(lm, prompt: str, code: str) -> tuple[float, int]:
    from ycode.lm.data import BODY_WEIGHT, InstructionExample, encode_example
    from ycode.lm.diagnose import _example_loss

    ids, w = encode_example(lm.tokenizer, InstructionExample(prompt, f"```python\n{code}\n```"), weighted=True)
    block = lm.model.cfg.block_size
    return _example_loss(lm.model, ids[:block], w[:block], lm.device, lm.precision(), only_weight=BODY_WEIGHT)


def main() -> None:
    import argparse

    from ycode.lm.evaluate import extract_code, run_tests
    from ycode.lm.generate import LocalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--set", default=str(ROOT / "ycode" / "lm" / "neighbours_v1.json"))
    ap.add_argument("--no-generate", action="store_true", help="margins only")
    ap.add_argument("--variants", default=None, help="comma-separated subset of the prompt variants")
    args = ap.parse_args()
    data = json.loads(Path(args.set).read_text())
    variants = args.variants.split(",") if args.variants else data["variants"]
    assert set(variants) <= set(data["variants"]), variants
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["name"], r["variant"]))
    lm = LocalLM(args.model, device="cpu")
    t0 = time.time()
    for variant in variants:  # one variant at a time over all tasks: partial runs stay comparable
        for t in data["tasks"]:
            if (t["name"], variant) in done:
                continue
            prompt = t["prompts"][variant]
            ref_loss, ref_n = code_loss(lm, prompt, t["reference"])
            wrong_loss, wrong_n = code_loss(lm, prompt, t["wrong"])
            row = {"set": data["version"], "model": args.model, "group": t["group"], "name": t["name"],
                   "variant": variant, "prompt": prompt, "ref_loss": round(ref_loss, 4), "ref_tokens": ref_n,
                   "wrong_loss": round(wrong_loss, 4), "wrong_tokens": wrong_n,
                   "margin": round(wrong_loss - ref_loss, 4)}
            if not args.no_generate:
                answer = lm.chat([("user", prompt)], max_new_tokens=200, temperature=0)
                code = extract_code(answer)
                got = outputs(code, t["name"], t["test_inputs"])
                row.update({"answer": answer, "code": code, "pass": run_tests(code, t["tests"]),
                            "outputs": got,
                            "wrong": got is not None and got == t["wrong_outputs"] and got != t["expected"]})
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{time.time() - t0:7.0f}s] {variant:10s} {t['name']:24s} pass={row.get('pass')} "
                  f"wrong={row.get('wrong')} margin={row['margin']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
