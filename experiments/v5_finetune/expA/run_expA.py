"""Experiment A: is V5's score reliable? Evaluate one checkpoint 3 times under identical conditions."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/user/Code-agent")
OUT = Path("/home/user/lmwork/expA")
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/user/lmwork/loop/anchor")
sys.path.insert(0, str(REPO))


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def manifest() -> dict:
    import torch
    payload = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    tok = hashlib.sha256(json.dumps(payload["tokenizer"], sort_keys=True).encode()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "ycode"], cwd=REPO, capture_output=True, text=True).stdout
    files = {f: sha256(REPO / "ycode" / "lm" / f)[:16] for f in
             ("evaluate.py", "devset.py", "generate.py", "cli.py", "model.py", "basics.py", "tokenizer.py")}
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), "?")
    return {
        "checkpoint": str(CKPT), "checkpoint_sha256": sha256(CKPT / "model.pt"),
        "checkpoint_info": json.loads((CKPT / "info.json").read_text()),
        "tokenizer_sha256": tok, "vocab_size": payload["config"]["vocab_size"],
        "decoding": {"method": "greedy (temperature 0; top_k/top_p unused)", "max_new_tokens": 200,
                     "no_repeat_ngram": 12, "stop": ["<|end|>", "<|endoftext|>", "<|user|>"],
                     "weights_dtype": "bfloat16 (LocalLM auto for >500M params)"},
        "seed": "torch.manual_seed(0) before the write problems; greedy decoding uses no randomness",
        "evaluator": {"git_commit": git, "uncommitted_changes_in_ycode": bool(dirty.strip()), "file_sha256_16": files,
                      "scoring": "pass@1: extracted code block must pass all unit tests (subprocess, 5 s timeout)"},
        "environment": {"torch": torch.__version__, "threads": torch.get_num_threads(), "cpu": cpu,
                        "amx_bf16": "amx_bf16" in Path("/proc/cpuinfo").read_text(), "python": platform.python_version()},
    }


def main():
    OUT.mkdir(exist_ok=True)
    m = manifest()
    (OUT / "manifest.json").write_text(json.dumps(m, indent=2))
    print("manifest written", m["checkpoint_sha256"][:16], flush=True)
    for run in (1, 2, 3):
        for suite in ("dev", "both"):
            tag = f"run{run}_{suite}"
            if (OUT / f"{tag}.log").is_file() and "[{" in (OUT / f"{tag}.log").read_text():
                continue  # finished before a restart
            t0 = time.time()
            with open(OUT / f"{tag}.log", "w") as f:
                code = subprocess.run([sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(CKPT),
                                       "--problems", suite, "--verbose", "--device", "cpu",
                                       "--answers-out", str(OUT / f"{tag}.jsonl")], cwd=REPO, stdout=f,
                                      stderr=subprocess.STDOUT).returncode
            print(f"{tag}: exit {code}, {time.time() - t0:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
