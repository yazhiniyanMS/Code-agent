"""Build one round's fine-tuning data. Run in a fresh process each round, so edits to the
generators (and to this mix) take effect without restarting the loop driver.

v5.1: the round data comes from ycode.lm.rounddata (the same production mix, with held-out
equivalents removed, skill families off by default, and a manifest + hygiene report per round).
The v5 version of this file is build_round.py.bak_v5."""

import sys
import time
from pathlib import Path

W = Path("/home/user/lmwork")
sys.path.insert(0, "/home/user/Code-agent")


def log(msg: str) -> None:
    with (W / "loop" / "loop.log").open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def main(r: int, out: Path) -> None:
    from ycode.lm.rounddata import write_round_dataset

    write_round_dataset(out, 1000 + r, W / "data" / "tokenizer.json", replay_dir=W / "data", log=log)


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
