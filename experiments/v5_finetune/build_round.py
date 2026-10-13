"""Build one round's fine-tuning data. Run in a fresh process each round, so edits to the
generators (and to this mix) take effect without restarting the loop driver."""

import sys
import time
from pathlib import Path

W = Path("/home/user/lmwork")
sys.path.insert(0, "/home/user/Code-agent")


def log(msg: str) -> None:
    with (W / "loop" / "loop.log").open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def main(r: int, out: Path) -> None:
    from ycode.lm import compose
    from ycode.lm.basics import basics_examples, write_basics_dataset

    seed = 1000 + r
    # Mix from round 22 on: more written functions, fewer repeated bug fixes (bug fixes held ~59% of
    # the loss weight while writing scores lagged), and half the old replay data.
    examples = (basics_examples(seed, write_per_concept=6, bugfix_per_concept=4)
                + compose.compose_examples(seed, per_exercise=3)
                + compose.compose_bugfix_examples(seed, per_exercise=1)
                + compose.contrast_examples(seed)
                + compose.novel_examples(seed))
    write_basics_dataset(out, W / "data" / "tokenizer.json", replay_dir=W / "data", replay=200, seed=seed,
                         examples=examples, log=log)


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
