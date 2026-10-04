"""``ycode-lm``: build, train and talk to your own from-scratch coding model.

    ycode-lm prepare  --source ~/code --out data/       # corpus + tokenizer + datasets
    ycode-lm train    --data data/ --out models/base    # pretrain on code
    ycode-lm sft      --data data/ --init-from models/base --out models/chat
    ycode-lm chat     --model models/chat               # ask it questions
    ycode-lm sample   --model models/base --prompt "def quicksort("
    ycode-lm info     --model models/chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ycode.config import default_local_model_dir as default_model_dir


def _require_torch() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
        print("ycode-lm needs PyTorch and NumPy. Install them with:\n\n    pip install -e \".[local]\"\n",
              file=sys.stderr)
        raise SystemExit(2)


def _train_args(p: argparse.ArgumentParser, *, sft: bool) -> None:
    p.add_argument("--data", required=True, type=Path, help="Directory produced by `ycode-lm prepare`.")
    p.add_argument("--out", type=Path, default=None, help="Where to write the model.")
    if sft:
        p.add_argument("--init-from", required=True, type=Path, help="Pretrained model directory.")
    else:
        p.add_argument("--preset", default="small", help="Model size: tiny, small, base, medium, large.")
        p.add_argument("--layers", type=int, help="Override number of layers.")
        p.add_argument("--heads", type=int, help="Override number of attention heads.")
        p.add_argument("--embd", type=int, help="Override embedding width.")
        p.add_argument("--context", type=int, help="Override context length (tokens).")
        p.add_argument("--resume", action="store_true", help="Continue training the model in --out.")
    p.add_argument("--steps", type=int, default=None, help="Optimizer steps.")
    p.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps.")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16"))
    p.add_argument("--seed", type=int, default=1337)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ycode-lm", description="Train your own coding LLM from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Collect code, train the tokenizer and encode datasets.")
    p.add_argument("--source", action="append", type=Path, default=[],
                   help="File or directory of code (repeatable). Default: the Python standard library.")
    p.add_argument("--sft-data", action="append", type=Path, default=[],
                   help="Extra instruction data: JSONL with {\"prompt\", \"response\"} per line (repeatable).")
    p.add_argument("--out", required=True, type=Path, help="Output data directory.")
    p.add_argument("--vocab-size", type=int, default=4096)

    _train_args(sub.add_parser("train", help="Pretrain a model on the code corpus."), sft=False)
    _train_args(sub.add_parser("sft", help="Instruction-tune a pretrained model so it answers requests."),
                sft=True)

    for name, help_text in (("chat", "Interactive Q&A with your model."),
                            ("sample", "Continue a code prompt."), ("info", "Show model details.")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--model", type=Path, default=None, help=f"Model directory (default {default_model_dir()}).")
        p.add_argument("--device", default="auto")
        if name != "info":
            p.add_argument("--temperature", type=float, default=0.7)
            p.add_argument("--max-tokens", type=int, default=300)
        if name == "sample":
            p.add_argument("--prompt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_torch()

    if args.command == "prepare":
        from ycode.lm.data import default_sources, prepare_dataset

        sources = args.source or default_sources()
        try:
            prepare_dataset(sources, args.out, vocab_size=args.vocab_size, extra_sft=args.sft_data)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command in ("train", "sft"):
        from ycode.lm.train import TrainConfig, train

        overrides = {}
        if args.command == "train":
            for key, attr in (("n_layer", "layers"), ("n_head", "heads"), ("n_embd", "embd"),
                              ("block_size", "context")):
                if getattr(args, attr) is not None:
                    overrides[key] = getattr(args, attr)
        sft = args.command == "sft"
        cfg = TrainConfig(
            stage="sft" if sft else "pretrain",
            preset=getattr(args, "preset", "small"),
            model_overrides=overrides,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr or (3e-4 if sft else 1e-3),
            min_lr=(args.lr or (3e-4 if sft else 1e-3)) / 10,
            warmup_steps=50 if sft else 100,
            max_steps=args.steps or (1000 if sft else 5000),
            max_minutes=args.minutes,
            eval_interval=args.eval_interval,
            device=args.device,
            precision=args.precision,
            seed=args.seed,
            init_from=str(args.init_from) if sft else None,
            resume=getattr(args, "resume", False),
        )
        out = args.out or default_model_dir()
        try:
            summary = train(args.data, out, cfg)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nInterrupted. The last checkpoint is in", out)
            return 130
        print(f"Saved model to {out} (val loss {summary['final']['val']:.3f})")
        return 0

    from ycode.lm.generate import LocalLM

    model_dir = args.model or default_model_dir()
    try:
        lm = LocalLM(model_dir, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command == "info":
        cfg = lm.model.cfg
        print(f"Model:      {model_dir}")
        print(f"Parameters: {lm.num_params / 1e6:.2f}M")
        print(f"Layers:     {cfg.n_layer}, heads {cfg.n_head}, width {cfg.n_embd}, context {cfg.block_size}")
        print(f"Vocab:      {cfg.vocab_size}")
        print(f"Stage:      {lm.stage} (step {lm.payload.get('step')}, val loss {lm.payload.get('val_loss')})")
        return 0

    def echo(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    if args.command == "sample":
        sys.stdout.write(args.prompt)
        lm.complete(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, on_text=echo)
        print()
        return 0

    print(f"YCode local model ({lm.num_params / 1e6:.1f}M parameters, {lm.stage}). Ctrl+D to quit.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        history.append(("user", question))
        sys.stdout.write("model> ")
        try:
            answer = lm.chat(history[-6:], max_new_tokens=args.max_tokens, temperature=args.temperature,
                             on_text=echo)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            history.pop()
            continue
        print()
        history.append(("assistant", answer))


if __name__ == "__main__":
    sys.exit(main())
