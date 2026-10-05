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
        p.add_argument("--resume", action="store_true",
                       help="Continue an interrupted SFT run in --out (falls back to --init-from).")
    else:
        p.add_argument("--preset", default="v2-small",
                       help="Model size: v3-40m, v2-tiny, v2-small (default), v2-base, v2-medium, v2-large, or v1 presets.")
        p.add_argument("--layers", type=int, help="Override number of layers.")
        p.add_argument("--heads", type=int, help="Override number of attention heads.")
        p.add_argument("--embd", type=int, help="Override embedding width.")
        p.add_argument("--context", type=int, help="Override context length (tokens).")
        p.add_argument("--resume", action="store_true", help="Continue training the model in --out.")
    p.add_argument("--steps", type=int, default=None, help="Optimizer steps (the LR schedule spans all of them).")
    p.add_argument("--until-step", type=int, default=None,
                   help="Stop at this step and save; continue later with --resume (segmented training).")
    p.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=None,
                   help="Also checkpoint every N steps without evaluating (cheap protection against crashes).")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps.")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16"))
    p.add_argument("--schedule", default="wsd", choices=("wsd", "cosine"),
                   help="LR schedule: warmup-stable-decay (default) or cosine.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (faster on many machines).")
    p.add_argument("--optimizer", default=None, choices=("adamw", "muon"),
                   help="adamw, or muon (Muon for hidden matrices + AdamW for the rest). "
                        "Default: muon for v3 presets, adamw otherwise.")
    p.add_argument("--muon-lr", type=float, default=0.02)
    if sft:
        p.add_argument("--no-pack", action="store_true", help="One example per row instead of packing.")
    else:
        p.add_argument("--anneal-mix", type=float, default=0.2,
                       help="Share of instruction data mixed in during the LR decay phase (0 disables).")
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
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--exclude", action="append", default=[],
                   help="Glob of paths to leave out, e.g. '*/tests/*' (repeatable). Use it to hold out eval data.")
    p.add_argument("--workers", type=int, default=None, help="Tokenizer processes (default: all CPU cores).")

    _train_args(sub.add_parser("train", help="Pretrain a model on the code corpus."), sft=False)
    _train_args(sub.add_parser("sft", help="Instruction-tune a pretrained model so it answers requests."),
                sft=True)

    p = sub.add_parser("eval", help="Score models: bits/byte on held-out code + pass rate on coding problems.")
    p.add_argument("--model", action="append", type=Path, default=[],
                   help="Model directory (repeat to compare several, e.g. v1 and v2).")
    p.add_argument("--heldout", action="append", type=Path, default=[],
                   help="Code the models never trained on, for bits-per-byte (repeatable).")
    p.add_argument("--samples", type=int, default=1, help="Samples per problem for pass@k (default 1 = greedy).")
    p.add_argument("--no-functional", action="store_true", help="Skip the coding-problem benchmark.")
    p.add_argument("--verbose", action="store_true", help="Show each problem's result.")
    p.add_argument("--device", default="auto")

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
            prepare_dataset(sources, args.out, vocab_size=args.vocab_size, extra_sft=args.sft_data,
                            exclude=args.exclude, workers=args.workers)
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
            preset=getattr(args, "preset", "v2-small"),
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
            schedule=args.schedule,
            compile=args.compile,
            optimizer=args.optimizer or ("muon" if str(getattr(args, "preset", "")).startswith("v3") else "adamw"),
            muon_lr=args.muon_lr,
            pack=not getattr(args, "no_pack", False),
            anneal_mix=getattr(args, "anneal_mix", 0.0),
            seed=args.seed,
            init_from=str(args.init_from) if sft else None,
            resume=getattr(args, "resume", False),
            until_step=args.until_step,
            save_interval=args.save_interval,
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

    if args.command == "eval":
        return _eval(args)

    model_dir = args.model or default_model_dir()
    try:
        lm = LocalLM(model_dir, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command == "info":
        cfg = lm.model.cfg
        print(f"Model:      {model_dir} (YCode-LM v{lm.version})")
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


def _eval(args) -> int:  # noqa: ANN001
    import json

    from ycode.lm.evaluate import PROBLEMS, bits_per_byte, functional_eval, load_heldout_texts
    from ycode.lm.generate import LocalLM

    models = args.model or [default_model_dir()]
    texts = load_heldout_texts(args.heldout) if args.heldout else []
    if args.heldout and not texts:
        print("error: no code found in --heldout", file=sys.stderr)
        return 1
    rows = []
    for model_dir in models:
        try:
            lm = LocalLM(model_dir, device=args.device)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Evaluating {model_dir} (YCode-LM v{lm.version}, {lm.num_params / 1e6:.1f}M params) ...")
        row = {"model": str(model_dir), "version": lm.version, "params_m": round(lm.num_params / 1e6, 2)}
        if texts:
            row["bits_per_byte"] = round(bits_per_byte(lm.model, lm.tokenizer, texts, device=lm.device), 4)
        if not args.no_functional:
            res = functional_eval(lm, samples=args.samples, log=print if args.verbose else None)
            row["pass@1"] = round(res.pass_at_1, 4)
            if res.pass_at_k is not None:
                row[f"pass@{res.k}"] = round(res.pass_at_k, 4)
            row["solved"] = res.solved
        rows.append(row)
    print()
    header = ["model", "version", "params_m", "bits_per_byte", "pass@1"] + sorted(
        {k for r in rows for k in r if k.startswith("pass@") and k != "pass@1"})
    print(" | ".join(h for h in header))
    for r in rows:
        print(" | ".join(str(r.get(h, "-")) for h in header))
    if not args.no_functional:
        print(f"\n({len(PROBLEMS)} problems; bits/byte: lower is better; pass@k: higher is better)")
    for r in rows:
        if r.get("solved"):
            print(f"{r['model']} solved: {', '.join(r['solved'])}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
