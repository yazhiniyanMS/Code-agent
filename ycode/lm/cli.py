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

from ycode.config import default_local_model_dir
from ycode.config import resolve_local_model as default_model_dir


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
        p.add_argument("--init-from", type=Path, default=None,
                       help="Start from this trained/grown model instead of random weights "
                            "(continued pretraining, e.g. after `ycode-lm grow`).")
    p.add_argument("--steps", type=int, default=None, help="Optimizer steps (the LR schedule spans all of them).")
    p.add_argument("--until-step", type=int, default=None,
                   help="Stop at this step and save; continue later with --resume (segmented training).")
    p.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20, help="Batches per evaluation (per split).")
    p.add_argument("--save-interval", type=int, default=None,
                   help="Also checkpoint every N steps without evaluating (cheap protection against crashes).")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps.")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recompute activations in backward: far less GPU memory, ~30%% slower.")
    p.add_argument("--warmup", type=int, default=None, help="Warmup steps (default 50 for sft, 100 for train).")
    p.add_argument("--train-layers", type=int, default=None,
                   help="Train only the top N transformer blocks; the rest stay frozen (fast fine-tuning).")
    p.add_argument("--shard", action="store_true",
                   help="With several GPUs (torchrun): split weights and optimizer state across them (FSDP) "
                        "so models too big for one GPU can train.")
    p.add_argument("--schedule", default="wsd", choices=("wsd", "cosine"),
                   help="LR schedule: warmup-stable-decay (default) or cosine.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (faster on many machines).")
    p.add_argument("--optimizer", default=None, choices=("adamw", "muon", "lion"),
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
    p.add_argument("--sft-only", action="store_true",
                   help="Only rebuild the instruction data, reusing --tokenizer (for re-tuning a pretrained model).")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="Reuse this tokenizer.json (needed to continue training an existing model).")
    p.add_argument("--max-mb", type=float, default=None, help="Cap the code corpus at this many MB.")

    _train_args(sub.add_parser("train", help="Pretrain a model on the code corpus."), sft=False)
    _train_args(sub.add_parser("sft", help="Instruction-tune a pretrained model so it answers requests."),
                sft=True)

    p = sub.add_parser("grow", help="Deepen a trained model; the grown model starts out computing the same function.")
    p.add_argument("--model", required=True, type=Path, help="Trained model directory.")
    p.add_argument("--out", required=True, type=Path, help="Output directory (use as --init-from for train).")
    p.add_argument("--layers", type=int, default=None, help="New number of layers.")
    p.add_argument("--width", type=int, default=None, help="New width (embedding size); heads scale with it.")
    p.add_argument("--version", type=int, default=None, help="Architecture version to record (e.g. 4).")

    p = sub.add_parser("export", help="Write a slim inference-only copy of a model (no optimizer state).")
    p.add_argument("--model", required=True, type=Path, help="Trained model directory.")
    p.add_argument("--out", required=True, type=Path, help="Output directory.")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"), help="bf16 halves the size (default).")
    p.add_argument("--max-shard-mb", type=float, default=None,
                   help="Split weights into files of at most this size (e.g. 45 to stay under GitHub's limits).")

    p = sub.add_parser("autotrain", help="Grow, train, tune, evaluate and publish a model across GPU sessions "
                                         "(Kaggle/Colab), syncing checkpoints to Hugging Face.")
    p.add_argument("--base-model", required=True, type=Path, help="Model to grow from, e.g. models/ycode-lm-v4.")
    p.add_argument("--preset", default="v5-385m", help="Target architecture (default v5-385m).")
    p.add_argument("--work", type=Path, default=Path("ycode-autotrain"), help="Local working directory.")
    p.add_argument("--hf-repo", default=None, help="Hugging Face repo for the finished model, e.g. you/ycode-lm-v5.")
    p.add_argument("--ckpt-repo", default=None, help="Private repo for checkpoints (default: <hf-repo>-checkpoints).")
    p.add_argument("--public", action="store_true", help="Make the finished-model repo public.")
    p.add_argument("--local-hub", type=Path, default=None, help="Use a local folder instead of Hugging Face.")
    p.add_argument("--pretrain-steps", type=int, default=6000)
    p.add_argument("--sft-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--nproc", type=int, default=None, help="GPUs to use (default: all visible).")
    p.add_argument("--hours", type=float, default=11.0, help="Session time budget (Kaggle sessions last 12 h).")
    p.add_argument("--segment-minutes", type=float, default=90.0, help="Checkpoint upload interval.")
    p.add_argument("--max-mb", type=float, default=1000.0, help="Cap on the code corpus size.")
    p.add_argument("--source", action="append", type=Path, default=[], help="Code to train on (default: "
                   "the Python standard library + installed packages).")
    p.add_argument("--no-grad-checkpoint", action="store_true", help="Faster but needs much more GPU memory.")
    p.add_argument("--shard", default="auto", choices=("auto", "yes", "no"),
                   help="Split the model across GPUs (FSDP). auto: for models over 600M parameters.")
    p.add_argument("--lr", type=float, default=None, help="Pretraining LR for AdamW-trained weights.")
    p.add_argument("--muon-lr", type=float, default=None, help="Pretraining LR for Muon-trained matrices.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    p.add_argument("--max-segments", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-interval", type=int, default=250)

    p = sub.add_parser("basics", help="Build an SFT dataset of basic exercises (write a function, fix a bug) "
                                      "with verified solutions.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path, help="The model's tokenizer.json.")
    p.add_argument("--replay-from", type=Path, default=None, help="Earlier SFT dataset to mix in.")
    p.add_argument("--replay", type=int, default=0, help="How many earlier examples to mix in.")

    p = sub.add_parser("push-hf", help="Upload an exported model folder to a Hugging Face model repo.")
    p.add_argument("--model", required=True, type=Path, help="Folder from `ycode-lm export`.")
    p.add_argument("--repo", required=True, help="Repo id, e.g. you/ycode-lm-v5.")
    p.add_argument("--public", action="store_true")

    p = sub.add_parser("eval", help="Score models: bits/byte on held-out code + pass rate on coding problems.")
    p.add_argument("--model", action="append", type=Path, default=[],
                   help="Model directory (repeat to compare several, e.g. v1 and v2).")
    p.add_argument("--heldout", action="append", type=Path, default=[],
                   help="Code the models never trained on, for bits-per-byte (repeatable).")
    p.add_argument("--samples", type=int, default=1, help="Samples per problem for pass@k (default 1 = greedy).")
    p.add_argument("--problems", default="standard", choices=("standard", "fresh", "both"),
                   help="Coding problems: the standard 30, 20 fresh ones, or both.")
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
        if args.sft_only:
            from ycode.lm.data import prepare_sft_dataset

            if args.tokenizer is None:
                print("error: --sft-only needs --tokenizer path/to/tokenizer.json", file=sys.stderr)
                return 2
            try:
                prepare_sft_dataset(sources, args.out, args.tokenizer, extra_sft=args.sft_data,
                                    exclude=args.exclude)
            except (ValueError, OSError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0
        try:
            prepare_dataset(sources, args.out, vocab_size=args.vocab_size, extra_sft=args.sft_data,
                            exclude=args.exclude, workers=args.workers, tokenizer_path=args.tokenizer,
                            max_mb=args.max_mb)
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
            warmup_steps=args.warmup if args.warmup is not None else (50 if sft else 100),
            max_steps=args.steps or (1000 if sft else 5000),
            max_minutes=args.minutes,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            device=args.device,
            precision=args.precision,
            schedule=args.schedule,
            compile=args.compile,
            grad_checkpoint=args.grad_checkpoint,
            shard=args.shard,
            train_layers=args.train_layers,
            optimizer=args.optimizer or ("muon" if str(getattr(args, "preset", "")).startswith("v3") else "adamw"),
            muon_lr=args.muon_lr,
            pack=not getattr(args, "no_pack", False),
            anneal_mix=getattr(args, "anneal_mix", 0.0),
            seed=args.seed,
            init_from=str(args.init_from) if args.init_from else None,
            resume=getattr(args, "resume", False),
            until_step=args.until_step,
            save_interval=args.save_interval,
        )
        out = args.out or default_local_model_dir()
        try:
            summary = train(args.data, out, cfg)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nInterrupted. The last checkpoint is in", out)
            return 130
        if summary:  # empty on non-zero ranks of a multi-GPU run; rank 0 reports
            print(f"Saved model to {out} (val loss {summary['final']['val']:.3f})")
        return 0

    if args.command == "autotrain":
        return _autotrain(args)

    if args.command == "push-hf":
        from ycode.lm.hub import HFHub

        try:
            hub = HFHub(args.repo, private=not args.public)
            hub.upload_folder(args.model, "", f"Upload {args.model.name}")
        except Exception as exc:  # noqa: BLE001 - network/auth errors become a clear message
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Uploaded {args.model} to {hub.url()}")
        return 0

    if args.command == "grow":
        from ycode.lm.grow import grow_depth, grow_width
        from ycode.lm.train import load_checkpoint, save_checkpoint

        try:
            model, tok, payload = load_checkpoint(args.model)
            if args.layers is None and args.width is None:
                raise ValueError("pass --layers and/or --width")
            grown = model
            if args.width is not None and args.width != model.cfg.n_embd:
                grown = grow_width(grown, args.width, arch_version=args.version)
            if args.layers is not None and args.layers != grown.cfg.n_layer:
                grown = grow_depth(grown, args.layers, arch_version=args.version)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        save_checkpoint(args.out, grown, tok, step=0, stage="grown", val_loss=None)
        print(f"Grew {model.cfg.n_layer}x{model.cfg.n_embd} -> {grown.cfg.n_layer}x{grown.cfg.n_embd}: "
              f"{model.num_params() / 1e6:.2f}M -> "
              f"{grown.num_params() / 1e6:.2f}M parameters, saved to {args.out}")
        return 0

    if args.command == "basics":
        from ycode.lm.basics import write_basics_dataset

        write_basics_dataset(args.out, args.tokenizer, replay_dir=args.replay_from, replay=args.replay)
        return 0

    if args.command == "export":
        from ycode.lm.train import export_checkpoint

        try:
            export_checkpoint(args.model, args.out, dtype=args.dtype, max_shard_mb=args.max_shard_mb)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        files = sorted(args.out.glob("model*.pt"))
        total = sum(f.stat().st_size for f in files) / 1e6
        print(f"Exported {args.model} -> {args.out} ({len(files)} file(s), {total:.1f} MB, {args.dtype})")
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


def _autotrain(args) -> int:  # noqa: ANN001
    from ycode.lm.autotrain import AutoTrainer, Plan, preset_params
    from ycode.lm.hub import HFHub, LocalHub

    import torch

    nproc = args.nproc if args.nproc is not None else max(1, torch.cuda.device_count())
    big = preset_params(args.preset) > 600e6
    shard = nproc > 1 and (args.shard == "yes" or (args.shard == "auto" and big))
    if big and not shard and torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gb < 40:
            print(f"warning: {args.preset} needs about 18 GB of GPU memory for weights and optimizer state "
                  f"alone; with one {gb:.0f} GB GPU it will likely run out of memory. Use 2+ GPUs "
                  "(sharded) or a 40 GB+ GPU.", file=sys.stderr)
    extra = {k: v for k, v in (("lr", args.lr), ("muon_lr", args.muon_lr)) if v is not None}
    plan = Plan(shard=shard, release_shard_mb=1000.0 if big else 45.0, **extra,base_model=str(args.base_model), preset=args.preset, sources=[str(s) for s in args.source],
                max_mb=args.max_mb, pretrain_steps=args.pretrain_steps, sft_steps=args.sft_steps,
                batch_size=args.batch_size, grad_accum=args.grad_accum, nproc=nproc, precision=args.precision,
                grad_checkpoint=not args.no_grad_checkpoint, compile=args.compile, session_hours=args.hours,
                segment_minutes=args.segment_minutes, max_segments=args.max_segments, device=args.device,
                eval_interval=args.eval_interval)
    try:
        if args.local_hub is not None:
            ckpt, release = LocalHub(args.local_hub, "checkpoints"), LocalHub(args.local_hub, "release")
        else:
            if not args.hf_repo:
                print("error: pass --hf-repo you/ycode-lm-v5 (or --local-hub DIR for a dry run)", file=sys.stderr)
                return 2
            ckpt = HFHub(args.ckpt_repo or f"{args.hf_repo}-checkpoints", private=True)
            release = HFHub(args.hf_repo, private=not args.public)
        state = AutoTrainer(plan, args.work, ckpt, release).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Phase: {state.phase} | pretrain {state.pretrain_step}/{plan.pretrain_steps} | "
          f"sft {state.sft_step}/{plan.sft_steps}")
    return 0


def _eval(args) -> int:  # noqa: ANN001
    import json

    from ycode.lm.evaluate import BUGGY, PROBLEMS, bits_per_byte, bugfix_eval, functional_eval, load_heldout_texts
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
            with lm.precision():
                row["bits_per_byte"] = round(bits_per_byte(lm.model, lm.tokenizer, texts, device=lm.device), 4)
        if not args.no_functional:
            from ycode.lm.evaluate import fresh_problems

            problems = {"standard": PROBLEMS, "fresh": fresh_problems(),
                        "both": PROBLEMS + fresh_problems()}[args.problems]
            res = functional_eval(lm, samples=args.samples, problems=problems,
                                  log=print if args.verbose else None)
            row["pass@1"] = round(res.pass_at_1, 4)
            if res.pass_at_k is not None:
                row[f"pass@{res.k}"] = round(res.pass_at_k, 4)
            row["solved"] = res.solved
            fix_rate, fixed = bugfix_eval(lm, log=print if args.verbose else None)
            row["fix@1"] = round(fix_rate, 4)
            row["fixed"] = fixed
        rows.append(row)
    print()
    header = ["model", "version", "params_m", "bits_per_byte", "pass@1"] + sorted(
        {k for r in rows for k in r if k.startswith("pass@") and k != "pass@1"}) + ["fix@1"]
    print(" | ".join(h for h in header))
    for r in rows:
        print(" | ".join(str(r.get(h, "-")) for h in header))
    if not args.no_functional:
        n_problems = {"standard": 30, "fresh": 20, "both": 50}[args.problems]
        print(f"\n({n_problems} problems, {len(BUGGY)} bug fixes; bits/byte: lower is better; "
              "pass@k / fix@1: higher is better)")
    for r in rows:
        if r.get("solved"):
            print(f"{r['model']} solved: {', '.join(r['solved'])}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
