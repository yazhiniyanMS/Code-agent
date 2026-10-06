"""`ycode-lm autotrain`: grow, train, tune, evaluate and publish a model across sessions.

Designed for free GPU notebooks (Kaggle, Colab) whose sessions end after a few hours. Every run
picks up where the last one stopped:

1. Download the latest checkpoint and progress from the (private) checkpoint repository.
2. Prepare the data (deterministic, so every session gets the same dataset).
3. Phase ``pretrain``: grow the base model to the target preset, then continued pretraining.
4. Phase ``sft``: instruction tuning.
5. Phase ``release``: evaluate, export bf16 shards, write a model card, upload the model.

Training runs in segments; after each one the checkpoint and progress are uploaded, so at most
one segment of work is lost when a session ends. A new segment only starts if it fits in the
remaining session time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ycode.lm.hub import Hub

Log = Callable[[str], None]
HELD_OUT = ("click", "anyio", "jsonschema", "markdown_it", "starlette", "uvicorn", "h11", "attr",
            "pluggy", "filelock", "idna")


@dataclass
class Plan:
    base_model: str  # trained model to grow from (e.g. models/ycode-lm-v4)
    preset: str = "v5-385m"
    sources: list[str] = field(default_factory=list)  # empty = Python stdlib + site-packages
    max_mb: float | None = 1000.0
    pretrain_steps: int = 6000
    sft_steps: int = 800
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 6e-4
    muon_lr: float = 0.01
    sft_lr: float = 2e-4
    nproc: int = 1  # GPUs (one process each)
    precision: str = "auto"
    grad_checkpoint: bool = True
    compile: bool = False
    session_hours: float = 11.0  # stop starting new segments after this
    segment_minutes: float = 90.0
    first_segment_steps: int = 50
    eval_interval: int = 250
    max_segments: int | None = None  # for tests / dry runs
    device: str = "auto"
    model_name: str = "YCode-LM v5"


@dataclass
class State:
    phase: str = "pretrain"  # pretrain | sft | release | done
    pretrain_step: int = 0
    sft_step: int = 0
    seconds_per_step: float | None = None
    sessions: int = 0
    data_uploaded: bool = False
    history: list[dict] = field(default_factory=list)


def _default_sources() -> list[str]:
    import site
    import sysconfig

    dirs = [sysconfig.get_paths()["stdlib"], *site.getsitepackages()]
    return [d for d in dict.fromkeys(dirs) if Path(d).is_dir()]


class AutoTrainer:
    def __init__(self, plan: Plan, work_dir: Path, ckpt_hub: Hub, release_hub: Hub | None,
                 *, log: Log = print) -> None:
        self.plan, self.work, self.hub, self.release_hub, self.log = plan, Path(work_dir), ckpt_hub, release_hub, log
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "state" / "state.json"
        self.started = time.time()
        self.segments = 0

    # ------------------------------------------------------------- state

    def load_state(self) -> State:
        self.hub.download_folder("state", self.work / "state")
        if self.state_path.is_file():
            return State(**json.loads(self.state_path.read_text()))
        return State()

    def save_state(self, state: State, message: str) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(state), indent=2))
        self.hub.upload_folder(self.state_path.parent, "state", message)

    def remaining_seconds(self) -> float:
        return self.plan.session_hours * 3600 - (time.time() - self.started)

    # -------------------------------------------------------------- steps

    def prepare_data(self, state: State) -> Path:
        data = self.work / "data"
        if (data / "train.bin").is_file() and (data / "sft_index.npy").is_file():
            return data
        # Preparing a large corpus takes a while; later sessions reuse the uploaded copy.
        if state.data_uploaded and self.hub.download_folder("data", data) and (data / "train.bin").is_file():
            self.log("Using the dataset prepared in an earlier session.")
            return data
        from ycode.lm.data import prepare_dataset
        from ycode.lm.train import load_checkpoint

        _, tok, _ = load_checkpoint(self.plan.base_model)
        tok_path = self.work / "base-tokenizer.json"
        tok.save(tok_path)
        sources = [Path(s) for s in (self.plan.sources or _default_sources())]
        exclude = [f"*/{name}/*" for name in HELD_OUT]
        self.log(f"Preparing data from {len(sources)} source(s), up to {self.plan.max_mb} MB ...")
        prepare_dataset(sources, data, tokenizer_path=tok_path, exclude=exclude, max_mb=self.plan.max_mb,
                        log=self.log)
        self.log("Uploading the prepared dataset so later sessions can skip this step ...")
        self.hub.upload_folder(data, "data", "prepared dataset")
        state.data_uploaded = True
        self.save_state(state, "dataset uploaded")
        return data

    def grow_initial(self) -> Path:
        init = self.work / "init"
        if (init / "model.pt").is_file():
            return init
        from ycode.lm.grow import grow_depth, grow_width
        from ycode.lm.model import PRESETS
        from ycode.lm.train import load_checkpoint, save_checkpoint

        target = PRESETS[self.plan.preset]
        model, tok, _ = load_checkpoint(self.plan.base_model)
        version = target.get("arch_version")
        grown = model
        if target["n_embd"] != model.cfg.n_embd:
            grown = grow_width(grown, target["n_embd"], new_heads=target["n_head"],
                               new_kv_heads=target.get("n_kv_head"), arch_version=version)
        if target["n_layer"] != grown.cfg.n_layer:
            grown = grow_depth(grown, target["n_layer"], arch_version=version)
        save_checkpoint(init, grown, tok, step=0, stage="grown", val_loss=None)
        self.log(f"Grew {self.plan.base_model} ({model.num_params() / 1e6:.1f}M) -> {self.plan.preset} "
                 f"({grown.num_params() / 1e6:.1f}M)")
        return init

    def _train_cmd(self, stage: str, data: Path, init: Path, out: Path, total: int, until: int) -> list[str]:
        p = self.plan
        launcher = ([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={p.nproc}"]
                    if p.nproc > 1 else [sys.executable])
        cmd = launcher + ["-m", "ycode.lm.cli", "train" if stage == "pretrain" else "sft",
                          "--data", str(data), "--init-from", str(init), "--out", str(out), "--resume",
                          "--steps", str(total), "--until-step", str(until), "--batch-size", str(p.batch_size),
                          "--grad-accum", str(p.grad_accum), "--eval-interval", str(p.eval_interval),
                          "--device", p.device, "--precision", p.precision]
        if stage == "pretrain":
            cmd += ["--lr", str(p.lr), "--muon-lr", str(p.muon_lr), "--optimizer", "muon"]
        else:
            cmd += ["--lr", str(p.sft_lr), "--optimizer", "adamw"]
        if p.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if p.compile:
            cmd.append("--compile")
        return cmd

    def run_phase(self, state: State, stage: str, data: Path, init: Path, out: Path, total: int) -> bool:
        """Train segments until done or out of time. Returns True when the phase is finished."""
        attr = "pretrain_step" if stage == "pretrain" else "sft_step"
        ckpt_name = "base" if stage == "pretrain" else "chat"
        while getattr(state, attr) < total:
            if self.plan.max_segments is not None and self.segments >= self.plan.max_segments:
                self.log("Segment limit reached for this run.")
                return False
            step = getattr(state, attr)
            if state.seconds_per_step:
                steps = int(self.plan.segment_minutes * 60 / state.seconds_per_step)
                budget = self.remaining_seconds() - 15 * 60  # keep time for the upload
                steps = min(steps, int(budget / state.seconds_per_step))
            else:
                steps = self.plan.first_segment_steps
            if steps < 1:
                self.log("Not enough session time left for another segment; stopping here.")
                return False
            until = min(total, step + max(1, steps))
            self.log(f"[{stage}] training steps {step} -> {until} of {total} ...")
            t0 = time.time()
            proc = subprocess.run(self._train_cmd(stage, data, init, out, total, until), env=os.environ.copy())
            if proc.returncode != 0:
                raise RuntimeError(f"training failed (exit {proc.returncode}); see the output above")
            done = json.loads((out / "info.json").read_text())["step"]
            if done > step:
                state.seconds_per_step = (time.time() - t0) / (done - step)
            setattr(state, attr, done)
            self.segments += 1
            self.log(f"[{stage}] at step {done}/{total}; uploading checkpoint ...")
            self.hub.upload_folder(out, f"checkpoint-{ckpt_name}", f"{stage} step {done}")
            self.save_state(state, f"{stage} step {done}")
            if hasattr(self.hub, "squash_history"):
                self.hub.squash_history()
        return True

    def release(self, state: State) -> None:
        from ycode.lm.evaluate import BUGGY, PROBLEMS, bits_per_byte, bugfix_eval, functional_eval, load_heldout_texts
        from ycode.lm.generate import LocalLM
        from ycode.lm.train import export_checkpoint

        chat = self.work / "chat"
        lm = LocalLM(chat, device=self.plan.device)
        results: dict = {}
        held = [Path(d) / name for d in _default_sources() for name in HELD_OUT if (Path(d) / name).is_dir()]
        texts = load_heldout_texts(held) if held else []
        if texts:
            results["bits_per_byte"] = round(bits_per_byte(lm.model, lm.tokenizer, texts, device=lm.device), 4)
        func = functional_eval(lm)
        results["pass@1"] = f"{len(func.solved)}/{len(PROBLEMS)}"
        fix_rate, fixed = bugfix_eval(lm)
        results["fix@1"] = f"{len(fixed)}/{len(BUGGY)}"
        self.log(f"Evaluation: {results}")
        release = self.work / "release"
        export_checkpoint(chat, release, dtype="bf16", max_shard_mb=45)
        (release / "README.md").write_text(model_card(self.plan, state, results, lm.num_params))
        (release / "eval.json").write_text(json.dumps(results, indent=2))
        target = self.release_hub or self.hub
        target.upload_folder(release, "", f"Release {self.plan.model_name}")
        self.log(f"Published {self.plan.model_name} to {target.url()}")
        state.history.append({"release": results})

    # --------------------------------------------------------------- main

    def run(self) -> State:
        state = self.load_state()
        state.sessions += 1
        self.log(f"Session {state.sessions}: phase {state.phase}, pretrain {state.pretrain_step}/"
                 f"{self.plan.pretrain_steps}, sft {state.sft_step}/{self.plan.sft_steps}")
        data = self.prepare_data(state)
        if state.phase == "pretrain":
            base = self.work / "base"
            if state.pretrain_step > 0:
                self.hub.download_folder("checkpoint-base", base)
            # Only the very first session needs the grown starting point; later ones resume `base`.
            init = self.work / "init" if (base / "model.pt").is_file() else self.grow_initial()
            if not self.run_phase(state, "pretrain", data, init, base, self.plan.pretrain_steps):
                return state
            state.phase = "sft"
            self.save_state(state, "pretraining finished")
        if state.phase == "sft":
            base, chat = self.work / "base", self.work / "chat"
            if not (base / "model.pt").is_file():
                self.hub.download_folder("checkpoint-base", base)
            if state.sft_step > 0:
                self.hub.download_folder("checkpoint-chat", chat)
            if not self.run_phase(state, "sft", data, base, chat, self.plan.sft_steps):
                return state
            state.phase = "release"
            self.save_state(state, "instruction tuning finished")
        if state.phase == "release":
            chat = self.work / "chat"
            if not (chat / "model.pt").is_file():
                self.hub.download_folder("checkpoint-chat", chat)
            self.release(state)
            state.phase = "done"
            self.save_state(state, "released")
        if state.phase == "done":
            self.log("All phases complete.")
        return state


def model_card(plan: Plan, state: State, results: dict, params: int) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in results.items())
    return f"""---
license: mit
library_name: ycode
tags:
- code
- python
- from-scratch
- ycode
---

# {plan.model_name} ({params / 1e6:.0f}M parameters)

A Python coding language model trained **from scratch** with the open-source
[YCode](https://github.com/yazhiniyanMS/Code-agent) `ycode-lm` pipeline: its own BPE tokenizer, its own
transformer and training loop, no pretrained weights from anyone else.

It was **grown** from YCode-LM v4 (100M parameters) to the `{plan.preset}` architecture. Growth keeps the
smaller model's function exactly at the start: new channels, attention heads and MLP units begin with
zeroed outputs and learn during training. Then it went through continued pretraining
({plan.pretrain_steps} steps with Muon) and instruction tuning ({plan.sft_steps} steps), on free GPU
notebooks, using `ycode-lm autotrain`.

## Evaluation

Measured with `ycode-lm eval`: bits per byte on Python packages excluded from training, plus
coding problems and bug fixes that are executed against unit tests.

| Metric | Value |
| --- | --- |
{rows}

This is a small research model. It writes well-formed Python and can fix simple bugs, but its code is
often wrong, so verify everything it produces.

## Use

```bash
git clone https://github.com/yazhiniyanMS/Code-agent && cd Code-agent
pip install -e ".[local]" huggingface_hub
huggingface-cli download <this repo> --local-dir models/ycode-lm-v5
ycode --local models/ycode-lm-v5          # or: ycode-lm chat --model models/ycode-lm-v5
```

The weights are bfloat16, split into shards of at most 45 MB; `model.pt` holds the config, the tokenizer
and the shard index. Training sessions used: {state.sessions}.
"""
