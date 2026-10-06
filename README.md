# YCode

```
██╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗
╚██╗ ██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ╚████╔╝ ██║     ██║   ██║██║  ██║█████╗
  ╚██╔╝  ██║     ██║   ██║██║  ██║██╔══╝
   ██║   ╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═╝    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

YCode is an AI coding agent that runs in your terminal. Start it inside a project, describe a task in plain English, and it works through it: it looks around the repository, makes a short plan, edits files, runs your tests and build, reads any failures, fixes them, and then tells you what changed.

It uses the Anthropic Claude API to do the reasoning. Everything else runs locally: the tool system, the permission checks and the terminal UI. The model sits behind a small provider interface, so you can add other LLM providers without rewriting the agent.

YCode also includes **YCode-LM**, a coding language model you train yourself from scratch: its own tokenizer, its own transformer, and training code, with no pretrained weights and no API key (see [Your own LLM](#your-own-llm-ycode-lm-no-api-key)).

```text
ycode> Fix the failing add() test

  ◇ Plan
    ▸ 1. Read calc.py
    ○ 2. Fix add
    ○ 3. Run tests

  ◇ Reading calc.py
  ✓ Read calc.py (2 of 2 lines)

  ◇ Editing calc.py
  ✓ Edited calc.py (+1 -1)
      @@ -1,2 +1,2 @@
       def add(a, b):
      -    return a - b
      +    return a + b

  ◇ Running `python -m pytest -q`
  ✓ Succeeded (exit code 0, 0.3s)

  Fixed `add` in calc.py; pytest passes (1 test).

  Changes:
   M calc.py

  Commands:
   ✓ python -m pytest -q

  ✓ YCode completed the task.  (5 steps)
```

## Features

- **Terminal first.** Uses Rich for output: spinners, streamed Markdown, syntax-highlighted diffs. Uses prompt_toolkit for input: persistent history, multiline editing, and tab completion for `/commands`.
- **Autonomous agent loop.** The model can make many tool calls per task, parallel calls included, until it is done. A configurable step limit (default 50) stops runaway loops.
- **Planning.** For non-trivial work the model writes a short plan through an `update_plan` tool and keeps it current. For trivial requests it skips the plan.
- **Verification.** The model picks verification commands from your project type (`npm test`, `pytest`, `cargo test`, ...), reads failures and iterates. It is told never to claim success it did not verify.
- **Tools:** `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `find_files`, `run_command`, `git_status`, `git_diff`, `git_log`, `git_branch`, `update_plan`.
- **Security.** Commands are classified as SAFE, REQUIRES_APPROVAL or BLOCKED. Risky ones need an explicit `y`. File access is confined to the workspace, and API keys are stripped from the environment that commands run in.
- **Repository awareness.** Detects Node.js, TypeScript, JavaScript, React, Next.js, Vue, Python, Java, Kotlin, C/C++, Rust, Go, Flutter/Dart, Ruby and PHP projects, plus the package manager and the likely test and build commands. Large generated directories are ignored.
- **Diff awareness.** After each task YCode summarizes the changed files and the commands it ran. `/diff` shows the full diff.
- **Project instructions.** A `YCODE.md` file in the repository is loaded into the agent's instructions.
- **Layered configuration:** global and per-project TOML files, environment variables and CLI flags.
- **Model-agnostic core.** The agent depends only on `LLMProvider`. There are two implementations: Claude, and your own local model.
- **Your own LLM.** `ycode-lm` builds a byte-level BPE tokenizer and a GPT-style transformer from scratch, pretrains it on code, instruction-tunes it to answer programming questions, and serves it to YCode with `ycode --local`. Everything runs offline.

## Architecture

```text
                ┌──────────────┐
                │   YCode CLI  │  cli.py, commands.py, ui/
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │ Agent Engine │  agent/loop.py, planner.py, context.py, prompts.py
                └──────┬───────┘
                       │
              ┌────────▼────────┐
              │   LLM Provider  │  llm/base.py (interface), llm/anthropic.py
              └────────┬────────┘
                       │
                  Claude API
                       │
              ┌────────▼────────┐
              │  Tool Execution │  tools/ (+ security/permissions.py)
              └────────┬────────┘
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Filesystem   Terminal       Git
```

```text
ycode/
├── __main__.py          # python -m ycode
├── cli.py               # argument parsing, REPL, graceful shutdown
├── app.py               # wires config, context, permissions, tools, provider, agent, UI
├── commands.py          # /help /clear /status /diff /files /model /plan /approval /exit
├── config.py            # layered TOML + env + CLI configuration
├── errors.py            # user-facing error hierarchy (no raw tracebacks)
├── agent/
│   ├── loop.py          # the agent loop, step limit, interruption handling
│   ├── planner.py       # Plan model + update_plan tool
│   ├── context.py       # project detection, YCODE.md loading
│   └── prompts.py       # system prompt
├── llm/
│   ├── base.py          # provider-neutral messages, ToolSpec, LLMProvider
│   ├── anthropic.py     # Claude via the official anthropic SDK (streaming)
│   └── local.py         # your own YCode-LM model (answer-only, no API key)
├── lm/                  # YCode-LM: the from-scratch model (needs the [local] extra)
│   ├── tokenizer.py     # byte-level BPE tokenizer for code
│   ├── model.py         # GPT transformer: RoPE, RMSNorm, SwiGLU, KV-cache sampling
│   ├── data.py          # corpus collection, instruction-pair extraction, encoding
│   ├── train.py         # pretraining + instruction tuning, checkpoints, resume
│   ├── optim.py         # Muon optimizer (v3) + AdamW wiring
│   ├── evaluate.py      # bits/byte on held-out code, pass@k on executed problems
│   ├── generate.py      # inference (chat + completion, streaming)
│   └── cli.py           # the `ycode-lm` command
├── tools/
│   ├── base.py          # Tool, ToolRegistry (validation + dispatch), ToolResult
│   ├── filesystem.py    # read/write/edit/list
│   ├── search.py        # regex content search, glob file search
│   ├── terminal.py      # run_command (timeouts, output capture, Ctrl+C)
│   └── git.py           # git helpers + read-only git tools
├── security/
│   └── permissions.py   # command classifier, PathGuard, PermissionManager
└── ui/
    ├── banner.py        # ASCII banner
    ├── console.py       # Rich renderer (implements AgentEvents)
    └── prompts.py       # prompt_toolkit input with history/multiline
```

The dependencies point one way. The agent knows nothing about Rich or Anthropic: it talks to an `LLMProvider` and reports progress through `AgentEvents` callbacks. The UI implements those callbacks. Tools don't know about the model.

**Adding a provider:** subclass `LLMProvider` in `ycode/llm/`, translate the neutral `UserMessage`/`AssistantMessage`/`ToolResultsMessage` types to your API's format, and register it in `ycode/llm/__init__.py:create_provider`. Assistant messages carry an opaque `provider_data` field. Your provider can use it to replay its own native content exactly, which Claude needs for its thinking blocks.

## Requirements

- Python 3.10+
- Git (optional, but needed for the git features)
- An Anthropic API key

## Installation

```bash
git clone <repo>
cd ycode
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # then put your key in .env
```

This installs a `ycode` command. `python -m ycode` works too.

## API key configuration

YCode reads `ANTHROPIC_API_KEY`. In order of precedence, the places it looks are:

1. Your shell environment: `export ANTHROPIC_API_KEY=sk-ant-...`
2. A `.env` file in the project you run YCode in
3. `~/.ycode/.env`, which applies to every project
4. The `.env` in the YCode checkout itself, which is handy with `pip install -e .`

Values that are already set are never overridden. Any of these work too: `ANTHROPIC_AUTH_TOKEN`, an `ant auth login` profile, or workload identity federation. If YCode finds no credentials, it shows the banner, prints a clear error and exits with status 1.

YCode never prints your key. It also removes `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the environment of commands it runs, so a stray `env` can't leak the key to the model. **Never commit `.env`.** The included `.gitignore` excludes it.

## Your own LLM (YCode-LM, no API key)

YCode-LM is a programming language model you train from scratch on your own machine. Nothing is downloaded: the tokenizer, the model architecture and the training loop are all in this repository (`ycode/lm/`), and the data is code you already have.

**What it is:**
- **Tokenizer:** byte-level BPE (`ycode/lm/tokenizer.py`) trained on your corpus. A code-aware pre-tokenizer turns indentation runs, identifiers and operators into compact tokens.
- **Model:** a decoder-only transformer (`ycode/lm/model.py`) with rotary position embeddings, RMSNorm, SwiGLU MLPs, tied embeddings and a KV cache for fast sampling. **Version 2** (the default) adds grouped-query attention, which shares key/value heads for a 4× smaller KV cache and faster generation, and QK-norm, which keeps attention stable at higher learning rates. v1 checkpoints still load.
- **Training data:** any code directories you point it at. With no `--source`, it uses the Python standard library, which every Python install has.
- **Two training stages:**
  1. *Pretraining:* next-token prediction on code.
  2. *Instruction tuning (SFT):* teaches the model to answer. The data is generated automatically from the real code in your corpus, so every answer is real code:
     - write a function from its description
     - explain a function
     - complete a function from its signature and docstring
     - write a docstring
     - explain a class
     - **find and fix a bug**: YCode-LM injects realistic one-token bugs (`<` vs `<=`, `+` vs `-`, off-by-one, `and`/`or`, `True`/`False`). The original code is the fix, so these examples are correct by construction.

     You can add your own pairs as JSONL (`{"prompt": ..., "response": ...}`) with `--sft-data`.

**What v2 improves:**

| | v1 | v2 |
| --- | --- | --- |
| Attention | multi-head | grouped-query (smaller cache) + QK-norm |
| LR schedule | cosine over steps; cut short by `--minutes` | warmup-stable-decay. Progress is the larger of steps done and time used, so a time-limited run always finishes its decay |
| End of pretraining | code only | anneals with 20% instruction data during the decay phase, so the model already knows the chat format before SFT |
| SFT batches | one example per row, padded | several examples packed per row, so no compute is wasted on padding |
| Data | write/explain pairs | + bug fixing, completion, docstrings, classes; de-duplicated prompts; licence headers stripped |
| Data prep | single process | tokenization on all CPU cores |
| Measurement | loss only | `ycode-lm eval`: bits per byte on held-out code + pass@k on 30 executed coding problems |

**Try the included models (no training needed).** The repository ships two trained models:

| Model | Params | Size | Card |
| --- | --- | --- | --- |
| **YCode-LM v4** | **100M** | 200 MB in 5 shards | [`models/ycode-lm-v4`](models/ycode-lm-v4/README.md) |
| YCode-LM v3 | 40.8M | 81.8 MB | [`models/ycode-lm-v3`](models/ycode-lm-v3/README.md) |
| YCode-LM v2 | 7.7M | 15.6 MB | [`models/ycode-lm-v2`](models/ycode-lm-v2/README.md) |

```bash
pip install -e ".[local]"
ycode --local models/ycode-lm-v4          # or: ycode-lm chat --model models/ycode-lm-v4
```

**Set up and train:**

```bash
pip install -e ".[local]"          # adds PyTorch + NumPy

# 1. Build the dataset: collect code, train the tokenizer, create instruction pairs.
#    --exclude keeps code out of training so you can evaluate on it later.
ycode-lm prepare --out data/ --source ~/my-projects --source /usr/lib/python3.11 --exclude '*/holdout/*'

# 2. Pretrain on code (bf16 on CPUs with AMX/AVX512-BF16, CUDA or Apple MPS when available)
ycode-lm train --data data/ --out models/base --preset v2-small --minutes 180

# 3. Instruction-tune it so it answers requests
ycode-lm sft --data data/ --init-from models/base --out ~/.ycode/models/ycode-lm --minutes 40

# 4. Measure it (compare several models by repeating --model)
ycode-lm eval --model ~/.ycode/models/ycode-lm --heldout path/to/unseen/code --samples 5

# 5. Use it
ycode-lm chat                                   # quick Q&A in the terminal
ycode --local                                   # inside YCode (default model dir ~/.ycode/models/ycode-lm)
ycode --local models/chat                       # or point to any trained model
```

Other commands: `ycode-lm sample --model models/base --prompt "def quicksort("` (raw code completion), `ycode-lm info`, `ycode-lm export --model DIR --out DIR` (slim bf16 copy for sharing, about 4× smaller than a training checkpoint), and `ycode-lm train --resume` (continue an interrupted run). Ctrl+C during training keeps the last checkpoint. Training options include `--schedule wsd|cosine`, `--anneal-mix`, `--no-pack`, `--compile` (torch.compile) and `--grad-accum`.

**Version 3: the 40M-parameter model.** `--preset v3-40m` uses the v2 architecture scaled to 40.8M parameters with a 1024-token context. It trains with the **Muon optimizer** by default: transformer weight matrices are updated with orthogonalized momentum (Newton–Schulz iterations), while embeddings and norms stay on AdamW. On small transformers, Muon reaches a given loss with fewer tokens, which matters most when compute is the bottleneck. Use `--optimizer adamw` to switch back.

```bash
ycode-lm train --data data/ --out models/v3-base --preset v3-40m --compile --minutes 480
ycode-lm sft   --data data/ --init-from models/v3-base --out ~/.ycode/models/ycode-lm --compile --minutes 60
```

In an A/B test (same model, data, seed and 600 steps), Muon reached a held-out loss of **4.13** against AdamW's **4.26**, for 14% more wall-clock per step at that tiny size; the overhead shrinks as models grow.

A 40M model wants far more data and compute than a CPU can supply. The compute-optimal budget is about 800M training tokens, while 8 hours on a 4-core CPU covers about 50M. On a single consumer GPU, the same run takes hours instead of days.

**Measured results (v1 → v4).** All four models were trained on a 4-core CPU and evaluated the same way. Bits per byte is measured on 11 packages none of them saw in training. The code tasks are *executed* against unit tests:

| | v1 | v2 | v3 | v4 |
| --- | --- | --- | --- | --- |
| Parameters | 6.9M | 7.7M | 40.8M | 100M |
| Training | ~41M tokens | ~82M tokens | ~60M tokens (Muon) | grown from v3 + ~24.6M tokens |
| Bits per byte on held-out code | 1.095 | 0.966 | 0.857 | **0.814** |
| Write a function, pass@1 (30 problems) | 0 / 30 | 0 / 30 | 0 / 30 | **1 / 30** |
| Fix an injected bug, fix@1 (15 functions) | 0 / 15 | **5 / 15** | 4 / 15 | **5 / 15** |

Each version models real code better than the last. v2–v4 write short, well-formed answers and can find and fix simple bugs ("The bug is in `result = 1`. It should be `result = 0`."), while v1 falls into repetition loops. v4 is the first to write a correct function from a description (`square` → `return x * x`). Writing functions reliably needs much more training data and compute than a CPU provides.

**Growing models.** `ycode-lm grow --model v3 --out v4-init --layers 34` deepens a trained model by inserting copies of existing blocks with zeroed output projections. The result computes exactly the same function, so `train --init-from v4-init` continues from everything the small model learned. v4 was built this way. For files over GitHub's 100 MB limit, `export --max-shard-mb 45` splits the weights into shards that load transparently.

For long runs on machines that may be interrupted, train in segments: `--until-step N` stops and saves, `--resume` continues with the same LR schedule, and `--save-interval N` saves cheaply between evaluations.

**Evaluation.** `ycode-lm eval` reports:
- **Bits per byte** on code the model never saw. Lower is better. It is comparable across tokenizers, so v1, v2 and v3 compare fairly.
- **pass@1 / pass@k** on 30 small programming problems, plus **fix@1** on 15 functions with one injected bug each. YCode-LM writes each solution or fix, and it is *executed* against unit tests in an isolated subprocess with a timeout. Only evaluate models you trained yourself, because generated code is run.

**Model sizes** (`--preset`; parameter counts are for an 8k vocabulary):

| Preset | Layers × width | Heads (KV) | Context | Params | Where to train |
| --- | --- | --- | --- | --- | --- |
| **`v4-100m`** | 34 × 512 | 8 (2) | 1024 | **100.0M** | grow from v3 (`ycode-lm grow`); GPU recommended for training |
| **`v3-40m`** | 13 × 512 | 8 (2) | 1024 | **40.8M** | GPU ideal; CPU works but slowly (measured ~2.1k tok/s on 4 Xeon cores with bf16 + compile) |
| `v3-tiny` | 3 × 64 | 4 (2) | 128 | ~0.7M | tests |
| `v2-tiny` | 2 × 64 | 4 (2) | 128 | ~0.6M | tests / smoke runs |
| **`v2-small`** (default) | 8 × 256 | 8 (2) | 512 | ~7.7M | laptop CPU, 2–4 h |
| `v2-base` | 12 × 512 | 8 (2) | 1024 | ~38M | GPU |
| `v2-medium` | 16 × 768 | 12 (4) | 1024 | ~107M | GPU, many hours |
| `v2-large` | 24 × 1024 | 16 (4) | 2048 | ~274M | multi-hour GPU runs |

The v1 presets (`tiny`, `small`, `base`, `medium`, `large`) are still available.

Use `--layers/--heads/--embd/--context` to customize a preset.

**Using it in YCode.** `ycode --local`, `YCODE_PROVIDER=local`, or `provider = "local"` in `config.toml` switches YCode to your model. Related settings: `local_model` (path), `local_max_tokens` and `local_temperature`. Nothing is sent over the network.

**Be realistic about quality.** Frontier coding models are trained on trillions of tokens with thousands of GPUs. A few-million-parameter model trained for an hour or two on a CPU learns real Python syntax, idioms and naming conventions, and can write small functions and short explanations. It will also often be wrong, repetitive or confused. Two consequences:

- **Answer-only mode.** In YCode, the local model runs in answer-only mode. It hasn't learned the tool-calling protocol, so it cannot read, edit or run files in your project. YCode shows a notice when it starts. Use Claude (the default provider) for autonomous coding tasks.
- **How to make it better:** more and better data (your own repositories), a bigger preset, a GPU and longer training. The loss figures in `train_log.json` show whether it is still improving.

## Usage

```bash
cd my-project
ycode                                  # interactive session
ycode "add a --verbose flag to the CLI" # one-shot: run the task, then exit
ycode -p "explain src/auth.ts"         # same, with -p
ycode -m claude-sonnet-5-5             # pick a model
ycode --approval-mode strict           # approve every write and non-read-only command
ycode --yes                            # auto-approve risky commands (blocked ones stay blocked)
ycode -C ../other-project              # run against another directory
```

In the interactive session:

| Key / command | Action |
| --- | --- |
| Enter | Send the request |
| Esc+Enter (or a line ending in `\`) | Insert a newline |
| Up / Down | History (stored in `~/.ycode/history`) |
| Ctrl+C | Interrupt the running task; YCode kills any running command and returns to the prompt |
| Ctrl+C twice / Ctrl+D | Exit |
| `/help` | List commands |
| `/status` | Project, branch, modified files, model, steps used, token usage |
| `/diff [full] [path...]` | Show the git diff, with a stat summary if it is very large |
| `/files` | Changed files: working tree plus files written this session |
| `/model [id]` | Show or switch the model for this session |
| `/plan` | Show the current plan |
| `/approval [strict\|normal\|auto]` | Show or change the approval mode |
| `/clear` | Forget the conversation and clear the screen |
| `/exit`, `/quit` | Quit |

Slash commands are handled locally and are never sent to the model. Each request continues the same conversation, so a follow-up like "now add tests for that" works.

## Examples

```text
ycode> Add dark mode to the settings page
ycode> Why does `npm run build` fail? Fix it.
ycode> Write unit tests for src/utils/date.ts and make sure they pass
ycode> Rename the `User.fullname` field to `full_name` everywhere
ycode> Explain how requests are authenticated in this project
ycode> Commit these changes with a good message      # git commit asks for approval
```

## Security model

YCode gives a language model the ability to edit files and run commands, so it treats both as risky.

**Command classification** (`ycode/security/permissions.py`). Every command line is split into segments (`&&`, `||`, `;`, `|`), and command substitutions (`$(...)` and backticks) and `bash -c "..."` strings are classified recursively. The most severe result wins.

| Level | Examples | Behavior |
| --- | --- | --- |
| SAFE | `npm test`, `pytest`, `git status`, `ls`, `cargo build` | Runs immediately |
| REQUIRES_APPROVAL | `rm`, `mv`, `sudo`, `chmod`, `chown`, `kill`, `git push`, `git reset --hard`, `git clean`, `git commit`, `git checkout -- .`, `npm publish`, `curl … \| sh`, `find -delete`, writes redirected outside the project, reading `.env`/key files | Shows `⚠ YCode wants to run: … Allow? [y/N/a]` |
| BLOCKED | `rm -rf /`, `rm -rf ~`, fork bombs, `mkfs`, `dd of=/dev/…`, `shutdown`, writes to raw devices | Never runs, even with `--yes` |

At the approval prompt, `y` allows the command once and `a` allows that exact command for the rest of the session. Anything else denies it, and the model is told not to retry.

**Approval modes:**
- `normal` (default): SAFE commands run, risky commands ask, blocked commands are refused.
- `strict`: also asks before every file write and every command that isn't read-only.
- `auto`: risky commands run without asking. Blocked commands are still refused.

**Filesystem confinement.** Every path is resolved, symlinks included, and has to stay inside the project directory. `../`, absolute paths and symlinks that point outside the project are rejected. Writes into `.git/` are always rejected. Reading files that look like secrets (`.env*`, `*.pem`, `id_rsa`, ...) asks first. `allow_outside_workspace = true` in your *global* config lifts the confinement.

**Untrusted repositories.** A project's `.ycode/config.toml` can only make settings *stricter*. It cannot switch on `auto` approvals or access outside the workspace, so cloning a repository can't silently loosen your safety settings.

**Git.** YCode never commits or pushes on its own. Those commands always need your approval.

**Limits.** This is a guardrail, not a sandbox. When the agent runs code it wrote (`python script.py`, `npm test`), that code can do anything your user account can. For untrusted code, run YCode inside a container or VM.

## Configuration

Settings are resolved in this order, with later sources winning:

1. Built-in defaults
2. Global config: `~/.ycode/config.toml` (move this directory with `YCODE_HOME`)
3. Project config: `<project>/.ycode/config.toml` (can only tighten security settings)
4. Environment: `YCODE_MODEL`, `YCODE_MAX_STEPS`, `YCODE_MAX_TOKENS`, `YCODE_EFFORT`, `YCODE_APPROVAL_MODE`, `YCODE_THEME`, `YCODE_COMMAND_TIMEOUT`, `YCODE_IGNORE_DIRS`
5. Command-line flags

```toml
# ~/.ycode/config.toml
model = "claude-opus-5-5"     # any Claude model ID
max_steps = 50                # model calls per task before YCode stops
max_tokens = 64000            # output token cap per model call
effort = "high"               # low | medium | high | xhigh | max (models that support it)
approval_mode = "normal"      # strict | normal | auto
theme = "default"             # default | mono (no colors)
command_timeout = 120         # seconds, per command
max_output_chars = 30000      # command/tool output sent back to the model
refusal_fallback = true       # retry declined requests on Anthropic's recommended fallback model
# ignore_dirs = [...]         # replaces the default ignore list
extra_ignore_dirs = ["generated", "vendor"]   # adds to it
```

The default model is `claude-opus-5-5`. You can set any model ID, for example `claude-sonnet-5-5`, `claude-fable-5-1`, `claude-haiku-4-5` or `claude-sonnet-4-5`. YCode adapts the request to the model. It turns on adaptive thinking and `effort` only on models that support them, and server-side refusal fallbacks only on Claude Fable 5.1, Opus 5.5, Opus 5 and Sonnet 5.5. If you route through a gateway with `ANTHROPIC_BASE_URL`, YCode sends a plainer request shape.

By default these directories are ignored: `node_modules .git dist build .next .nuxt .cache venv .venv env __pycache__ .pytest_cache .mypy_cache .ruff_cache target coverage .gradle .idea .dart_tool .tox .eggs`.

## YCODE.md instructions

Put a `YCODE.md` at the root of your repository (or at `.ycode/YCODE.md`) and YCode adds it to the agent's instructions for every task in that project:

```markdown
# YCode Project Instructions

Use TypeScript.

Run npm test before finishing.

Do not modify generated files.

Use functional React components.
```

The banner and `/status` show when instructions are loaded. Files longer than 20,000 characters are truncated.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

Set `YCODE_DEBUG=1` to see full tracebacks for unexpected internal errors. Normal user errors are always shown as short messages, never as tracebacks.

Some guidelines:
- Keep the agent free of UI and provider imports. Communicate through `AgentEvents` and `LLMProvider`.
- Tools return a `ToolResult` and never raise for expected failures. The registry turns exceptions into error results the model can react to.
- New command patterns belong in `security/permissions.py`, with tests in `tests/test_security.py`.

## Testing

The test suite needs no API key and makes no network calls:

- `tests/test_anthropic_provider.py` mocks the Anthropic client: request shape, stream parsing, error mapping and missing credentials.
- `tests/test_anthropic_wire.py` runs the real `anthropic` SDK against a mock HTTP transport that returns canned server-sent events. It checks the exact JSON YCode sends across a full tool-use round trip.
- `tests/test_lm.py` covers YCode-LM: tokenizer round-trips (including Unicode), special tokens, a KV-cache check against full recomputation, answer-only loss masking, dataset preparation, pretraining that measurably lowers loss, SFT, resume, checkpoint round trips and the `ycode-lm` CLI end to end. These tests are skipped if PyTorch isn't installed.
- `tests/test_local_provider.py` checks the local provider: no API key, no tools sent, `--local` from the CLI.
- `tests/test_agent_loop.py` drives the agent with a scripted fake provider: multi-step tasks, parallel tool calls, the step limit, interruption, truncated tool calls and refusals.
- The other test files cover configuration, the banner, file tools, path security, command classification and approvals, command execution (timeouts, exit codes, secret stripping), git, project detection and YCODE.md, slash commands, and the CLI and REPL.

```bash
python -m pytest -q
```

## Roadmap

- YCode-LM: tool-calling training data, so the local model can drive the agent
- YCode-LM: multi-GPU / distributed training, larger presets
- More providers: OpenAI-compatible, local models through Ollama
- Session persistence and resume (`ycode --continue`)
- Context compaction for very long sessions
- Optional sandboxed command execution (containers)
- `/undo` that restores files to the state before the last task
- MCP server support for extra tools
- A richer TUI mode (Textual) as an optional front-end
