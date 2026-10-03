# YCode Project Instructions

This repository is YCode itself (Python 3.10+).

- Run `python -m pytest -q` before finishing; all tests must pass without an API key or network.
- Keep layers separate: `agent/` must not import from `ui/` or a specific provider in `llm/`.
- Tools return `ToolResult` objects; don't let expected failures raise out of a tool.
- Any change to command classification in `security/permissions.py` needs a test in `tests/test_security.py`.
- Never print or log API keys.
