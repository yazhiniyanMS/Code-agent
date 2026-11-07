"""The agent loop: model -> tool calls -> results -> model ... -> final answer."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field

from ycode.agent.planner import Plan, UpdatePlanTool
from ycode.errors import YCodeError
from ycode.llm.base import (
    AssistantMessage,
    LLMProvider,
    Message,
    StopReason,
    StreamCallbacks,
    ToolCall,
    ToolResultBlock,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)
from ycode.tools.base import ToolContext, ToolRegistry, ToolResult


class TaskStatus(enum.Enum):
    COMPLETED = "completed"
    STEP_LIMIT = "step_limit"
    INTERRUPTED = "interrupted"
    REFUSED = "refused"
    ERROR = "error"
    STUCK = "stuck"  # the same tool call kept failing


@dataclass
class TaskResult:
    status: TaskStatus
    final_text: str = ""
    steps: int = 0
    changed_files: list[str] = field(default_factory=list)
    commands_run: list[tuple[str, int | None]] = field(default_factory=list)
    error: YCodeError | None = None


class AgentEvents:
    """Callbacks the UI implements. All methods are optional no-ops."""

    def on_step_start(self, step: int, max_steps: int) -> None: ...
    def on_text(self, chunk: str) -> None: ...
    def on_tool_call_streaming(self, name: str) -> None: ...
    def on_response_done(self, message: AssistantMessage) -> None: ...
    def on_tool_start(self, call: ToolCall, description: str) -> None: ...
    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None: ...
    def on_plan(self, plan: Plan) -> None: ...
    def on_notice(self, message: str, level: str = "info") -> None: ...


_TRUNCATED_TOOL_INPUT = (
    "Your response hit the output token limit while writing this tool call, so its input was "
    "truncated and it was NOT executed. Retry with smaller pieces (e.g. several edit_file calls "
    "instead of one huge write_file)."
)
_CONTINUE_AFTER_MAX_TOKENS = (
    "Your previous response was cut off by the output token limit. Continue exactly where you left off."
)
_INTERRUPTED_NOTE = "[Note: the user interrupted the previous request before it finished.]"
# The same call (tool and input) failing again: the model is told from the REPEAT_WARN-th failure, and the
# task stops at the REPEAT_STOP-th instead of spending the remaining steps on it.
REPEAT_WARN = 2
REPEAT_STOP = 3
# The conversation sent to the model is kept under this many characters (~4 per token: 100k tokens, half
# of a 200k-token context, leaving room for the system prompt, tool specs and the answer). Past it, the
# oldest tool outputs are replaced by a note; the most recent KEEP_RECENT_RESULTS stay whole.
MAX_CONTEXT_CHARS = 400_000
KEEP_RECENT_RESULTS = 2
_ELIDED = ("[Output removed to keep the conversation within the model's context ({n:,} characters). "
           "Run the tool again if you need it.]")


def _size(message: Message) -> int:
    if isinstance(message, ToolResultsMessage):
        return sum(len(r.content) for r in message.results)
    if isinstance(message, AssistantMessage):
        return len(message.text or "") + sum(len(json.dumps(c.arguments, default=str)) for c in message.tool_calls)
    return len(getattr(message, "text", "") or "")


def _call_key(call: ToolCall) -> tuple[str, str]:
    try:
        args = json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = repr(call.arguments)
    return call.name, args


class Agent:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        tool_context: ToolContext,
        system_prompt: str,
        max_steps: int = 50,
        events: AgentEvents | None = None,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> None:
        self.provider = provider
        self.max_context_chars = max_context_chars
        self.registry = registry
        self.ctx = tool_context
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.events = events or AgentEvents()
        self.messages: list[Message] = []
        self.usage = Usage()
        self.last_steps = 0
        self.total_steps = 0
        self._pending_note: str | None = None
        self._failures: dict[tuple[str, str], int] = {}  # failed calls this task, by tool and input
        self._stuck: str | None = None

        self.plan_tool = UpdatePlanTool(on_update=self.events.on_plan)
        if registry.get(self.plan_tool.name) is None:
            registry.register(self.plan_tool)
        else:
            existing = registry.get(self.plan_tool.name)
            if isinstance(existing, UpdatePlanTool):
                self.plan_tool = existing
                existing.on_update = self.events.on_plan

    # ------------------------------------------------------------- state

    @property
    def plan(self) -> Plan:
        return self.plan_tool.plan

    def reset(self) -> None:
        """Forget the conversation (the /clear command)."""
        self.messages.clear()
        self.plan_tool.plan = Plan()
        self.last_steps = 0
        self._pending_note = None

    def tool_specs(self) -> list[ToolSpec]:
        return [ToolSpec(s["name"], s["description"], s["parameters"]) for s in self.registry.specs()]

    # -------------------------------------------------------------- loop

    def run(self, task: str) -> TaskResult:
        text = task.strip()
        if self._pending_note:
            text = f"{self._pending_note}\n\n{text}"
            self._pending_note = None
        self.messages.append(UserMessage(text))
        self.ctx.commands_run = []
        changed_before = set(self.ctx.changed_files)
        self.plan_tool.plan = Plan()
        self._failures, self._stuck = {}, None

        result = self._loop()
        result.changed_files = sorted(self.ctx.changed_files - changed_before)
        result.commands_run = list(self.ctx.commands_run)
        self.last_steps = result.steps
        self.total_steps += result.steps
        return result

    def _loop(self) -> TaskResult:
        tools = self.tool_specs() if self.provider.supports_tools else []
        callbacks = StreamCallbacks(on_text=self.events.on_text, on_tool_call_start=self.events.on_tool_call_streaming)
        steps = 0
        final_text = ""
        while True:
            if steps >= self.max_steps:
                self.events.on_notice(
                    f"Reached the step limit ({self.max_steps}). Say 'continue' to keep going.", "warning"
                )
                return TaskResult(TaskStatus.STEP_LIMIT, final_text, steps)
            steps += 1
            self.events.on_step_start(steps, self.max_steps)

            self._fit_context()
            try:
                response = self.provider.complete(
                    system=self.system_prompt, messages=self.messages, tools=tools, callbacks=callbacks
                )
            except KeyboardInterrupt:
                self._pending_note = _INTERRUPTED_NOTE
                return TaskResult(TaskStatus.INTERRUPTED, final_text, steps)
            except YCodeError as exc:
                return TaskResult(TaskStatus.ERROR, final_text, steps, error=exc)

            self.usage.add(response.usage)
            message = response.message
            self.messages.append(message)
            self.events.on_response_done(message)
            if message.text:
                final_text = message.text

            if response.stop_reason is StopReason.REFUSAL:
                detail = f" ({response.stop_detail})" if response.stop_detail else ""
                self.events.on_notice(f"The model declined this request{detail}.", "error")
                # Drop the refused turn so the conversation stays usable.
                self.messages.pop()
                return TaskResult(TaskStatus.REFUSED, final_text, steps)

            if not message.tool_calls:
                if response.stop_reason is StopReason.MAX_TOKENS:
                    self.messages.append(UserMessage(_CONTINUE_AFTER_MAX_TOKENS))
                    continue
                if response.stop_reason is StopReason.PAUSE:
                    continue
                return TaskResult(TaskStatus.COMPLETED, final_text, steps)

            truncated = response.stop_reason is StopReason.MAX_TOKENS
            try:
                results = self._execute(message.tool_calls, truncated=truncated)
            except _Interrupted as interrupted:
                self.messages.append(ToolResultsMessage(interrupted.results))
                self._pending_note = _INTERRUPTED_NOTE
                return TaskResult(TaskStatus.INTERRUPTED, final_text, steps)
            self.messages.append(ToolResultsMessage(results))
            if self._stuck:
                self.events.on_notice(self._stuck, "error")
                return TaskResult(TaskStatus.STUCK, final_text, steps)

    def _fit_context(self) -> None:
        """Keep the conversation under max_context_chars: replace the oldest tool outputs (never the
        KEEP_RECENT_RESULTS newest, never a user or assistant message) by a note. Without this a long
        session outgrows the model's context, and every request fails from then on."""
        total = sum(_size(m) for m in self.messages)
        if total <= self.max_context_chars:
            return
        results = [i for i, m in enumerate(self.messages) if isinstance(m, ToolResultsMessage)]
        freed = 0
        for i in results[:-KEEP_RECENT_RESULTS] if KEEP_RECENT_RESULTS else results:
            if total - freed <= self.max_context_chars:
                break
            old = self.messages[i]
            new = [ToolResultBlock(r.tool_call_id, _ELIDED.format(n=len(r.content)), r.is_error)
                   if len(r.content) > len(_ELIDED) + 20 else r for r in old.results]
            freed += _size(old) - sum(len(r.content) for r in new)
            self.messages[i] = ToolResultsMessage(new)  # a new message: what was already sent stays as sent
        if freed:
            self.events.on_notice(f"Shortened older tool outputs by {freed:,} characters to stay within the "
                                  "model's context.", "info")

    def _execute(self, calls: list[ToolCall], *, truncated: bool) -> list[ToolResultBlock]:
        results: list[ToolResultBlock] = []
        first: dict[tuple[str, str], int] = {}  # each distinct call in this response, by tool and input
        for index, call in enumerate(calls):
            if truncated:
                results.append(ToolResultBlock(call.id, _TRUNCATED_TOOL_INPUT, is_error=True))
                continue
            key = _call_key(call)
            if key in first:  # e.g. the same command twice: it runs once
                results.append(ToolResultBlock(
                    call.id, f"Not run: identical to call {first[key] + 1} in this response (same tool and "
                    "input). Its result is above.", is_error=True))
                continue
            first[key] = index
            self.events.on_tool_start(call, self.registry.describe(call.name, call.arguments))
            try:
                outcome = self.registry.dispatch(call.name, call.arguments, self.ctx)
            except KeyboardInterrupt:
                done = results + [ToolResultBlock(call.id, "Interrupted by the user.", is_error=True)]
                done += [
                    ToolResultBlock(c.id, "Not run: the user interrupted the task.", is_error=True)
                    for c in calls[index + 1:]
                ]
                interrupted_result = ToolResult.error("interrupted by user", summary="Interrupted")
                self.events.on_tool_end(call, interrupted_result)
                raise _Interrupted(done) from None
            self.events.on_tool_end(call, outcome)
            content = outcome.content
            if outcome.is_error:
                failures = self._failures[key] = self._failures.get(key, 0) + 1
                if failures >= REPEAT_WARN:
                    content += (f"\n\nThis exact call has failed {failures} times with the same input. Do not repeat "
                                "it: change the input or take another approach.")
                if failures >= REPEAT_STOP and not self._stuck:
                    first_line = outcome.content.splitlines()[0] if outcome.content else ""
                    self._stuck = (f"Stopped: {call.name} failed {failures} times with the same input "
                                   f"({first_line[:200]}).")
            else:
                self._failures.pop(key, None)
            results.append(ToolResultBlock(call.id, content, outcome.is_error))
        return results


class _Interrupted(Exception):
    def __init__(self, results: list[ToolResultBlock]) -> None:
        super().__init__("interrupted")
        self.results = results


def describe_status(result: TaskResult) -> str:
    return {
        TaskStatus.COMPLETED: "YCode completed the task.",
        TaskStatus.STEP_LIMIT: "Stopped: step limit reached.",
        TaskStatus.INTERRUPTED: "Interrupted.",
        TaskStatus.REFUSED: "The model declined the request.",
        TaskStatus.ERROR: "Stopped because of an error.",
        TaskStatus.STUCK: "Stopped: the same tool call kept failing.",
    }[result.status]

