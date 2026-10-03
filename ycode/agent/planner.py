"""Planning: a structured plan the model maintains through the update_plan tool.

The model decides how much planning a task needs (the system prompt tells it
to skip plans for trivial requests). When it does plan, the plan is shown to
the user and kept up to date as steps complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ycode.errors import ToolError
from ycode.tools.base import Tool, ToolContext, ToolResult

STATUSES = ("pending", "in_progress", "done", "skipped")


@dataclass
class PlanStep:
    text: str
    status: str = "pending"


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def render_text(self) -> str:
        marks = {"pending": " ", "in_progress": "~", "done": "x", "skipped": "-"}
        return "\n".join(f"{i}. [{marks[s.status]}] {s.text}" for i, s in enumerate(self.steps, 1))


def parse_steps(raw: Any) -> list[PlanStep]:
    if not isinstance(raw, list) or not raw:
        raise ToolError("steps must be a non-empty list")
    steps: list[PlanStep] = []
    for item in raw:
        if isinstance(item, str):
            steps.append(PlanStep(item.strip()))
            continue
        if not isinstance(item, dict) or not isinstance(item.get("step"), str):
            raise ToolError("each step must be a string or an object with a 'step' string")
        status = item.get("status", "pending")
        if status not in STATUSES:
            raise ToolError(f"invalid status {status!r}; use one of {', '.join(STATUSES)}")
        steps.append(PlanStep(item["step"].strip(), status))
    if len(steps) > 20:
        raise ToolError("keep plans to 20 steps or fewer")
    return steps


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = (
        "Record or update your implementation plan for the current task. Call it before making "
        "significant changes (skip it for trivial requests), and again to mark steps in_progress "
        "or done. Pass the full list of steps every time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "Ordered plan steps.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "description": "Short imperative description."},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["step"],
                },
            },
        },
        "required": ["steps"],
    }

    def __init__(self, on_update: Callable[[Plan], None] | None = None) -> None:
        self.plan = Plan()
        self.on_update = on_update

    def describe(self, args: dict[str, Any]) -> str:
        return "Updating plan"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.plan = Plan(parse_steps(args["steps"]))
        if self.on_update:
            self.on_update(self.plan)
        done = sum(s.status == "done" for s in self.plan.steps)
        return ToolResult(
            content="Plan recorded:\n" + self.plan.render_text(),
            summary=f"Plan: {done}/{len(self.plan.steps)} steps done",
            display={"plan": self.plan},
        )
