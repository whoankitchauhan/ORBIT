"""Tool registry, permission model and safe execution.

Two ideas from the project document are enforced here rather than merely
described:

*   **Least privilege.** A tool is registered with the set of agents allowed to
    call it. The Research Agent cannot reach `create_replacement_request` even
    if a prompt injection convinces it to try, because the check happens in
    Python before the function is looked up.
*   **Risk classification.** Every tool carries a risk level. `high` means the
    workflow must pause for human approval before the call executes — the
    policy lives on the tool, not in a prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.graph.state import ToolCall

RiskLevel = str  # "low" | "medium" | "high"


class ToolError(RuntimeError):
    """Raised when a tool cannot run: bad arguments, denied, or failed."""


class PermissionDenied(ToolError):
    """Raised when an agent calls a tool it is not entitled to use."""


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    risk: RiskLevel = "low"
    allowed_agents: tuple[str, ...] = ("research", "analysis", "action", "supervisor")
    parameters: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()

    @property
    def needs_approval(self) -> bool:
        return self.risk == "high"

    def signature(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({args})"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        risk: RiskLevel = "low",
        allowed_agents: tuple[str, ...] = ("research", "analysis", "action", "supervisor"),
        parameters: dict[str, str] | None = None,
        required: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: keeps a tool's metadata next to its implementation."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    fn=fn,
                    risk=risk,
                    allowed_agents=allowed_agents,
                    parameters=parameters or {},
                    required=required,
                )
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool {name!r}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def for_agent(self, agent: str) -> list[Tool]:
        return [t for t in self.all() if agent in t.allowed_agents]

    def describe_for_agent(self, agent: str) -> str:
        """The tool menu injected into an agent's prompt."""
        lines = [
            f"- {t.signature()} [risk: {t.risk}] — {t.description}"
            for t in self.for_agent(agent)
        ]
        return "\n".join(lines) if lines else "(no tools available)"

    # ------------------------------------------------------------ execution

    def execute(
        self, name: str, agent: str, arguments: dict[str, Any] | None = None
    ) -> ToolCall:
        """Run a tool under permission, validation and error control.

        Always returns a `ToolCall`; failures are recorded on it rather than
        raised, so one bad tool call degrades a step instead of killing the run.
        """
        arguments = dict(arguments or {})
        started = time.perf_counter()

        try:
            tool = self.get(name)
        except ToolError as exc:
            return ToolCall(
                name=name, agent=agent, arguments=arguments,
                status="error", error=str(exc), duration_ms=0,
            )

        call = ToolCall(name=name, agent=agent, arguments=arguments, risk=tool.risk)

        if agent not in tool.allowed_agents:
            call.status = "denied"
            call.error = (
                f"the {agent} agent is not permitted to call {name!r} "
                f"(allowed: {', '.join(tool.allowed_agents)})"
            )
            return call

        missing = [p for p in tool.required if p not in arguments or arguments[p] in (None, "")]
        if missing:
            call.status = "error"
            call.error = f"missing required argument(s): {', '.join(missing)}"
            return call

        try:
            call.result = tool.fn(**arguments)
            call.status = "ok"
        except TypeError as exc:
            call.status = "error"
            call.error = f"invalid arguments: {exc}"
        except Exception as exc:
            call.status = "error"
            call.error = f"{type(exc).__name__}: {exc}"

        call.duration_ms = int((time.perf_counter() - started) * 1000)
        return call


registry = ToolRegistry()


def load_all_tools() -> ToolRegistry:
    """Import every tool module so decorators run. Safe to call repeatedly."""
    from app.tools import calculator, database, documents, external_api, search  # noqa: F401

    return registry
