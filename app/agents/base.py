"""Shared behaviour for every ORBIT agent.

An agent is an LLM plus a role, a permitted tool set, and an output contract.
This base class supplies the parts that are identical across agents — calling
tools under the permission model, recording what happened, and returning a
uniform result — so each subclass contains only its actual reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.graph.state import Event, OrbitState, ToolCall, new_id
from app.llm.provider import LLMClient, get_llm
from app.memory.store import get_store
from app.tools.registry import load_all_tools


@dataclass
class AgentResult:
    """What an agent hands back to the graph node that invoked it."""

    agent: str
    output: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    latency_ms: int = 0
    status: str = "ok"
    error: str = ""


class Agent:
    """Base class. Subclasses implement `run`."""

    name: str = "agent"
    system_prompt: str = ""
    description: str = ""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm()
        self.registry = load_all_tools()

    # ------------------------------------------------------------ tool use

    @property
    def tools(self) -> list[Any]:
        return self.registry.for_agent(self.name)

    def tool_menu(self) -> str:
        return self.registry.describe_for_agent(self.name)

    def call_tool(
        self, name: str, arguments: dict[str, Any], state: OrbitState | None = None
    ) -> ToolCall:
        """Invoke a tool and persist the call for the audit trail."""
        call = self.registry.execute(name, agent=self.name, arguments=arguments)
        if state is not None:
            state.tool_calls.append(call)
            detail = call.error if call.status != "ok" else "ok"
            state.log(
                self.name, "tool", f"{name} → {call.status}",
                tool=name, arguments=arguments, status=call.status, detail=detail,
            )
            try:
                get_store().save_tool_call(state.task_id, call)
            except Exception:
                pass  # the audit write must never break the workflow
        return call

    # ------------------------------------------------------------ recording

    def record_run(
        self,
        state: OrbitState,
        step_index: int,
        input_payload: Any,
        output_payload: Any,
        provider: str,
        latency_ms: int,
        status: str = "ok",
    ) -> None:
        try:
            get_store().save_agent_run(
                {
                    "id": new_id("run"),
                    "task_id": state.task_id,
                    "agent_name": self.name,
                    "step_index": step_index,
                    "input": input_payload,
                    "output": output_payload,
                    "status": status,
                    "provider": provider,
                    "latency_ms": latency_ms,
                    "started_at": time.time() - latency_ms / 1000,
                    "completed_at": time.time(),
                }
            )
        except Exception:
            pass

    # ------------------------------------------------------------- contract

    def run(self, state: OrbitState, **kwargs: Any) -> AgentResult:  # pragma: no cover
        raise NotImplementedError

    def card(self) -> dict[str, Any]:
        """Agent metadata for the UI's architecture panel."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [
                {"name": t.name, "risk": t.risk, "description": t.description}
                for t in self.tools
            ],
        }
