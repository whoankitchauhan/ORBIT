"""The shared state that travels through the ORBIT workflow.

Every node receives the whole state and returns a *partial update* — a plain
dict of the keys it changed. The engine merges those updates, which keeps
agents from stepping on each other's fields and makes each transition
auditable: the diff between two checkpoints is exactly what one agent did.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskStatus = Literal[
    "pending",
    "running",
    "awaiting_approval",
    "completed",
    "rejected",
    "failed",
]

RiskLevel = Literal["low", "medium", "high"]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class PlanStep:
    """One unit of delegated work produced by the Supervisor."""

    index: int
    agent: str                      # research | analysis | action
    instruction: str
    status: str = "pending"         # pending | running | done | skipped | failed
    output_key: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    """A single line in the execution trace shown in the UI."""

    ts: float
    actor: str
    kind: str                       # routing | agent | tool | memory | policy | error | system
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    agent: str
    arguments: dict[str, Any]
    result: Any = None
    status: str = "ok"
    error: str = ""
    risk: RiskLevel = "low"
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalRequest:
    """The payload a human sees when the workflow pauses."""

    id: str
    action: str
    summary: str
    reason: str
    risk: RiskLevel
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"          # pending | approved | rejected
    decided_by: str = ""
    note: str = ""
    requested_at: float = field(default_factory=time.time)
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrbitState:
    """Everything the workflow knows about one objective."""

    task_id: str
    objective: str
    user_id: str = "demo-user"
    status: TaskStatus = "pending"

    # Supervisor output
    plan: list[PlanStep] = field(default_factory=list)
    cursor: int = 0                  # index of next pending step
    hops: int = 0                    # supervisor visits, guards against loops

    # Agent outputs
    research: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    action_result: dict[str, Any] = field(default_factory=dict)

    # Memory
    retrieved: list[dict[str, Any]] = field(default_factory=list)

    # Control
    risk: RiskLevel = "low"
    confidence: float = 0.0
    approval: ApprovalRequest | None = None
    pending_action: dict[str, Any] | None = None

    # Trace
    events: list[Event] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Result
    final_answer: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- utils

    def log(self, actor: str, kind: str, message: str, **payload: Any) -> Event:
        event = Event(ts=time.time(), actor=actor, kind=kind, message=message, payload=payload)
        self.events.append(event)
        self.updated_at = event.ts
        return event

    def current_step(self) -> PlanStep | None:
        for step in self.plan:
            if step.status in {"pending", "running"}:
                return step
        return None

    def context_digest(self, limit: int = 1800) -> str:
        """A compact text summary of prior work, fed to downstream agents."""
        parts: list[str] = []
        if self.retrieved:
            joined = "\n".join(f"- {m['text']}" for m in self.retrieved[:5])
            parts.append(f"Recalled memory:\n{joined}")
        if self.research.get("summary"):
            parts.append(f"Research findings:\n{self.research['summary']}")
        if self.analysis.get("summary"):
            parts.append(f"Analysis:\n{self.analysis['summary']}")
        text = "\n\n".join(parts)
        return text[:limit]

    # -------------------------------------------------------- (de)serialise

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrbitState":
        data = dict(data)
        data["plan"] = [PlanStep(**s) for s in data.get("plan", [])]
        data["events"] = [Event(**e) for e in data.get("events", [])]
        data["tool_calls"] = [ToolCall(**t) for t in data.get("tool_calls", [])]
        approval = data.get("approval")
        data["approval"] = ApprovalRequest(**approval) if approval else None
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
