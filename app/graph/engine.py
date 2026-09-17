"""A small stateful graph engine with checkpointing and interrupts.

ORBIT targets LangGraph, and the node functions here are written against the
same contract LangGraph uses: a node takes the state and returns a partial
update, edges may be conditional, and a node may *interrupt* the run so a
human can decide what happens next.

This module provides that contract directly, with no external dependency, for
three reasons:

1.  The demo must run on a machine where only Python is installed.
2.  The pause/resume semantics are the intellectual core of the project, so
    implementing them makes the mechanism inspectable rather than magic.
3.  The node functions stay portable — `workflow.py` can hand the exact same
    callables to LangGraph when it is installed.

Semantics
---------
* A run advances one node at a time from the entry point.
* After each node the merged state is written to the checkpointer.
* A node raising `Interrupt` freezes the run *before* its effect is applied and
  records which node to re-enter on resume.
* `resume()` reloads the checkpoint, merges the human's decision into the
  state, and re-enters the interrupted node.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from app.graph.state import OrbitState

START = "__start__"
END = "__end__"

NodeFn = Callable[[OrbitState], dict[str, Any] | None]
RouterFn = Callable[[OrbitState], str]


class Interrupt(Exception):
    """Raised by a node to suspend the graph and wait for a human decision."""

    def __init__(self, reason: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


class GraphError(RuntimeError):
    """Raised for malformed graphs — missing nodes, dangling edges."""


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    thread_id: str
    next_node: str
    state: dict[str, Any]
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "next_node": self.next_node,
            "state": self.state,
            "step": self.step,
        }


class BaseCheckpointer:
    def put(self, checkpoint: Checkpoint) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get(self, thread_id: str) -> Checkpoint | None:  # pragma: no cover
        raise NotImplementedError


class MemoryCheckpointer(BaseCheckpointer):
    """In-process checkpoints. Fast, and lost when the process exits."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._lock = threading.Lock()

    def put(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._store[checkpoint.thread_id] = checkpoint

    def get(self, thread_id: str) -> Checkpoint | None:
        with self._lock:
            return self._store.get(thread_id)


class FileCheckpointer(BaseCheckpointer):
    """Durable checkpoints on disk, so a paused workflow survives a restart.

    This is what makes the approval demo honest: you can stop the Streamlit
    process while a task waits for approval, start it again, and the task is
    still there waiting.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, thread_id: str) -> Path:
        safe = "".join(c for c in thread_id if c.isalnum() or c in "-_")
        return self.directory / f"{safe}.json"

    def put(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            path = self._path(checkpoint.thread_id)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(checkpoint.to_dict(), default=str), encoding="utf-8")
            tmp.replace(path)          # atomic; never leaves a half-written file

    def get(self, thread_id: str) -> Checkpoint | None:
        path = self._path(thread_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Checkpoint(**raw)

    def list_threads(self) -> list[str]:
        return [p.stem for p in self.directory.glob("*.json")]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class _ConditionalEdge:
    router: RouterFn
    mapping: dict[str, str]


@dataclass
class RunResult:
    state: OrbitState
    interrupted: bool = False
    interrupt_reason: str = ""
    next_node: str = END
    steps_run: list[str] = field(default_factory=list)


class Graph:
    """Builder for a node/edge workflow."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, _ConditionalEdge] = {}
        self.entry: str = ""

    def add_node(self, name: str, fn: NodeFn) -> "Graph":
        if name in {START, END}:
            raise GraphError(f"{name!r} is reserved")
        self.nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> "Graph":
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: RouterFn, mapping: dict[str, str]
    ) -> "Graph":
        self.conditional[source] = _ConditionalEdge(router=router, mapping=mapping)
        return self

    def set_entry_point(self, name: str) -> "Graph":
        self.entry = name
        return self

    def validate(self) -> None:
        if not self.entry:
            raise GraphError("no entry point set")
        targets: Iterable[str] = list(self.edges.values()) + [
            t for edge in self.conditional.values() for t in edge.mapping.values()
        ]
        for target in targets:
            if target != END and target not in self.nodes:
                raise GraphError(f"edge points at unknown node {target!r}")
        if self.entry not in self.nodes:
            raise GraphError(f"entry point {self.entry!r} is not a node")

    def compile(self, checkpointer: BaseCheckpointer | None = None) -> "CompiledGraph":
        self.validate()
        return CompiledGraph(self, checkpointer or MemoryCheckpointer())

    def to_mermaid(self) -> str:
        """Render the graph for documentation and the UI's architecture tab."""
        lines = ["flowchart TD", f"    {START}([Start])"]
        for node in self.nodes:
            lines.append(f"    {node}[{node.replace('_', ' ').title()}]")
        lines.append(f"    {END}([End])")
        lines.append(f"    {START} --> {self.entry}")
        for src, dst in self.edges.items():
            lines.append(f"    {src} --> {dst}")
        for src, edge in self.conditional.items():
            for label, dst in edge.mapping.items():
                lines.append(f"    {src} -->|{label}| {dst}")
        return "\n".join(lines)


class CompiledGraph:
    """An executable graph bound to a checkpointer."""

    def __init__(self, graph: Graph, checkpointer: BaseCheckpointer) -> None:
        self.graph = graph
        self.checkpointer = checkpointer

    # ------------------------------------------------------------- merging

    @staticmethod
    def _apply(state: OrbitState, update: dict[str, Any] | None) -> OrbitState:
        """Merge a node's partial update into the state.

        List fields append rather than replace, so a node adding one event
        does not have to carry the whole history back with it.
        """
        if not update:
            return state
        for key, value in update.items():
            if not hasattr(state, key):
                continue
            if key in {"events", "tool_calls", "errors", "retrieved"} and isinstance(value, list):
                getattr(state, key).extend(value)
            else:
                setattr(state, key, value)
        return state

    def _next_node(self, node: str, state: OrbitState) -> str:
        if node in self.graph.conditional:
            edge = self.graph.conditional[node]
            key = edge.router(state)
            if key not in edge.mapping:
                raise GraphError(f"router for {node!r} returned unmapped key {key!r}")
            return edge.mapping[key]
        return self.graph.edges.get(node, END)

    # ------------------------------------------------------------- running

    def invoke(
        self,
        state: OrbitState,
        thread_id: str | None = None,
        start_at: str | None = None,
        on_step: Callable[[str, OrbitState], None] | None = None,
        max_steps: int = 80,
    ) -> RunResult:
        """Run from `start_at` (default: the entry point) until END or Interrupt."""
        thread_id = thread_id or state.task_id
        node = start_at or self.graph.entry
        executed: list[str] = []

        for step in range(max_steps):
            if node == END:
                break
            fn = self.graph.nodes.get(node)
            if fn is None:
                raise GraphError(f"unknown node {node!r}")

            try:
                update = fn(state)
            except Interrupt as interrupt:
                # Freeze *before* the node's effect lands. On resume we
                # re-enter this same node with the decision in state.
                self.checkpointer.put(
                    Checkpoint(
                        thread_id=thread_id,
                        next_node=node,
                        state=state.to_dict(),
                        step=step,
                    )
                )
                return RunResult(
                    state=state,
                    interrupted=True,
                    interrupt_reason=interrupt.reason,
                    next_node=node,
                    steps_run=executed,
                )

            self._apply(state, update)
            executed.append(node)
            if on_step:
                on_step(node, state)

            node = self._next_node(node, state)
            self.checkpointer.put(
                Checkpoint(thread_id=thread_id, next_node=node, state=state.to_dict(), step=step)
            )
        else:
            raise GraphError(f"graph exceeded {max_steps} steps — probable cycle")

        return RunResult(state=state, interrupted=False, next_node=END, steps_run=executed)

    def resume(
        self,
        thread_id: str,
        update: dict[str, Any] | None = None,
        on_step: Callable[[str, OrbitState], None] | None = None,
    ) -> RunResult:
        """Continue a suspended run after a human decision."""
        checkpoint = self.checkpointer.get(thread_id)
        if checkpoint is None:
            raise GraphError(f"no checkpoint for thread {thread_id!r}")
        state = OrbitState.from_dict(checkpoint.state)
        self._apply(state, update)
        return self.invoke(
            state, thread_id=thread_id, start_at=checkpoint.next_node, on_step=on_step
        )

    def get_state(self, thread_id: str) -> OrbitState | None:
        checkpoint = self.checkpointer.get(thread_id)
        return OrbitState.from_dict(checkpoint.state) if checkpoint else None
