"""Assembly of the ORBIT workflow graph.

    plan
      ↓
    supervisor ──┬── research ──┐
                 ├── analysis ──┤→ back to supervisor
                 └── action ────┴──→ validation
                                       ↓
                             pending action?
                          yes ↓             ↓ no
                         human gate      finalise → END
                     approve ↓  ↓ reject
                      execute    finalise
                         ↓
                    supervisor

The loop back through the supervisor after each specialist is what makes this
an orchestrated system rather than a fixed pipeline: routing is re-decided from
the current state on every hop, so a plan can be cut short, re-ordered or
abandoned mid-run.
"""

from __future__ import annotations

from app.config import DATA_DIR, settings
from app.graph import nodes
from app.graph.engine import END, CompiledGraph, FileCheckpointer, Graph


def build_graph() -> Graph:
    graph = Graph(name="orbit")

    graph.add_node("plan", nodes.plan_node)
    graph.add_node("supervisor", nodes.supervisor_node)
    graph.add_node("research", nodes.research_node)
    graph.add_node("analysis", nodes.analysis_node)
    graph.add_node("action", nodes.action_propose_node)
    graph.add_node("validation", nodes.validation_node)
    graph.add_node("gate", nodes.human_gate_node)
    graph.add_node("execute", nodes.action_execute_node)
    graph.add_node("finalise", nodes.finalise_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        nodes.route_from_supervisor,
        {
            "research": "research",
            "analysis": "analysis",
            "action": "action",
            "validation": "validation",
        },
    )

    # Specialists report back so routing is re-decided with fresh state.
    graph.add_edge("research", "supervisor")
    graph.add_edge("analysis", "supervisor")

    # A proposed action is audited before anyone is asked to approve it.
    graph.add_edge("action", "validation")

    graph.add_conditional_edges(
        "validation",
        nodes.route_after_validation,
        {"gate": "gate", "finalise": "finalise"},
    )
    graph.add_conditional_edges(
        "gate",
        nodes.route_after_gate,
        {"execute": "execute", "finalise": "finalise"},
    )

    graph.add_edge("execute", "supervisor")
    graph.add_edge("finalise", END)
    return graph


_compiled: CompiledGraph | None = None


def get_workflow() -> CompiledGraph:
    """Compile once and reuse. Checkpoints persist to disk.

    File-backed checkpointing is what allows a task to sit in
    `awaiting_approval` across a process restart — the pause is durable, not a
    variable held in memory.
    """
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(
            checkpointer=FileCheckpointer(DATA_DIR / "checkpoints")
        )
    return _compiled


def graph_diagram() -> str:
    """Mermaid source for the UI's architecture view."""
    return build_graph().to_mermaid()


def orchestration_backend() -> str:
    return "LangGraph" if settings.has_langgraph else "Built-in graph engine"
