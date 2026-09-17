"""The graph engine: routing, merging, interrupts and checkpoint durability."""

from __future__ import annotations

import pytest

from app.graph.engine import (
    END,
    FileCheckpointer,
    Graph,
    GraphError,
    Interrupt,
    MemoryCheckpointer,
)
from app.graph.state import OrbitState


def make_state(objective: str = "test objective") -> OrbitState:
    return OrbitState(task_id="task-test", objective=objective)


def test_linear_graph_runs_in_order():
    order: list[str] = []
    graph = Graph()
    graph.add_node("a", lambda s: order.append("a") or {"cursor": 1})
    graph.add_node("b", lambda s: order.append("b") or {"cursor": 2})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = graph.compile().invoke(make_state())
    assert order == ["a", "b"]
    assert result.state.cursor == 2
    assert not result.interrupted


def test_list_fields_append_rather_than_replace():
    """A node reporting one error must not wipe the errors already recorded."""
    graph = Graph()
    graph.add_node("a", lambda s: {"errors": ["first"]})
    graph.add_node("b", lambda s: {"errors": ["second"]})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = graph.compile().invoke(make_state())
    assert result.state.errors == ["first", "second"]


def test_conditional_edges_route_on_state():
    graph = Graph()
    graph.add_node("start", lambda s: {"risk": "high"})
    graph.add_node("high", lambda s: {"final_answer": "escalated"})
    graph.add_node("low", lambda s: {"final_answer": "auto"})
    graph.set_entry_point("start")
    graph.add_conditional_edges("start", lambda s: s.risk, {"high": "high", "low": "low"})
    graph.add_edge("high", END)
    graph.add_edge("low", END)

    result = graph.compile().invoke(make_state())
    assert result.state.final_answer == "escalated"


def test_router_returning_an_unmapped_key_is_an_error():
    graph = Graph()
    graph.add_node("start", lambda s: {})
    graph.add_node("other", lambda s: {})
    graph.set_entry_point("start")
    graph.add_conditional_edges("start", lambda s: "nonsense", {"ok": "other"})
    graph.add_edge("other", END)

    with pytest.raises(GraphError):
        graph.compile().invoke(make_state())


def test_dangling_edge_is_caught_at_compile_time():
    graph = Graph()
    graph.add_node("a", lambda s: {})
    graph.set_entry_point("a")
    graph.add_edge("a", "does_not_exist")
    with pytest.raises(GraphError):
        graph.compile()


def test_runaway_cycle_is_cut_off():
    graph = Graph()
    graph.add_node("loop", lambda s: {"hops": s.hops + 1})
    graph.set_entry_point("loop")
    graph.add_edge("loop", "loop")

    with pytest.raises(GraphError, match="probable cycle"):
        graph.compile().invoke(make_state(), max_steps=20)


# ------------------------------------------------------------ interrupts


def build_gated_graph(should_pause: list[bool]) -> Graph:
    """A graph whose middle node interrupts until a flag is flipped."""

    def gate(state: OrbitState) -> dict:
        if should_pause[0]:
            raise Interrupt("needs a human", {"where": "gate"})
        return {"final_answer": "released"}

    graph = Graph()
    graph.add_node("before", lambda s: {"research": {"summary": "gathered"}})
    graph.add_node("gate", gate)
    graph.add_node("after", lambda s: {"status": "completed"})
    graph.set_entry_point("before")
    graph.add_edge("before", "gate")
    graph.add_edge("gate", "after")
    graph.add_edge("after", END)
    return graph


def test_interrupt_freezes_the_run_before_the_node_takes_effect():
    flag = [True]
    compiled = build_gated_graph(flag).compile()
    result = compiled.invoke(make_state(), thread_id="t1")

    assert result.interrupted
    assert result.next_node == "gate"
    assert result.interrupt_reason == "needs a human"
    # The node's own effect must not have landed.
    assert result.state.final_answer == ""
    # Work completed before the gate is preserved.
    assert result.state.research["summary"] == "gathered"


def test_resume_re_enters_the_interrupted_node():
    flag = [True]
    compiled = build_gated_graph(flag).compile()
    compiled.invoke(make_state(), thread_id="t2")

    flag[0] = False                      # the human decided
    resumed = compiled.resume("t2")
    assert not resumed.interrupted
    assert resumed.state.final_answer == "released"
    assert resumed.state.status == "completed"


def test_resume_merges_the_decision_into_state():
    flag = [True]
    compiled = build_gated_graph(flag).compile()
    compiled.invoke(make_state(), thread_id="t3")
    flag[0] = False

    resumed = compiled.resume("t3", update={"risk": "low", "confidence": 0.9})
    assert resumed.state.risk == "low"
    assert resumed.state.confidence == 0.9


def test_resuming_an_unknown_thread_is_an_error():
    compiled = build_gated_graph([False]).compile()
    with pytest.raises(GraphError):
        compiled.resume("no-such-thread")


# ----------------------------------------------------------- checkpointing


def test_memory_checkpointer_round_trips_state():
    compiled = build_gated_graph([True]).compile(MemoryCheckpointer())
    compiled.invoke(make_state("remember me"), thread_id="t4")
    restored = compiled.get_state("t4")
    assert restored is not None
    assert restored.objective == "remember me"


def test_file_checkpoint_survives_a_new_process(tmp_path):
    """A paused workflow must outlive the process that paused it."""
    checkpointer = FileCheckpointer(tmp_path / "cp")
    compiled = build_gated_graph([True]).compile(checkpointer)
    compiled.invoke(make_state("durable objective"), thread_id="t5")

    # A fresh checkpointer over the same directory stands in for a restart.
    reopened = FileCheckpointer(tmp_path / "cp")
    checkpoint = reopened.get("t5")
    assert checkpoint is not None
    assert checkpoint.next_node == "gate"
    assert OrbitState.from_dict(checkpoint.state).objective == "durable objective"


def test_state_survives_a_serialisation_round_trip():
    state = make_state()
    state.log("supervisor", "system", "hello")
    state.research = {"findings": ["a", "b"]}
    restored = OrbitState.from_dict(state.to_dict())

    assert restored.events[0].message == "hello"
    assert restored.research["findings"] == ["a", "b"]
    assert restored.task_id == state.task_id
