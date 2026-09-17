"""End-to-end workflow behaviour: planning, escalation, approval and memory."""

from __future__ import annotations

import uuid

import pytest

from app import runtime
from app.agents.supervisor import _normalise_plan
from app.config import settings
from app.graph.state import OrbitState, PlanStep
from app.memory.store import get_store
from app.memory.vector import get_memory
from app.policy import assess_action
from app.seed import seed_all


@pytest.fixture(scope="module", autouse=True)
def seeded():
    seed_all(verbose=False)


# -------------------------------------------------------------- planning


def test_plan_puts_research_before_analysis():
    plan = _normalise_plan([PlanStep(index=0, agent="analysis", instruction="compare things")])
    assert [s.agent for s in plan][:2] == ["research", "analysis"]


def test_plan_puts_analysis_before_any_action():
    """An action with no reasoning behind it has nothing to justify approval."""
    plan = _normalise_plan([PlanStep(index=0, agent="action", instruction="send an email")])
    agents = [s.agent for s in plan]
    assert agents.index("analysis") < agents.index("action")
    assert agents.index("research") < agents.index("analysis")


def test_plan_steps_are_reindexed_after_insertion():
    plan = _normalise_plan([PlanStep(index=0, agent="action", instruction="raise a ticket")])
    assert [s.index for s in plan] == list(range(len(plan)))


def test_a_question_is_not_planned_as_an_action():
    """'What is our refund policy?' contains 'refund' but orders nothing."""
    state = runtime.run_task_sync("What is our refund policy for a delivery 90 days ago?")
    assert "action" not in [s.agent for s in state.plan]
    assert state.status == "completed"


# ---------------------------------------------------------------- policy


def _state_with_confidence(confidence: float) -> OrbitState:
    state = OrbitState(task_id="task-policy", objective="test")
    state.validation = {"confidence": confidence, "issues": []}
    state.analysis = {"confidence": confidence}
    return state


def test_high_risk_tool_always_requires_approval():
    decision = assess_action(
        _state_with_confidence(0.99),
        {"tool": "create_replacement_request",
         "arguments": {"customer_id": "CUST-001", "order_id": "ORD-1001", "reason": "fault"}},
    )
    assert decision.requires_approval
    assert decision.risk == "high"


def test_low_confidence_escalates_even_a_safe_tool():
    below = settings.approval_confidence_threshold - 0.2
    decision = assess_action(
        _state_with_confidence(below), {"tool": "knowledge_search", "arguments": {"query": "x"}}
    )
    assert decision.requires_approval
    assert any("threshold" in r for r in decision.reasons)


def test_validation_issues_escalate():
    state = _state_with_confidence(0.95)
    state.validation["issues"] = ["evidence is thin"]
    decision = assess_action(state, {"tool": "knowledge_search", "arguments": {"query": "x"}})
    assert decision.requires_approval


def test_unregistered_tool_is_treated_as_high_risk():
    decision = assess_action(_state_with_confidence(0.99), {"tool": "made_up_tool", "arguments": {}})
    assert decision.requires_approval
    assert any("not a registered tool" in r for r in decision.reasons)


def test_missing_arguments_escalate_rather_than_failing_silently():
    decision = assess_action(
        _state_with_confidence(0.99),
        {"tool": "issue_refund", "arguments": {"order_id": "", "amount": 10, "reason": "x"}},
    )
    assert decision.requires_approval
    assert any("missing" in r for r in decision.reasons)


# ------------------------------------------------------------- workflow


def test_consequential_objective_pauses_for_a_human():
    state = runtime.run_task_sync(
        "Check the complaints for CUST-001 and create a replacement request if eligible"
    )
    assert state.status == "awaiting_approval"
    assert state.approval is not None
    assert state.approval.status == "pending"
    assert state.approval.summary                     # the reviewer sees plain language
    assert state.approval.reason                      # and why they were asked
    # Nothing has been executed yet.
    assert not state.action_result


def test_approval_resumes_the_workflow_and_executes():
    state = runtime.run_task_sync(
        "Check the complaints for CUST-002 and raise a replacement request if eligible"
    )
    assert state.status == "awaiting_approval"

    runtime.approve(state.task_id, decided_by="examiner", note="defect confirmed", background=False)
    final = runtime.get_state(state.task_id)

    assert final.status == "completed"
    assert final.action_result["status"] == "executed"
    assert final.approval.decided_by == "examiner"
    assert final.final_answer


def test_rejection_stops_the_action():
    state = runtime.run_task_sync(
        "Check the complaints for CUST-003 and raise a replacement request if eligible"
    )
    before = len(get_store().list_actions(100))

    runtime.reject(state.task_id, decided_by="examiner", note="needs inspection", background=False)
    final = runtime.get_state(state.task_id)

    assert final.status == "rejected"
    assert not final.action_result
    assert len(get_store().list_actions(100)) == before   # nothing reached the ledger
    assert final.final_answer                              # the user still gets an explanation


def test_every_run_records_an_auditable_trace():
    state = runtime.run_task_sync("What is covered by the warranty on a monitor?")
    assert state.events
    assert {e.actor for e in state.events} & {"supervisor", "research"}
    assert get_store().list_agent_runs(state.task_id)


def test_read_only_objective_completes_without_a_gate():
    state = runtime.run_task_sync("What is covered by the warranty on a keyboard?")
    assert state.status == "completed"
    assert state.approval is None


# --------------------------------------------------------------- memory


def test_retrieval_prefers_the_record_that_was_asked_for():
    """Identifier bigrams must stop a near-miss record outranking the right one."""
    results = get_memory().search("complaints for CUST-002 keyboard", k=3)
    assert results
    assert "CUST-002" in results[0].text


def test_policy_retrieval_finds_the_governing_rule():
    results = get_memory().search("when is a customer eligible for a free replacement", k=3)
    assert any("Replacement Policy" in r.text for r in results)


def test_finished_tasks_are_written_back_as_episodes():
    before = get_memory().count()
    runtime.run_task_sync("What is our escalation policy?")
    assert get_memory().count() > before
    assert get_memory().search("escalation policy", k=5, where={"kind": "episode"}) is not None


def test_chunking_splits_on_paragraph_boundaries():
    memory = get_memory()
    before = memory.count()
    # Unique content each run: record ids are content hashes, so storing the
    # same text twice upserts rather than adding, and a fixed string would
    # make this test pass or fail depending on what ran before it.
    marker = uuid.uuid4().hex
    text = "\n\n".join(
        f"Paragraph {i} about orbital mechanics and agents, run {marker}." for i in range(12)
    )
    chunks = memory.chunk_and_add(text, source=f"test/chunking-{marker}")

    assert chunks >= 1
    assert memory.count() == before + chunks
    assert memory.search(f"orbital mechanics {marker}", k=1)
