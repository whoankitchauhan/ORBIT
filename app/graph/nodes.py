"""Graph nodes: the functions the engine actually executes.

Each node takes the state and returns a partial update. Keeping the agents
themselves free of graph concepts means the same agent classes can be driven
by the built-in engine, by LangGraph, or by a unit test calling them directly.

The approval gate lives in `human_gate`. It raises `Interrupt`, which freezes
the run *before* the action executes and hands control to a person.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.action_agent import ActionAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.research_agent import ResearchAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.validation_agent import ValidationAgent
from app.config import settings
from app.graph.engine import Interrupt
from app.graph.state import ApprovalRequest, OrbitState, new_id
from app.memory.retrieval import remember_task_outcome
from app.memory.store import get_store
from app.policy import assess_action, summarise_action

supervisor = SupervisorAgent()
research = ResearchAgent()
analysis = AnalysisAgent()
action = ActionAgent()
validation = ValidationAgent()

AGENTS = {
    "supervisor": supervisor,
    "research": research,
    "analysis": analysis,
    "action": action,
    "validation": validation,
}


def _persist(state: OrbitState) -> None:
    """Mirror state into the structured store; never fatal if it fails."""
    try:
        get_store().save_task(state)
    except Exception as exc:
        state.errors.append(f"persistence: {exc}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def plan_node(state: OrbitState) -> dict[str, Any]:
    """Supervisor decomposes the objective into delegated steps."""
    state.status = "running"
    state.log("supervisor", "system", "Task received; decomposing objective")
    _persist(state)

    result = supervisor.plan(state)
    plan = result.output["plan"]
    state.log(
        "supervisor", "routing",
        f"Planned {len(plan)} step(s): " + " → ".join(f"{s.agent}" for s in plan),
        rationale=result.output.get("rationale", ""),
        provider=result.provider,
    )
    return {
        "plan": plan,
        "status": "running",
        "retrieved": result.output.get("retrieved", []),
    }


def supervisor_node(state: OrbitState) -> dict[str, Any]:
    """Advance the cursor and announce the routing decision."""
    state.hops += 1
    step = state.current_step()
    if step is None:
        state.log("supervisor", "routing", "All steps complete; sending to validation")
        return {"hops": state.hops}

    if state.hops > settings.max_supervisor_hops:
        # Circuit breaker: a plan that will not terminate gets cut off rather
        # than burning tokens until the step limit trips.
        state.log("supervisor", "error", "Hop limit reached; ending delegation early")
        for pending in state.plan:
            if pending.status == "pending":
                pending.status = "skipped"
        return {"hops": state.hops, "plan": state.plan}

    step.status = "running"
    step.started_at = time.time()
    state.log(
        "supervisor", "routing",
        f"Step {step.index + 1}/{len(state.plan)} → {step.agent} agent",
        instruction=step.instruction,
    )
    return {"hops": state.hops, "plan": state.plan, "cursor": step.index}


def research_node(state: OrbitState) -> dict[str, Any]:
    step = state.current_step()
    instruction = step.instruction if step else state.objective
    result = research.run(state, instruction=instruction, step_index=step.index if step else 0)

    if step:
        step.status = "done"
        step.finished_at = time.time()
    state.log("research", "agent", "Evidence gathered",
              findings=len(result.output["research"].get("findings", [])),
              provider=result.provider, latency_ms=result.latency_ms)
    return {**result.output, "plan": state.plan}


def analysis_node(state: OrbitState) -> dict[str, Any]:
    step = state.current_step()
    instruction = step.instruction if step else state.objective
    result = analysis.run(state, instruction=instruction, step_index=step.index if step else 0)

    if step:
        step.status = "done"
        step.finished_at = time.time()
    state.log("analysis", "agent",
              f"Conclusion reached (confidence {result.output['analysis']['confidence']:.0%})",
              provider=result.provider, latency_ms=result.latency_ms)
    return {**result.output, "plan": state.plan}


def action_propose_node(state: OrbitState) -> dict[str, Any]:
    """Decide what the action would be, and how risky it is. Nothing executes."""
    step = state.current_step()
    instruction = step.instruction if step else state.objective
    proposal = action.propose(state, instruction)

    decision = assess_action(state, proposal)
    state.log(
        "action", "policy",
        f"Proposed {proposal['tool']} — risk {decision.risk}",
        tool=proposal["tool"], arguments=proposal["arguments"], reasons=decision.reasons,
    )

    if step:
        # The step is not done: it is staged. It completes only after the gate
        # clears and `action_execute_node` actually runs the call.
        step.status = "running"

    update: dict[str, Any] = {
        "pending_action": proposal,
        "risk": decision.risk,
        "plan": state.plan,
    }

    if decision.requires_approval:
        approval = ApprovalRequest(
            id=new_id("apr"),
            action=proposal["tool"],
            summary=summarise_action(proposal),
            reason=decision.reason_text,
            risk=decision.risk,
            arguments=proposal.get("arguments", {}),
        )
        update["approval"] = approval
        update["status"] = "awaiting_approval"
    return update


def validation_node(state: OrbitState) -> dict[str, Any]:
    result = validation.run(state)
    validation_output = result.output["validation"]
    state.log(
        "validation", "agent",
        "Audit passed" if validation_output["passed"]
        else f"Audit raised {len(validation_output['issues'])} issue(s)",
        issues=validation_output["issues"], provider=result.provider,
    )
    return result.output


def human_gate_node(state: OrbitState) -> dict[str, Any]:
    """Pause for a human decision, or let a cleared action through.

    Re-entered on resume: the engine restarts this node with the decision
    already merged into state, so the same code both raises the interrupt and
    consumes its answer.
    """
    approval = state.approval

    if approval is None or approval.status == "approved":
        if approval:
            state.log("human", "policy",
                      f"Approved by {approval.decided_by or 'reviewer'}", note=approval.note)
        return {"status": "running"}

    if approval.status == "rejected":
        state.log("human", "policy",
                  f"Rejected by {approval.decided_by or 'reviewer'}", note=approval.note)
        for step in state.plan:
            if step.status in {"pending", "running"}:
                step.status = "skipped"
        return {"status": "rejected", "plan": state.plan, "pending_action": None}

    # Still pending — freeze the workflow here.
    state.status = "awaiting_approval"
    state.log("human", "policy", f"Awaiting approval: {approval.summary}",
              risk=approval.risk, reason=approval.reason)
    try:
        get_store().save_approval(state.task_id, approval)
    except Exception:
        pass
    _persist(state)
    raise Interrupt(approval.summary, {"approval_id": approval.id})


def action_execute_node(state: OrbitState) -> dict[str, Any]:
    """Run the approved call."""
    proposal = state.pending_action
    if not proposal:
        # Mark the staged step resolved before returning. Leaving it "running"
        # would make the supervisor select it again on the next hop.
        step = state.current_step()
        if step and step.agent == "action":
            step.status = "skipped"
        state.log("action", "system", "No action staged; nothing to execute")
        return {"plan": state.plan}

    result = action.execute(state, proposal, step_index=state.cursor)
    step = state.current_step()
    if step:
        step.status = "done" if result.status == "ok" else "failed"
        step.finished_at = time.time()

    outcome = result.output["action_result"]
    state.log("action", "agent",
              f"{outcome['tool']} {outcome['status']}"
              + (f" — {outcome['error']}" if outcome["error"] else ""),
              reference=(outcome.get("result") or {}).get("reference")
              if isinstance(outcome.get("result"), dict) else None)
    return {**result.output, "plan": state.plan, "pending_action": None}


def finalise_node(state: OrbitState) -> dict[str, Any]:
    """Write the final answer, store the episode, close the task."""
    result = supervisor.synthesise(state)
    final_answer = result.output["final_answer"]

    status = "rejected" if state.status == "rejected" else "completed"
    confidence = float(state.validation.get("confidence", state.analysis.get("confidence", 0.0)))

    state.final_answer = final_answer
    state.status = status
    state.confidence = confidence

    try:
        memory_id = remember_task_outcome(state)
        if memory_id:
            state.log("supervisor", "memory", "Outcome written to long-term memory",
                      memory_id=memory_id)
    except Exception as exc:
        state.errors.append(f"memory write: {exc}")

    state.log("supervisor", "system", f"Task {status} (confidence {confidence:.0%})")
    _persist(state)
    try:
        get_store().save_workflow_state(state.task_id, "final", str(state.to_dict())[:100000])
        if state.approval:
            get_store().save_approval(state.task_id, state.approval)
    except Exception:
        pass

    return {"final_answer": final_answer, "status": status, "confidence": confidence}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


def route_from_supervisor(state: OrbitState) -> str:
    """Conditional edge: which specialist runs next, or are we done?"""
    if state.hops > settings.max_supervisor_hops:
        return "validation"
    step = state.current_step()
    if step is None:
        return "validation"
    return step.agent if step.agent in {"research", "analysis", "action"} else "validation"


def route_after_validation(state: OrbitState) -> str:
    """After the audit: gate a staged action, or finish."""
    return "gate" if state.pending_action else "finalise"


def route_after_gate(state: OrbitState) -> str:
    """After a human decides: execute, or stop."""
    if state.status == "rejected":
        return "finalise"
    return "execute"
