"""The service layer every front end talks to.

Streamlit, the FastAPI routes and the Celery worker all call this module, so
task submission, approval and resumption behave identically no matter which
door the request comes through.

Long-running work is dispatched to Celery when Redis is reachable and to a
thread pool otherwise. The calling code does not change: `submit_task` returns
a task id immediately either way, and progress is read back from the store.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.config import settings
from app.graph.engine import GraphError
from app.graph.state import ApprovalRequest, OrbitState, new_id
from app.graph.workflow import get_workflow
from app.memory.store import get_store

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orbit")
_live: dict[str, dict[str, Any]] = {}      # task_id -> progress snapshot
_lock = threading.Lock()


def _mark(task_id: str, **fields: Any) -> None:
    with _lock:
        entry = _live.setdefault(task_id, {})
        entry.update(fields)
        entry["updated_at"] = time.time()


def progress(task_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_live.get(task_id, {}))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_workflow(state: OrbitState, on_step: Callable[[str, OrbitState], None] | None = None) -> OrbitState:
    """Drive the graph and normalise the outcome."""
    workflow = get_workflow()

    def step_callback(node: str, current: OrbitState) -> None:
        _mark(current.task_id, node=node, status=current.status,
              events=len(current.events), steps_done=sum(1 for s in current.plan if s.status == "done"))
        if on_step:
            on_step(node, current)

    try:
        result = workflow.invoke(state, thread_id=state.task_id, on_step=step_callback)
    except GraphError as exc:
        state.status = "failed"
        state.errors.append(str(exc))
        state.log("system", "error", f"Workflow error: {exc}")
        get_store().save_task(state)
        _mark(state.task_id, status="failed", error=str(exc))
        return state
    except Exception as exc:
        state.status = "failed"
        state.errors.append(f"{type(exc).__name__}: {exc}")
        state.log("system", "error", f"Unhandled failure: {exc}",
                  trace=traceback.format_exc()[-1500:])
        get_store().save_task(state)
        _mark(state.task_id, status="failed", error=str(exc))
        return state

    final = result.state
    if result.interrupted:
        final.status = "awaiting_approval"
        _mark(final.task_id, status="awaiting_approval", node=result.next_node,
              reason=result.interrupt_reason)
    else:
        _mark(final.task_id, status=final.status, node="finalise")
    get_store().save_task(final)
    return final


def submit_task(objective: str, user_id: str = "demo-user", background: bool = True) -> str:
    """Create a task and start it. Returns the task id immediately."""
    objective = (objective or "").strip()
    if not objective:
        raise ValueError("an objective is required")

    state = OrbitState(task_id=new_id("task"), objective=objective, user_id=user_id)
    get_store().save_task(state)
    _mark(state.task_id, status="queued", node="plan", objective=objective)

    if background and settings.has_celery:
        from app.workers.tasks import run_task_async

        run_task_async.delay(state.task_id, objective, user_id)
        _mark(state.task_id, dispatch="celery")
    elif background:
        _executor.submit(_run_workflow, state)
        _mark(state.task_id, dispatch="thread")
    else:
        _mark(state.task_id, dispatch="inline")
        _run_workflow(state)

    return state.task_id


def run_task_sync(objective: str, user_id: str = "demo-user") -> OrbitState:
    """Run a task to completion in the caller's thread. Used by tests and the CLI."""
    state = OrbitState(task_id=new_id("task"), objective=objective.strip(), user_id=user_id)
    get_store().save_task(state)
    return _run_workflow(state)


def run_task_streaming(
    objective: str,
    user_id: str = "demo-user",
    on_step: Callable[[str, OrbitState], None] | None = None,
) -> OrbitState:
    """Run inline, invoking `on_step` after every node.

    The Streamlit UI uses this so the trace appears as the workflow advances
    rather than arriving all at once at the end. Running in the caller's thread
    also keeps Streamlit's session state consistent, which background threads
    do not.
    """
    objective = (objective or "").strip()
    if not objective:
        raise ValueError("an objective is required")
    state = OrbitState(task_id=new_id("task"), objective=objective, user_id=user_id)
    get_store().save_task(state)
    _mark(state.task_id, status="running", node="plan", objective=objective)
    return _run_workflow(state, on_step=on_step)


def decide_streaming(
    task_id: str,
    approved: bool,
    decided_by: str = "reviewer",
    note: str = "",
    on_step: Callable[[str, OrbitState], None] | None = None,
) -> OrbitState:
    """Record a decision and resume inline, reporting each node as it runs."""
    workflow = get_workflow()
    state = workflow.get_state(task_id)
    if state is None:
        raise ValueError(f"no workflow checkpoint for task {task_id}")
    if state.approval is None:
        raise ValueError(f"task {task_id} is not waiting on an approval")

    approval = state.approval
    approval.status = "approved" if approved else "rejected"
    approval.decided_by = decided_by or "reviewer"
    approval.note = note
    approval.resolved_at = time.time()
    try:
        get_store().save_approval(task_id, approval)
    except Exception:
        pass

    def step_callback(node: str, current: OrbitState) -> None:
        _mark(current.task_id, node=node, status=current.status)
        if on_step:
            on_step(node, current)

    result = workflow.resume(
        task_id,
        update={"approval": approval, "status": "running" if approved else "rejected"},
        on_step=step_callback,
    )
    final = result.state
    if result.interrupted:
        final.status = "awaiting_approval"
    get_store().save_task(final)
    _mark(task_id, status=final.status)
    return final


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def _decide(task_id: str, approved: bool, decided_by: str, note: str, background: bool) -> str:
    workflow = get_workflow()
    state = workflow.get_state(task_id)
    if state is None:
        raise ValueError(f"no workflow checkpoint for task {task_id}")
    if state.approval is None:
        raise ValueError(f"task {task_id} is not waiting on an approval")

    approval: ApprovalRequest = state.approval
    approval.status = "approved" if approved else "rejected"
    approval.decided_by = decided_by or "reviewer"
    approval.note = note
    approval.resolved_at = time.time()

    try:
        get_store().save_approval(task_id, approval)
    except Exception:
        pass

    update = {"approval": approval, "status": "running" if approved else "rejected"}

    def resume() -> OrbitState:
        try:
            result = workflow.resume(
                task_id, update=update,
                on_step=lambda node, s: _mark(s.task_id, node=node, status=s.status),
            )
            final = result.state
            if result.interrupted:
                final.status = "awaiting_approval"
            get_store().save_task(final)
            _mark(task_id, status=final.status, node="finalise")
            return final
        except Exception as exc:
            _mark(task_id, status="failed", error=str(exc))
            raise

    _mark(task_id, status="resuming", decision=approval.status)
    if background:
        _executor.submit(resume)
    else:
        resume()
    return approval.status


def approve(task_id: str, decided_by: str = "reviewer", note: str = "", background: bool = True) -> str:
    """Approve the pending action and resume the workflow."""
    return _decide(task_id, True, decided_by, note, background)


def reject(task_id: str, decided_by: str = "reviewer", note: str = "", background: bool = True) -> str:
    """Reject the pending action; the workflow closes out without executing it."""
    return _decide(task_id, False, decided_by, note, background)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_state(task_id: str) -> OrbitState | None:
    """The live workflow state, which is richer than the tasks table row."""
    return get_workflow().get_state(task_id)


def list_tasks(limit: int = 50) -> list[dict[str, Any]]:
    return get_store().list_tasks(limit)


def pending_approvals() -> list[dict[str, Any]]:
    return get_store().list_pending_approvals()


def task_detail(task_id: str) -> dict[str, Any]:
    """Everything the UI needs for one task, from both stores."""
    store = get_store()
    state = get_state(task_id)
    return {
        "task": store.get_task(task_id),
        "state": state.to_dict() if state else None,
        "agent_runs": store.list_agent_runs(task_id),
        "tool_calls": store.list_tool_calls(task_id),
        "progress": progress(task_id),
    }


def system_stats() -> dict[str, Any]:
    from app.memory.vector import get_memory

    stats = get_store().stats()
    stats["memories"] = get_memory().count()
    return stats
