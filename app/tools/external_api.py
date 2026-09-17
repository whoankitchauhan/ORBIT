"""External action tools — the consequential end of the system.

Everything here is marked `risk="high"` and restricted to the Action Agent,
which means the workflow cannot reach any of these functions without passing
the approval gate first. The gate is enforced in `app/graph/nodes.py`; the risk
label on each tool is what triggers it.

The side effects are written to a local `action_ledger` rather than sent to a
real CRM. That is a deliberate choice for an academic prototype: the approval
mechanism, the audit trail and the permission model are all genuine, while the
blast radius of a mistake during a demonstration is zero. Swapping the body of
`_dispatch` for a real HTTP call is the only change needed to go live.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from app.memory.store import get_store
from app.tools.registry import registry

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

# Current task id, set by the Action node so ledger rows are attributable.
_task_context: dict[str, str] = {"task_id": "unknown"}


def set_task_context(task_id: str) -> None:
    _task_context["task_id"] = task_id


def _dispatch(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Record an external action. Replace with a real API client for production."""
    reference = f"{action.upper()[:3]}-{uuid.uuid4().hex[:8].upper()}"
    outcome = {
        "reference": reference,
        "action": action,
        "status": "recorded",
        "executed_at": time.time(),
        "payload": payload,
        "note": "Recorded in the local action ledger (simulated external system).",
    }
    try:
        get_store().record_action(_task_context["task_id"], action, payload, reference)
    except Exception as exc:
        # Never report success when persistence failed — that is the exact
        # failure mode the project document warns about.
        outcome["status"] = "failed"
        outcome["error"] = f"could not persist action: {exc}"
    return outcome


@registry.tool(
    name="create_replacement_request",
    description="Raise a product replacement request for a customer order.",
    risk="high",
    allowed_agents=("action",),
    parameters={"customer_id": "str", "order_id": "str", "reason": "str"},
    required=("customer_id", "order_id", "reason"),
)
def create_replacement_request(customer_id: str, order_id: str, reason: str) -> dict[str, Any]:
    orders = get_store().query(
        "SELECT * FROM orders WHERE id = ? AND customer_id = ?", (order_id, customer_id)
    )
    if not orders:
        return {
            "status": "failed",
            "error": f"order {order_id} does not belong to customer {customer_id}",
        }
    return _dispatch(
        "replacement_request",
        {"customer_id": customer_id, "order_id": order_id,
         "product": orders[0].get("product"), "reason": reason[:500]},
    )


@registry.tool(
    name="create_ticket",
    description="Open a support ticket and assign it to a queue.",
    risk="high",
    allowed_agents=("action",),
    parameters={"customer_id": "str", "subject": "str", "detail": "str", "priority": "str = 'normal'"},
    required=("customer_id", "subject"),
)
def create_ticket(
    customer_id: str, subject: str, detail: str = "", priority: str = "normal"
) -> dict[str, Any]:
    if priority not in {"low", "normal", "high", "urgent"}:
        return {"status": "failed", "error": f"invalid priority {priority!r}"}
    return _dispatch(
        "support_ticket",
        {"customer_id": customer_id, "subject": subject[:200],
         "detail": detail[:2000], "priority": priority},
    )


@registry.tool(
    name="send_email",
    description="Send an email to a customer or colleague.",
    risk="high",
    allowed_agents=("action",),
    parameters={"to": "str", "subject": "str", "body": "str"},
    required=("to", "subject", "body"),
)
def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    if not EMAIL_RE.match(to.strip()):
        return {"status": "failed", "error": f"{to!r} is not a valid email address"}
    return _dispatch(
        "email", {"to": to.strip(), "subject": subject[:200], "body": body[:5000]}
    )


@registry.tool(
    name="issue_refund",
    description="Issue a monetary refund against an order.",
    risk="high",
    allowed_agents=("action",),
    parameters={"order_id": "str", "amount": "float", "reason": "str"},
    required=("order_id", "amount", "reason"),
)
def issue_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"status": "failed", "error": "amount must be numeric"}
    if amount <= 0:
        return {"status": "failed", "error": "refund amount must be positive"}

    orders = get_store().query("SELECT * FROM orders WHERE id = ?", (order_id,))
    if not orders:
        return {"status": "failed", "error": f"unknown order {order_id}"}
    order_total = float(orders[0].get("amount") or 0)
    if amount > order_total:
        # A refund larger than the order is almost always an agent arithmetic
        # error; refuse rather than approve an over-refund.
        return {
            "status": "failed",
            "error": f"refund {amount:.2f} exceeds order total {order_total:.2f}",
        }
    return _dispatch(
        "refund",
        {"order_id": order_id, "amount": round(amount, 2),
         "order_total": order_total, "reason": reason[:500]},
    )


@registry.tool(
    name="update_record",
    description="Update the status of a complaint or order record.",
    risk="high",
    allowed_agents=("action",),
    parameters={"table": "str", "record_id": "str", "status": "str"},
    required=("table", "record_id", "status"),
)
def update_record(table: str, record_id: str, status: str) -> dict[str, Any]:
    if table not in {"complaints", "orders"}:
        return {"status": "failed", "error": f"table {table!r} is not writable"}
    store = get_store()
    existing = store.query(f"SELECT id FROM {table} WHERE id = ?", (record_id,))
    if not existing:
        return {"status": "failed", "error": f"no {table[:-1]} with id {record_id}"}
    store.execute(f"UPDATE {table} SET status = ? WHERE id = ?", (status, record_id))
    return _dispatch("record_update", {"table": table, "record_id": record_id, "status": status})
