"""Read-only structured-data tools.

Agents get named lookups, not a free-form SQL endpoint. An agent that can emit
arbitrary SQL is an agent that can drop a table when a prompt tells it to, so
every query here is written in advance and only its parameters are bound.
"""

from __future__ import annotations

from typing import Any

from app.memory.store import get_store
from app.tools.registry import registry

# Whitelist for the deliberately narrow `table_preview` tool.
READABLE_TABLES = {"customers", "orders", "complaints", "tasks", "approvals", "action_ledger"}


@registry.tool(
    name="lookup_customer",
    description="Fetch a customer record by id, email or name.",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"identifier": "str"},
    required=("identifier",),
)
def lookup_customer(identifier: str) -> list[dict[str, Any]]:
    store = get_store()
    like = f"%{identifier}%"
    return store.query(
        """SELECT * FROM customers
           WHERE id = ? OR email = ? OR name LIKE ? OR id LIKE ?
           LIMIT 5""",
        (identifier, identifier, like, like),
    )


@registry.tool(
    name="customer_orders",
    description="List a customer's orders, most recent first.",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"customer_id": "str", "limit": "int = 10"},
    required=("customer_id",),
)
def customer_orders(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    return get_store().query(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY ordered_at DESC LIMIT ?",
        (customer_id, int(limit)),
    )


@registry.tool(
    name="customer_complaints",
    description="Retrieve a customer's complaint history — the support case file.",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"customer_id": "str", "limit": "int = 10"},
    required=("customer_id",),
)
def customer_complaints(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    return get_store().query(
        "SELECT * FROM complaints WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
        (customer_id, int(limit)),
    )


@registry.tool(
    name="table_preview",
    description="Preview rows from an approved table. Read-only and whitelisted.",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"table": "str", "limit": "int = 5"},
    required=("table",),
)
def table_preview(table: str, limit: int = 5) -> dict[str, Any]:
    if table not in READABLE_TABLES:
        return {
            "error": f"table {table!r} is not readable",
            "readable_tables": sorted(READABLE_TABLES),
        }
    # `table` is validated against a fixed set, never interpolated from raw input.
    rows = get_store().query(f"SELECT * FROM {table} LIMIT ?", (int(limit),))
    return {"table": table, "row_count": len(rows), "rows": rows}


@registry.tool(
    name="count_rows",
    description="Count rows in an approved table, optionally filtered by status.",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"table": "str", "status": "str = ''"},
    required=("table",),
)
def count_rows(table: str, status: str = "") -> dict[str, Any]:
    if table not in READABLE_TABLES:
        return {"error": f"table {table!r} is not readable"}
    if status:
        rows = get_store().query(
            f"SELECT COUNT(*) AS n FROM {table} WHERE status = ?", (status,)
        )
    else:
        rows = get_store().query(f"SELECT COUNT(*) AS n FROM {table}", ())
    return {"table": table, "status": status or "any", "count": rows[0]["n"] if rows else 0}
