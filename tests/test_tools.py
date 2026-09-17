"""Tool registry, permission model and tool behaviour."""

from __future__ import annotations

import pytest

from app.tools.registry import load_all_tools

registry = load_all_tools()


# --------------------------------------------------------------- permissions


def test_research_agent_cannot_reach_high_risk_actions():
    """The permission model must hold even if an agent asks for the tool by name."""
    call = registry.execute(
        "issue_refund", agent="research",
        arguments={"order_id": "ORD-1001", "amount": 100, "reason": "test"},
    )
    assert call.status == "denied"
    assert "not permitted" in call.error


def test_action_agent_may_reach_its_own_tools():
    tool = registry.get("create_replacement_request")
    assert "action" in tool.allowed_agents
    assert tool.risk == "high"
    assert tool.needs_approval


def test_every_external_action_is_high_risk():
    """A new action tool must not slip in below the approval gate."""
    for name in ("create_replacement_request", "issue_refund", "send_email",
                 "create_ticket", "update_record"):
        assert registry.get(name).risk == "high", f"{name} would bypass approval"
        assert registry.get(name).allowed_agents == ("action",)


def test_unknown_tool_is_reported_not_raised():
    call = registry.execute("definitely_not_a_tool", agent="research", arguments={})
    assert call.status == "error"
    assert "unknown tool" in call.error


def test_missing_required_arguments_are_caught_before_execution():
    call = registry.execute("issue_refund", agent="action", arguments={"order_id": "ORD-1001"})
    assert call.status == "error"
    assert "missing required argument" in call.error


# ---------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    "expression,expected",
    [("2 + 3 * 4", 14), ("(1240 - 980) / 980 * 100", pytest.approx(26.53, abs=0.01)),
     ("sqrt(144)", 12.0), ("round(3.14159, 2)", 3.14)],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    call = registry.execute("calculate", agent="analysis", arguments={"expression": expression})
    assert call.status == "ok"
    assert call.result["result"] == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "open('/etc/passwd').read()",
     "(1).__class__.__bases__", "9**9**9", "1/0"],
)
def test_calculator_refuses_anything_that_is_not_arithmetic(expression):
    """The calculator must never become a code-execution path."""
    call = registry.execute("calculate", agent="analysis", arguments={"expression": expression})
    assert call.status == "ok"          # handled, not crashed
    assert "error" in call.result


def test_summarise_numbers_handles_a_single_value():
    call = registry.execute("summarise_numbers", agent="analysis", arguments={"values": [7]})
    assert call.result["stdev"] == 0.0   # one point has no spread
    assert call.result["mean"] == 7


def test_compare_options_ranks_and_reports_margin():
    call = registry.execute(
        "compare_options", agent="analysis",
        arguments={
            "options": {"A": {"cost": 3, "speed": 9}, "B": {"cost": 8, "speed": 4}},
            "weights": {"cost": 1.0, "speed": 2.0},
        },
    )
    assert call.result["winner"] == "A"
    assert call.result["margin"] > 0


# ----------------------------------------------------------------- database


def test_table_preview_rejects_tables_outside_the_whitelist():
    call = registry.execute(
        "table_preview", agent="research", arguments={"table": "approvals; DROP TABLE users"}
    )
    assert "error" in call.result


def test_document_reads_stay_inside_the_project():
    call = registry.execute(
        "read_document", agent="research", arguments={"path": "/etc/passwd"}
    )
    assert "error" in call.result
    assert "not permitted" in call.result["error"]


# ----------------------------------------------------------------- integrity


def test_refund_cannot_exceed_the_order_total():
    """An arithmetic slip must not become an over-refund."""
    from app.seed import seed_all

    seed_all(verbose=False)
    call = registry.execute(
        "issue_refund", agent="action",
        arguments={"order_id": "ORD-1002", "amount": 999999, "reason": "test"},
    )
    assert call.result["status"] == "failed"
    assert "exceeds order total" in call.result["error"]


def test_replacement_requires_the_order_to_belong_to_the_customer():
    from app.seed import seed_all

    seed_all(verbose=False)
    call = registry.execute(
        "create_replacement_request", agent="action",
        arguments={"customer_id": "CUST-002", "order_id": "ORD-1001", "reason": "test"},
    )
    assert call.result["status"] == "failed"
