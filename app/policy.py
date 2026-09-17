"""The escalation policy: when must a human decide?

This is deliberately ordinary Python rather than a prompt. A policy expressed
as an instruction to a model can be argued out of; a policy expressed as a
function cannot. Every rule below returns a reason string, so the approval card
can always tell the reviewer *why* they are being asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.graph.state import OrbitState
from app.tools.registry import load_all_tools

# Monetary ceilings above which a human always decides, whatever the model
# thinks of its own confidence.
REFUND_CEILING = 5000.0
EXTERNAL_RECIPIENT_TOOLS = {"send_email"}


@dataclass
class PolicyDecision:
    requires_approval: bool
    risk: str                 # low | medium | high
    reasons: list[str]

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no escalation triggers matched"


def assess_action(state: OrbitState, proposal: dict[str, Any]) -> PolicyDecision:
    """Decide whether a proposed action can proceed automatically."""
    registry = load_all_tools()
    reasons: list[str] = []
    risk = "low"

    tool_name = proposal.get("tool", "")
    arguments = proposal.get("arguments", {}) or {}

    # 1. The tool's own risk label.
    if registry.has(tool_name):
        tool = registry.get(tool_name)
        risk = tool.risk
        if tool.risk == "high":
            reasons.append(f"'{tool_name}' is classified as a high-risk external action")
        elif tool.risk == "medium":
            reasons.append(f"'{tool_name}' writes to a shared system")
    else:
        risk = "high"
        reasons.append(f"'{tool_name}' is not a registered tool")

    # 2. Confidence floor — thin evidence means a person checks the work.
    confidence = float(state.validation.get("confidence", state.analysis.get("confidence", 0.0)))
    if confidence < settings.approval_confidence_threshold:
        risk = "high"
        reasons.append(
            f"confidence {confidence:.0%} is below the {settings.approval_confidence_threshold:.0%} threshold"
        )

    # 3. Unresolved validation issues.
    issues = state.validation.get("issues", [])
    if issues:
        risk = "high"
        reasons.append(f"validation raised {len(issues)} unresolved issue(s)")

    # 4. Money.
    amount = arguments.get("amount")
    if amount is not None:
        try:
            if float(amount) > REFUND_CEILING:
                risk = "high"
                reasons.append(f"amount {float(amount):,.2f} exceeds the {REFUND_CEILING:,.0f} ceiling")
        except (TypeError, ValueError):
            risk = "high"
            reasons.append("the monetary amount could not be parsed")

    # 5. Anything leaving the organisation.
    if tool_name in EXTERNAL_RECIPIENT_TOOLS:
        risk = "high"
        reasons.append("this action sends a message to an external recipient")

    # 6. Missing required arguments — never ask a human to approve a call that
    #    is going to fail validation anyway.
    if registry.has(tool_name):
        missing = [
            p for p in registry.get(tool_name).required
            if not arguments.get(p) and arguments.get(p) != 0
        ]
        if missing:
            risk = "high"
            reasons.append(f"required argument(s) missing: {', '.join(missing)}")

    return PolicyDecision(requires_approval=risk == "high", risk=risk, reasons=reasons)


def summarise_action(proposal: dict[str, Any]) -> str:
    """One plain sentence describing what will happen if a reviewer approves."""
    tool = proposal.get("tool", "unknown action")
    arguments = proposal.get("arguments", {}) or {}

    templates = {
        "create_replacement_request": "Raise a replacement for order {order_id} belonging to {customer_id}.",
        "issue_refund": "Refund {amount} against order {order_id}.",
        "send_email": "Send an email to {to} with the subject “{subject}”.",
        "create_ticket": "Open a {priority}-priority ticket for {customer_id}: {subject}.",
        "update_record": "Set {table} record {record_id} to status “{status}”.",
    }
    template = templates.get(tool)
    if not template:
        pairs = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:4])
        return f"Run {tool} with {pairs or 'no arguments'}."
    try:
        return template.format(**{k: arguments.get(k, "—") for k in arguments} | 
                               {k: arguments.get(k, "—") for k in
                                ("order_id", "customer_id", "amount", "to", "subject",
                                 "priority", "table", "record_id", "status")})
    except (KeyError, IndexError):
        return f"Run {tool}."
