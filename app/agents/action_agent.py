"""The Action Agent — the only agent that changes anything outside ORBIT.

It works in two separate phases, and the split is the point:

1.  `propose()` decides *what* it would do and returns a description. Nothing
    happens to the outside world.
2.  `execute()` runs that exact call, and is only ever reached after the
    approval gate has cleared.

Because proposal and execution are different methods with the graph's approval
node between them, there is no code path where an agent "decides and acts" in
one breath.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agents.base import Agent, AgentResult
from app.graph.state import OrbitState
from app.llm import prompts
from app.tools import external_api

CUSTOMER_RE = re.compile(r"\b(CUST[-_]?\d+)\b", re.I)
ORDER_RE = re.compile(r"\b(ORD[-_]?\d+)\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b")
AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr|\$)\s?([\d,]+\.?\d*)", re.I)

INTENT_TOOLS = [
    ({"replace", "replacement", "swap"}, "create_replacement_request"),
    ({"refund", "reimburse", "money back"}, "issue_refund"),
    ({"email", "mail", "write to", "notify"}, "send_email"),
    ({"ticket", "case", "escalate", "raise"}, "create_ticket"),
    ({"update", "close", "mark", "set status"}, "update_record"),
]


class ActionAgent(Agent):
    name = "action"
    system_prompt = prompts.ACTION
    description = "Executes one approved external operation. Never acts without clearance."

    # ------------------------------------------------------------- proposal

    def _heuristic_proposal(self, instruction: str, state: OrbitState) -> dict[str, Any]:
        text = f"{instruction} {state.objective}"
        lowered = text.lower()

        tool = "create_ticket"
        for keywords, candidate in INTENT_TOOLS:
            if any(k in lowered for k in keywords):
                tool = candidate
                break

        customer = CUSTOMER_RE.search(text)
        order = ORDER_RE.search(text)
        email = EMAIL_RE.search(text)
        amount = AMOUNT_RE.search(text)

        customer_id = customer.group(1).upper() if customer else self._infer_customer(state)
        order_id = order.group(1).upper() if order else self._infer_order(state)
        reason = (state.analysis.get("conclusion") or instruction)[:300]

        if tool == "create_replacement_request":
            arguments = {"customer_id": customer_id or "", "order_id": order_id or "", "reason": reason}
        elif tool == "issue_refund":
            value = 0.0
            if amount:
                try:
                    value = float(amount.group(1).replace(",", ""))
                except ValueError:
                    value = 0.0
            if not value:
                value = self._infer_order_amount(state, order_id)
            arguments = {"order_id": order_id or "", "amount": value, "reason": reason}
        elif tool == "send_email":
            arguments = {
                "to": email.group(0) if email else self._infer_email(state),
                "subject": f"Update on your request ({state.task_id})",
                "body": (state.analysis.get("conclusion") or state.objective)[:1200],
            }
        elif tool == "update_record":
            arguments = {"table": "complaints", "record_id": self._infer_complaint(state) or "",
                         "status": "resolved"}
        else:
            arguments = {
                "customer_id": customer_id or "unknown",
                "subject": instruction[:120],
                "detail": reason,
                "priority": "high" if state.risk == "high" else "normal",
            }
        return {"tool": tool, "arguments": arguments, "intent": instruction[:200]}

    # Helpers that mine the research evidence for identifiers the objective
    # did not state explicitly.
    def _rows(self, state: OrbitState) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in state.research.get("tool_results", []):
            result = entry.get("result")
            if isinstance(result, list):
                rows.extend(r for r in result if isinstance(r, dict))
            elif isinstance(result, dict) and "rows" in result:
                rows.extend(r for r in result["rows"] if isinstance(r, dict))
        return rows

    def _infer_customer(self, state: OrbitState) -> str:
        for row in self._rows(state):
            for key in ("customer_id", "id"):
                value = str(row.get(key, ""))
                if value.upper().startswith("CUST"):
                    return value.upper()
        return ""

    def _infer_order(self, state: OrbitState) -> str:
        for row in self._rows(state):
            for key in ("order_id", "id"):
                value = str(row.get(key, ""))
                if value.upper().startswith("ORD"):
                    return value.upper()
        return ""

    def _infer_complaint(self, state: OrbitState) -> str:
        for row in self._rows(state):
            value = str(row.get("id", ""))
            if value.upper().startswith("CMP"):
                return value.upper()
        return ""

    def _infer_email(self, state: OrbitState) -> str:
        for row in self._rows(state):
            email = str(row.get("email", ""))
            if "@" in email:
                return email
        return ""

    def _infer_order_amount(self, state: OrbitState, order_id: str) -> float:
        for row in self._rows(state):
            if str(row.get("id", "")).upper() == (order_id or "").upper():
                try:
                    return float(row.get("amount") or 0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def propose(self, state: OrbitState, instruction: str = "") -> dict[str, Any]:
        """Decide what to do — without doing it."""
        instruction = instruction or state.objective
        heuristic = self._heuristic_proposal(instruction, state)
        if self.llm.provider == "simulated":
            return heuristic

        prompt = (
            f"Objective: {state.objective}\n"
            f"Approved step: {instruction}\n\n"
            f"Analysis conclusion: {state.analysis.get('conclusion', '(none)')}\n"
            f"Evidence: {state.research.get('summary', '(none)')[:800]}\n\n"
            f"Tools you may use:\n{self.tool_menu()}\n\n"
            "Propose exactly one tool call to carry out this step."
        )
        payload, _ = self.llm.complete_json(
            prompt, system=self.system_prompt, task="generic", fallback=heuristic
        )
        tool = str(payload.get("tool", "")).strip()
        if not (self.registry.has(tool) and self.name in self.registry.get(tool).allowed_agents):
            return heuristic
        return {
            "tool": tool,
            "arguments": payload.get("arguments") or {},
            "intent": payload.get("intent", instruction)[:200],
        }

    # ------------------------------------------------------------ execution

    def execute(self, state: OrbitState, proposal: dict[str, Any], step_index: int = 0) -> AgentResult:
        """Run the approved call. Reached only after the approval gate clears."""
        started = time.perf_counter()
        external_api.set_task_context(state.task_id)

        call = self.call_tool(proposal["tool"], proposal.get("arguments", {}), state)
        latency = int((time.perf_counter() - started) * 1000)

        succeeded = call.status == "ok" and not (
            isinstance(call.result, dict) and call.result.get("status") == "failed"
        )
        action_result = {
            "tool": call.name,
            "arguments": call.arguments,
            "result": call.result,
            "status": "executed" if succeeded else "failed",
            "error": call.error or (
                call.result.get("error", "") if isinstance(call.result, dict) else ""
            ),
        }

        self.record_run(state, step_index, proposal, action_result, self.llm.provider, latency,
                        status="ok" if succeeded else "error")
        return AgentResult(
            agent=self.name,
            output={"action_result": action_result},
            provider=self.llm.provider,
            latency_ms=latency,
            status="ok" if succeeded else "error",
            error=action_result["error"],
        )

    def run(self, state: OrbitState, instruction: str = "", step_index: int = 0, **kwargs: Any) -> AgentResult:
        proposal = self.propose(state, instruction)
        return AgentResult(agent=self.name, output={"pending_action": proposal})
