"""The Research Agent — evidence gathering only.

It selects tools, runs them, retrieves from semantic memory, and reports what
it found. It draws no conclusions and takes no actions; that separation is
what lets the Analysis Agent be held to a calibrated confidence and the Action
Agent be held behind an approval gate.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agents.base import Agent, AgentResult
from app.config import settings
from app.graph.state import OrbitState
from app.llm import prompts
from app.memory.retrieval import retrieve_context

# Identifier shapes that appear in support objectives, e.g. "CUST-001", "ORD-1042".
CUSTOMER_RE = re.compile(r"\b(CUST[-_]?\d+)\b", re.I)
ORDER_RE = re.compile(r"\b(ORD[-_]?\d+)\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b")

POLICY_HINTS = {"policy", "eligible", "eligibility", "entitled", "rule", "warranty",
                "refund", "replacement", "return", "sla", "escalation"}
HISTORY_HINTS = {"history", "previous", "past", "earlier", "prior", "complaint",
                 "complaints", "order", "orders", "record"}


class ResearchAgent(Agent):
    name = "research"
    system_prompt = prompts.RESEARCH
    description = "Gathers evidence from memory, documents and the database. Never decides or acts."

    # ------------------------------------------------------- tool selection

    def _heuristic_tools(self, instruction: str, state: OrbitState) -> list[dict[str, Any]]:
        """Pick tools from the text when no model is available to choose.

        Identifier patterns are strong signals: an objective naming CUST-001
        almost certainly wants that customer's record, whatever else it asks.
        """
        text = f"{instruction} {state.objective}"
        words = set(re.findall(r"[a-z]+", text.lower()))
        plan: list[dict[str, Any]] = [
            {"tool": "knowledge_search", "arguments": {"query": instruction[:200], "k": 4}}
        ]

        customer = CUSTOMER_RE.search(text)
        email = EMAIL_RE.search(text)
        identifier = customer.group(1).upper() if customer else (email.group(0) if email else None)
        if identifier:
            plan.append({"tool": "lookup_customer", "arguments": {"identifier": identifier}})
            if words & HISTORY_HINTS:
                customer_id = customer.group(1).upper() if customer else identifier
                plan.append({"tool": "customer_complaints", "arguments": {"customer_id": customer_id}})

        # Policy comes before order history: an eligibility question is decided
        # by the rule, and the four-call budget should not be spent before the
        # rule has been fetched.
        if words & POLICY_HINTS:
            plan.append({"tool": "policy_lookup", "arguments": {"topic": instruction[:150]}})

        if identifier and (words & HISTORY_HINTS):
            customer_id = customer.group(1).upper() if customer else identifier
            plan.append({"tool": "customer_orders", "arguments": {"customer_id": customer_id}})

        # For general-knowledge questions that don't match policy/customer/order
        # patterns, enable web search so the agent can find real information.
        is_general = not (words & POLICY_HINTS) and not (words & HISTORY_HINTS) and not identifier
        if is_general and (settings.allow_live_web_search or True):
            plan.append({"tool": "web_search", "arguments": {"query": instruction[:200]}})

        return plan[:4]  # a research step should not fan out indefinitely

    def _choose_tools(self, instruction: str, state: OrbitState) -> list[dict[str, Any]]:
        """Ask the model which tools to use; fall back to heuristics."""
        heuristic = self._heuristic_tools(instruction, state)
        if self.llm.provider == "simulated":
            return heuristic

        prompt = (
            f"Objective: {state.objective}\n"
            f"Your current step: {instruction}\n\n"
            f"Tools available to you:\n{self.tool_menu()}\n\n"
            "Choose up to 4 tool calls that will gather the evidence this step needs. "
            'Return {"calls": [{"tool": str, "arguments": object}]}.'
        )
        payload, _ = self.llm.complete_json(
            prompt,
            system=self.system_prompt,
            task="generic",
            fallback={"calls": heuristic},
        )
        calls = payload.get("calls") or []

        chosen: list[dict[str, Any]] = []
        for call in calls[:4]:
            name = str(call.get("tool", "")).strip()
            # Validate against the registry: a hallucinated tool name must not
            # reach execution, and neither must one this agent cannot use.
            if self.registry.has(name) and self.name in self.registry.get(name).allowed_agents:
                chosen.append({"tool": name, "arguments": call.get("arguments") or {}})
        return chosen or heuristic

    # ------------------------------------------------------------- execution

    def run(self, state: OrbitState, instruction: str = "", step_index: int = 0, **kwargs: Any) -> AgentResult:
        started = time.perf_counter()
        instruction = instruction or state.objective

        records = retrieve_context(instruction, k=settings.retrieval_top_k)
        # Filter out low-relevance records (e.g. old episodes about totally
        # different topics that just happen to share a few stopwords).
        # A score below 0.15 (hashed-embedding space) is almost certainly noise.
        records = [r for r in records if r.score > 0.15]
        retrieved = [r.to_dict() for r in records]
        if retrieved:
            state.log(self.name, "memory", f"Recalled {len(retrieved)} memory item(s)",
                      top_score=retrieved[0]["score"])

        tool_results: list[dict[str, Any]] = []
        for call_spec in self._choose_tools(instruction, state):
            call = self.call_tool(call_spec["tool"], call_spec["arguments"], state)
            if call.status == "ok" and call.result:
                tool_results.append({"name": call.name, "result": call.result})

        prompt = (
            f"Step: {instruction}\n\n"
            f"Retrieved memory:\n"
            + ("\n".join(f"[{i+1}] {r['text'][:400]}" for i, r in enumerate(retrieved)) or "(none)")
            + "\n\nTool results:\n"
            + ("\n".join(f"{t['name']}: {str(t['result'])[:500]}" for t in tool_results) or "(none)")
            + "\n\nReport your findings, grounded strictly in the material above."
        )

        payload, response = self.llm.complete_json(
            prompt,
            system=self.system_prompt,
            task="research",
            context={"instruction": instruction, "retrieved": retrieved, "tool_results": tool_results},
            fallback={
                "summary": "Research output could not be parsed.",
                "findings": [], "sources": [], "coverage": "none",
            },
        )

        # Merge with anything an earlier research step already established, so
        # a multi-step plan accumulates evidence instead of overwriting it.
        existing = state.research or {}
        findings = list(existing.get("findings", [])) + list(payload.get("findings", []))
        sources = sorted(set(existing.get("sources", [])) | set(payload.get("sources", [])))
        summary = "\n\n".join(filter(None, [existing.get("summary", ""), payload.get("summary", "")]))

        research = {
            "summary": summary.strip(),
            "findings": findings[:12],
            "sources": sources,
            "coverage": payload.get("coverage", "partial"),
            "tool_results": (existing.get("tool_results", []) + tool_results)[:10],
        }

        latency = int((time.perf_counter() - started) * 1000)
        self.record_run(state, step_index, instruction, research, response.provider, latency)
        return AgentResult(
            agent=self.name,
            output={"research": research, "retrieved": retrieved},
            provider=response.provider,
            latency_ms=latency,
        )
