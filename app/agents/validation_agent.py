"""The Validation Agent — the last automated check before a human is involved.

It audits the other agents' work for grounding, completeness and internal
consistency. It has no tools: an auditor that can also fetch new evidence
tends to fix problems quietly instead of reporting them.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agents.base import Agent, AgentResult
from app.graph.state import OrbitState
from app.llm import prompts

STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "into", "are",
             "was", "will", "have", "has", "been", "check", "please", "would"}


class ValidationAgent(Agent):
    name = "validation"
    system_prompt = prompts.VALIDATION
    description = "Audits grounding, completeness and consistency before anything consequential happens."

    def _deterministic_checks(self, state: OrbitState) -> list[str]:
        """Structural checks that do not depend on a model's judgement.

        These run regardless of provider, so the audit has a floor even when
        the reasoning layer is degraded.
        """
        issues: list[str] = []

        if not state.research.get("findings"):
            issues.append("No evidence was gathered for this objective.")
        if not state.research.get("sources"):
            issues.append("Findings carry no source attribution.")
        if state.research.get("coverage") == "partial":
            issues.append("Evidence coverage is partial; the conclusion rests on limited sources.")
        if not state.analysis.get("conclusion"):
            issues.append("No explicit conclusion was recorded.")
        if state.analysis.get("verdict") == "insufficient-evidence":
            issues.append("The Analysis Agent reported insufficient evidence.")

        failed_tools = [c for c in state.tool_calls if c.status in {"error", "denied"}]
        if failed_tools:
            names = ", ".join(sorted({c.name for c in failed_tools}))
            issues.append(f"{len(failed_tools)} tool call(s) did not succeed: {names}.")

        # Objective coverage: if a distinctive term from the request never
        # appears anywhere in the work, some part of the request was dropped.
        terms = {
            w for w in re.findall(r"[a-zA-Z]{4,}", state.objective.lower())
            if w not in STOPWORDS
        }
        corpus = " ".join(
            [state.research.get("summary", ""), state.analysis.get("summary", ""),
             " ".join(str(f) for f in state.research.get("findings", []))]
        ).lower()
        missed = [t for t in terms if t not in corpus]
        if terms and len(missed) > len(terms) * 0.7:
            issues.append("Most terms from the objective do not appear in the work produced.")

        incomplete = [s for s in state.plan if s.status in {"pending", "failed"}]
        if incomplete:
            issues.append(f"{len(incomplete)} planned step(s) did not complete.")

        return issues

    def run(self, state: OrbitState, **kwargs: Any) -> AgentResult:
        started = time.perf_counter()
        deterministic = self._deterministic_checks(state)

        prompt = (
            f"Objective: {state.objective}\n\n"
            f"Research summary:\n{state.research.get('summary', '(none)')[:1500]}\n\n"
            f"Findings: {state.research.get('findings', [])[:8]}\n"
            f"Sources: {state.research.get('sources', [])}\n\n"
            f"Analysis:\n{state.analysis.get('summary', '(none)')[:1500]}\n"
            f"Conclusion: {state.analysis.get('conclusion', '(none)')}\n"
            f"Reported confidence: {state.analysis.get('confidence', 0)}\n\n"
            f"Automated checks already flagged: {deterministic or 'nothing'}\n\n"
            "Audit this work."
        )

        payload, response = self.llm.complete_json(
            prompt,
            system=self.system_prompt,
            task="validation",
            context={"research": state.research, "analysis": state.analysis},
            fallback={"passed": not deterministic, "issues": deterministic,
                      "confidence": state.analysis.get("confidence", 0.3), "notes": ""},
        )

        # Union the two check sets: the model may spot a reasoning flaw the
        # structural checks cannot, and vice versa. Neither overrides the other.
        issues = list(dict.fromkeys(deterministic + list(payload.get("issues", []))))
        try:
            confidence = float(payload.get("confidence", state.analysis.get("confidence", 0.3)))
        except (TypeError, ValueError):
            confidence = 0.3
        # Each unresolved issue costs confidence; five issues should not leave
        # a workflow claiming it is 90% sure.
        confidence = max(0.05, min(1.0, confidence - 0.07 * len(issues)))

        validation = {
            "passed": not issues,
            "issues": issues[:8],
            "confidence": round(confidence, 2),
            "notes": payload.get("notes", "") or (
                "All checks passed." if not issues else f"{len(issues)} issue(s) recorded."
            ),
        }

        latency = int((time.perf_counter() - started) * 1000)
        self.record_run(state, 98, "validation", validation, response.provider, latency)
        return AgentResult(
            agent=self.name,
            output={"validation": validation, "confidence": validation["confidence"]},
            provider=response.provider,
            latency_ms=latency,
        )
