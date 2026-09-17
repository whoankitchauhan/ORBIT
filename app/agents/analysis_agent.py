"""The Analysis Agent — reasoning over gathered evidence.

Its confidence score is load-bearing rather than decorative: the approval
policy escalates to a human whenever confidence falls below the configured
threshold, so an honest low score is what triggers oversight.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agents.base import Agent, AgentResult
from app.graph.state import OrbitState
from app.llm import prompts

# Standalone numbers only. A bare `\d+` match pulls "5005" out of "CMP-5005"
# and, because of the hyphen, reports it as negative — which is how an analysis
# ends up describing a mean of -272 over a set of complaint ids. The trailing
# lookahead also excludes the components of an ISO date, so "2026-07-02" is
# skipped entirely rather than averaged in as a measurement.
NUMBER_RE = re.compile(r"(?<![\w.,-])\d+(?:,\d{3})*(?:\.\d+)?(?![\w-])")


class AnalysisAgent(Agent):
    name = "analysis"
    system_prompt = prompts.ANALYSIS
    description = "Compares, calculates and concludes over the Research Agent's evidence."

    def _numeric_support(self, state: OrbitState) -> dict[str, Any] | None:
        """Run real statistics when the evidence contains numbers.

        Letting a language model do arithmetic in prose is how analyses end up
        quietly wrong, so any numbers found are pushed through the calculator
        tool and the tool's answer is what gets reported.
        """
        blob = " ".join(str(f) for f in state.research.get("findings", []))[:3000]
        raw = NUMBER_RE.findall(blob)
        values: list[float] = []
        for token in raw:
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            values.append(value)
        if len(values) < 2:
            return None

        call = self.call_tool(
            "summarise_numbers",
            {"values": values[:40], "label": "figures found in evidence"},
            state,
        )
        return call.result if call.status == "ok" else None

    def run(self, state: OrbitState, instruction: str = "", step_index: int = 0, **kwargs: Any) -> AgentResult:
        started = time.perf_counter()
        instruction = instruction or state.objective

        stats = self._numeric_support(state)

        findings_block = "\n".join(
            f"- {f}" for f in state.research.get("findings", [])[:10]
        ) or "(no findings were gathered)"

        prompt = (
            f"Step: {instruction}\n"
            f"Overall objective: {state.objective}\n\n"
            f"Evidence gathered:\n{findings_block}\n\n"
            f"Sources: {', '.join(state.research.get('sources', [])) or 'none'}\n"
            f"Evidence coverage: {state.research.get('coverage', 'unknown')}\n"
            + (f"\nComputed statistics: {stats}\n" if stats else "")
            + "\nAnalyse this and state a conclusion with a calibrated confidence."
        )

        payload, response = self.llm.complete_json(
            prompt,
            system=self.system_prompt,
            task="analysis",
            context={"instruction": instruction, "research": state.research, "stats": stats},
            fallback={
                "summary": "Analysis output could not be parsed.",
                "observations": [], "conclusion": "", "confidence": 0.2,
                "verdict": "insufficient-evidence",
            },
        )

        try:
            confidence = float(payload.get("confidence", 0.4))
        except (TypeError, ValueError):
            confidence = 0.4
        confidence = max(0.0, min(1.0, confidence))

        analysis = {
            "summary": payload.get("summary", ""),
            "observations": payload.get("observations", [])[:8],
            "conclusion": payload.get("conclusion", ""),
            "confidence": round(confidence, 2),
            "verdict": payload.get("verdict", "supported"),
            "statistics": stats,
        }

        latency = int((time.perf_counter() - started) * 1000)
        self.record_run(state, step_index, instruction, analysis, response.provider, latency)
        return AgentResult(
            agent=self.name,
            output={"analysis": analysis, "confidence": analysis["confidence"]},
            provider=response.provider,
            latency_ms=latency,
        )
