"""Offline reasoning routines used when no model API is configured.

Each routine reads the same context an agent would hand a hosted model and
emits output in the same shape, so the graph, the tools, the memory layer and
the approval gate all behave identically whether or not a key is present.

These are heuristics, not a language model. They are honest about that: the
client marks every offline result `provider="simulated"` and the UI badges it.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Verbs that suggest the objective wants something *done*, not just answered.
ACTION_VERBS = {
    "create", "raise", "open", "file", "submit", "send", "email", "notify",
    "issue", "refund", "replace", "cancel", "update", "delete", "schedule",
    "book", "escalate", "assign", "register", "approve", "transfer", "pay",
}

ANALYSIS_VERBS = {
    "compare", "analyse", "analyze", "evaluate", "assess", "determine",
    "calculate", "compute", "rank", "score", "decide", "judge", "measure",
    "eligible", "eligibility", "whether", "recommend", "forecast", "trend",
}

RESEARCH_VERBS = {
    "research", "find", "search", "look", "retrieve", "gather", "check",
    "review", "read", "investigate", "summarise", "summarize", "history",
    "background", "policy", "document", "who", "what", "when", "where",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "my", "is",
    "are", "was", "with", "that", "this", "it", "if", "then", "please", "i",
    "me", "we", "our", "us", "be", "can", "do", "does", "about", "into",
}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 2]


def _clauses(objective: str) -> list[str]:
    """Split an objective into candidate subtasks.

    Users write compound objectives joined by 'and then', commas or newlines;
    each clause usually maps to one delegated step.
    """
    text = re.sub(r"\s+", " ", objective.strip())
    parts = re.split(r",\s*(?:and\s+)?|\band then\b|\bthen\b|\bafter that\b|;\s*", text, flags=re.I)
    clauses = [p.strip(" .") for p in parts if p and len(p.strip()) > 3]
    return clauses or [text]


QUESTION_STARTERS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should",
    "would", "will", "tell", "explain", "describe",
}


def _is_question(clause: str) -> bool:
    """Is this clause asking something rather than ordering something?

    "What is our refund policy?" contains the word "refund", which would
    otherwise class it as an action and send a question down the path that
    issues money.
    """
    stripped = clause.strip()
    if stripped.endswith("?"):
        return True
    words = re.findall(r"[a-z]+", stripped.lower())
    return bool(words) and words[0] in QUESTION_STARTERS


def _classify(clause: str) -> str:
    """Decide which specialist should own a clause."""
    words = set(re.findall(r"[a-z]+", clause.lower()))
    if _is_question(clause):
        return "analysis" if words & ANALYSIS_VERBS else "research"
    if words & ACTION_VERBS:
        return "action"
    if words & ANALYSIS_VERBS:
        return "analysis"
    return "research"


def _keywords(text: str, limit: int = 8) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text.lower()) if w not in STOPWORDS]
    seen: dict[str, int] = {}
    for word in words:
        if len(word) > 2:
            seen[word] = seen.get(word, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


# ---------------------------------------------------------------------------
# Routines
# ---------------------------------------------------------------------------


def _plan(objective: str, context: dict[str, Any]) -> str:
    clauses = _clauses(objective)
    steps: list[dict[str, Any]] = []
    for clause in clauses:
        agent = _classify(clause)
        steps.append({"agent": agent, "instruction": clause[0].upper() + clause[1:]})

    # Research must precede analysis: an analysis step with nothing to analyse
    # is the commonest way an agent plan goes wrong.
    if any(s["agent"] == "analysis" for s in steps) and not any(
        s["agent"] == "research" for s in steps
    ):
        steps.insert(
            0,
            {
                "agent": "research",
                "instruction": f"Gather background needed for: {objective[:120]}",
            },
        )
    # Likewise an action needs something to justify it.
    if steps and steps[0]["agent"] == "action":
        steps.insert(
            0,
            {
                "agent": "research",
                "instruction": f"Verify the facts required before acting on: {objective[:120]}",
            },
        )
    for i, step in enumerate(steps):
        step["index"] = i
    return json.dumps(
        {
            "steps": steps[:6],
            "rationale": (
                f"Split the objective into {min(len(steps), 6)} delegated steps by "
                "matching each clause to the specialist that owns that kind of work."
            ),
        }
    )


def _research(prompt: str, context: dict[str, Any]) -> str:
    instruction = context.get("instruction", "")
    findings: list[str] = []
    sources: list[str] = []

    for memory in context.get("retrieved", [])[:5]:
        findings.append(memory.get("text", "")[:260])
        sources.append(memory.get("source", "semantic-memory"))

    for call in context.get("tool_results", []):
        result = call.get("result")
        name = call.get("name", "tool")
        if isinstance(result, list):
            for row in result[:4]:
                findings.append(f"{name}: {json.dumps(row, default=str)[:240]}")
            sources.append(name)
        elif isinstance(result, dict):
            findings.append(f"{name}: {json.dumps(result, default=str)[:260]}")
            sources.append(name)
        elif result:
            findings.append(f"{name}: {str(result)[:260]}")
            sources.append(name)

    if not findings:
        findings.append(
            f"No stored record matched '{instruction[:80]}'. Treating this as a gap "
            "rather than a negative finding."
        )

    summary_lines = [f"- {f}" for f in findings[:6]]
    summary = (
        f"Retrieved {len(findings)} item(s) relevant to: {instruction[:120]}\n"
        + "\n".join(summary_lines)
    )
    return json.dumps(
        {
            "summary": summary,
            "findings": findings[:6],
            "sources": sorted(set(sources)) or ["none"],
            "coverage": "partial" if len(findings) < 3 else "good",
        }
    )


def _analysis(prompt: str, context: dict[str, Any]) -> str:
    instruction = context.get("instruction", "")
    research = context.get("research", {}) or {}
    findings = research.get("findings", [])

    observations: list[str] = []
    # The Analysis Agent has already pushed any numbers through the calculator
    # tool. Re-extracting them here would duplicate that work with a looser
    # regex and produce a second, contradictory set of figures.
    stats = context.get("stats") or {}
    if stats and not stats.get("error"):
        observations.append(
            f"{stats['count']} figure(s) in the evidence: range {stats['min']:g} to "
            f"{stats['max']:g}, mean {stats['mean']:.2f}."
        )
    for finding in findings[:4]:
        observations.append(f"Considered: {str(finding)[:200]}")

    # Confidence tracks evidence volume, because an analysis resting on one
    # retrieved row should not be presented with the same certainty as one
    # resting on five.
    confidence = min(0.92, 0.35 + 0.12 * len(findings))
    if research.get("coverage") == "partial":
        confidence -= 0.1
    confidence = round(max(0.15, confidence), 2)

    verdict = "supported" if confidence >= 0.6 else "insufficient-evidence"
    conclusion = (
        f"Based on {len(findings)} retrieved item(s), the request '{instruction[:100]}' "
        f"is assessed as {verdict}."
    )
    return json.dumps(
        {
            "summary": conclusion + ("\n" + "\n".join(f"- {o}" for o in observations) if observations else ""),
            "observations": observations[:6],
            "conclusion": conclusion,
            "confidence": confidence,
            "verdict": verdict,
        }
    )


def _validation(prompt: str, context: dict[str, Any]) -> str:
    research = context.get("research", {}) or {}
    analysis = context.get("analysis", {}) or {}
    issues: list[str] = []

    if not research.get("findings"):
        issues.append("No research findings were produced for this objective.")
    if research.get("coverage") == "partial":
        issues.append("Evidence coverage is partial; conclusions rest on limited sources.")
    if not analysis.get("conclusion"):
        issues.append("No explicit conclusion was recorded by the Analysis Agent.")
    if analysis.get("verdict") == "insufficient-evidence":
        issues.append("Analysis reported insufficient evidence to support the request.")

    confidence = float(analysis.get("confidence", 0.4))
    confidence = round(max(0.1, confidence - 0.08 * len(issues)), 2)
    return json.dumps(
        {
            "passed": not issues,
            "issues": issues,
            "confidence": confidence,
            "notes": (
                "All checks passed."
                if not issues
                else f"{len(issues)} quality issue(s) recorded; escalating for review."
            ),
        }
    )


def _final(prompt: str, context: dict[str, Any]) -> str:
    objective = context.get("objective", "")
    research = context.get("research", {}) or {}
    analysis = context.get("analysis", {}) or {}
    validation = context.get("validation", {}) or {}
    action = context.get("action_result", {}) or {}
    approval = context.get("approval") or {}

    lines = [f"**Objective** — {objective}", ""]

    if research.get("findings"):
        lines.append("**What was found**")
        lines.extend(f"- {f}" for f in research["findings"][:5])
        lines.append("")
    if analysis.get("conclusion"):
        lines.append("**What it means**")
        lines.append(analysis["conclusion"])
        for observation in analysis.get("observations", [])[:3]:
            lines.append(f"- {observation}")
        lines.append("")
    if approval:
        decision = approval.get("status", "pending")
        lines.append("**Human decision**")
        note = approval.get("note") or "no note recorded"
        lines.append(f"The `{approval.get('action')}` action was {decision} ({note}).")
        lines.append("")
    if action:
        lines.append("**What was done**")
        lines.append(f"`{action.get('tool', 'action')}` returned: {json.dumps(action.get('result'), default=str)[:400]}")
        lines.append("")

    confidence = validation.get("confidence", analysis.get("confidence", 0.0))
    lines.append(f"**Confidence** — {float(confidence):.0%}")
    if validation.get("issues"):
        lines.append("")
        lines.append("**Caveats**")
        lines.extend(f"- {i}" for i in validation["issues"][:4])
    return "\n".join(lines)


def _generic(prompt: str, context: dict[str, Any]) -> str:
    sentences = _sentences(prompt)
    keywords = _keywords(prompt)
    head = sentences[0][:300] if sentences else prompt[:300]
    return (
        f"{head}\n\nKey terms: {', '.join(keywords) if keywords else 'none identified'}."
    )


_ROUTINES = {
    "plan": lambda p, c: _plan(c.get("objective", p), c),
    "research": _research,
    "analysis": _analysis,
    "validation": _validation,
    "final": _final,
    "generic": _generic,
}


def respond(task: str, prompt: str, context: dict[str, Any]) -> str:
    routine = _ROUTINES.get(task, _generic)
    return routine(prompt, context)
