"""Retrieval tools for the Research Agent."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from app.config import settings
from app.memory.retrieval import recall_similar_tasks, retrieve_context
from app.tools.registry import registry


@registry.tool(
    name="knowledge_search",
    description="Semantic search over the ingested knowledge base and stored documents.",
    risk="low",
    allowed_agents=("research", "analysis", "supervisor"),
    parameters={"query": "str", "k": "int = 4", "kind": "str = ''"},
    required=("query",),
)
def knowledge_search(query: str, k: int = 4, kind: str = "") -> list[dict[str, Any]]:
    kinds = [kind] if kind else None
    records = retrieve_context(query, k=int(k), kinds=kinds)
    return [r.to_dict() for r in records]


@registry.tool(
    name="policy_lookup",
    description="Find the policy text governing a decision (refunds, replacements, escalation).",
    risk="low",
    allowed_agents=("research", "analysis"),
    parameters={"topic": "str"},
    required=("topic",),
)
def policy_lookup(topic: str) -> list[dict[str, Any]]:
    records = retrieve_context(topic, k=3, kinds=["policy"])
    if not records:
        # Fall back to the whole corpus rather than reporting "no policy",
        # which would read as "no policy exists" instead of "none tagged".
        records = retrieve_context(f"policy {topic}", k=3)
    return [r.to_dict() for r in records]


@registry.tool(
    name="recall_past_tasks",
    description="Recall earlier ORBIT tasks similar to the current objective.",
    risk="low",
    allowed_agents=("research", "supervisor", "analysis"),
    parameters={"objective": "str", "k": "int = 3"},
    required=("objective",),
)
def recall_past_tasks(objective: str, k: int = 3) -> list[dict[str, Any]]:
    return [r.to_dict() for r in recall_similar_tasks(objective, k=int(k))]


@registry.tool(
    name="web_search",
    description="Public web search. Disabled unless ALLOW_LIVE_WEB_SEARCH is enabled.",
    risk="medium",
    allowed_agents=("research",),
    parameters={"query": "str", "max_results": "int = 5"},
    required=("query",),
)
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Query DuckDuckGo's instant-answer endpoint.

    Off by default: an offline lab machine would otherwise stall on a network
    timeout in the middle of a live demonstration.
    """
    if not settings.allow_live_web_search:
        return {
            "enabled": False,
            "note": "Live web search is off. Set ALLOW_LIVE_WEB_SEARCH=true to enable it.",
            "results": [],
        }

    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ORBIT/1.0"})
        with urllib.request.urlopen(request, timeout=settings.tool_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"enabled": True, "error": f"{type(exc).__name__}: {exc}", "results": []}

    results: list[dict[str, str]] = []
    if payload.get("AbstractText"):
        results.append(
            {
                "title": payload.get("Heading", query),
                "snippet": payload["AbstractText"],
                "url": payload.get("AbstractURL", ""),
            }
        )
    for topic in payload.get("RelatedTopics", []):
        if len(results) >= int(max_results):
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(
                {
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                }
            )
    return {"enabled": True, "query": query, "results": results}
