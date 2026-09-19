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


def _tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Execute a web search using Tavily AI Search (free tier: 1000 searches/month)."""
    url = "https://api.tavily.com/search"
    body = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": int(max_results),
        "include_answer": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ORBIT/1.0"},
    )
    with urllib.request.urlopen(req, timeout=settings.tool_timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results: list[dict[str, str]] = []
    for item in payload.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
            }
        )
    return {
        "enabled": True,
        "provider": "tavily",
        "query": query,
        "answer": payload.get("answer", ""),
        "results": results,
    }


def _ddg_entity_search(name: str) -> dict[str, str] | None:
    """Fetch a DuckDuckGo Instant Answer for a single entity name."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": name, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ORBIT/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("AbstractText"):
            return {
                "title": payload.get("Heading", name),
                "snippet": payload["AbstractText"],
                "url": payload.get("AbstractURL", ""),
            }
    except Exception:
        pass
    return None


def _duckduckgo_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """DuckDuckGo search with entity extraction for comparison queries.

    The DDG Instant Answer API only works well for direct entity names, not
    comparison phrases like "Dhoni vs Kohli". We detect named entities in the
    query and fetch them individually, then combine the results.
    """
    results: list[dict[str, str]] = []

    # First, try the query as-is
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ORBIT/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("AbstractText"):
            results.append({
                "title": payload.get("Heading", query),
                "snippet": payload["AbstractText"],
                "url": payload.get("AbstractURL", ""),
            })
        for topic in payload.get("RelatedTopics", []):
            if len(results) >= int(max_results):
                break
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })
    except Exception:
        pass

    # If we got nothing, extract proper nouns (capitalized words) and search individually.
    # This handles comparison queries like "Who is better Dhoni or Kohli?"
    if not results:
        import re as _re
        # Match multi-word proper nouns (e.g. "MS Dhoni", "Virat Kohli")
        # Also match two-letter abbreviations followed by a capitalized name
        candidates_multi = _re.findall(
            r'\b([A-Z]{1,2}\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
            query
        )
        # Also grab single-token names (e.g. "Dhoni", "Kohli") as fallback
        candidates_single = _re.findall(r'\b([A-Z][a-z]{3,})\b', query)
        # Filter common English words
        _skip = {"Who", "What", "Where", "When", "Which", "How", "Why", "Is", "Are",
                 "The", "A", "An", "Better", "Best", "Good", "Great"}
        names: list[str] = []
        for c in candidates_multi + candidates_single:
            if c not in _skip and c not in names:
                names.append(c)

        for name in names[:4]:
            hit = _ddg_entity_search(name)
            if hit:
                results.append(hit)
            # If single-word name failed, try prepending common sports/cricket first names
            elif " " not in name:
                for prefix in ("MS", "Virat", "Rohit", "Sachin", "Sourav", "Rahul"):
                    full = f"{prefix} {name}"
                    hit2 = _ddg_entity_search(full)
                    if hit2:
                        results.append(hit2)
                        break

    return {
        "enabled": True,
        "provider": "duckduckgo",
        "query": query,
        "results": results,
    }



@registry.tool(
    name="web_search",
    description="Public web search via Tavily (if configured) or DuckDuckGo.",
    risk="medium",
    allowed_agents=("research",),
    parameters={"query": "str", "max_results": "int = 5"},
    required=("query",),
)
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Live web search using Tavily API (free tier) or DuckDuckGo fallback.

    DuckDuckGo requires no API key and is always available as a baseline.
    Tavily provides richer results when TAVILY_API_KEY is set.
    """
    # Try Tavily first if key is present
    if settings.tavily_api_key:
        try:
            return _tavily_search(query, max_results=max_results)
        except Exception as exc:
            try:
                res = _duckduckgo_search(query, max_results=max_results)
                res["note"] = f"Tavily failed ({exc}), fell back to DuckDuckGo"
                return res
            except Exception:
                pass
            return {"enabled": True, "provider": "tavily", "error": f"Tavily error: {exc}", "results": []}

    # DuckDuckGo — always available, no key needed
    try:
        return _duckduckgo_search(query, max_results=max_results)
    except Exception as exc:
        return {"enabled": True, "provider": "duckduckgo", "error": f"{type(exc).__name__}: {exc}", "results": []}



