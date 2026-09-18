"""Tests for free and free-tier LLM providers and web search integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.llm.provider import LLMClient, _extract_json
from app.tools.search import web_search


def test_extract_json_parses_plain_and_fenced():
    assert _extract_json('{"key": "value"}') == {"key": "value"}
    assert _extract_json('```json\n{"key": "value"}\n```') == {"key": "value"}
    assert _extract_json('Some text before {"key": "value"} and after') == {"key": "value"}
    assert _extract_json('```\n[1, 2, 3]\n```') == [1, 2, 3]


def test_free_provider_resolution():
    s = Settings(gemini_api_key="test-gemini", llm_provider="auto")
    assert s.has_gemini
    assert s.resolved_llm_provider() == "gemini"

    s_groq = Settings(groq_api_key="test-groq", llm_provider="groq")
    assert s_groq.has_groq
    assert s_groq.resolved_llm_provider() == "groq"

    s_xai = Settings(xai_api_key="test-xai", llm_provider="xai")
    assert s_xai.has_xai
    assert s_xai.resolved_llm_provider() == "xai"

    s_openrouter = Settings(openrouter_api_key="test-or", llm_provider="openrouter")
    assert s_openrouter.has_openrouter
    assert s_openrouter.resolved_llm_provider() == "openrouter"


def test_free_provider_auto_priority():
    # Gemini should be prioritized over Groq and OpenAI when multiple free keys are provided
    s = Settings(
        gemini_api_key="test-gemini",
        groq_api_key="test-groq",
        openai_api_key="test-openai",
        llm_provider="auto",
    )
    assert s.resolved_llm_provider() == "gemini"


def test_capabilities_readout_formats_free_tiers():
    s = Settings(gemini_api_key="test-gemini", tavily_api_key="test-tavily")
    caps = s.capabilities()
    assert caps["Reasoning"]["mode"] == "live"
    assert "Gemini (Free tier)" in caps["Reasoning"]["detail"]
    assert caps["Web search"]["mode"] == "live"
    assert "Tavily" in caps["Web search"]["detail"]


def test_llm_client_fallback_on_network_failure():
    client = LLMClient(provider="gemini")
    # complete() should not raise an exception; it should return a simulated fallback
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        resp = LLMClient(provider="gemini").complete("Hello", task="generic")
        assert resp.simulated
        assert "fallback" in resp.text


def test_web_search_disabled_when_not_configured():
    with patch("app.config.settings.tavily_api_key", ""), patch("app.config.settings.allow_live_web_search", False):
        res = web_search("latest AI news")
        assert not res["enabled"]
        assert "off" in res["note"]


def test_tavily_web_search_mock():
    mock_payload = {
        "results": [
            {"title": "Free API News", "content": "Generous free tiers available.", "url": "https://example.com/api"}
        ],
        "answer": "Summary of free APIs.",
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("app.config.settings.tavily_api_key", "mock-tavily-key"):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = web_search("AI agent research")
            assert res["enabled"]
            assert res["provider"] == "tavily"
            assert len(res["results"]) == 1
            assert res["results"][0]["title"] == "Free API News"
