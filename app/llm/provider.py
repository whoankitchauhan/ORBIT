"""Model access for every agent, behind one interface.

`LLMClient.complete()` routes to Anthropic, OpenAI, or an offline planner. The
offline planner is not a stub that returns "TODO" — it is a rule-based reasoner
that reads the same context the real model would and produces structurally
valid output. That keeps the graph, the tools, the memory and the approval
gate fully exercisable with no API key and no network, which is what makes the
project demonstrable in a lab.

Anything produced offline is marked `provider="simulated"` and the UI labels it,
so a result is never passed off as model reasoning when it was not.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.llm import offline


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def simulated(self) -> bool:
        return self.provider == "simulated"


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model response.

    Models wrap JSON in prose or fences more often than they should, so a
    bare `json.loads` is not safe here.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("response did not contain parsable JSON")


class LLMClient:
    """One client for all agents; the provider is chosen at construction."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.resolved_llm_provider()
        self._client: Any = None

    # ----------------------------------------------------------- backends

    def _anthropic(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        import anthropic  # imported lazily so the package stays optional

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = {
            "input_tokens": getattr(message.usage, "input_tokens", 0),
            "output_tokens": getattr(message.usage, "output_tokens", 0),
        }
        return text, usage

    def _openai(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        from openai import OpenAI  # lazy import

        if self._client is None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        completion = self._client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = completion.choices[0].message.content or ""
        usage = {
            "input_tokens": getattr(completion.usage, "prompt_tokens", 0),
            "output_tokens": getattr(completion.usage, "completion_tokens", 0),
        }
        return text, usage

    # ------------------------------------------------------------- public

    def complete(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        task: str = "generic",
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        `task` and `context` are ignored by the hosted providers and used by
        the offline planner to pick the right reasoning routine.
        """
        started = time.perf_counter()
        provider = self.provider
        usage: dict[str, int] = {}

        try:
            if provider == "anthropic":
                text, usage = self._anthropic(system, prompt)
                model = settings.anthropic_model
            elif provider == "openai":
                text, usage = self._openai(system, prompt)
                model = settings.openai_model
            else:
                text = offline.respond(task, prompt, context or {})
                model = "orbit-offline-planner"
        except Exception as exc:  # network down, bad key, rate limit
            # Falling back keeps a live demo alive rather than dropping the
            # whole workflow because one call failed.
            text = offline.respond(task, prompt, context or {})
            provider, model = "simulated", "orbit-offline-planner"
            text = f"{text}\n\n[fallback: {type(exc).__name__}]"

        latency = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            text=text, provider=provider, model=model, latency_ms=latency, usage=usage
        )

    def complete_json(
        self,
        prompt: str,
        system: str = "Reply with JSON only.",
        task: str = "generic",
        context: dict[str, Any] | None = None,
        fallback: Any = None,
    ) -> tuple[Any, LLMResponse]:
        """Generate and parse structured output, with a safe fallback.

        Structured-output parse failure is a named risk in the project
        document; this is where it is handled rather than allowed to crash a
        node mid-workflow.
        """
        system = f"{system}\nRespond with valid JSON only. No prose, no code fences."
        response = self.complete(prompt, system=system, task=task, context=context)
        try:
            return _extract_json(response.text), response
        except LLMError:
            if fallback is not None:
                return fallback, response
            raise


_default_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """Process-wide client. Cheap to share; holds no per-request state."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_llm() -> None:
    """Drop the cached client so a settings change takes effect."""
    global _default_client
    _default_client = None
