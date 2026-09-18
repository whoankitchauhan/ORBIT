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

    def _gemini(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        import time as _time
        import urllib.error
        import urllib.request

        model = settings.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={settings.gemini_api_key}"
        )
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        body_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(3):  # up to 3 attempts
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={"Content-Type": "application/json", "User-Agent": "ORBIT/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=settings.tool_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break  # success
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    # Rate limited — parse retry-after if available, else back off
                    err_body = e.read().decode("utf-8", errors="ignore")
                    wait = 15  # default wait
                    import re as _re
                    m = _re.search(r"retry in ([0-9.]+)s", err_body)
                    if m:
                        wait = min(30, float(m.group(1)) + 1)
                    _time.sleep(wait)
                    continue
                raise  # re-raise on non-429 or after all retries

        candidates = data.get("candidates", [])
        if not candidates:
            raise LLMError("Gemini API returned no completion candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "input_tokens": usage_meta.get("promptTokenCount", 0),
            "output_tokens": usage_meta.get("candidatesTokenCount", 0),
        }
        return text, usage


    def _http_chat(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system: str,
        prompt: str,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Generic OpenAI-compatible HTTP completion with zero hard package dependencies."""
        import urllib.request

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'dummy-key'}",
            "User-Agent": "ORBIT/1.0",
            **(extra_headers or {}),
        }
        body = {
            "model": model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=settings.tool_timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices", [])
        if not choices:
            raise LLMError(f"API returned no completion choices from {base_url}")
        text = choices[0].get("message", {}).get("content") or ""
        usage = {
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        }
        return text, usage

    def _ollama(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        """Call local Ollama instance (gemma3:4b, llama3, etc.) completely offline."""
        import urllib.request

        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        body = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ORBIT/1.0"},
        )
        timeout = max(45.0, settings.tool_timeout_seconds)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data.get("message", {}).get("content") or ""
        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }
        return text, usage

    def _groq(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        return self._http_chat(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            system=system,
            prompt=prompt,
        )

    def _xai(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        return self._http_chat(
            base_url="https://api.x.ai/v1",
            api_key=settings.xai_api_key,
            model=settings.xai_model,
            system=system,
            prompt=prompt,
        )

    def _openrouter(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        return self._http_chat(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            system=system,
            prompt=prompt,
            extra_headers={
                "HTTP-Referer": "https://github.com/orbit-agent/orbit",
                "X-Title": "ORBIT Agentic System",
            },
        )

    def _openai_compatible(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        return self._http_chat(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model or "default",
            system=system,
            prompt=prompt,
        )

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
            if provider == "ollama":
                text, usage = self._ollama(system, prompt)
                model = f"ollama/{settings.ollama_model}"
            elif provider == "gemini":
                text, usage = self._gemini(system, prompt)
                model = settings.gemini_model
            elif provider == "groq":
                text, usage = self._groq(system, prompt)
                model = settings.groq_model
            elif provider == "xai":
                text, usage = self._xai(system, prompt)
                model = settings.xai_model
            elif provider == "openrouter":
                text, usage = self._openrouter(system, prompt)
                model = settings.openrouter_model
            elif provider == "openai_compatible":
                text, usage = self._openai_compatible(system, prompt)
                model = settings.openai_model or "custom-model"
            elif provider == "anthropic":
                text, usage = self._anthropic(system, prompt)
                model = settings.anthropic_model
            elif provider == "openai":
                text, usage = self._openai(system, prompt)
                model = settings.openai_model
            else:
                text = offline.respond(task, prompt, context or {})
                model = "orbit-offline-planner"
        except Exception as exc:  # network down, bad key, rate limit
            # Try alternate real providers before falling back to the offline planner.
            _primary_exc = exc
            _fell_back_to_real = False

            # If primary was not Ollama and Ollama is running, fall back to local model immediately
            if provider != "ollama" and settings.has_ollama:
                try:
                    text, usage = self._ollama(system, prompt)
                    provider, model = "ollama", f"ollama/{settings.ollama_model}"
                    _fell_back_to_real = True
                except Exception:
                    pass

            # If still failed, try other keys
            if not _fell_back_to_real and provider != "gemini" and settings.gemini_api_key:
                try:
                    text, usage = self._gemini(system, prompt)
                    provider, model = "gemini", settings.gemini_model
                    _fell_back_to_real = True
                except Exception:
                    pass

            if not _fell_back_to_real and provider != "groq" and settings.groq_api_key:
                try:
                    text, usage = self._groq(system, prompt)
                    provider, model = "groq", settings.groq_model
                    _fell_back_to_real = True
                except Exception:
                    pass

            if not _fell_back_to_real:
                text = offline.respond(task, prompt, context or {})
                provider, model = "simulated", "orbit-offline-planner"
                text = f"{text}\n\n[fallback: {type(_primary_exc).__name__}]"

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
