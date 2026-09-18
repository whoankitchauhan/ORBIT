"""Central configuration for ORBIT.

Every external dependency named in the synopsis (PostgreSQL, ChromaDB, Redis,
Celery, OpenAI, Anthropic, LangGraph) is *optional at runtime*. ORBIT detects
what is actually available and reports it honestly through `capabilities()`,
falling back to an embedded equivalent so the system always runs end to end.

That design is deliberate: a demo that silently fails because Redis is not
running teaches nobody anything, and a viva examiner can unplug any single
component and watch the platform degrade rather than die.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# .env loading (no hard dependency on python-dotenv)
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
# Relocatable so the databases can live off a read-only or network-mounted
# checkout, e.g. ORBIT_DATA_DIR=/var/lib/orbit.
DATA_DIR = Path(os.getenv("ORBIT_DATA_DIR", str(ROOT / "data"))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables always win over the file.
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass
class Settings:
    """Runtime settings, read once at import time."""

    # --- identity -------------------------------------------------------
    app_name: str = "ORBIT"
    app_full_name: str = "Orchestrated Reasoning & Behavioral Intelligence Technology"
    version: str = "1.0.0"

    # --- LLM ------------------------------------------------------------
    # Free / Free-tier providers (primary)
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    xai_api_key: str = field(
        default_factory=lambda: os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))
    )
    xai_model: str = field(
        default_factory=lambda: os.getenv("XAI_MODEL", os.getenv("GROK_MODEL", "grok-2-latest"))
    )
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    )
    # Local / Offline Ollama provider
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma3:4b")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", os.getenv("LOCAL_LLM_URL", ""))
    )

    # Legacy / optional paid providers
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1200")))

    # --- search & tools -------------------------------------------------
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # --- structured store -----------------------------------------------
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    sqlite_path: Path = field(default_factory=lambda: DATA_DIR / "orbit.db")

    # --- vector store ----------------------------------------------------
    chroma_path: Path = field(default_factory=lambda: DATA_DIR / "chroma")
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "orbit_memory")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))

    # --- queue ------------------------------------------------------------
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    use_celery: bool = field(default_factory=lambda: _flag("USE_CELERY", False))

    # --- policy -----------------------------------------------------------
    approval_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("APPROVAL_CONFIDENCE_THRESHOLD", "0.65"))
    )
    max_supervisor_hops: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUPERVISOR_HOPS", "12"))
    )
    tool_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "20"))
    )
    allow_live_web_search: bool = field(default_factory=lambda: _flag("ALLOW_LIVE_WEB_SEARCH", False))

    # --- api ---------------------------------------------------------------
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # ------------------------------------------------------------------
    # Derived capability detection
    # ------------------------------------------------------------------

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_xai(self) -> bool:
        return bool(self.xai_api_key)

    @property
    def has_ollama(self) -> bool:
        if not self.ollama_base_url:
            return False
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.ollama_base_url.rstrip('/')}/api/tags",
                headers={"User-Agent": "ORBIT/1.0"}
            )
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_openai_compatible(self) -> bool:
        return bool(self.openai_base_url)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key) and _module_available("anthropic")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key) and _module_available("openai")

    @property
    def has_live_llm(self) -> bool:
        return (
            self.has_ollama
            or self.has_gemini
            or self.has_groq
            or self.has_xai
            or self.has_openrouter
            or self.has_openai_compatible
            or self.has_anthropic
            or self.has_openai
        )

    @property
    def has_postgres(self) -> bool:
        if not self.database_url:
            return False
        return _module_available("psycopg") or _module_available("psycopg2")

    @property
    def has_chroma(self) -> bool:
        return _module_available("chromadb")

    @property
    def has_langgraph(self) -> bool:
        return _module_available("langgraph")

    @property
    def has_redis(self) -> bool:
        if not _module_available("redis"):
            return False
        try:  # an installed client is not a reachable server
            import redis  # type: ignore

            redis.Redis.from_url(self.redis_url, socket_connect_timeout=0.4).ping()
            return True
        except Exception:
            return False

    @property
    def has_celery(self) -> bool:
        return self.use_celery and _module_available("celery") and self.has_redis

    def resolved_llm_provider(self) -> str:
        """Which LLM backend will actually answer a call."""
        choice = (self.llm_provider or "auto").lower()
        if choice in {"ollama", "local", "gemma"} and self.has_ollama:
            return "ollama"
        if choice in {"gemini", "google"} and self.has_gemini:
            return "gemini"
        if choice == "groq" and self.has_groq:
            return "groq"
        if choice in {"xai", "grok"} and self.has_xai:
            return "xai"
        if choice == "openrouter" and self.has_openrouter:
            return "openrouter"
        if choice in {"openai_compatible"} and self.has_openai_compatible:
            return "openai_compatible"
        if choice == "anthropic" and self.has_anthropic:
            return "anthropic"
        if choice == "openai" and self.has_openai:
            return "openai"
        if choice == "simulated":
            return "simulated"
        if choice == "auto":
            if self.has_ollama:
                return "ollama"
            if self.has_gemini:
                return "gemini"
            if self.has_groq:
                return "groq"
            if self.has_xai:
                return "xai"
            if self.has_openrouter:
                return "openrouter"
            if self.has_openai_compatible:
                return "openai_compatible"
            if self.has_anthropic:
                return "anthropic"
            if self.has_openai:
                return "openai"
        return "simulated"

    def capabilities(self) -> dict[str, dict[str, str]]:
        """A human-readable map of what is live versus what is standing in.

        The Streamlit sidebar renders this directly, so the operator always
        knows whether a result came from a real model or the offline planner.
        """
        provider = self.resolved_llm_provider()
        llm_detail_map = {
            "ollama": f"Ollama Local (Offline) · {self.ollama_model}",
            "gemini": f"Google Gemini (Free tier) · {self.gemini_model}",
            "groq": f"Groq (Free tier) · {self.groq_model}",
            "xai": f"xAI / Grok · {self.xai_model}",
            "openrouter": f"OpenRouter · {self.openrouter_model}",
            "openai_compatible": f"Local / Custom LLM · {self.openai_model}",
            "anthropic": f"Anthropic · {self.anthropic_model}",
            "openai": f"OpenAI · {self.openai_model}",
            "simulated": "Offline planner (no API key set)",
        }
        search_detail = (
            "Tavily (Free tier)"
            if self.has_tavily
            else ("DuckDuckGo (Free web)" if self.allow_live_web_search else "Disabled (offline safe)")
        )
        return {
            "Reasoning": {
                "mode": "live" if provider != "simulated" else "fallback",
                "detail": llm_detail_map.get(provider, "Offline planner"),
            },
            "Web search": {
                "mode": "live" if (self.has_tavily or self.allow_live_web_search) else "fallback",
                "detail": search_detail,
            },
            "Orchestration": {
                "mode": "live" if self.has_langgraph else "fallback",
                "detail": "LangGraph" if self.has_langgraph else "Built-in graph engine",
            },
            "Structured store": {
                "mode": "live" if self.has_postgres else "fallback",
                "detail": "PostgreSQL" if self.has_postgres else "SQLite (data/orbit.db)",
            },
            "Semantic memory": {
                "mode": "live" if self.has_chroma else "fallback",
                "detail": "ChromaDB" if self.has_chroma else "Embedded vector index",
            },
            "Background work": {
                "mode": "live" if self.has_celery else "fallback",
                "detail": "Celery + Redis" if self.has_celery else "In-process thread pool",
            },
        }


settings = Settings()
