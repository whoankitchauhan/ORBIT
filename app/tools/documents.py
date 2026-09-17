"""Document ingestion and reading tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import ROOT
from app.memory.vector import get_memory
from app.tools.registry import registry

# Ingestion is confined to the project tree so a path in a model-generated
# argument cannot walk out to /etc or a user's home directory.
ALLOWED_ROOTS = [ROOT / "data", ROOT / "docs"]
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".rst"}
MAX_BYTES = 2_000_000


def _resolve_safely(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    for root in ALLOWED_ROOTS:
        try:
            path.relative_to(root.resolve())
            return path
        except ValueError:
            continue
    raise PermissionError(
        f"reading {path} is not permitted; allowed roots: "
        + ", ".join(str(r) for r in ALLOWED_ROOTS)
    )


@registry.tool(
    name="read_document",
    description="Read a text document from the project's data or docs directory.",
    risk="low",
    allowed_agents=("research",),
    parameters={"path": "str", "max_chars": "int = 4000"},
    required=("path",),
)
def read_document(path: str, max_chars: int = 4000) -> dict[str, Any]:
    try:
        resolved = _resolve_safely(path)
    except PermissionError as exc:
        return {"error": str(exc)}
    if not resolved.exists():
        return {"error": f"no such file: {resolved.name}"}
    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        return {"error": f"unsupported file type {resolved.suffix!r}"}
    if resolved.stat().st_size > MAX_BYTES:
        return {"error": "file exceeds the 2 MB read limit"}

    text = resolved.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(resolved.relative_to(ROOT)),
        "characters": len(text),
        "truncated": len(text) > int(max_chars),
        "content": text[: int(max_chars)],
    }


@registry.tool(
    name="ingest_document",
    description="Chunk a document and store it in semantic memory for later retrieval.",
    risk="medium",
    allowed_agents=("research",),
    parameters={"path": "str", "kind": "str = 'document'"},
    required=("path",),
)
def ingest_document(path: str, kind: str = "document") -> dict[str, Any]:
    result = read_document(path=path, max_chars=MAX_BYTES)
    if "error" in result:
        return result
    chunks = get_memory().chunk_and_add(
        result["content"], source=result["path"], kind=kind
    )
    return {"path": result["path"], "chunks_stored": chunks, "kind": kind}


@registry.tool(
    name="list_documents",
    description="List documents available for reading or ingestion.",
    risk="low",
    allowed_agents=("research", "supervisor"),
    parameters={},
)
def list_documents() -> dict[str, Any]:
    files: list[str] = []
    for root in ALLOWED_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.append(str(path.relative_to(ROOT)))
    return {"count": len(files), "documents": files[:100]}


def ingest_text(text: str, source: str, kind: str = "document") -> int:
    """Direct ingestion used by the UI's upload panel (not exposed to agents)."""
    return get_memory().chunk_and_add(text, source=source, kind=kind)
