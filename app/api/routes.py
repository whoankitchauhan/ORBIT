"""FastAPI backend for ORBIT.

The Streamlit UI calls `app.runtime` directly, so this API is not required for
the demonstration. It exists because the architecture in the project document
places FastAPI between the frontend and the orchestration layer, and because it
is what a React frontend or an external system would integrate against.

Run with:
    uvicorn app.api.routes:api --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from app import runtime
from app.config import settings
from app.graph.workflow import graph_diagram, orchestration_backend
from app.memory.vector import get_memory
from app.tools.registry import load_all_tools

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI is not installed. Install it with: pip install fastapi uvicorn"
    ) from exc


api = FastAPI(
    title="ORBIT",
    description="Orchestrated Reasoning & Behavioral Intelligence Technology",
    version=settings.version,
)


class TaskRequest(BaseModel):
    objective: str = Field(..., min_length=3, max_length=4000)
    user_id: str = "api-user"
    background: bool = True


class DecisionRequest(BaseModel):
    decided_by: str = "reviewer"
    note: str = ""


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = "api-upload"
    kind: str = "document"


@api.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": settings.version,
        "orchestration": orchestration_backend(),
        "capabilities": settings.capabilities(),
    }


@api.post("/tasks", status_code=201)
def create_task(request: TaskRequest) -> dict[str, Any]:
    try:
        task_id = runtime.submit_task(
            request.objective, user_id=request.user_id, background=request.background
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "accepted"}


@api.get("/tasks")
def list_tasks(limit: int = 50) -> dict[str, Any]:
    return {"tasks": runtime.list_tasks(limit)}


@api.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    detail = runtime.task_detail(task_id)
    if detail["task"] is None and detail["state"] is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    return detail


@api.get("/approvals")
def list_approvals() -> dict[str, Any]:
    return {"pending": runtime.pending_approvals()}


@api.post("/approvals/{task_id}/approve")
def approve(task_id: str, request: DecisionRequest) -> dict[str, Any]:
    try:
        status = runtime.approve(task_id, request.decided_by, request.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "decision": status}


@api.post("/approvals/{task_id}/reject")
def reject(task_id: str, request: DecisionRequest) -> dict[str, Any]:
    try:
        status = runtime.reject(task_id, request.decided_by, request.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "decision": status}


@api.get("/tools")
def list_tools() -> dict[str, Any]:
    registry = load_all_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "risk": t.risk,
                "allowed_agents": list(t.allowed_agents),
                "signature": t.signature(),
            }
            for t in registry.all()
        ]
    }


@api.post("/memory")
def ingest(request: IngestRequest) -> dict[str, Any]:
    chunks = get_memory().chunk_and_add(request.text, source=request.source, kind=request.kind)
    return {"source": request.source, "chunks_stored": chunks, "total": get_memory().count()}


@api.get("/memory/search")
def search_memory(q: str, k: int = 5) -> dict[str, Any]:
    return {"results": [r.to_dict() for r in get_memory().search(q, k=k)]}


@api.get("/stats")
def stats() -> dict[str, Any]:
    return runtime.system_stats()


@api.get("/graph")
def graph() -> dict[str, str]:
    return {"mermaid": graph_diagram(), "backend": orchestration_backend()}
