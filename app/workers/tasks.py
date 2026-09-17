"""Background jobs executed by Celery workers.

Each task is written so that it can also be called directly as a plain
function, which is how the thread-pool fallback and the test suite use it.
"""

from __future__ import annotations

from typing import Any

from app.graph.state import OrbitState
from app.workers.celery_app import celery_app


def _execute(task_id: str, objective: str, user_id: str) -> dict[str, Any]:
    from app.runtime import _run_workflow

    state = OrbitState(task_id=task_id, objective=objective, user_id=user_id)
    final = _run_workflow(state)
    return {"task_id": final.task_id, "status": final.status, "confidence": final.confidence}


def _ingest(text: str, source: str, kind: str) -> dict[str, Any]:
    from app.memory.vector import get_memory

    chunks = get_memory().chunk_and_add(text, source=source, kind=kind)
    return {"source": source, "chunks": chunks}


if celery_app is not None:

    @celery_app.task(name="orbit.run_task", bind=True, max_retries=2)
    def run_task_async(self, task_id: str, objective: str, user_id: str = "demo-user"):  # type: ignore[no-untyped-def]
        """Run a full agent workflow off the request thread."""
        try:
            return _execute(task_id, objective, user_id)
        except Exception as exc:
            # Back off before retrying: an immediate retry against a rate-limited
            # model API just burns the remaining attempts.
            raise self.retry(exc=exc, countdown=10)

    @celery_app.task(name="orbit.ingest_document")
    def ingest_document_async(text: str, source: str, kind: str = "document"):  # type: ignore[no-untyped-def]
        """Chunk and embed a large document without blocking the UI."""
        return _ingest(text, source, kind)

else:  # pragma: no cover - no Celery installed

    class _DirectTask:
        """Stand-in exposing `.delay()` so callers need no branching."""

        def __init__(self, fn):
            self._fn = fn

        def delay(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    run_task_async = _DirectTask(_execute)
    ingest_document_async = _DirectTask(_ingest)
