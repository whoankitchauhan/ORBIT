"""Celery application for asynchronous workflow execution.

Import is safe even when Celery is not installed: `celery_app` is then `None`
and `app/runtime.py` routes work to its thread pool instead. That keeps the
import graph intact on a machine with no Redis.

Run a worker with:
    celery -A app.workers.celery_app.celery_app worker --loglevel=info
"""

from __future__ import annotations

from app.config import settings

celery_app = None

try:
    from celery import Celery

    celery_app = Celery(
        "orbit",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.workers.tasks"],
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Kolkata",
        enable_utc=True,
        # An agent workflow that has run for ten minutes is stuck, not slow.
        task_time_limit=600,
        task_soft_time_limit=540,
        worker_prefetch_multiplier=1,     # long tasks; don't hoard them
        task_acks_late=True,              # redeliver if a worker dies mid-task
        result_expires=86400,
    )
except ImportError:  # pragma: no cover - exercised only without Celery
    celery_app = None
