"""Structured persistence for tasks, agent runs, tool calls and approvals.

PostgreSQL is the target (per the synopsis) and is used whenever `DATABASE_URL`
is set and a driver is installed. Otherwise the same schema runs on SQLite in
`data/orbit.db`. The SQL is written to work on both: no dialect-specific types,
and parameters are bound rather than interpolated.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        role TEXT DEFAULT 'analyst',
        created_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        objective TEXT NOT NULL,
        status TEXT NOT NULL,
        risk TEXT,
        confidence DOUBLE PRECISION DEFAULT 0,
        final_answer TEXT,
        created_at DOUBLE PRECISION,
        updated_at DOUBLE PRECISION,
        completed_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        step_index INTEGER,
        input TEXT,
        output TEXT,
        status TEXT,
        provider TEXT,
        latency_ms INTEGER,
        started_at DOUBLE PRECISION,
        completed_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workflow_states (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        checkpoint TEXT,
        state_json TEXT,
        created_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        action TEXT,
        summary TEXT,
        reason TEXT,
        risk TEXT,
        arguments TEXT,
        status TEXT,
        decided_by TEXT,
        note TEXT,
        requested_at DOUBLE PRECISION,
        resolved_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_name TEXT,
        tool_name TEXT,
        arguments TEXT,
        result TEXT,
        status TEXT,
        risk TEXT,
        duration_ms INTEGER,
        created_at DOUBLE PRECISION
    )
    """,
    # --- demo business data the Database tool reads -----------------------
    """
    CREATE TABLE IF NOT EXISTS customers (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        tier TEXT,
        joined_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        customer_id TEXT,
        product TEXT,
        amount DOUBLE PRECISION,
        status TEXT,
        ordered_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS complaints (
        id TEXT PRIMARY KEY,
        customer_id TEXT,
        order_id TEXT,
        subject TEXT,
        detail TEXT,
        status TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS action_ledger (
        id TEXT PRIMARY KEY,
        task_id TEXT,
        action TEXT,
        payload TEXT,
        outcome TEXT,
        created_at DOUBLE PRECISION
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_task ON agent_runs(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_tools_task ON tool_calls(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_approvals_task ON approvals(task_id)",
]


class Store:
    """Thin data-access layer shared by the agents, the API and the UI."""

    def __init__(self) -> None:
        self.backend = "postgres" if settings.has_postgres else "sqlite"
        self._lock = threading.Lock()
        self._sqlite: sqlite3.Connection | None = None
        if self.backend == "sqlite":
            self._sqlite = sqlite3.connect(
                settings.sqlite_path, check_same_thread=False, timeout=15
            )
            self._sqlite.row_factory = sqlite3.Row
            # WAL gives better concurrency for the UI reading while a worker
            # writes, but it relies on shared memory mapping that some network
            # and container filesystems do not provide. Fall back rather than
            # fail on those.
            try:
                self._sqlite.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                self._sqlite.execute("PRAGMA journal_mode=DELETE")
            self._sqlite.execute("PRAGMA busy_timeout=10000")
        self.init_schema()

    # ------------------------------------------------------------ plumbing

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Yield a cursor, committing on success and rolling back on error."""
        if self.backend == "postgres":
            import psycopg  # lazy: only needed on the Postgres path

            with psycopg.connect(settings.database_url) as conn:
                with conn.cursor() as cur:
                    yield cur
                conn.commit()
        else:
            assert self._sqlite is not None
            with self._lock:
                cur = self._sqlite.cursor()
                try:
                    yield cur
                    self._sqlite.commit()
                except Exception:
                    self._sqlite.rollback()
                    raise
                finally:
                    cur.close()

    def _q(self, sql: str) -> str:
        """Translate the `?` placeholder style to `%s` for psycopg."""
        return sql.replace("?", "%s") if self.backend == "postgres" else sql

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._cursor() as cur:
            cur.execute(self._q(sql), params)

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(self._q(sql), params)
            if cur.description is None:
                return []
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def init_schema(self) -> None:
        for statement in SCHEMA:
            self.execute(statement)
        for statement in INDEXES:
            try:
                self.execute(statement)
            except Exception:
                pass  # index creation is best-effort

    # --------------------------------------------------------------- tasks

    def save_task(self, state: Any) -> None:
        existing = self.query("SELECT id FROM tasks WHERE id = ?", (state.task_id,))
        completed = time.time() if state.status in {"completed", "rejected", "failed"} else None
        if existing:
            self.execute(
                """UPDATE tasks SET status=?, risk=?, confidence=?, final_answer=?,
                   updated_at=?, completed_at=? WHERE id=?""",
                (
                    state.status, state.risk, state.confidence, state.final_answer,
                    time.time(), completed, state.task_id,
                ),
            )
        else:
            self.execute(
                """INSERT INTO tasks (id, user_id, objective, status, risk, confidence,
                   final_answer, created_at, updated_at, completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    state.task_id, state.user_id, state.objective, state.status,
                    state.risk, state.confidence, state.final_answer,
                    state.created_at, time.time(), completed,
                ),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return rows[0] if rows else None

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    # ---------------------------------------------------------- agent runs

    def save_agent_run(self, run: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO agent_runs (id, task_id, agent_name, step_index, input, output,
               status, provider, latency_ms, started_at, completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run["id"], run["task_id"], run["agent_name"], run.get("step_index", 0),
                json.dumps(run.get("input"), default=str)[:8000],
                json.dumps(run.get("output"), default=str)[:8000],
                run.get("status", "ok"), run.get("provider", ""),
                run.get("latency_ms", 0), run.get("started_at", time.time()),
                run.get("completed_at", time.time()),
            ),
        )

    def list_agent_runs(self, task_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM agent_runs WHERE task_id = ? ORDER BY started_at", (task_id,)
        )

    # ---------------------------------------------------------- tool calls

    def save_tool_call(self, task_id: str, call: Any) -> None:
        self.execute(
            """INSERT INTO tool_calls (id, task_id, agent_name, tool_name, arguments,
               result, status, risk, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                f"tc-{int(time.time() * 1e6) % 10**12}-{call.name[:6]}", task_id,
                call.agent, call.name,
                json.dumps(call.arguments, default=str)[:4000],
                json.dumps(call.result, default=str)[:8000],
                call.status, call.risk, call.duration_ms, time.time(),
            ),
        )

    def list_tool_calls(self, task_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM tool_calls WHERE task_id = ? ORDER BY created_at", (task_id,)
        )

    # ----------------------------------------------------------- approvals

    def save_approval(self, task_id: str, approval: Any) -> None:
        existing = self.query("SELECT id FROM approvals WHERE id = ?", (approval.id,))
        if existing:
            self.execute(
                "UPDATE approvals SET status=?, decided_by=?, note=?, resolved_at=? WHERE id=?",
                (approval.status, approval.decided_by, approval.note,
                 approval.resolved_at, approval.id),
            )
        else:
            self.execute(
                """INSERT INTO approvals (id, task_id, action, summary, reason, risk,
                   arguments, status, decided_by, note, requested_at, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    approval.id, task_id, approval.action, approval.summary,
                    approval.reason, approval.risk,
                    json.dumps(approval.arguments, default=str)[:4000],
                    approval.status, approval.decided_by, approval.note,
                    approval.requested_at, approval.resolved_at,
                ),
            )

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        return self.query(
            """SELECT a.*, t.objective FROM approvals a
               LEFT JOIN tasks t ON t.id = a.task_id
               WHERE a.status = 'pending' ORDER BY a.requested_at""",
        )

    def list_approvals(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM approvals ORDER BY requested_at DESC LIMIT ?", (limit,)
        )

    # ------------------------------------------------------ workflow state

    def save_workflow_state(self, task_id: str, checkpoint: str, state_json: str) -> None:
        self.execute(
            """INSERT INTO workflow_states (id, task_id, checkpoint, state_json, created_at)
               VALUES (?,?,?,?,?)""",
            (f"ws-{int(time.time() * 1e6) % 10**12}", task_id, checkpoint,
             state_json[:200000], time.time()),
        )

    # -------------------------------------------------------- action audit

    def record_action(self, task_id: str, action: str, payload: dict, outcome: str) -> None:
        self.execute(
            "INSERT INTO action_ledger (id, task_id, action, payload, outcome, created_at) VALUES (?,?,?,?,?,?)",
            (f"al-{int(time.time() * 1e6) % 10**12}", task_id, action,
             json.dumps(payload, default=str)[:4000], outcome, time.time()),
        )

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM action_ledger ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    # --------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        def scalar(sql: str, params: tuple = ()) -> int:
            rows = self.query(sql, params)
            return int(list(rows[0].values())[0]) if rows else 0

        return {
            "tasks": scalar("SELECT COUNT(*) FROM tasks"),
            "completed": scalar("SELECT COUNT(*) FROM tasks WHERE status='completed'"),
            "awaiting": scalar("SELECT COUNT(*) FROM tasks WHERE status='awaiting_approval'"),
            "rejected": scalar("SELECT COUNT(*) FROM tasks WHERE status='rejected'"),
            "agent_runs": scalar("SELECT COUNT(*) FROM agent_runs"),
            "tool_calls": scalar("SELECT COUNT(*) FROM tool_calls"),
            "actions": scalar("SELECT COUNT(*) FROM action_ledger"),
        }


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
