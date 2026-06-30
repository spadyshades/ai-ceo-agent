"""SQLite persistence for agent runs, plans, tool calls, recommendations, validations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from src.agent.schema import (
    AgentState,
    Plan,
    Recommendation,
    ToolCallResult,
    ValidationResult,
)
from src.config import DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    error TEXT,
    state_json TEXT
);

CREATE TABLE IF NOT EXISTS agent_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    replan_index INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_id INTEGER NOT NULL,
    tool TEXT NOT NULL,
    params_json TEXT,
    summary TEXT,
    error TEXT,
    called_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    title TEXT NOT NULL,
    priority TEXT,
    confidence REAL,
    recommendation_json TEXT NOT NULL,
    validated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    issues_json TEXT NOT NULL,
    validated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_plans_run_id ON agent_plans(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run_id ON agent_tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_recommendations_run_id ON agent_recommendations(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_validations_run_id ON agent_validations(run_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def start_run(goal: str) -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO agent_runs (id, goal, started_at, status) VALUES (?, ?, ?, ?)",
            (run_id, goal, _now_iso(), "running"),
        )
    return run_id


def record_plan(run_id: str, plan: Plan, replan_index: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO agent_plans (run_id, replan_index, plan_json, created_at)
            VALUES (?, ?, ?, ?)""",
            (run_id, replan_index, plan.model_dump_json(), _now_iso()),
        )


def record_tool_call(run_id: str, result: ToolCallResult) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO agent_tool_calls
            (run_id, step_id, tool, params_json, summary, error, called_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result.step_id,
                result.tool,
                json.dumps(result.params, default=str),
                result.summary,
                result.error or None,
                _now_iso(),
            ),
        )


def record_recommendations(
    run_id: str,
    recommendations: list[Recommendation],
    validated: bool,
) -> None:
    created_at = _now_iso()
    with get_connection() as conn:
        for rec in recommendations:
            conn.execute(
                """INSERT INTO agent_recommendations
                (run_id, title, priority, confidence, recommendation_json, validated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    rec.title,
                    rec.priority,
                    rec.confidence,
                    rec.model_dump_json(),
                    int(validated),
                    created_at,
                ),
            )


def record_validation(run_id: str, validation: ValidationResult) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO agent_validations
            (run_id, passed, issues_json, validated_at)
            VALUES (?, ?, ?, ?)""",
            (
                run_id,
                int(validation.passed),
                json.dumps(validation.issues),
                _now_iso(),
            ),
        )


def finish_run(
    run_id: str,
    status: str,
    state: AgentState,
    error: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE agent_runs
            SET finished_at = ?, status = ?, error = ?, state_json = ?
            WHERE id = ?""",
            (_now_iso(), status, error, state.model_dump_json(), run_id),
        )


def get_latest_run() -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_run(run_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, goal, started_at, finished_at, status
            FROM agent_runs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
