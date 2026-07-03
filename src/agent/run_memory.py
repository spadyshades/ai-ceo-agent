"""Cross-run memory: provides context from the most recent previous run."""

from __future__ import annotations

import json
import sqlite3

from src.config import DB_PATH


def get_previous_run_summary(exclude_run_id: str = "") -> str:
    """Return a brief summary of the most recent completed agent run."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = (
        "SELECT id, goal, state_json FROM agent_runs "
        "WHERE status = 'success' AND id != ? "
        "ORDER BY started_at DESC LIMIT 1"
    )
    row = conn.execute(query, (exclude_run_id,)).fetchone()
    conn.close()

    if not row or not row["state_json"]:
        return ""

    try:
        state = json.loads(row["state_json"])
    except json.JSONDecodeError:
        return ""

    recs = state.get("recommendations", [])
    if not recs:
        return ""

    lines = [f"Previous analysis goal: {row['goal'][:120]}"]
    lines.append(f"Previous recommendations ({len(recs)}):")
    for i, rec in enumerate(recs[:5], 1):
        priority = rec.get("priority", "Medium")
        title = rec.get("title", "")
        lines.append(f"  {i}. [{priority}] {title}")

    briefing = state.get("briefing", "")
    if briefing:
        lines.append(f"Previous briefing excerpt: {briefing[:200]}")

    return "\n".join(lines)
