"""memory.py – SQLite-backed conversation memory."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name   TEXT NOT NULL,
            department      TEXT,
            issue           TEXT NOT NULL,
            response        TEXT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_conversation(name: str, department: str, issue: str, response: str = ""):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO conversations (customer_name, department, issue, response) VALUES (?, ?, ?, ?)",
        (name, department, issue, response),
    )
    conn.commit()
    conn.close()


def get_conversation_history(name: str, limit: int = 5) -> list[dict]:
    """Return the last `limit` conversations for a customer."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT department, issue, response, timestamp
        FROM conversations
        WHERE customer_name = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (name, limit),
    ).fetchall()
    conn.close()
    return [
        {"department": r[0], "issue": r[1], "response": r[2], "timestamp": r[3]}
        for r in rows
    ]


def get_last_issue(name: str) -> str:
    history = get_conversation_history(name, limit=1)
    if history:
        h = history[0]
        return (
            f"Your most recent support issue (logged {h['timestamp']}):\n"
            f"Department: {h['department']}\n"
            f"Issue: {h['issue']}\n"
            f"Response given: {h['response']}"
        )
    return "No previous support interactions found for your account."
