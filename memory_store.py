from __future__ import annotations

from pathlib import Path
import sqlite3
import json
from datetime import datetime


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                input_text TEXT,
                source_nodes TEXT NOT NULL,
                activated_nodes TEXT NOT NULL,
                traversed_edges TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)

    def add_memory(
        self,
        kind: str,
        input_text: str,
        source_nodes: list[int],
        activated_nodes: list[int],
        traversed_edges: list[tuple[int, int]],
        importance: float = 1.0,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories
                (created_at, kind, input_text, source_nodes, activated_nodes, traversed_edges, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    kind,
                    input_text,
                    json.dumps(source_nodes),
                    json.dumps(activated_nodes),
                    json.dumps(traversed_edges),
                    importance,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["source_nodes"] = json.loads(item["source_nodes"])
            item["activated_nodes"] = json.loads(item["activated_nodes"])
            item["traversed_edges"] = json.loads(item["traversed_edges"])
            result.append(item)
        return result

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def recent_context_nodes(self, memory_limit: int = 5, node_limit: int = 16) -> list[int]:
        memories = self.recent(memory_limit)
        scores: dict[int, int] = {}
        for memory in memories:
            for node in memory["activated_nodes"]:
                scores[node] = scores.get(node, 0) + 1
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:node_limit]

    def set_value(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_value(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if row is None else str(row["value"])
