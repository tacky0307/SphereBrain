from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import local
import json
import sqlite3


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._local = local()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Return one SQLite connection per worker thread."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.connection = conn
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
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

            CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_kind_id
                ON memories(kind, id DESC);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.commit()

    @staticmethod
    def _encode(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def add_memory(
        self,
        kind: str,
        input_text: str,
        source_nodes: list[int],
        activated_nodes: list[int],
        traversed_edges: list[tuple[int, int]],
        importance: float = 1.0,
    ) -> int:
        return self.add_memories(
            [
                {
                    "kind": kind,
                    "input_text": input_text,
                    "source_nodes": source_nodes,
                    "activated_nodes": activated_nodes,
                    "traversed_edges": traversed_edges,
                    "importance": importance,
                }
            ]
        )[0]

    def add_memories(self, items: list[dict]) -> list[int]:
        """Insert a group of memories in one transaction."""
        if not items:
            return []

        conn = self._connect()
        created_at = datetime.now().isoformat(timespec="milliseconds")
        ids: list[int] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            for item in items:
                cursor = conn.execute(
                    """
                    INSERT INTO memories
                    (created_at, kind, input_text, source_nodes, activated_nodes,
                     traversed_edges, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("created_at", created_at),
                        item["kind"],
                        item.get("input_text", ""),
                        self._encode(item["source_nodes"]),
                        self._encode(item["activated_nodes"]),
                        self._encode(item["traversed_edges"]),
                        float(item.get("importance", 1.0)),
                    ),
                )
                ids.append(int(cursor.lastrowid))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return ids

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._connect().execute(
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
        return int(
            self._connect().execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        )

    def recent_context_nodes(
        self, memory_limit: int = 5, node_limit: int = 16
    ) -> list[int]:
        memories = self.recent(memory_limit)
        scores: dict[int, int] = {}
        for memory in memories:
            for node in memory["activated_nodes"]:
                node_id = int(node)
                scores[node_id] = scores.get(node_id, 0) + 1
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:node_limit]

    def set_value(self, key: str, value: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()

    def get_value(self, key: str, default: str = "") -> str:
        row = self._connect().execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return default if row is None else str(row["value"])

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None
