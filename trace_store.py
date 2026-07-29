from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3

from brain import SignalResult


class TraceStore:
    """Coreの活動Traceと外部入力ログを分離して保存する。"""

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
            CREATE TABLE IF NOT EXISTS external_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                external_text TEXT,
                encoder_name TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_id INTEGER,
                parent_trace_id INTEGER,
                mode TEXT NOT NULL,
                source_nodes TEXT NOT NULL,
                activated_nodes TEXT NOT NULL,
                traversed_edges TEXT NOT NULL,
                activation_history TEXT NOT NULL,
                final_activation TEXT NOT NULL,
                learn_enabled INTEGER NOT NULL,
                FOREIGN KEY(event_id) REFERENCES external_events(id),
                FOREIGN KEY(parent_trace_id) REFERENCES traces(id)
            );
            """)

    def add_external_event(
        self,
        kind: str,
        external_text: str,
        encoder_name: str,
        importance: float,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO external_events
                (created_at, kind, external_text, encoder_name, importance)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    kind,
                    external_text,
                    encoder_name,
                    importance,
                ),
            )
            return int(cursor.lastrowid)

    def add_trace(
        self,
        result: SignalResult,
        mode: str,
        learn_enabled: bool,
        event_id: int | None = None,
        parent_trace_id: int | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO traces
                (created_at, event_id, parent_trace_id, mode, source_nodes,
                 activated_nodes, traversed_edges, activation_history,
                 final_activation, learn_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    event_id,
                    parent_trace_id,
                    mode,
                    json.dumps(result.source_nodes),
                    json.dumps(result.activated_nodes),
                    json.dumps(result.traversed_edges),
                    json.dumps(result.activation_history),
                    json.dumps(result.final_activation.tolist()),
                    int(learn_enabled),
                ),
            )
            return int(cursor.lastrowid)

    def get_trace(self, trace_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        return None if row is None else self._decode_trace(row)

    def latest_trace(self, mode: str | None = None) -> dict | None:
        sql = "SELECT * FROM traces"
        params: tuple = ()
        if mode is not None:
            sql += " WHERE mode=?"
            params = (mode,)
        sql += " ORDER BY id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return None if row is None else self._decode_trace(row)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.*, e.kind, e.external_text, e.importance
                FROM traces t
                LEFT JOIN external_events e ON e.id=t.event_id
                ORDER BY t.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._decode_trace(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0])

    def recent_context_nodes(self, trace_limit: int = 7, node_limit: int = 18) -> list[int]:
        traces = self.recent(trace_limit)
        scores: dict[int, int] = {}
        for trace in traces:
            for node in trace["activated_nodes"]:
                scores[int(node)] = scores.get(int(node), 0) + 1
        return sorted(scores, key=scores.get, reverse=True)[:node_limit]

    @staticmethod
    def _decode_trace(row: sqlite3.Row) -> dict:
        item = dict(row)
        for key in (
            "source_nodes",
            "activated_nodes",
            "traversed_edges",
            "activation_history",
            "final_activation",
        ):
            item[key] = json.loads(item[key])
        item["input_text"] = item.get("external_text") or ""
        item["kind"] = item.get("kind") or item.get("mode", "internal")
        return item
