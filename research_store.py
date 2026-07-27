from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import hashlib
import json
import sqlite3
import uuid

SCHEMA_VERSION = "1.0.0"
ENGINE_VERSION = "0.4.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialHandle:
    trial_id: str
    session_id: str
    sequence_no: int


class ResearchStore:
    """Append-oriented research log for Sphere Brain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    hypothesis TEXT,
                    protocol_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    engine_version TEXT NOT NULL,
                    structure_version TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    random_seed INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS inputs (
                    input_id TEXT PRIMARY KEY,
                    input_type TEXT NOT NULL,
                    raw_value TEXT NOT NULL,
                    encoding TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_inputs_hash ON inputs(content_hash);
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    sequence_no INTEGER NOT NULL,
                    input_id TEXT REFERENCES inputs(input_id),
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    source_nodes_json TEXT NOT NULL,
                    final_nodes_json TEXT,
                    error_text TEXT,
                    UNIQUE(session_id, sequence_no)
                );
                CREATE TABLE IF NOT EXISTS path_steps (
                    path_step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    step_no INTEGER NOT NULL,
                    from_node INTEGER NOT NULL,
                    to_node INTEGER NOT NULL,
                    activation REAL,
                    weight_before REAL,
                    weight_after REAL,
                    selected_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(trial_id, step_no, from_node, to_node)
                );
                CREATE INDEX IF NOT EXISTS idx_path_trial ON path_steps(trial_id, step_no);
                CREATE INDEX IF NOT EXISTS idx_path_edge ON path_steps(from_node, to_node);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    step_no INTEGER NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outputs (
                    output_id TEXT PRIMARY KEY,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    output_type TEXT NOT NULL,
                    raw_value_json TEXT NOT NULL,
                    decoder_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    calculation_version TEXT NOT NULL,
                    calculated_at TEXT NOT NULL,
                    parameters_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_metric_trial ON metrics(trial_id);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_versions(version, applied_at) VALUES(?, ?)",
                (SCHEMA_VERSION, utc_now()),
            )

    def create_experiment(self, name: str, purpose: str, hypothesis: str = "",
                          protocol_version: str = "1.0", metadata: dict | None = None) -> str:
        experiment_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments VALUES(?,?,?,?,?,?,?)",
                (experiment_id, name, purpose, hypothesis, protocol_version, utc_now(), canonical_json(metadata or {})),
            )
        return experiment_id

    def start_session(self, experiment_id: str, structure_version: str, config_version: str,
                      random_seed: int | None = None, metadata: dict | None = None) -> str:
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
                (session_id, experiment_id, utc_now(), None, ENGINE_VERSION, structure_version,
                 config_version, random_seed, canonical_json(metadata or {})),
            )
        return session_id

    def finish_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET ended_at=? WHERE session_id=? AND ended_at IS NULL", (utc_now(), session_id))

    def add_input(self, raw_value: str, input_type: str = "text", encoding: str = "utf-8",
                  source: str = "manual", metadata: dict | None = None) -> str:
        payload = {"input_type": input_type, "raw_value": raw_value, "encoding": encoding}
        digest = content_hash(payload)
        input_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO inputs VALUES(?,?,?,?,?,?,?,?)",
                (input_id, input_type, raw_value, encoding, source, digest, utc_now(), canonical_json(metadata or {})),
            )
        return input_id

    def start_trial(self, session_id: str, sequence_no: int, input_id: str | None,
                    source_nodes: Iterable[int]) -> TrialHandle:
        trial_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trials VALUES(?,?,?,?,?,?,?,?,?,?)",
                (trial_id, session_id, sequence_no, input_id, utc_now(), None, "running",
                 canonical_json(list(source_nodes)), None, None),
            )
        return TrialHandle(trial_id, session_id, sequence_no)

    def add_path_step(self, trial_id: str, step_no: int, from_node: int, to_node: int,
                      activation: float | None = None, weight_before: float | None = None,
                      weight_after: float | None = None, selected_by: str = "propagation",
                      metadata: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO path_steps
                (trial_id,step_no,from_node,to_node,activation,weight_before,weight_after,selected_by,created_at,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (trial_id, step_no, from_node, to_node, activation, weight_before, weight_after,
                 selected_by, utc_now(), canonical_json(metadata or {})),
            )

    def add_snapshot(self, trial_id: str, step_no: int, snapshot_type: str, state: object) -> str:
        snapshot_id = str(uuid.uuid4())
        state_json = canonical_json(state)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)",
                (snapshot_id, trial_id, step_no, snapshot_type, state_json,
                 hashlib.sha256(state_json.encode("utf-8")).hexdigest(), utc_now()),
            )
        return snapshot_id

    def finish_trial(self, trial_id: str, final_nodes: Iterable[int], status: str = "completed",
                     error_text: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trials SET ended_at=?,status=?,final_nodes_json=?,error_text=? WHERE trial_id=?",
                (utc_now(), status, canonical_json(list(final_nodes)), error_text, trial_id),
            )

    def add_output(self, trial_id: str, value: object, output_type: str = "nodes",
                   decoder_version: str = "raw-1.0", metadata: dict | None = None) -> str:
        output_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO outputs VALUES(?,?,?,?,?,?,?)",
                (output_id, trial_id, output_type, canonical_json(value), decoder_version,
                 utc_now(), canonical_json(metadata or {})),
            )
        return output_id

    def add_metric(self, trial_id: str, name: str, value: float,
                   calculation_version: str = "1.0", parameters: dict | None = None) -> str:
        metric_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics VALUES(?,?,?,?,?,?,?)",
                (metric_id, trial_id, name, float(value), calculation_version,
                 utc_now(), canonical_json(parameters or {})),
            )
        return metric_id

    def trial_path(self, trial_id: str) -> list[tuple[int, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT from_node, to_node FROM path_steps
                   WHERE trial_id=? ORDER BY step_no, path_step_id""",
                (trial_id,),
            ).fetchall()
        return [(int(row["from_node"]), int(row["to_node"])) for row in rows]

    def repeated_input_comparison(self, raw_value: str, source: str = "input", limit: int = 10) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.trial_id, t.started_at
                   FROM trials t JOIN inputs i ON i.input_id=t.input_id
                   WHERE i.raw_value=? AND i.source=? AND t.status='completed'
                   ORDER BY t.started_at DESC LIMIT ?""",
                (raw_value, source, max(2, int(limit))),
            ).fetchall()

        trials = [{"trial_id": str(row["trial_id"]), "started_at": str(row["started_at"])} for row in reversed(rows)]
        for trial in trials:
            trial["path"] = self.trial_path(trial["trial_id"])
            trial["unique_edges"] = sorted(set(trial["path"]))

        comparisons = []
        for previous, current in zip(trials, trials[1:]):
            prev_edges = set(previous["unique_edges"])
            curr_edges = set(current["unique_edges"])
            union = prev_edges | curr_edges
            shared = prev_edges & curr_edges
            new = curr_edges - prev_edges
            lost = prev_edges - curr_edges
            jaccard = len(shared) / len(union) if union else 1.0

            prefix = 0
            for old_step, new_step in zip(previous["path"], current["path"]):
                if old_step != new_step:
                    break
                prefix += 1
            prefix_base = max(1, min(len(previous["path"]), len(current["path"])))

            comparisons.append({
                "from_trial_id": previous["trial_id"],
                "to_trial_id": current["trial_id"],
                "shared_edges": len(shared),
                "new_edges": len(new),
                "lost_edges": len(lost),
                "shared_edge_list": sorted(shared),
                "new_edge_list": sorted(new),
                "lost_edge_list": sorted(lost),
                "edge_similarity": jaccard,
                "matching_prefix_steps": prefix,
                "ordered_similarity": prefix / prefix_base,
            })

        return {
            "input": raw_value,
            "source": source,
            "trial_count": len(trials),
            "trials": trials,
            "comparisons": comparisons,
            "latest": comparisons[-1] if comparisons else None,
        }

    def summary(self) -> dict[str, int]:
        tables = ("experiments", "sessions", "inputs", "trials", "path_steps", "snapshots", "outputs", "metrics")
        with self._connect() as conn:
            return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
