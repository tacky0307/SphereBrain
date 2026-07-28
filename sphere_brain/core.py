"""SQLite persistence used by the Sphere Brain learning process.

The rows describing a run's configuration and topology are immutable.  A new
session may therefore safely refer to rows created by an earlier process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SetupResult:
    """Identifiers selected (or created) for one process session."""

    project_id: int
    experiment_id: int
    configuration_id: int
    structure_id: int
    session_id: int


class Store:
    """An SQLite store whose :meth:`setup` operation is restart-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _id_for(
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        lookup: Mapping[str, object],
        create: Mapping[str, object],
    ) -> tuple[int, bool]:
        """Return a matching id, atomically creating the row if necessary."""
        where = " AND ".join(f"{column} = ?" for column in lookup)
        row = connection.execute(
            f"SELECT {id_column} FROM {table} WHERE {where}", tuple(lookup.values())
        ).fetchone()
        if row is not None:
            return int(row[id_column]), False

        columns = tuple(create)
        placeholders = ", ".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(create.values()),
        )
        return int(cursor.lastrowid), True

    def setup(
        self,
        *,
        config_version: str,
        configuration: object,
        structure_version: str,
        nodes: Iterable[Mapping[str, object] | Sequence[object]] = (),
        edges: Iterable[Mapping[str, object] | Sequence[object]] = (),
        project_name: str = "Sphere Brain",
        experiment_name: str = "continuous-learning",
    ) -> SetupResult:
        """Prepare a new session without discarding previously learned data.

        Projects and experiments use their stable names as natural keys.
        Configurations are identified by version and a hash of their canonical
        JSON, while structures are identified by version.  Nodes and edges are
        inserted only along with a genuinely new structure.
        """
        configuration_json = _json(configuration)
        configuration_hash = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()
        now = _utc_now()

        with self._connect() as connection:
            self._create_schema(connection)
            project_id, _ = self._id_for(
                connection,
                "projects",
                "project_id",
                {"name": project_name},
                {"name": project_name, "created_at": now},
            )
            experiment_id, _ = self._id_for(
                connection,
                "experiments",
                "experiment_id",
                {"project_id": project_id, "name": experiment_name},
                {"project_id": project_id, "name": experiment_name, "created_at": now},
            )
            configuration_id, _ = self._id_for(
                connection,
                "configurations",
                "configuration_id",
                {"config_version": config_version, "content_hash": configuration_hash},
                {
                    "config_version": config_version,
                    "content_hash": configuration_hash,
                    "content_json": configuration_json,
                    "created_at": now,
                },
            )
            structure_id, structure_created = self._id_for(
                connection,
                "structures",
                "structure_id",
                {"structure_version": structure_version},
                {"structure_version": structure_version, "created_at": now},
            )

            if structure_created:
                self._insert_nodes(connection, structure_id, nodes)
                self._insert_edges(connection, structure_id, edges)

            cursor = connection.execute(
                """INSERT INTO sessions
                   (project_id, experiment_id, configuration_id, structure_id, started_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (project_id, experiment_id, configuration_id, structure_id, now),
            )
            session_id = int(cursor.lastrowid)

        return SetupResult(
            project_id, experiment_id, configuration_id, structure_id, session_id
        )

    @staticmethod
    def _insert_nodes(
        connection: sqlite3.Connection,
        structure_id: int,
        nodes: Iterable[Mapping[str, object] | Sequence[object]],
    ) -> None:
        for position, node in enumerate(nodes):
            if isinstance(node, Mapping):
                node_key = str(node.get("node_key", node.get("id", position)))
                content = _json(node)
            else:
                values = tuple(node)
                node_key = str(values[0] if values else position)
                content = _json(values)
            connection.execute(
                "INSERT INTO nodes (structure_id, node_key, content_json) VALUES (?, ?, ?)",
                (structure_id, node_key, content),
            )

    @staticmethod
    def _insert_edges(
        connection: sqlite3.Connection,
        structure_id: int,
        edges: Iterable[Mapping[str, object] | Sequence[object]],
    ) -> None:
        for edge in edges:
            if isinstance(edge, Mapping):
                source = edge.get("source", edge.get("from_node"))
                target = edge.get("target", edge.get("to_node"))
                content = _json(edge)
            else:
                values = tuple(edge)
                if len(values) < 2:
                    raise ValueError("an edge must contain a source and target")
                source, target = values[:2]
                content = _json(values)
            if source is None or target is None:
                raise ValueError("an edge must contain a source and target")
            connection.execute(
                """INSERT INTO edges
                   (structure_id, source_node_key, target_node_key, content_json)
                   VALUES (?, ?, ?, ?)""",
                (structure_id, str(source), str(target), content),
            )

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, name)
            );
            CREATE TABLE IF NOT EXISTS configurations (
                configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(config_version, content_hash)
            );
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_version TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_id INTEGER NOT NULL REFERENCES structures(structure_id),
                node_key TEXT NOT NULL,
                content_json TEXT NOT NULL,
                UNIQUE(structure_id, node_key)
            );
            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_id INTEGER NOT NULL REFERENCES structures(structure_id),
                source_node_key TEXT NOT NULL,
                target_node_key TEXT NOT NULL,
                content_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id),
                configuration_id INTEGER NOT NULL REFERENCES configurations(configuration_id),
                structure_id INTEGER NOT NULL REFERENCES structures(structure_id),
                started_at TEXT NOT NULL
            );
            """
        )
