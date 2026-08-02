from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3

from semantic_encoder_v2 import DB_FILE
from semantic_encoder_v2_contextual import observe_contextual


@dataclass(frozen=True)
class StoredExperience:
    subject: str
    relation: str
    content: str
    subject_nodes: set[int]
    relation_nodes: set[int]
    content_nodes: set[int]
    subject_edges: set[tuple[int, int]]
    relation_edges: set[tuple[int, int]]
    content_edges: set[tuple[int, int]]
    repetitions: int

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"

    @property
    def all_nodes(self) -> set[int]:
        return self.subject_nodes | self.relation_nodes | self.content_nodes

    @property
    def all_edges(self) -> set[tuple[int, int]]:
        return self.subject_edges | self.relation_edges | self.content_edges


def _nodes(value: str) -> set[int]:
    return {int(v) for v in json.loads(value)}


def _edges(value: str) -> set[tuple[int, int]]:
    return {tuple(sorted((int(v[0]), int(v[1])))) for v in json.loads(value)}


def load_stored_experiences() -> list[StoredExperience]:
    if not DB_FILE.exists():
        raise ValueError("semantic_v2.db が見つかりません。先にSemantic Encoderで経験を保存してください。")

    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT * FROM semantic_experiences ORDER BY id").fetchall())

    if not rows:
        raise ValueError("保存済み経験がありません。")

    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (str(row["subject"]), str(row["relation"]), str(row["content"]))
        bucket = grouped.setdefault(key, {
            "subject_nodes": set(), "relation_nodes": set(), "content_nodes": set(),
            "subject_edges": set(), "relation_edges": set(), "content_edges": set(),
            "repetitions": 0,
        })
        bucket["subject_nodes"].update(_nodes(row["subject_nodes"]))
        bucket["relation_nodes"].update(_nodes(row["relation_nodes"]))
        bucket["content_nodes"].update(_nodes(row["content_nodes"]))
        bucket["subject_edges"].update(_edges(row["subject_edges"]))
        bucket["relation_edges"].update(_edges(row["relation_edges"]))
        bucket["content_edges"].update(_edges(row["content_edges"]))
        bucket["repetitions"] += 1

    return [
        StoredExperience(subject, relation, content, repetitions=data["repetitions"],
            subject_nodes=data["subject_nodes"], relation_nodes=data["relation_nodes"], content_nodes=data["content_nodes"],
            subject_edges=data["subject_edges"], relation_edges=data["relation_edges"], content_edges=data["content_edges"])
        for (subject, relation, content), data in grouped.items()
    ]


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _containment(probe: set, stored: set) -> float:
    return len(probe & stored) / len(probe) if probe else 0.0


def _stage_signature(experience, stage: str) -> tuple[set[int], set[tuple[int, int]]]:
    result = {
        "subject": experience.subject_result,
        "relation": experience.relation_result,
        "content": experience.content_result,
    }[stage]
    return set(int(v) for v in result.activated_nodes), {
        tuple(sorted((int(edge[0]), int(edge[1])))) for edge in result.traversed_edges
    }


def _stored_signature(item: StoredExperience, stage: str) -> tuple[set[int], set[tuple[int, int]]]:
    if stage == "subject":
        return item.subject_nodes, item.subject_edges
    if stage == "relation":
        return item.relation_nodes, item.relation_edges
    if stage == "content":
        return item.content_nodes, item.content_edges
    return item.all_nodes, item.all_edges


def _score(probe_nodes: set[int], probe_edges: set[tuple[int, int]], stored_nodes: set[int], stored_edges: set[tuple[int, int]]) -> dict:
    node_jaccard = _jaccard(probe_nodes, stored_nodes)
    edge_jaccard = _jaccard(probe_edges, stored_edges)
    node_containment = _containment(probe_nodes, stored_nodes)
    edge_containment = _containment(probe_edges, stored_edges)
    score = 0.22 * node_jaccard + 0.18 * node_containment + 0.35 * edge_jaccard + 0.25 * edge_containment
    return {
        "score": score,
        "node_jaccard": node_jaccard,
        "edge_jaccard": edge_jaccard,
        "node_containment": node_containment,
        "edge_containment": edge_containment,
        "common_nodes": len(probe_nodes & stored_nodes),
        "common_edges": len(probe_edges & stored_edges),
        "probe_only_nodes": len(probe_nodes - stored_nodes),
        "probe_only_edges": len(probe_edges - stored_edges),
    }


def observe_novel(subject: str, relation: str, content: str, *, limit: int = 10) -> dict:
    subject, relation, content = subject.strip(), relation.strip(), content.strip()
    if not subject or not relation or not content:
        raise ValueError("主体・関係・内容をすべて入力してください。")

    stored = load_stored_experiences()
    probe = observe_contextual(subject, relation, content)
    stages = ["subject", "relation", "content"]
    probe_signatures = {stage: _stage_signature(probe, stage) for stage in stages}
    probe_signatures["all"] = (
        set().union(*(probe_signatures[s][0] for s in stages)),
        set().union(*(probe_signatures[s][1] for s in stages)),
    )

    rankings: dict[str, list[dict]] = {}
    for stage in ["subject", "relation", "content", "all"]:
        probe_nodes, probe_edges = probe_signatures[stage]
        rows = []
        for item in stored:
            stored_nodes, stored_edges = _stored_signature(item, stage)
            metrics = _score(probe_nodes, probe_edges, stored_nodes, stored_edges)
            rows.append({
                "label": item.label,
                "subject": item.subject,
                "relation": item.relation,
                "content": item.content,
                "repetitions": item.repetitions,
                **metrics,
            })
        rows.sort(key=lambda row: (-row["score"], -row["common_edges"], -row["common_nodes"], row["label"]))
        rankings[stage] = rows[: max(1, int(limit))]

    stage_connections = []
    first_connection = None
    for stage in stages:
        best = rankings[stage][0]
        connected = best["common_edges"] > 0 or best["common_nodes"] >= 3
        if connected and first_connection is None:
            first_connection = stage
        stage_connections.append({"stage": stage, "connected": connected, "best": best})

    exact_exists = any(item.subject == subject and item.relation == relation and item.content == content for item in stored)
    return {
        "probe": {"subject": subject, "relation": relation, "content": content, "label": f"{subject}｜{relation}｜{content}"},
        "exact_exists": exact_exists,
        "stored_count": len(stored),
        "first_connection": first_connection,
        "stage_connections": stage_connections,
        "rankings": rankings,
        "probe_sizes": {
            stage: {"nodes": len(probe_signatures[stage][0]), "edges": len(probe_signatures[stage][1])}
            for stage in ["subject", "relation", "content", "all"]
        },
    }
