from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable
import json
import sqlite3

import numpy as np

from semantic_encoder_v2 import (
    DB_FILE,
    _context_tail,
    component_nodes,
    initialize_db,
    jaccard,
    load_brain,
)


@dataclass
class RawOutput:
    subject: str
    relation: str
    source_nodes: list[int]
    final_node: int | None
    final_value: float
    active_nodes: list[tuple[int, float]]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    convergence_step: int
    stopped_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _top_active(activation: np.ndarray, limit: int = 16) -> list[tuple[int, float]]:
    indexes = np.flatnonzero(activation > 0)
    ranked = sorted(
        ((int(index), float(activation[index])) for index in indexes),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: max(1, int(limit))]


def _continue_focused(
    brain,
    initial_activation: np.ndarray,
    *,
    steps: int = 24,
    threshold: float = 0.18,
) -> tuple[np.ndarray, list[list[int]], list[tuple[int, int]], str]:
    """Continue Core activity from a prior numeric state without learning.

    This intentionally mirrors the current focused propagation rule, but starts
    from relation_result.final_activation instead of injecting a new semantic cue.
    """
    activation = np.asarray(initial_activation, dtype=float).copy()
    history: list[list[int]] = [np.flatnonzero(activation > 0).astype(int).tolist()]
    traversed_edges: set[tuple[int, int]] = set()
    stopped_reason = "step_limit"

    for _ in range(max(1, int(steps))):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            stopped_reason = "no_active_nodes"
            break

        candidates: dict[int, tuple[float, int]] = {}
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue

            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for local_index in best_indices:
                target = int(neighbors[local_index])
                value = float(scores[local_index]) * brain.signal_decay
                if value < threshold:
                    continue
                previous = candidates.get(target)
                if previous is None or value > previous[0]:
                    candidates[target] = (value, int(source))

        if not candidates:
            stopped_reason = "below_threshold"
            break

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        selected = ranked[: min(brain.max_active_per_step, len(ranked))]
        next_activation = np.zeros(brain.node_count, dtype=float)

        for target, (value, source) in selected:
            value = float(np.clip(value, 0.0, 1.0))
            if value < threshold:
                continue
            next_activation[target] = max(next_activation[target], value)
            traversed_edges.add(tuple(sorted((source, target))))

        active_now = np.flatnonzero(next_activation > 0).astype(int).tolist()
        if not active_now:
            stopped_reason = "no_next_activation"
            break

        history.append(active_now)
        activation = next_activation
    else:
        stopped_reason = "step_limit"

    return activation, history, sorted(traversed_edges), stopped_reason


def _ensure_output_table() -> None:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_raw_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                raw_output TEXT NOT NULL,
                decoder_candidates TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_output_input "
            "ON semantic_raw_outputs(subject, relation)"
        )


def _known_content_rows(relation: str) -> list[sqlite3.Row]:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        return list(
            conn.execute(
                "SELECT content, content_nodes, content_edges "
                "FROM semantic_experiences WHERE relation=? ORDER BY id",
                (relation,),
            ).fetchall()
        )


def decode_candidates(raw: RawOutput, limit: int = 8) -> list[dict]:
    """Observer-side translation candidates; never fed back into Core."""
    rows = _known_content_rows(raw.relation)
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["content"]), []).append(row)

    raw_nodes = {node for node, _ in raw.active_nodes}
    raw_edges = set(raw.traversed_edges)
    results: list[dict] = []

    for content, items in grouped.items():
        scores: list[float] = []
        for row in items:
            stored_nodes = {int(v) for v in json.loads(row["content_nodes"])}
            stored_edges = {
                tuple(int(x) for x in edge)
                for edge in json.loads(row["content_edges"])
            }
            node_score = jaccard(raw_nodes, stored_nodes)
            edge_score = jaccard(raw_edges, stored_edges)
            scores.append(0.45 * node_score + 0.55 * edge_score)
        results.append(
            {
                "content": content,
                "score": max(scores, default=0.0),
                "experiences": len(items),
            }
        )

    results.sort(key=lambda item: (-item["score"], -item["experiences"], item["content"]))
    return results[: max(1, int(limit))]


def observe_once(
    subject: str,
    relation: str,
    *,
    output_steps: int = 24,
    threshold: float = 0.18,
    top_nodes: int = 16,
    save: bool = True,
) -> dict:
    subject = subject.strip()
    relation = relation.strip()
    if not subject or not relation:
        raise ValueError("主体と関係を入力してください。")

    brain = load_brain()
    subject_sources = (
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", subject, 3)
    )
    subject_result = brain.propagate(
        subject_sources,
        steps=8,
        threshold=threshold,
        noise=0.0,
        learn=False,
    )

    relation_sources = (
        component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", relation, 3)
    )
    relation_result = brain.propagate(
        relation_sources,
        steps=10,
        threshold=threshold,
        noise=0.0,
        learn=False,
        context_nodes=_context_tail(subject_result),
    )

    final_activation, free_history, free_edges, stopped_reason = _continue_focused(
        brain,
        relation_result.final_activation,
        steps=output_steps,
        threshold=threshold,
    )
    active_nodes = _top_active(final_activation, top_nodes)
    final_node = active_nodes[0][0] if active_nodes else None
    final_value = active_nodes[0][1] if active_nodes else 0.0

    raw = RawOutput(
        subject=subject,
        relation=relation,
        source_nodes=relation_result.source_nodes,
        final_node=final_node,
        final_value=final_value,
        active_nodes=active_nodes,
        traversed_edges=free_edges,
        activation_history=free_history,
        convergence_step=max(0, len(free_history) - 1),
        stopped_reason=stopped_reason,
    )
    candidates = decode_candidates(raw)

    if save:
        _ensure_output_table()
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            conn.execute(
                "INSERT INTO semantic_raw_outputs "
                "(created_at, subject, relation, raw_output, decoder_candidates) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    subject,
                    relation,
                    json.dumps(raw.to_dict(), ensure_ascii=False),
                    json.dumps(candidates, ensure_ascii=False),
                ),
            )

    return {"raw_output": raw.to_dict(), "decoder_candidates": candidates}


def _weighted_signature(active_nodes: Iterable[tuple[int, float]]) -> dict[int, float]:
    return {int(node): float(value) for node, value in active_nodes}


def output_similarity(left: dict, right: dict) -> float:
    a = _weighted_signature(left["raw_output"]["active_nodes"])
    b = _weighted_signature(right["raw_output"]["active_nodes"])
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denominator = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def observe_repeated(
    subject: str,
    relation: str,
    *,
    repeats: int = 5,
    output_steps: int = 24,
    threshold: float = 0.18,
) -> dict:
    runs = [
        observe_once(
            subject,
            relation,
            output_steps=output_steps,
            threshold=threshold,
            save=True,
        )
        for _ in range(max(1, int(repeats)))
    ]

    pair_scores: list[float] = []
    for index, left in enumerate(runs):
        for right in runs[index + 1 :]:
            pair_scores.append(output_similarity(left, right))

    final_nodes = [run["raw_output"]["final_node"] for run in runs]
    counts: dict[int | None, int] = {}
    for node in final_nodes:
        counts[node] = counts.get(node, 0) + 1

    return {
        "subject": subject.strip(),
        "relation": relation.strip(),
        "runs": runs,
        "repeat_count": len(runs),
        "mean_similarity": sum(pair_scores) / len(pair_scores) if pair_scores else 1.0,
        "final_node_counts": sorted(
            ({"node": node, "count": count} for node, count in counts.items()),
            key=lambda item: (-item["count"], -1 if item["node"] is None else item["node"]),
        ),
    }
