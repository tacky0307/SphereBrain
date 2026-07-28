from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from dormant_surface_flow_v14 import Edge, PromotedCooccurrenceTrackingBrain


@dataclass(frozen=True)
class ActivitySnapshot:
    epoch: int
    region: int
    input_key: float
    contributions: dict[Edge, float]


class ExperienceClusterTransitionBrain(PromotedCooccurrenceTrackingBrain):
    """Observation-only experience clustering and directed transition tracking."""

    def __init__(
        self,
        *args,
        cluster_similarity_threshold: float = 0.985,
        cluster_min_shared_inputs: int = 3,
        active_contribution_threshold: float = 1e-12,
        transition_min_distinct_input_pairs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cluster_similarity_threshold = float(cluster_similarity_threshold)
        self.cluster_min_shared_inputs = int(cluster_min_shared_inputs)
        self.active_contribution_threshold = float(active_contribution_threshold)
        self.transition_min_distinct_input_pairs = int(transition_min_distinct_input_pairs)
        self.activity_snapshots: list[ActivitySnapshot] = []
        self._snapshot_seen: set[tuple[int, int, float]] = set()

    def _record_activity_snapshot(self, peak_contributions: np.ndarray) -> None:
        if self.contribution_phase != "recovery":
            return
        region = self.current_experience_region
        input_key = self.current_experience_input_key
        if region is None or input_key is None:
            return
        dedupe_key = (self.cooccurrence_epoch, int(region), float(input_key))
        if dedupe_key in self._snapshot_seen:
            return
        contributions = {
            edge: float(peak_contributions[edge])
            for edge in self.selective_promoted_edges
            if float(peak_contributions[edge]) > self.active_contribution_threshold
        }
        self._snapshot_seen.add(dedupe_key)
        self.activity_snapshots.append(
            ActivitySnapshot(self.cooccurrence_epoch, int(region), float(input_key), contributions)
        )

    def _record_existing_promoted(self, peak_contributions: np.ndarray) -> None:
        super()._record_existing_promoted(peak_contributions)
        self._record_activity_snapshot(peak_contributions)

    def _activity_matrix(self) -> tuple[list[Edge], list[tuple[int, float]], np.ndarray]:
        edges = sorted(self.selective_promoted_edges)
        keys = sorted({(s.region, s.input_key) for s in self.activity_snapshots})
        totals = np.zeros((len(edges), len(keys)), dtype=float)
        counts = np.zeros_like(totals)
        edge_index = {edge: i for i, edge in enumerate(edges)}
        key_index = {key: i for i, key in enumerate(keys)}
        for snapshot in self.activity_snapshots:
            column = key_index[(snapshot.region, snapshot.input_key)]
            for edge, value in snapshot.contributions.items():
                if edge in edge_index:
                    row = edge_index[edge]
                    totals[row, column] += value
                    counts[row, column] += 1.0
        matrix = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0.0)
        return edges, keys, matrix

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)

    def derive_experience_clusters(self) -> list[list[Edge]]:
        edges, _, matrix = self._activity_matrix()
        adjacency: dict[Edge, set[Edge]] = defaultdict(set)
        for i, edge_a in enumerate(edges):
            active_a = matrix[i] > self.active_contribution_threshold
            for j in range(i + 1, len(edges)):
                edge_b = edges[j]
                active_b = matrix[j] > self.active_contribution_threshold
                if int(np.sum(active_a & active_b)) < self.cluster_min_shared_inputs:
                    continue
                if self._cosine(matrix[i], matrix[j]) < self.cluster_similarity_threshold:
                    continue
                adjacency[edge_a].add(edge_b)
                adjacency[edge_b].add(edge_a)
        clusters: list[list[Edge]] = []
        visited: set[Edge] = set()
        for edge in edges:
            if edge in visited:
                continue
            stack = [edge]
            cluster: list[Edge] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                stack.extend(adjacency[current] - visited)
            clusters.append(sorted(cluster))
        clusters.sort(key=lambda cluster: (-len(cluster), cluster))
        return clusters

    def cluster_profiles(self) -> list[dict[str, object]]:
        edges, keys, matrix = self._activity_matrix()
        edge_index = {edge: i for i, edge in enumerate(edges)}
        rows: list[dict[str, object]] = []
        for cluster_id, cluster in enumerate(self.derive_experience_clusters()):
            indices = [edge_index[edge] for edge in cluster]
            profile = np.mean(matrix[indices], axis=0) if indices else np.zeros(len(keys))
            dominant = sorted(zip(keys, profile), key=lambda item: float(item[1]), reverse=True)[:3]
            rows.append({
                "cluster_id": cluster_id,
                "edges": cluster,
                "size": len(cluster),
                "dominant_inputs": [
                    (int(region), float(input_key), float(value))
                    for (region, input_key), value in dominant
                    if float(value) > 0.0
                ],
            })
        return rows

    def _snapshot_cluster_scores(self, snapshot: ActivitySnapshot, clusters: list[list[Edge]]) -> list[float]:
        return [float(sum(snapshot.contributions.get(edge, 0.0) for edge in cluster)) for cluster in clusters]

    def transition_rows(self) -> list[dict[str, object]]:
        clusters = self.derive_experience_clusters()
        if len(clusters) < 2:
            return []
        assigned: list[tuple[ActivitySnapshot, int, float]] = []
        for snapshot in self.activity_snapshots:
            scores = self._snapshot_cluster_scores(snapshot, clusters)
            if scores and max(scores) > 0.0:
                cluster_id = int(np.argmax(scores))
                assigned.append((snapshot, cluster_id, float(scores[cluster_id])))
        event_counts: dict[tuple[int, int], int] = defaultdict(int)
        input_pairs: dict[tuple[int, int], set[tuple[float, float]]] = defaultdict(set)
        epochs: dict[tuple[int, int], set[int]] = defaultdict(set)
        strengths: dict[tuple[int, int], list[float]] = defaultdict(list)
        outgoing: dict[int, int] = defaultdict(int)
        target_total: dict[int, int] = defaultdict(int)
        for _, cluster_id, _ in assigned:
            target_total[cluster_id] += 1
        for previous, current in zip(assigned, assigned[1:]):
            previous_snapshot, source_id, _ = previous
            current_snapshot, target_id, target_strength = current
            if previous_snapshot.epoch != current_snapshot.epoch:
                continue
            outgoing[source_id] += 1
            if source_id == target_id:
                continue
            key = (source_id, target_id)
            event_counts[key] += 1
            input_pairs[key].add((previous_snapshot.input_key, current_snapshot.input_key))
            epochs[key].add(current_snapshot.epoch)
            strengths[key].append(target_strength)
        total = max(len(assigned), 1)
        rows: list[dict[str, object]] = []
        for (source_id, target_id), events in event_counts.items():
            conditional = events / max(outgoing[source_id], 1)
            baseline = target_total[target_id] / total
            lift = conditional / baseline if baseline > 1e-12 else 0.0
            distinct_pairs = len(input_pairs[(source_id, target_id)])
            distinct_epochs = len(epochs[(source_id, target_id)])
            rows.append({
                "source_cluster": source_id,
                "target_cluster": target_id,
                "events": events,
                "distinct_input_pairs": distinct_pairs,
                "distinct_epochs": distinct_epochs,
                "conditional_probability": conditional,
                "baseline_probability": baseline,
                "transition_lift": lift,
                "target_strength_mean": float(np.mean(strengths[(source_id, target_id)])),
                "candidate": int(distinct_pairs >= self.transition_min_distinct_input_pairs and distinct_epochs >= 2 and lift > 1.0),
            })
        rows.sort(key=lambda row: (int(row["candidate"]), float(row["transition_lift"]), int(row["distinct_input_pairs"]), int(row["events"])), reverse=True)
        return rows

    def cluster_transition_stats(self) -> dict[str, float]:
        clusters = self.derive_experience_clusters()
        transitions = self.transition_rows()
        candidates = [row for row in transitions if int(row["candidate"]) == 1]
        return {
            "experience_clusters": float(len(clusters)),
            "largest_experience_cluster": float(max((len(c) for c in clusters), default=0)),
            "singleton_clusters": float(sum(len(c) == 1 for c in clusters)),
            "directed_transitions": float(len(transitions)),
            "candidate_cluster_bridges": float(len(candidates)),
            "candidate_mean_lift": float(np.mean([float(r["transition_lift"]) for r in candidates])) if candidates else 0.0,
            "activity_snapshots": float(len(self.activity_snapshots)),
        }
