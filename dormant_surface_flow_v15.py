from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dormant_surface_flow_v14b import ActivitySnapshot, ExperienceClusterTransitionBrain
from dormant_surface_flow_v14 import Edge


@dataclass(frozen=True)
class ExperienceIdentity:
    task_id: int
    task_name: str
    x: float


class MultiExperienceTransitionBrain(ExperienceClusterTransitionBrain):
    """v15a observation-only brain for multiple contextual experiences.

    A context cue is treated as part of sensory input, never as a teacher wake
    signal.  The class only labels recovery observations so that experience
    specialization and cross-experience transitions can be measured.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_experience_identity: ExperienceIdentity | None = None
        self.snapshot_identity: dict[tuple[int, int, float], ExperienceIdentity] = {}

    def set_multi_experience(
        self,
        region: int | None,
        task_id: int | None,
        task_name: str | None,
        x: float | None,
    ) -> None:
        if region is None or task_id is None or task_name is None or x is None:
            self.current_experience_identity = None
            self.set_experience(None, None)
            return
        identity = ExperienceIdentity(int(task_id), str(task_name), float(x))
        self.current_experience_identity = identity
        # Keep each task/x combination distinct without changing region semantics.
        composite_key = float(task_id) * 10.0 + float(x)
        self.set_experience(int(region), composite_key)

    def _record_activity_snapshot(self, peak_contributions: np.ndarray) -> None:
        before = len(self.activity_snapshots)
        super()._record_activity_snapshot(peak_contributions)
        if len(self.activity_snapshots) == before:
            return
        identity = self.current_experience_identity
        if identity is None:
            return
        snapshot = self.activity_snapshots[-1]
        key = (snapshot.epoch, snapshot.region, snapshot.input_key)
        self.snapshot_identity[key] = identity

    def identity_for_snapshot(self, snapshot: ActivitySnapshot) -> ExperienceIdentity | None:
        return self.snapshot_identity.get(
            (snapshot.epoch, snapshot.region, snapshot.input_key)
        )

    def cluster_experience_profiles(self) -> list[dict[str, object]]:
        clusters = self.derive_experience_clusters()
        rows: list[dict[str, object]] = []
        for cluster_id, cluster in enumerate(clusters):
            task_strength: dict[str, float] = {}
            task_events: dict[str, int] = {}
            x_strength: dict[str, list[tuple[float, float]]] = {}
            for snapshot in self.activity_snapshots:
                identity = self.identity_for_snapshot(snapshot)
                if identity is None:
                    continue
                strength = float(
                    sum(snapshot.contributions.get(edge, 0.0) for edge in cluster)
                )
                if strength <= 0.0:
                    continue
                task_strength[identity.task_name] = (
                    task_strength.get(identity.task_name, 0.0) + strength
                )
                task_events[identity.task_name] = task_events.get(identity.task_name, 0) + 1
                x_strength.setdefault(identity.task_name, []).append((identity.x, strength))

            total_strength = sum(task_strength.values())
            shares = {
                name: (value / total_strength if total_strength > 1e-12 else 0.0)
                for name, value in task_strength.items()
            }
            ordered = sorted(shares.items(), key=lambda item: item[1], reverse=True)
            dominant_name = ordered[0][0] if ordered else "none"
            dominant_share = ordered[0][1] if ordered else 0.0
            entropy = 0.0
            for share in shares.values():
                if share > 1e-12:
                    entropy -= share * float(np.log(share))
            normalized_entropy = (
                entropy / float(np.log(len(shares))) if len(shares) > 1 else 0.0
            )
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "edges": cluster,
                    "size": len(cluster),
                    "task_strength": task_strength,
                    "task_events": task_events,
                    "task_shares": shares,
                    "dominant_task": dominant_name,
                    "dominant_share": dominant_share,
                    "specialization": 1.0 - normalized_entropy,
                    "x_strength": x_strength,
                }
            )
        return rows

    def cross_experience_transition_rows(self) -> list[dict[str, object]]:
        clusters = self.derive_experience_clusters()
        if len(clusters) < 2:
            return []

        assigned: list[tuple[ActivitySnapshot, int, float, ExperienceIdentity]] = []
        for snapshot in self.activity_snapshots:
            identity = self.identity_for_snapshot(snapshot)
            if identity is None:
                continue
            scores = self._snapshot_cluster_scores(snapshot, clusters)
            if not scores or max(scores) <= 0.0:
                continue
            cluster_id = int(np.argmax(scores))
            assigned.append((snapshot, cluster_id, float(scores[cluster_id]), identity))

        event_counts: dict[tuple[int, int, str, str], int] = {}
        input_pairs: dict[tuple[int, int, str, str], set[tuple[float, float]]] = {}
        epochs: dict[tuple[int, int, str, str], set[int]] = {}
        strengths: dict[tuple[int, int, str, str], list[float]] = {}
        outgoing: dict[tuple[int, str], int] = {}
        target_total: dict[tuple[int, str], int] = {}

        for _, cluster_id, _, identity in assigned:
            target_key = (cluster_id, identity.task_name)
            target_total[target_key] = target_total.get(target_key, 0) + 1

        for previous, current in zip(assigned, assigned[1:]):
            prev_snapshot, source_id, _, prev_identity = previous
            curr_snapshot, target_id, target_strength, curr_identity = current
            if prev_snapshot.epoch != curr_snapshot.epoch:
                continue
            source_key = (source_id, prev_identity.task_name)
            outgoing[source_key] = outgoing.get(source_key, 0) + 1
            if prev_identity.task_name == curr_identity.task_name:
                continue
            if source_id == target_id:
                continue
            key = (
                source_id,
                target_id,
                prev_identity.task_name,
                curr_identity.task_name,
            )
            event_counts[key] = event_counts.get(key, 0) + 1
            input_pairs.setdefault(key, set()).add((prev_identity.x, curr_identity.x))
            epochs.setdefault(key, set()).add(curr_snapshot.epoch)
            strengths.setdefault(key, []).append(target_strength)

        total = max(len(assigned), 1)
        rows: list[dict[str, object]] = []
        for key, events in event_counts.items():
            source_id, target_id, source_task, target_task = key
            conditional = events / max(outgoing.get((source_id, source_task), 0), 1)
            baseline = target_total.get((target_id, target_task), 0) / total
            lift = conditional / baseline if baseline > 1e-12 else 0.0
            distinct_pairs = len(input_pairs.get(key, set()))
            distinct_epochs = len(epochs.get(key, set()))
            candidate = int(
                distinct_pairs >= self.transition_min_distinct_input_pairs
                and distinct_epochs >= 2
                and lift > 1.0
            )
            rows.append(
                {
                    "source_cluster": source_id,
                    "target_cluster": target_id,
                    "source_task": source_task,
                    "target_task": target_task,
                    "events": events,
                    "distinct_input_pairs": distinct_pairs,
                    "distinct_epochs": distinct_epochs,
                    "conditional_probability": conditional,
                    "baseline_probability": baseline,
                    "transition_lift": lift,
                    "target_strength_mean": float(np.mean(strengths.get(key, [0.0]))),
                    "candidate": candidate,
                }
            )
        rows.sort(
            key=lambda row: (
                int(row["candidate"]),
                float(row["transition_lift"]),
                int(row["distinct_input_pairs"]),
                int(row["events"]),
            ),
            reverse=True,
        )
        return rows

    def multi_experience_stats(self) -> dict[str, float]:
        profiles = self.cluster_experience_profiles()
        transitions = self.cross_experience_transition_rows()
        candidates = [row for row in transitions if int(row["candidate"]) == 1]
        specialized = [row for row in profiles if float(row["dominant_share"]) >= 0.60]
        return {
            "experience_clusters": float(len(profiles)),
            "specialized_clusters": float(len(specialized)),
            "mean_dominant_task_share": float(
                np.mean([float(row["dominant_share"]) for row in profiles])
            ) if profiles else 0.0,
            "mean_specialization": float(
                np.mean([float(row["specialization"]) for row in profiles])
            ) if profiles else 0.0,
            "cross_experience_transitions": float(len(transitions)),
            "cross_experience_bridge_candidates": float(len(candidates)),
            "candidate_mean_lift": float(
                np.mean([float(row["transition_lift"]) for row in candidates])
            ) if candidates else 0.0,
        }
