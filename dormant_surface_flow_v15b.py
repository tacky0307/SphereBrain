from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT
from dormant_surface_flow_v14 import Edge
from dormant_surface_flow_v15 import MultiExperienceTransitionBrain


@dataclass(frozen=True)
class FrozenBridge:
    source_cluster: int
    target_cluster: int
    source_task: str
    target_task: str
    strength: float
    source_edges: tuple[Edge, ...]
    target_edges: tuple[Edge, ...]


class ExperienceBridgeFacilitationBrain(MultiExperienceTransitionBrain):
    """v15b: transient, experience-derived facilitation between frozen clusters.

    Bridges never add graph edges, never change stored weights, never assist the
    teacher phase, and never boost a dormant or disabled pathway. A matching
    previous experience cluster merely gives enabled, non-dormant edges in the
    learned target cluster a tiny propagation multiplier during observation.
    """

    def __init__(
        self,
        *args,
        bridge_facilitation: float = 0.02,
        bridge_mode: str = "off",
        bridge_random_seed: int = 271828,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if bridge_facilitation < 0.0:
            raise ValueError("bridge_facilitation must be non-negative")
        if bridge_mode not in {"off", "learned", "shuffled"}:
            raise ValueError("bridge_mode must be off, learned, or shuffled")
        self.bridge_facilitation = float(bridge_facilitation)
        self.bridge_mode = bridge_mode
        self.bridge_random_seed = int(bridge_random_seed)
        self.bridge_runtime_enabled = False
        self.frozen_clusters: list[tuple[Edge, ...]] = []
        self.frozen_bridges: list[FrozenBridge] = []
        self.previous_frozen_cluster: int | None = None
        self.previous_task: str | None = None
        self.active_bridge_targets: dict[Edge, float] = {}
        self.active_bridge_count = 0
        self.bridge_application_events = 0
        self.bridge_target_edge_opportunities = 0
        self.bridge_target_edges_traversed = 0
        self.bridge_dormant_edges_skipped = 0
        self.bridge_disabled_edges_skipped = 0

    def set_bridge_runtime(self, enabled: bool) -> None:
        self.bridge_runtime_enabled = bool(enabled)

    def freeze_experience_bridges(self) -> None:
        clusters = [tuple(cluster) for cluster in self.derive_experience_clusters()]
        candidates = [
            row for row in self.cross_experience_transition_rows()
            if int(row["candidate"]) == 1
        ]
        self.frozen_clusters = clusters
        self.frozen_bridges = []
        if self.bridge_mode == "off" or not candidates or len(clusters) < 2:
            self.previous_frozen_cluster = None
            self.previous_task = None
            return

        target_map = list(range(len(clusters)))
        if self.bridge_mode == "shuffled":
            rng = random.Random(self.bridge_random_seed)
            rng.shuffle(target_map)
            if all(index == target_map[index] for index in range(len(target_map))):
                target_map = target_map[1:] + target_map[:1]

        for row in candidates:
            source_id = int(row["source_cluster"])
            learned_target_id = int(row["target_cluster"])
            target_id = learned_target_id if self.bridge_mode == "learned" else target_map[learned_target_id]
            if source_id >= len(clusters) or target_id >= len(clusters) or source_id == target_id:
                continue
            lift = max(1.0, float(row["transition_lift"]))
            strength = self.bridge_facilitation * min(1.0, (lift - 1.0) / 0.5)
            if strength <= 0.0:
                continue
            self.frozen_bridges.append(
                FrozenBridge(
                    source_cluster=source_id,
                    target_cluster=target_id,
                    source_task=str(row["source_task"]),
                    target_task=str(row["target_task"]),
                    strength=strength,
                    source_edges=clusters[source_id],
                    target_edges=clusters[target_id],
                )
            )
        self.previous_frozen_cluster = None
        self.previous_task = None

    def _assign_frozen_cluster(self, contributions: dict[Edge, float]) -> int | None:
        if not self.frozen_clusters:
            return None
        scores = [float(sum(contributions.get(edge, 0.0) for edge in cluster)) for cluster in self.frozen_clusters]
        if not scores or max(scores) <= 0.0:
            return None
        return int(np.argmax(scores))

    def _prepare_active_bridge_targets(self) -> None:
        self.active_bridge_targets = {}
        self.active_bridge_count = 0
        if not self.bridge_runtime_enabled:
            return
        identity = self.current_experience_identity
        if identity is None or self.previous_frozen_cluster is None or self.previous_task is None:
            return
        for bridge in self.frozen_bridges:
            if bridge.source_cluster != self.previous_frozen_cluster:
                continue
            if bridge.source_task != self.previous_task or bridge.target_task != identity.task_name:
                continue
            self.active_bridge_count += 1
            for edge in bridge.target_edges:
                self.active_bridge_targets[edge] = max(
                    self.active_bridge_targets.get(edge, 0.0), bridge.strength
                )

    def _effective_weight_matrix(self) -> np.ndarray:
        effective = super()._effective_weight_matrix()
        if not self.active_bridge_targets:
            return effective
        for (source, target), strength in self.active_bridge_targets.items():
            if not self.edge_enabled[source, target]:
                self.bridge_disabled_edges_skipped += 1
                continue
            if self.pathway_state[source, target] == PATHWAY_DORMANT:
                self.bridge_dormant_edges_skipped += 1
                continue
            effective[source, target] *= 1.0 + strength
            self.bridge_target_edge_opportunities += 1
        return effective

    def propagate(self, *args, **kwargs):
        self._prepare_active_bridge_targets()
        if self.active_bridge_targets:
            self.bridge_application_events += 1
        result = super().propagate(*args, **kwargs)
        if self.active_bridge_targets:
            traversed = set(result.traversed_edges)
            self.bridge_target_edges_traversed += len(traversed.intersection(self.active_bridge_targets))
        return result

    def _record_activity_snapshot(self, peak_contributions: np.ndarray) -> None:
        before = len(self.activity_snapshots)
        super()._record_activity_snapshot(peak_contributions)
        if len(self.activity_snapshots) == before:
            return
        snapshot = self.activity_snapshots[-1]
        identity = self.identity_for_snapshot(snapshot)
        cluster_id = self._assign_frozen_cluster(snapshot.contributions)
        if identity is not None and cluster_id is not None:
            self.previous_frozen_cluster = cluster_id
            self.previous_task = identity.task_name

    def bridge_stats(self) -> dict[str, float]:
        return {
            "frozen_clusters": float(len(self.frozen_clusters)),
            "frozen_bridges": float(len(self.frozen_bridges)),
            "bridge_application_events": float(self.bridge_application_events),
            "bridge_target_edge_opportunities": float(self.bridge_target_edge_opportunities),
            "bridge_target_edges_traversed": float(self.bridge_target_edges_traversed),
            "bridge_dormant_edges_skipped": float(self.bridge_dormant_edges_skipped),
            "bridge_disabled_edges_skipped": float(self.bridge_disabled_edges_skipped),
        }
