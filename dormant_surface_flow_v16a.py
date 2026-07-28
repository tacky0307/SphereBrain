from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT
from dormant_surface_flow_v14 import Edge
from dormant_surface_flow_v15b import ExperienceBridgeFacilitationBrain


@dataclass(frozen=True)
class FrontierExplorationEvent:
    epoch: int
    task_name: str
    source_cluster: int
    target_cluster: int
    eligible_edges: int
    selected_edges: tuple[Edge, ...]
    traversed_edges: tuple[Edge, ...]


class ActivityFrontierExploratoryBrain(ExperienceBridgeFacilitationBrain):
    """v16a: explore only from the activity frontier observed moments earlier.

    A normal sensory probe first reveals where activity actually reached. Exploration
    can then transiently facilitate enabled, non-dormant outgoing edges whose source
    lies on that frontier. Guided mode further restricts the frontier to the learned
    bridge target neighbourhood. No edge is created, no stored weight is changed,
    dormant pathways are never assisted, and teacher propagation never explores.
    """

    def __init__(
        self,
        *args,
        frontier_mode: str = "off",
        exploration_probability: float = 0.01,
        exploration_facilitation: float = 0.08,
        exploration_edge_limit: int = 8,
        reproduction_threshold: int = 2,
        exploration_seed: int = 161803,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if frontier_mode not in {"off", "guided-frontier", "random-frontier"}:
            raise ValueError("invalid frontier_mode")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        if exploration_edge_limit < 1:
            raise ValueError("exploration_edge_limit must be at least 1")

        import random

        self.frontier_mode = frontier_mode
        self.exploration_probability = float(exploration_probability)
        self.exploration_facilitation = float(exploration_facilitation)
        self.exploration_edge_limit = int(exploration_edge_limit)
        self.reproduction_threshold = int(reproduction_threshold)
        self.exploration_rng = random.Random(int(exploration_seed))

        self.frontier_runtime_enabled = False
        self.frontier_epoch = -1
        self.active_frontier_edges: set[Edge] = set()
        self.pending_frontier_pass = False
        self.pending_source_cluster = -1
        self.pending_target_cluster = -1
        self.pending_eligible_count = 0

        self.frontier_observations = 0
        self.eligible_opportunities = 0
        self.eligible_edges_total = 0
        self.exploration_draws = 0
        self.exploration_triggered = 0
        self.selected_edges_total = 0
        self.traversed_edges_total = 0
        self.frontier_events: list[FrontierExplorationEvent] = []
        self.edge_epochs: dict[Edge, set[int]] = defaultdict(set)
        self.edge_tasks: dict[Edge, set[str]] = defaultdict(set)

    def set_frontier_runtime(self, enabled: bool, epoch: int | None = None) -> None:
        self.frontier_runtime_enabled = bool(enabled)
        if epoch is not None:
            self.frontier_epoch = int(epoch)
        if not enabled:
            self.active_frontier_edges = set()
            self.pending_frontier_pass = False

    def _bridge_for_current_transition(self):
        identity = self.current_experience_identity
        if identity is None or self.previous_frozen_cluster is None or self.previous_task is None:
            return None
        matches = [
            bridge for bridge in self.frozen_bridges
            if bridge.source_cluster == self.previous_frozen_cluster
            and bridge.source_task == self.previous_task
            and bridge.target_task == identity.task_name
        ]
        return max(matches, key=lambda row: row.strength) if matches else None

    def _frontier_candidates(self, probe_edges: set[Edge]) -> tuple[list[Edge], int, int]:
        reached_nodes = {target for _, target in probe_edges}
        used = set(probe_edges)
        bridge = self._bridge_for_current_transition()
        source_cluster = bridge.source_cluster if bridge is not None else -1
        target_cluster = bridge.target_cluster if bridge is not None else -1

        if self.frontier_mode == "guided-frontier":
            if bridge is None:
                return [], source_cluster, target_cluster
            bridge_nodes = {node for edge in bridge.target_edges for node in edge}
            neighbourhood = set(bridge_nodes)
            for node in tuple(bridge_nodes):
                neighbourhood.update(int(v) for v in np.flatnonzero(self.adjacency[node]))
            reached_nodes.intersection_update(neighbourhood)

        candidates: list[Edge] = []
        for source in reached_nodes:
            for target_raw in np.flatnonzero(self.adjacency[source]):
                edge = (int(source), int(target_raw))
                if edge in used:
                    continue
                if not self.edge_enabled[edge]:
                    continue
                if self.pathway_state[edge] == PATHWAY_DORMANT:
                    continue
                candidates.append(edge)

        if self.frontier_mode == "random-frontier":
            self.exploration_rng.shuffle(candidates)
        else:
            candidates.sort(key=lambda edge: (int(self.usage[edge]), float(self.weights[edge])))
        return candidates, source_cluster, target_cluster

    def prepare_from_probe(self, probe_result) -> bool:
        """Prepare one exploratory pass from a completed normal observation."""
        self.active_frontier_edges = set()
        self.pending_frontier_pass = False
        self.frontier_observations += 1
        if not self.frontier_runtime_enabled or self.frontier_mode == "off":
            return False

        candidates, source_cluster, target_cluster = self._frontier_candidates(
            set(probe_result.traversed_edges)
        )
        if not candidates:
            return False

        self.eligible_opportunities += 1
        self.eligible_edges_total += len(candidates)
        self.exploration_draws += 1
        if self.exploration_rng.random() >= self.exploration_probability:
            return False

        chosen = candidates[: self.exploration_edge_limit]
        self.active_frontier_edges = set(chosen)
        self.pending_frontier_pass = True
        self.pending_source_cluster = source_cluster
        self.pending_target_cluster = target_cluster
        self.pending_eligible_count = len(candidates)
        self.exploration_triggered += 1
        self.selected_edges_total += len(chosen)
        return True

    def _effective_weight_matrix(self) -> np.ndarray:
        effective = super()._effective_weight_matrix()
        for source, target in self.active_frontier_edges:
            if not self.edge_enabled[source, target]:
                continue
            if self.pathway_state[source, target] == PATHWAY_DORMANT:
                continue
            effective[source, target] *= 1.0 + self.exploration_facilitation
        return effective

    def propagate(self, *args, **kwargs):
        exploratory_pass = self.pending_frontier_pass
        result = super().propagate(*args, **kwargs)
        if not exploratory_pass:
            return result

        identity = self.current_experience_identity
        traversed = tuple(sorted(set(result.traversed_edges).intersection(self.active_frontier_edges)))
        self.traversed_edges_total += len(traversed)
        if identity is not None and traversed:
            self.frontier_events.append(
                FrontierExplorationEvent(
                    epoch=self.frontier_epoch,
                    task_name=identity.task_name,
                    source_cluster=self.pending_source_cluster,
                    target_cluster=self.pending_target_cluster,
                    eligible_edges=self.pending_eligible_count,
                    selected_edges=tuple(sorted(self.active_frontier_edges)),
                    traversed_edges=traversed,
                )
            )
            for edge in traversed:
                self.edge_epochs[edge].add(self.frontier_epoch)
                self.edge_tasks[edge].add(identity.task_name)

        self.active_frontier_edges = set()
        self.pending_frontier_pass = False
        return result

    def frontier_stats(self) -> dict[str, float]:
        reproduced = [
            edge for edge, epochs in self.edge_epochs.items()
            if len(epochs) >= self.reproduction_threshold
        ]
        emergent = [edge for edge in reproduced if len(self.edge_tasks[edge]) >= 2]
        vanished = [edge for edge, epochs in self.edge_epochs.items() if len(epochs) == 1]
        conversion = (
            self.traversed_edges_total / self.selected_edges_total
            if self.selected_edges_total else 0.0
        )
        return {
            "frontier_observations": float(self.frontier_observations),
            "eligible_opportunities": float(self.eligible_opportunities),
            "eligible_edges_total": float(self.eligible_edges_total),
            "exploration_draws": float(self.exploration_draws),
            "exploration_triggered": float(self.exploration_triggered),
            "selected_edges": float(self.selected_edges_total),
            "traversed_edges": float(self.traversed_edges_total),
            "selection_conversion": float(conversion),
            "events": float(len(self.frontier_events)),
            "unique_edges": float(len(self.edge_epochs)),
            "reproduced_edges": float(len(reproduced)),
            "emergent_candidates": float(len(emergent)),
            "vanished_edges": float(len(vanished)),
        }
