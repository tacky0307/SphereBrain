from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT
from dormant_surface_flow_v14 import Edge
from dormant_surface_flow_v15b import ExperienceBridgeFacilitationBrain


@dataclass(frozen=True)
class ExplorationEvent:
    epoch: int
    task_name: str
    source_cluster: int
    target_cluster: int
    selected_edges: tuple[Edge, ...]
    traversed_edges: tuple[Edge, ...]


class ExploratoryFlowBrain(ExperienceBridgeFacilitationBrain):
    """v16: sparse exploratory flow near learned experience bridges.

    Exploration does not create edges, alter stored weights, wake dormant paths,
    or assist teacher propagation.  On a small fraction of sensory observations,
    enabled and non-dormant underused edges one or two graph steps beyond a
    learned bridge target receive a transient facilitation.
    """

    def __init__(
        self,
        *args,
        exploration_mode: str = "off",
        exploration_probability: float = 0.01,
        exploration_facilitation: float = 0.08,
        exploration_edge_limit: int = 8,
        exploration_reproduction_threshold: int = 2,
        exploration_seed: int = 161803,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if exploration_mode not in {"off", "guided", "random"}:
            raise ValueError("exploration_mode must be off, guided, or random")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        if exploration_facilitation < 0.0:
            raise ValueError("exploration_facilitation must be non-negative")
        if exploration_edge_limit < 1:
            raise ValueError("exploration_edge_limit must be at least 1")
        if exploration_reproduction_threshold < 2:
            raise ValueError("exploration_reproduction_threshold must be at least 2")

        self.exploration_mode = exploration_mode
        self.exploration_probability = float(exploration_probability)
        self.exploration_facilitation = float(exploration_facilitation)
        self.exploration_edge_limit = int(exploration_edge_limit)
        self.exploration_reproduction_threshold = int(exploration_reproduction_threshold)
        self.exploration_rng = random.Random(int(exploration_seed))
        self.exploration_runtime_enabled = False
        self.exploration_epoch = -1
        self.active_exploration_edges: set[Edge] = set()
        self.active_exploration_source_cluster: int | None = None
        self.active_exploration_target_cluster: int | None = None
        self.exploration_attempts = 0
        self.exploration_triggered = 0
        self.exploration_selected_edges = 0
        self.exploration_traversed_edges = 0
        self.exploration_events: list[ExplorationEvent] = []
        self.exploration_edge_epochs: dict[Edge, set[int]] = defaultdict(set)
        self.exploration_edge_tasks: dict[Edge, set[str]] = defaultdict(set)

    def set_exploration_runtime(self, enabled: bool, epoch: int | None = None) -> None:
        self.exploration_runtime_enabled = bool(enabled)
        if epoch is not None:
            self.exploration_epoch = int(epoch)

    def _bridge_for_current_transition(self):
        identity = self.current_experience_identity
        if identity is None or self.previous_frozen_cluster is None or self.previous_task is None:
            return None
        matching = [
            bridge for bridge in self.frozen_bridges
            if bridge.source_cluster == self.previous_frozen_cluster
            and bridge.source_task == self.previous_task
            and bridge.target_task == identity.task_name
        ]
        if not matching:
            return None
        return max(matching, key=lambda bridge: bridge.strength)

    def _guided_candidates(self, target_edges: tuple[Edge, ...]) -> list[Edge]:
        target_nodes = {node for edge in target_edges for node in edge}
        one_hop = set(target_nodes)
        for node in tuple(target_nodes):
            one_hop.update(int(value) for value in np.flatnonzero(self.adjacency[node]))
        two_hop = set(one_hop)
        for node in tuple(one_hop):
            two_hop.update(int(value) for value in np.flatnonzero(self.adjacency[node]))

        target_set = set(target_edges)
        candidates: list[Edge] = []
        for source in one_hop:
            for target_raw in np.flatnonzero(self.adjacency[source]):
                target = int(target_raw)
                edge = (int(source), target)
                if target not in two_hop or edge in target_set:
                    continue
                if not self.edge_enabled[edge] or self.pathway_state[edge] == PATHWAY_DORMANT:
                    continue
                candidates.append(edge)
        candidates.sort(key=lambda edge: (int(self.usage[edge]), float(self.weights[edge])))
        return candidates

    def _random_candidates(self) -> list[Edge]:
        rows, cols = np.nonzero(
            self.adjacency
            & self.edge_enabled
            & (self.pathway_state != PATHWAY_DORMANT)
        )
        candidates = [(int(source), int(target)) for source, target in zip(rows, cols)]
        self.exploration_rng.shuffle(candidates)
        return candidates

    def _prepare_exploration(self) -> None:
        self.active_exploration_edges = set()
        self.active_exploration_source_cluster = None
        self.active_exploration_target_cluster = None
        if not self.exploration_runtime_enabled or self.exploration_mode == "off":
            return

        self.exploration_attempts += 1
        if self.exploration_rng.random() >= self.exploration_probability:
            return

        bridge = self._bridge_for_current_transition()
        if self.exploration_mode == "guided":
            if bridge is None:
                return
            candidates = self._guided_candidates(bridge.target_edges)
            self.active_exploration_source_cluster = bridge.source_cluster
            self.active_exploration_target_cluster = bridge.target_cluster
        else:
            candidates = self._random_candidates()
            if bridge is not None:
                self.active_exploration_source_cluster = bridge.source_cluster
                self.active_exploration_target_cluster = bridge.target_cluster

        if not candidates:
            return
        chosen = candidates[: self.exploration_edge_limit]
        self.active_exploration_edges = set(chosen)
        self.exploration_triggered += 1
        self.exploration_selected_edges += len(chosen)

    def _effective_weight_matrix(self) -> np.ndarray:
        effective = super()._effective_weight_matrix()
        for source, target in self.active_exploration_edges:
            if not self.edge_enabled[source, target]:
                continue
            if self.pathway_state[source, target] == PATHWAY_DORMANT:
                continue
            effective[source, target] *= 1.0 + self.exploration_facilitation
        return effective

    def propagate(self, *args, **kwargs):
        self._prepare_exploration()
        result = super().propagate(*args, **kwargs)
        if not self.active_exploration_edges:
            return result

        identity = self.current_experience_identity
        traversed = tuple(sorted(set(result.traversed_edges).intersection(self.active_exploration_edges)))
        self.exploration_traversed_edges += len(traversed)
        if identity is not None and traversed:
            event = ExplorationEvent(
                epoch=self.exploration_epoch,
                task_name=identity.task_name,
                source_cluster=self.active_exploration_source_cluster if self.active_exploration_source_cluster is not None else -1,
                target_cluster=self.active_exploration_target_cluster if self.active_exploration_target_cluster is not None else -1,
                selected_edges=tuple(sorted(self.active_exploration_edges)),
                traversed_edges=traversed,
            )
            self.exploration_events.append(event)
            for edge in traversed:
                self.exploration_edge_epochs[edge].add(self.exploration_epoch)
                self.exploration_edge_tasks[edge].add(identity.task_name)
        return result

    def exploratory_flow_stats(self) -> dict[str, float]:
        reproduced = [
            edge for edge, epochs in self.exploration_edge_epochs.items()
            if len(epochs) >= self.exploration_reproduction_threshold
        ]
        emergent = [
            edge for edge in reproduced
            if len(self.exploration_edge_tasks.get(edge, set())) >= 2
        ]
        vanished = [
            edge for edge, epochs in self.exploration_edge_epochs.items()
            if len(epochs) == 1
        ]
        return {
            "exploration_attempts": float(self.exploration_attempts),
            "exploration_triggered": float(self.exploration_triggered),
            "exploration_events": float(len(self.exploration_events)),
            "exploration_selected_edges": float(self.exploration_selected_edges),
            "exploration_traversed_edges": float(self.exploration_traversed_edges),
            "exploration_unique_edges": float(len(self.exploration_edge_epochs)),
            "exploration_reproduced_edges": float(len(reproduced)),
            "exploration_emergent_candidates": float(len(emergent)),
            "exploration_vanished_edges": float(len(vanished)),
        }
