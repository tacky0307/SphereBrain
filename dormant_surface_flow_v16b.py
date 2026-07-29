from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from dormant_surface_flow_v14 import Edge
from dormant_surface_flow_v16a import ActivityFrontierExploratoryBrain


class PromisingFrontierPlayBrain(ActivityFrontierExploratoryBrain):
    """v16b: repeatedly touch the most promising live frontier.

    This is not a random search over the sphere. A normal probe first reveals
    the live activity frontier. Candidate edges are then ranked by three local
    clues: source recurrence in the probe, existing transmission strength, and
    proximity to experience bridges. The touch is transient: no physical edge,
    stored weight, dormant state, or teacher route is changed directly.
    """

    def __init__(
        self,
        *args,
        touch_probability: float = 0.20,
        touch_facilitation: float = 1.50,
        touch_edge_limit: int = 4,
        **kwargs,
    ) -> None:
        kwargs["frontier_mode"] = "random-frontier"
        kwargs["exploration_probability"] = touch_probability
        kwargs["exploration_facilitation"] = touch_facilitation
        kwargs["exploration_edge_limit"] = touch_edge_limit
        super().__init__(*args, **kwargs)
        self.frontier_mode = "promising-frontier"
        self.last_probe_source_hits: dict[int, int] = {}
        self.promising_scores: dict[Edge, float] = {}

    def _bridge_neighbourhood(self) -> set[int]:
        nodes: set[int] = set()
        for bridge in self.frozen_bridges:
            for source, target in bridge.source_edges + bridge.target_edges:
                nodes.add(int(source))
                nodes.add(int(target))
        expanded = set(nodes)
        for node in tuple(nodes):
            expanded.update(int(v) for v in np.flatnonzero(self.adjacency[node]))
        return expanded

    def _frontier_candidates(self, probe_edges: set[Edge]) -> tuple[list[Edge], int, int]:
        reached_nodes = {target for _, target in probe_edges}
        used = set(probe_edges)
        bridge = self._bridge_for_current_transition()
        source_cluster = bridge.source_cluster if bridge is not None else -1
        target_cluster = bridge.target_cluster if bridge is not None else -1
        bridge_near = self._bridge_neighbourhood()

        max_hits = max(self.last_probe_source_hits.values(), default=1)
        connected_usage = self.usage[self.edge_enabled]
        max_usage = max(int(np.max(connected_usage)) if connected_usage.size else 0, 1)

        candidates: list[Edge] = []
        scores: dict[Edge, float] = {}
        for source in reached_nodes:
            recurrence = self.last_probe_source_hits.get(int(source), 0) / max_hits
            for target_raw in np.flatnonzero(self.adjacency[source]):
                edge = (int(source), int(target_raw))
                if edge in used or not self.edge_enabled[edge]:
                    continue
                if self.pathway_state[edge] == 2:
                    continue

                weight_score = float(self.weights[edge])
                usage_score = math.log1p(int(self.usage[edge])) / math.log1p(max_usage)
                bridge_score = 1.0 if source in bridge_near or int(target_raw) in bridge_near else 0.0
                # High probability first: live repeatedly, already transmittable,
                # and close to structures formed by experience.
                score = 0.45 * recurrence + 0.35 * weight_score + 0.15 * bridge_score + 0.05 * usage_score
                candidates.append(edge)
                scores[edge] = score

        candidates.sort(key=lambda edge: scores[edge], reverse=True)
        self.promising_scores = scores
        return candidates, source_cluster, target_cluster

    def prepare_from_probe(self, probe_result) -> bool:
        hits: dict[int, int] = defaultdict(int)
        for step_nodes in probe_result.activation_history:
            for node in step_nodes:
                hits[int(node)] += 1
        self.last_probe_source_hits = dict(hits)
        return super().prepare_from_probe(probe_result)
