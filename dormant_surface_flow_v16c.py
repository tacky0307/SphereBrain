from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from dormant_surface_flow_v14 import Edge
from dormant_surface_flow_v16b import PromisingFrontierPlayBrain


class CachedPromisingFrontierPlayBrain(PromisingFrontierPlayBrain):
    """v16c: v16b behaviour with read-only topology caches.

    The graph, weights, pathway states, learning rules, and propagation rules are
    unchanged. Only repeated discovery work is cached:

    - adjacency rows become immutable neighbour tuples;
    - the frozen bridge neighbourhood is calculated once after bridge freezing.

    These caches affect execution cost only. They do not create routes, change
    scores, wake pathways, or alter stored memory.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._neighbour_cache: tuple[tuple[int, ...], ...] = tuple(
            tuple(int(value) for value in np.flatnonzero(self.adjacency[node]))
            for node in range(self.node_count)
        )
        self._frozen_bridge_neighbourhood_cache: frozenset[int] | None = None

    def freeze_experience_bridges(self):
        result = super().freeze_experience_bridges()
        self._frozen_bridge_neighbourhood_cache = frozenset(
            self._calculate_bridge_neighbourhood()
        )
        return result

    def _calculate_bridge_neighbourhood(self) -> set[int]:
        nodes: set[int] = set()
        for bridge in self.frozen_bridges:
            for source, target in bridge.source_edges + bridge.target_edges:
                nodes.add(int(source))
                nodes.add(int(target))

        expanded = set(nodes)
        for node in nodes:
            expanded.update(self._neighbour_cache[node])
        return expanded

    def _bridge_neighbourhood(self) -> set[int]:
        cached = self._frozen_bridge_neighbourhood_cache
        if cached is None:
            cached = frozenset(self._calculate_bridge_neighbourhood())
            self._frozen_bridge_neighbourhood_cache = cached
        # The caller only performs membership tests. Returning a set preserves the
        # v16b method contract while keeping the stored cache immutable.
        return set(cached)

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
        log_max_usage = math.log1p(max_usage)

        candidates: list[Edge] = []
        scores: dict[Edge, float] = {}
        for source_raw in reached_nodes:
            source = int(source_raw)
            recurrence = self.last_probe_source_hits.get(source, 0) / max_hits
            for target in self._neighbour_cache[source]:
                edge = (source, target)
                if edge in used or not self.edge_enabled[edge]:
                    continue
                if self.pathway_state[edge] == 2:
                    continue

                weight_score = float(self.weights[edge])
                usage_score = math.log1p(int(self.usage[edge])) / log_max_usage
                bridge_score = 1.0 if source in bridge_near or target in bridge_near else 0.0
                score = (
                    0.45 * recurrence
                    + 0.35 * weight_score
                    + 0.15 * bridge_score
                    + 0.05 * usage_score
                )
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
        return super(PromisingFrontierPlayBrain, self).prepare_from_probe(probe_result)
