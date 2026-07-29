from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable

import numpy as np

from surface_flow import SurfaceFlowBrain, SurfaceFlowResult, SurfacePattern


PATHWAY_NORMAL = 0
PATHWAY_PROTECTED = 1
PATHWAY_DORMANT = 2


class DormantSurfaceFlowBrain(SurfaceFlowBrain):
    """SurfaceFlowBrain with reversible pathway-state plasticity.

    Pathway weights remain long-term memory. Plasticity changes pathway state:

    - normal: available with ordinary transmission
    - protected: recently useful and temporarily exempt from dormancy
    - dormant: memory is preserved and transmits as a low-power standby route

    Dormant pathways can return either through teacher-guided relearning or when
    sufficiently strong propagation repeatedly recruits them. During recovery
    mode, new dormancy is suspended so alternative routes are allowed to form.
    """

    def __init__(
        self,
        *args,
        dormancy_after: int = 160,
        protection_period: int = 36,
        dormant_transmission: float = 0.40,
        dormant_search_penalty: float = 1.2,
        reactivation_boost: float = 0.025,
        auto_reactivation_traversals: int = 2,
        state_activity_decay: float = 0.92,
        overuse_threshold: float = 0.55,
        overuse_penalty_gain: float = 0.55,
        overuse_penalty_decay: float = 0.82,
        **kwargs,
    ) -> None:
        kwargs["bidirectional_plasticity_enabled"] = False
        super().__init__(*args, **kwargs)

        if dormancy_after < 1:
            raise ValueError("dormancy_after must be at least 1")
        if protection_period < 0:
            raise ValueError("protection_period must be non-negative")
        if auto_reactivation_traversals < 1:
            raise ValueError("auto_reactivation_traversals must be at least 1")
        for name, value in (
            ("dormant_transmission", dormant_transmission),
            ("reactivation_boost", reactivation_boost),
            ("state_activity_decay", state_activity_decay),
            ("overuse_threshold", overuse_threshold),
            ("overuse_penalty_gain", overuse_penalty_gain),
            ("overuse_penalty_decay", overuse_penalty_decay),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if dormant_search_penalty < 0.0:
            raise ValueError("dormant_search_penalty must be non-negative")

        self.dormancy_after = int(dormancy_after)
        self.protection_period = int(protection_period)
        self.dormant_transmission = float(dormant_transmission)
        self.dormant_search_penalty = float(dormant_search_penalty)
        self.reactivation_boost = float(reactivation_boost)
        self.auto_reactivation_traversals = int(auto_reactivation_traversals)
        self.state_activity_decay = float(state_activity_decay)
        self.overuse_threshold = float(overuse_threshold)
        self.overuse_penalty_gain = float(overuse_penalty_gain)
        self.overuse_penalty_decay = float(overuse_penalty_decay)
        self.recovery_mode = False

        shape = (self.node_count, self.node_count)
        self.pathway_state = np.full(shape, PATHWAY_NORMAL, dtype=np.uint8)
        self.protection_remaining = np.zeros(shape, dtype=np.int32)
        self.inactive_age = np.zeros(shape, dtype=np.int32)
        self.state_recent_activity = np.zeros(shape, dtype=float)
        self.homeostatic_penalty = np.zeros(shape, dtype=float)
        self.reactivation_count = np.zeros(shape, dtype=np.int32)
        self.auto_reactivation_count = np.zeros(shape, dtype=np.int32)
        self.last_reactivated: set[tuple[int, int]] = set()
        self.last_auto_reactivated: set[tuple[int, int]] = set()

        self.pathway_state[~self.adjacency] = PATHWAY_DORMANT

    def set_recovery_mode(self, enabled: bool) -> None:
        """Suspend new dormancy while damaged pathways are being relearned."""
        self.recovery_mode = bool(enabled)

    def pathway_state_stats(self) -> dict[str, float]:
        connected = self.adjacency
        return {
            "normal": float(np.count_nonzero(connected & (self.pathway_state == PATHWAY_NORMAL))),
            "protected": float(
                np.count_nonzero(connected & (self.pathway_state == PATHWAY_PROTECTED))
            ),
            "dormant": float(np.count_nonzero(connected & (self.pathway_state == PATHWAY_DORMANT))),
            "reactivations": float(np.sum(self.reactivation_count[connected])),
            "auto_reactivations": float(np.sum(self.auto_reactivation_count[connected])),
            "mean_homeostatic_penalty": float(np.mean(self.homeostatic_penalty[connected])),
            "recovery_mode": float(self.recovery_mode),
        }

    def _effective_weight_matrix(self) -> np.ndarray:
        multipliers = np.ones_like(self.weights)
        dormant = self.pathway_state == PATHWAY_DORMANT
        multipliers[dormant] *= self.dormant_transmission
        multipliers *= 1.0 - np.clip(self.homeostatic_penalty, 0.0, 0.95)
        return self.weights * multipliers

    def _reactivate_pathway(self, source: int, target: int, automatic: bool) -> None:
        if not self.edge_enabled[source, target]:
            return
        if self.pathway_state[source, target] != PATHWAY_DORMANT:
            return
        self.pathway_state[source, target] = PATHWAY_PROTECTED
        self.protection_remaining[source, target] = self.protection_period
        self.inactive_age[source, target] = 0
        self.reactivation_count[source, target] += 1
        self.last_reactivated.add((source, target))
        if automatic:
            self.auto_reactivation_count[source, target] += 1
            self.last_auto_reactivated.add((source, target))

    def _auto_reactivate_from_result(self, result: SurfaceFlowResult) -> None:
        """Wake dormant standby routes repeatedly recruited by strong flow."""
        self.last_auto_reactivated = set()
        counts = Counter(result.traversed_edges)
        for (source, target), count in counts.items():
            if count < self.auto_reactivation_traversals:
                continue
            self._reactivate_pathway(source, target, automatic=True)

    def propagate(
        self,
        input_pattern: SurfacePattern,
        steps: int = 24,
        threshold: float = 0.08,
        noise: float = 0.006,
        use_activation_field: bool | None = None,
        update_activation_field: bool = False,
    ) -> SurfaceFlowResult:
        stored_weights = self.weights
        self.weights = self._effective_weight_matrix()
        try:
            result = super().propagate(
                input_pattern=input_pattern,
                steps=steps,
                threshold=threshold,
                noise=noise,
                use_activation_field=use_activation_field,
                update_activation_field=update_activation_field,
            )
        finally:
            self.weights = stored_weights
        self._auto_reactivate_from_result(result)
        return result

    def _shortest_path(
        self,
        start: int,
        goal: int,
        temporarily_used: set[tuple[int, int]] | None = None,
    ) -> list[int]:
        queue: list[tuple[float, int]] = [(0.0, start)]
        costs = {start: 0.0}
        previous: dict[int, int] = {}
        temporarily_used = temporarily_used or set()

        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if cost != costs.get(node):
                continue

            for neighbor_raw in self.edge_enabled[node].nonzero()[0]:
                neighbor = int(neighbor_raw)
                weight = max(float(self.weights[node, neighbor]), 1e-9)
                weight_cost = 1.0 / weight
                edge_congestion = self.route_usage_penalty * np.log1p(
                    self.usage[node, neighbor]
                )
                node_congestion = self.node_usage_penalty * np.log1p(
                    self.node_usage[neighbor]
                )
                temporary_congestion = (
                    self.same_experience_penalty
                    if (node, neighbor) in temporarily_used
                    else 0.0
                )
                dormant_penalty = (
                    self.dormant_search_penalty
                    if self.pathway_state[node, neighbor] == PATHWAY_DORMANT
                    else 0.0
                )
                homeostatic = float(self.homeostatic_penalty[node, neighbor])
                edge_cost = weight_cost * (
                    1.0
                    + edge_congestion
                    + node_congestion
                    + temporary_congestion
                    + dormant_penalty
                    + homeostatic
                )
                new_cost = cost + edge_cost
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    previous[neighbor] = node
                    heapq.heappush(queue, (new_cost, neighbor))

        if goal not in costs:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def _update_pathway_states(self, reinforced: set[tuple[int, int]]) -> None:
        connected_enabled = self.adjacency & self.edge_enabled
        self.last_reactivated = set()

        self.state_recent_activity *= self.state_activity_decay
        self.homeostatic_penalty *= self.overuse_penalty_decay
        self.protection_remaining[self.protection_remaining > 0] -= 1

        protected_expired = (
            connected_enabled
            & (self.pathway_state == PATHWAY_PROTECTED)
            & (self.protection_remaining <= 0)
        )
        self.pathway_state[protected_expired] = PATHWAY_NORMAL

        self.inactive_age[connected_enabled] += 1
        for source, target in reinforced:
            if not self.edge_enabled[source, target]:
                continue
            self.state_recent_activity[source, target] += 1.0
            self.inactive_age[source, target] = 0

            if self.pathway_state[source, target] == PATHWAY_DORMANT:
                self._reactivate_pathway(source, target, automatic=False)
            else:
                self.pathway_state[source, target] = PATHWAY_PROTECTED
                self.protection_remaining[source, target] = max(
                    self.protection_remaining[source, target], self.protection_period
                )

        if not self.recovery_mode:
            normal = connected_enabled & (self.pathway_state == PATHWAY_NORMAL)
            become_dormant = normal & (self.inactive_age >= self.dormancy_after)
            self.pathway_state[become_dormant] = PATHWAY_DORMANT

        overused = connected_enabled & (self.state_recent_activity > self.overuse_threshold)
        excess = np.zeros_like(self.state_recent_activity)
        excess[overused] = (
            self.state_recent_activity[overused] - self.overuse_threshold
        ) / max(1.0 - self.overuse_threshold, 1e-9)
        self.homeostatic_penalty[overused] = np.maximum(
            self.homeostatic_penalty[overused],
            np.clip(self.overuse_penalty_gain * excess[overused], 0.0, 0.85),
        )

    def _reinforce(self, edges: Iterable[tuple[int, int]]) -> None:
        edge_set = set(edges)
        self._update_pathway_states(edge_set)
        self.weights[self.edge_enabled] *= 1.0 - self.decay_rate

        for source, target in edge_set:
            if not self.edge_enabled[source, target]:
                continue
            current = self.weights[source, target]
            saturation = 1.0 / (1.0 + 0.08 * self.usage[source, target])
            delta = self.learning_rate * saturation * (1.0 - current)
            if (source, target) in self.last_reactivated:
                delta += self.reactivation_boost * (1.0 - current)
            self.weights[source, target] = min(1.0, current + delta)
            self.usage[source, target] += 1
            self.node_usage[source] += 1
            self.node_usage[target] += 1

    def lesion_most_used_edges(
        self,
        fraction: float = 0.05,
        bidirectional: bool = True,
    ) -> list[tuple[int, int]]:
        disabled = super().lesion_most_used_edges(
            fraction=fraction,
            bidirectional=bidirectional,
        )
        for source, target in disabled:
            self.protection_remaining[source, target] = 0
            self.homeostatic_penalty[source, target] = 0.0
        return disabled
