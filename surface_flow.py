from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from flow_bias import FlowBias, FlowBiasStats


SurfacePattern = Mapping[int, float]


@dataclass
class SurfaceFlowResult:
    source_nodes: list[int]
    output_history: list[dict[int, float]]
    activation_history: list[list[int]]
    traversed_edges: list[tuple[int, int]]
    final_activation: np.ndarray

    @property
    def output_nodes(self) -> list[int]:
        return sorted({node for step in self.output_history for node in step})


class SurfaceFlowBrain:
    """Language-independent experimental SphereBrain core.

    The core knows only numeric surface activity, graph propagation, pathway
    weights, fatigue, usage, optional short-term activity, and optional
    bidirectional plasticity. External encoders alone know text or scalar labels.
    """

    def __init__(
        self,
        node_count: int = 600,
        neighbors_per_node: int = 8,
        seed: int = 42,
        learning_rate: float = 0.035,
        decay_rate: float = 0.0004,
        surface_radius: float = 0.78,
        fatigue_gain: float = 0.32,
        fatigue_decay: float = 0.72,
        transmission_gain: float = 0.86,
        edge_activity_ratio: float = 0.35,
        route_usage_penalty: float = 0.18,
        node_usage_penalty: float = 0.035,
        same_experience_penalty: float = 0.45,
        activation_field_enabled: bool = False,
        activation_field_influence: float = 0.05,
        activation_field_decay: float = 0.90,
        activation_field_gain: float = 0.10,
        bidirectional_plasticity_enabled: bool = False,
        recent_activity_decay: float = 0.90,
        unused_activity_threshold: float = 0.015,
        unused_weakening_rate: float = 0.00015,
        overuse_activity_threshold: float = 0.35,
        overuse_weakening_rate: float = 0.0020,
        minimum_edge_weight: float = 0.02,
        flow_bias_enabled: bool = False,
        flow_bias_initial: float = 1.0,
        flow_bias_gain: float = 0.25,
        flow_bias_decay: float = 0.92,
        flow_bias_maximum: float = 3.0,
        flow_bias_weight_power: float = 1.0,
    ) -> None:
        self.node_count = node_count
        self.neighbors_per_node = neighbors_per_node
        self.seed = seed
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.surface_radius = surface_radius
        self.fatigue_gain = fatigue_gain
        self.fatigue_decay = fatigue_decay
        self.transmission_gain = transmission_gain
        self.edge_activity_ratio = edge_activity_ratio
        self.route_usage_penalty = route_usage_penalty
        self.node_usage_penalty = node_usage_penalty
        self.same_experience_penalty = same_experience_penalty
        self.activation_field_enabled = activation_field_enabled
        self.activation_field_influence = activation_field_influence
        self.activation_field_decay = activation_field_decay
        self.activation_field_gain = activation_field_gain
        self.bidirectional_plasticity_enabled = bidirectional_plasticity_enabled
        self.recent_activity_decay = recent_activity_decay
        self.unused_activity_threshold = unused_activity_threshold
        self.unused_weakening_rate = unused_weakening_rate
        self.overuse_activity_threshold = overuse_activity_threshold
        self.overuse_weakening_rate = overuse_weakening_rate
        self.minimum_edge_weight = minimum_edge_weight
        self.flow_bias_enabled = flow_bias_enabled
        self.flow_bias_initial = flow_bias_initial
        self.flow_bias_gain = flow_bias_gain
        self.flow_bias_decay = flow_bias_decay
        self.flow_bias_maximum = flow_bias_maximum
        self.flow_bias_weight_power = flow_bias_weight_power
        self._validate_parameters()
        self.rng = np.random.default_rng(seed)

        self.positions = self._generate_points_in_sphere(node_count)
        self.adjacency = np.zeros((node_count, node_count), dtype=bool)
        self.weights = np.zeros((node_count, node_count), dtype=float)
        self.usage = np.zeros((node_count, node_count), dtype=int)
        self.node_usage = np.zeros(node_count, dtype=int)
        self.activation_field = np.zeros(node_count, dtype=float)
        self.recent_edge_activity = np.zeros((node_count, node_count), dtype=float)
        self.flow_bias = FlowBias(
            node_count,
            initial_bias=self.flow_bias_initial,
            reinforce_gain=self.flow_bias_gain,
            decay=self.flow_bias_decay,
            maximum_bias=self.flow_bias_maximum,
        )
        self._connect_nearest_nodes()
        self.edge_enabled = self.adjacency.copy()

        radii = np.linalg.norm(self.positions, axis=1)
        surface = radii >= surface_radius
        self.input_nodes = np.flatnonzero(surface & (self.positions[:, 0] < 0)).tolist()
        self.output_nodes = np.flatnonzero(surface & (self.positions[:, 0] >= 0)).tolist()
        if len(self.input_nodes) < 4 or len(self.output_nodes) < 4:
            raise ValueError("Not enough surface nodes; lower surface_radius or increase node_count.")

    def _validate_parameters(self) -> None:
        unit_values = (
            ("activation_field_influence", self.activation_field_influence),
            ("activation_field_decay", self.activation_field_decay),
            ("activation_field_gain", self.activation_field_gain),
            ("recent_activity_decay", self.recent_activity_decay),
            ("unused_activity_threshold", self.unused_activity_threshold),
            ("overuse_activity_threshold", self.overuse_activity_threshold),
        )
        for name, value in unit_values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (
            ("unused_weakening_rate", self.unused_weakening_rate),
            ("overuse_weakening_rate", self.overuse_weakening_rate),
            ("minimum_edge_weight", self.minimum_edge_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.flow_bias_initial <= 0.0:
            raise ValueError("flow_bias_initial must be positive")
        if self.flow_bias_gain < 0.0:
            raise ValueError("flow_bias_gain must be non-negative")
        if not 0.0 <= self.flow_bias_decay <= 1.0:
            raise ValueError("flow_bias_decay must be in [0, 1]")
        if self.flow_bias_maximum < self.flow_bias_initial:
            raise ValueError("flow_bias_maximum must be >= flow_bias_initial")
        if self.flow_bias_weight_power <= 0.0:
            raise ValueError("flow_bias_weight_power must be positive")

    def reset_activation_field(self) -> None:
        self.activation_field.fill(0.0)

    def activation_field_stats(self) -> dict[str, float]:
        return {
            "mean": float(np.mean(self.activation_field)),
            "max": float(np.max(self.activation_field)),
            "active_ratio": float(np.mean(self.activation_field > 1e-6)),
            "energy": float(np.sum(self.activation_field)),
        }

    def reset_flow_bias(self) -> None:
        """Reset temporary pathway memory without changing long-term weights."""
        self.flow_bias.reset()

    def flow_bias_stats(self) -> FlowBiasStats:
        """Return statistics for currently enabled graph edges only."""
        return self.flow_bias.stats()

    def effective_weights(self, use_flow_bias: bool | None = None) -> np.ndarray:
        """Return transmission weights with optional temporary flow bias applied.

        This method does not mutate the brain. ``flow_bias_weight_power`` can
        sharpen long-term pathway differences while the temporary multiplier
        encourages recently traversed directions.
        """
        if use_flow_bias is None:
            use_flow_bias = self.flow_bias_enabled
        base = np.power(self.weights, self.flow_bias_weight_power)
        if use_flow_bias:
            base = base * self.flow_bias.multiplier()
        return np.where(self.edge_enabled, base, 0.0)

    def pathway_stats(self) -> dict[str, float]:
        connected = self.adjacency
        enabled = self.edge_enabled
        enabled_weights = self.weights[enabled]
        return {
            "connected_edges": float(np.count_nonzero(connected)),
            "enabled_edges": float(np.count_nonzero(enabled)),
            "disabled_edges": float(np.count_nonzero(connected & ~enabled)),
            "mean_enabled_weight": (
                float(np.mean(enabled_weights)) if enabled_weights.size else 0.0
            ),
            "max_recent_activity": float(np.max(self.recent_edge_activity)),
        }

    def _update_activation_field(self, trace: np.ndarray) -> None:
        if not self.activation_field_enabled:
            return
        trace = np.clip(np.asarray(trace, dtype=float), 0.0, 1.0)
        self.activation_field *= self.activation_field_decay
        self.activation_field += self.activation_field_gain * trace
        self.activation_field = np.clip(self.activation_field, 0.0, 1.0)

    def _generate_points_in_sphere(self, count: int) -> np.ndarray:
        directions = self.rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = self.rng.random(count) ** (1.0 / 3.0)
        return directions * radii[:, None]

    def _connect_nearest_nodes(self) -> None:
        differences = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)

        for node in range(self.node_count):
            nearest = np.argsort(distances[node])[: self.neighbors_per_node]
            for other_raw in nearest:
                other = int(other_raw)
                self.adjacency[node, other] = True
                self.adjacency[other, node] = True
                if self.weights[node, other] == 0.0:
                    base = 0.22 + 0.42 * np.exp(-2.0 * distances[node, other])
                    forward = min(0.92, base + float(self.rng.uniform(0.0, 0.07)))
                    reverse = np.clip(forward + self.rng.normal(0.0, 0.015), 0.05, 0.92)
                    self.weights[node, other] = forward
                    self.weights[other, node] = reverse

    @staticmethod
    def _validate_pattern(
        pattern: SurfacePattern,
        allowed_nodes: Iterable[int],
        name: str,
    ) -> dict[int, float]:
        allowed = set(allowed_nodes)
        normalized: dict[int, float] = {}
        for node_raw, value_raw in pattern.items():
            node = int(node_raw)
            value = float(value_raw)
            if node not in allowed:
                raise ValueError(f"{name} contains node {node} outside its surface")
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} activity must be finite and in [0, 1]")
            if value > 0.0:
                normalized[node] = value
        if not normalized:
            raise ValueError(f"{name} has no positive activity")
        return normalized

    def propagate(
        self,
        input_pattern: SurfacePattern,
        steps: int = 24,
        threshold: float = 0.08,
        noise: float = 0.006,
        use_activation_field: bool | None = None,
        update_activation_field: bool = False,
        use_flow_bias: bool | None = None,
        update_flow_bias: bool = False,
    ) -> SurfaceFlowResult:
        sources = self._validate_pattern(input_pattern, self.input_nodes, "input_pattern")
        activation = np.zeros(self.node_count, dtype=float)
        fatigue = np.zeros(self.node_count, dtype=float)

        for node, value in sources.items():
            activation[node] = value

        if use_activation_field is None:
            use_activation_field = self.activation_field_enabled
        if use_activation_field and self.activation_field_enabled:
            field_activity = np.clip(
                self.activation_field * self.activation_field_influence, 0.0, 1.0
            )
            activation = 1.0 - (1.0 - activation) * (1.0 - field_activity)

        trace = activation.copy()
        output_mask = np.zeros(self.node_count, dtype=bool)
        output_mask[self.output_nodes] = True
        output_history: list[dict[int, float]] = []
        history: list[list[int]] = [np.flatnonzero(activation > 0).tolist()]
        traversed: list[tuple[int, int]] = []

        if use_flow_bias is None:
            use_flow_bias = self.flow_bias_enabled

        for _ in range(steps):
            if use_flow_bias and update_flow_bias:
                self.flow_bias.decay()

            effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
            step_weights = self.effective_weights(use_flow_bias=use_flow_bias)
            contributions = effective[:, None] * step_weights * self.transmission_gain
            contributions[~self.edge_enabled] = 0.0
            contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)

            next_activation = 1.0 - np.prod(1.0 - contributions, axis=0)
            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)
            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0
            trace = np.maximum(trace, next_activation)

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)

            edge_threshold = threshold * self.edge_activity_ratio
            active_edges = np.argwhere(contributions >= edge_threshold)
            traversed_step = {
                (int(source), int(target)) for source, target in active_edges
            }
            traversed.extend(sorted(traversed_step))
            if use_flow_bias and update_flow_bias and traversed_step:
                self.flow_bias.reinforce(traversed_step)

            visible = np.flatnonzero((next_activation > 0) & output_mask)
            output_history.append({int(n): float(next_activation[n]) for n in visible})

            fatigue = fatigue * self.fatigue_decay
            fatigue += next_activation * self.fatigue_gain
            fatigue = np.clip(fatigue, 0.0, 0.95)
            activation = next_activation
            if not active_now:
                break

        if update_activation_field:
            self._update_activation_field(trace)

        return SurfaceFlowResult(
            source_nodes=sorted(sources),
            output_history=output_history,
            activation_history=history,
            traversed_edges=traversed,
            final_activation=activation.copy(),
        )

    def experience(
        self,
        input_pattern: SurfacePattern,
        target_pattern: SurfacePattern,
        update_activation_field: bool = False,
    ) -> set[tuple[int, int]]:
        inputs = self._validate_pattern(input_pattern, self.input_nodes, "input_pattern")
        targets = self._validate_pattern(target_pattern, self.output_nodes, "target_pattern")

        input_rank = sorted(inputs, key=lambda node: inputs[node], reverse=True)
        target_rank = sorted(targets, key=lambda node: targets[node], reverse=True)
        pair_count = max(len(input_rank), len(target_rank))
        edges: set[tuple[int, int]] = set()

        for index in range(pair_count):
            source = input_rank[index % len(input_rank)]
            target = target_rank[index % len(target_rank)]
            path = self._shortest_path(source, target, temporarily_used=edges)
            edges.update(zip(path, path[1:]))

        self._reinforce(edges)

        if update_activation_field and self.activation_field_enabled:
            trace = np.zeros(self.node_count, dtype=float)
            for node, value in inputs.items():
                trace[node] = max(trace[node], value)
            for node, value in targets.items():
                trace[node] = max(trace[node], value)
            for source, target in edges:
                route_activity = min(1.0, self.weights[source, target])
                trace[source] = max(trace[source], route_activity)
                trace[target] = max(trace[target], route_activity)
            self._update_activation_field(trace)

        return edges

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
                weight_cost = 1.0 / max(float(self.weights[node, neighbor]), 1e-9)
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
                edge_cost = weight_cost * (
                    1.0 + edge_congestion + node_congestion + temporary_congestion
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

    def _apply_bidirectional_plasticity(self, reinforced: set[tuple[int, int]]) -> None:
        if not self.bidirectional_plasticity_enabled:
            self.weights[self.edge_enabled] *= 1.0 - self.decay_rate
            return

        self.recent_edge_activity *= self.recent_activity_decay
        for source, target in reinforced:
            self.recent_edge_activity[source, target] += 1.0
        self.recent_edge_activity = np.clip(self.recent_edge_activity, 0.0, 1.0)

        enabled = self.edge_enabled
        recent = self.recent_edge_activity
        unused = enabled & (recent < self.unused_activity_threshold)
        overused = enabled & (recent > self.overuse_activity_threshold)

        self.weights[unused] *= 1.0 - self.unused_weakening_rate
        excess = np.zeros_like(recent)
        excess[overused] = (
            recent[overused] - self.overuse_activity_threshold
        ) / max(1.0 - self.overuse_activity_threshold, 1e-9)
        self.weights[overused] *= 1.0 - self.overuse_weakening_rate * excess[overused]
        self.weights[enabled] = np.maximum(self.weights[enabled], self.minimum_edge_weight)

    def _reinforce(self, edges: Iterable[tuple[int, int]]) -> None:
        edge_set = set(edges)
        self._apply_bidirectional_plasticity(edge_set)
        for source, target in edge_set:
            if not self.edge_enabled[source, target]:
                continue
            current = self.weights[source, target]
            saturation = 1.0 / (1.0 + 0.08 * self.usage[source, target])
            delta = self.learning_rate * saturation * (1.0 - current)
            self.weights[source, target] = min(1.0, current + delta)
            self.usage[source, target] += 1
            self.node_usage[source] += 1
            self.node_usage[target] += 1

    def lesion_most_used_edges(
        self,
        fraction: float = 0.05,
        bidirectional: bool = True,
    ) -> list[tuple[int, int]]:
        """Disable the most-used active pathways and return disabled directions."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        candidates = np.argwhere(self.edge_enabled & (self.usage > 0))
        if candidates.size == 0:
            return []
        ranked = sorted(
            ((int(s), int(t)) for s, t in candidates),
            key=lambda edge: (
                int(self.usage[edge]),
                float(self.weights[edge]),
            ),
            reverse=True,
        )
        count = max(1, int(np.ceil(len(ranked) * fraction)))
        disabled: set[tuple[int, int]] = set()
        for source, target in ranked[:count]:
            if self.edge_enabled[source, target]:
                self.edge_enabled[source, target] = False
                disabled.add((source, target))
            if bidirectional and self.edge_enabled[target, source]:
                self.edge_enabled[target, source] = False
                disabled.add((target, source))
        return sorted(disabled)

    def restore_edges(self, edges: Iterable[tuple[int, int]]) -> None:
        for source, target in edges:
            if self.adjacency[source, target]:
                self.edge_enabled[source, target] = True

    def output_vector(self, result: SurfaceFlowResult, max_steps: int = 24) -> np.ndarray:
        index = {node: i for i, node in enumerate(self.output_nodes)}
        vector = np.zeros((max_steps, len(self.output_nodes)), dtype=float)
        for step_no, step in enumerate(result.output_history[:max_steps]):
            for node, value in step.items():
                vector[step_no, index[node]] = value
        return vector.ravel()

    def output_similarity(self, left: SurfaceFlowResult, right: SurfaceFlowResult) -> float:
        a = self.output_vector(left)
        b = self.output_vector(right)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)

    @staticmethod
    def target_score(result: SurfaceFlowResult, target_pattern: SurfacePattern) -> float:
        targets = set(target_pattern)
        return sum(
            value
            for step in result.output_history
            for node, value in step.items()
            if node in targets
        )
