from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


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

    The core knows only:
    - numeric activity on input-surface nodes,
    - multi-path propagation through the spherical graph,
    - numeric activity on output-surface nodes,
    - directed weights, fatigue, usage and reinforcement.

    Text, images, sounds and scalar values must be converted into surface
    patterns by external encoders. The core never receives semantic labels.
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
        self.rng = np.random.default_rng(seed)

        self.positions = self._generate_points_in_sphere(node_count)
        self.adjacency = np.zeros((node_count, node_count), dtype=bool)
        self.weights = np.zeros((node_count, node_count), dtype=float)
        self.usage = np.zeros((node_count, node_count), dtype=int)
        self.node_usage = np.zeros(node_count, dtype=int)
        self._connect_nearest_nodes()

        radii = np.linalg.norm(self.positions, axis=1)
        surface = radii >= surface_radius
        self.input_nodes = np.flatnonzero(surface & (self.positions[:, 0] < 0)).tolist()
        self.output_nodes = np.flatnonzero(surface & (self.positions[:, 0] >= 0)).tolist()
        if len(self.input_nodes) < 4 or len(self.output_nodes) < 4:
            raise ValueError("Not enough surface nodes; lower surface_radius or increase node_count.")

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
    ) -> SurfaceFlowResult:
        """Propagate a numeric surface pattern through many paths at once.

        Every active directed edge contributes. Incoming contributions are
        combined as a bounded probabilistic union rather than selecting only
        the strongest parent. Thus several weak routes can jointly activate a
        node while all activities remain in [0, 1].
        """
        sources = self._validate_pattern(input_pattern, self.input_nodes, "input_pattern")
        activation = np.zeros(self.node_count, dtype=float)
        fatigue = np.zeros(self.node_count, dtype=float)

        for node, value in sources.items():
            activation[node] = value

        output_mask = np.zeros(self.node_count, dtype=bool)
        output_mask[self.output_nodes] = True
        output_history: list[dict[int, float]] = []
        history: list[list[int]] = [np.flatnonzero(activation > 0).tolist()]
        traversed: list[tuple[int, int]] = []

        for _ in range(steps):
            effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
            contributions = effective[:, None] * self.weights * self.transmission_gain
            contributions[~self.adjacency] = 0.0
            contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)

            # 1 - product(1 - contribution) is a bounded superposition. It
            # preserves each route's influence and rewards simultaneous arrival.
            next_activation = 1.0 - np.prod(1.0 - contributions, axis=0)
            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)
            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)

            edge_threshold = threshold * self.edge_activity_ratio
            active_edges = np.argwhere(contributions >= edge_threshold)
            traversed.extend((int(source), int(target)) for source, target in active_edges)

            visible = np.flatnonzero((next_activation > 0) & output_mask)
            output_history.append({int(n): float(next_activation[n]) for n in visible})

            fatigue = fatigue * self.fatigue_decay
            fatigue += next_activation * self.fatigue_gain
            fatigue = np.clip(fatigue, 0.0, 0.95)
            activation = next_activation
            if not active_now:
                break

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
    ) -> set[tuple[int, int]]:
        """Learn one numeric input-target experience.

        This first implementation is teacher-guided: active input and target
        surface populations are paired by activity rank, and their strongest
        existing routes are reinforced. The teacher supplies only two numeric
        surface patterns; it supplies no words or semantic meaning.
        """
        inputs = self._validate_pattern(input_pattern, self.input_nodes, "input_pattern")
        targets = self._validate_pattern(target_pattern, self.output_nodes, "target_pattern")

        input_rank = sorted(inputs, key=lambda node: inputs[node], reverse=True)
        target_rank = sorted(targets, key=lambda node: targets[node], reverse=True)
        pair_count = max(len(input_rank), len(target_rank))
        edges: set[tuple[int, int]] = set()

        for index in range(pair_count):
            source = input_rank[index % len(input_rank)]
            target = target_rank[index % len(target_rank)]
            path = self._shortest_path(source, target)
            edges.update(zip(path, path[1:]))

        self._reinforce(edges)
        return edges

    def _shortest_path(self, start: int, goal: int) -> list[int]:
        queue: list[tuple[float, int]] = [(0.0, start)]
        costs = {start: 0.0}
        previous: dict[int, int] = {}

        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if cost != costs.get(node):
                continue
            for neighbor_raw in self.adjacency[node].nonzero()[0]:
                neighbor = int(neighbor_raw)
                edge_cost = 1.0 / max(float(self.weights[node, neighbor]), 1e-9)
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

    def _reinforce(self, edges: Iterable[tuple[int, int]]) -> None:
        self.weights[self.adjacency] *= 1.0 - self.decay_rate
        for source, target in edges:
            current = self.weights[source, target]
            saturation = 1.0 / (1.0 + 0.08 * self.usage[source, target])
            delta = self.learning_rate * saturation * (1.0 - current)
            self.weights[source, target] = min(1.0, current + delta)
            self.usage[source, target] += 1
            self.node_usage[source] += 1
            self.node_usage[target] += 1

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
