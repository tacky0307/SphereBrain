from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np


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
    """Experimental SphereBrain propagation model.

    - Core means the complete spherical medium, not a center node.
    - Inputs enter through one part of the surface.
    - Outputs are time-varying activation patterns on another surface area.
    - Directed weights and temporary fatigue prevent one hub from dominating.
    - Associative learning can reinforce only paths that reach a named target pattern.
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
    ) -> None:
        self.node_count = node_count
        self.neighbors_per_node = neighbors_per_node
        self.seed = seed
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.surface_radius = surface_radius
        self.fatigue_gain = fatigue_gain
        self.fatigue_decay = fatigue_decay
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
    def _text_to_nodes(text: str, candidates: list[int], count: int) -> list[int]:
        clean = text.strip()
        if not clean:
            raise ValueError("Text is empty.")
        digest = sha256(clean.encode("utf-8")).digest()
        selected: list[int] = []
        offset = 0
        while len(selected) < min(count, len(candidates)):
            value = int.from_bytes(digest[offset : offset + 4], "big")
            node = candidates[value % len(candidates)]
            if node not in selected:
                selected.append(node)
            offset += 4
            if offset + 4 > len(digest):
                digest = sha256(digest).digest()
                offset = 0
        return selected

    def stimulus_to_inputs(self, stimulus: str, count: int = 4) -> list[int]:
        return self._text_to_nodes(stimulus, self.input_nodes, count)

    def concept_to_outputs(self, concept: str, count: int = 4) -> list[int]:
        """Assign a stable output-surface pattern to a word or concept label."""
        return self._text_to_nodes(concept, self.output_nodes, count)

    def propagate(
        self,
        source_nodes: Iterable[int],
        steps: int = 24,
        threshold: float = 0.08,
        noise: float = 0.006,
        learn: bool = True,
        target_output_nodes: Iterable[int] | None = None,
    ) -> SurfaceFlowResult:
        sources = list(source_nodes)
        targets = None if target_output_nodes is None else set(target_output_nodes)
        activation = np.zeros(self.node_count, dtype=float)
        fatigue = np.zeros(self.node_count, dtype=float)
        parent = np.full(self.node_count, -1, dtype=int)

        for index, node in enumerate(sources):
            activation[node] = max(activation[node], 1.0 - index * 0.06)

        output_mask = np.zeros(self.node_count, dtype=bool)
        output_mask[self.output_nodes] = True
        output_history: list[dict[int, float]] = []
        history: list[list[int]] = [np.flatnonzero(activation > 0).tolist()]
        traversed: list[tuple[int, int]] = []

        for _ in range(steps):
            effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
            transmitted = effective[:, None] * self.weights
            transmitted[~self.adjacency] = 0.0

            best_parent = np.argmax(transmitted, axis=0)
            next_activation = transmitted[best_parent, np.arange(self.node_count)] * 0.86
            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)
            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)
            for target in active_now:
                source = int(best_parent[target])
                if transmitted[source, target] >= threshold:
                    parent[target] = source
                    traversed.append((source, target))

            visible = np.flatnonzero((next_activation > 0) & output_mask)
            output_history.append({int(n): float(next_activation[n]) for n in visible})

            fatigue = fatigue * self.fatigue_decay
            fatigue += next_activation * self.fatigue_gain
            fatigue = np.clip(fatigue, 0.0, 0.95)
            activation = next_activation
            if not active_now:
                break

        successful_edges = self._successful_paths(parent, output_history, sources, targets)
        if learn and successful_edges:
            self._reinforce(successful_edges)

        return SurfaceFlowResult(
            source_nodes=sources,
            output_history=output_history,
            activation_history=history,
            traversed_edges=traversed,
            final_activation=activation.copy(),
        )

    def _successful_paths(
        self,
        parent: np.ndarray,
        output_history: list[dict[int, float]],
        sources: list[int],
        target_output_nodes: set[int] | None = None,
    ) -> set[tuple[int, int]]:
        source_set = set(sources)
        reached = {node for step in output_history for node, value in step.items() if value > 0}
        if target_output_nodes is not None:
            reached &= target_output_nodes
        edges: set[tuple[int, int]] = set()
        for node in reached:
            current = node
            seen: set[int] = set()
            while current not in source_set and current not in seen:
                seen.add(current)
                previous = int(parent[current])
                if previous < 0:
                    break
                edges.add((previous, current))
                current = previous
        return edges

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
    def target_score(result: SurfaceFlowResult, target_nodes: Iterable[int]) -> float:
        """Total output activation that appeared on the requested target pattern."""
        targets = set(target_nodes)
        return sum(value for step in result.output_history for node, value in step.items() if node in targets)
