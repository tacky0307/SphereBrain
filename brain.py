from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import numpy as np


@dataclass
class SignalResult:
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    final_activation: np.ndarray


class SphereBrain:
    """数値刺激だけを扱うSphere BrainのCore。"""

    def __init__(
        self,
        node_count: int = 240,
        neighbors_per_node: int = 7,
        seed: int = 42,
        learning_rate: float = 0.07,
        decay_rate: float = 0.0008,
    ) -> None:
        self.node_count = node_count
        self.neighbors_per_node = neighbors_per_node
        self.seed = seed
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed)

        self.positions = self._generate_points_in_sphere(node_count)
        self.adjacency = np.zeros((node_count, node_count), dtype=bool)
        self.weights = np.zeros((node_count, node_count), dtype=float)
        self.usage = np.zeros((node_count, node_count), dtype=int)
        self.node_usage = np.zeros(node_count, dtype=int)

        self._connect_nearest_nodes()

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
            for other in nearest:
                a, b = sorted((node, int(other)))
                self.adjacency[a, b] = True
                self.adjacency[b, a] = True
                if self.weights[a, b] == 0:
                    base = 0.22 + 0.42 * np.exp(-2.0 * distances[a, b])
                    weight = min(0.92, base + float(self.rng.uniform(0.0, 0.07)))
                    self.weights[a, b] = weight
                    self.weights[b, a] = weight

    def _propagation_step(
        self,
        activation: np.ndarray,
        threshold: float,
        noise: float,
        carry: float = 0.0,
    ) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
        transmitted = activation[:, None] * self.weights
        next_activation = transmitted.max(axis=0) * 0.82
        if carry:
            next_activation = np.maximum(next_activation, activation * carry)
        if noise:
            next_activation += self.rng.normal(0.0, noise, self.node_count)

        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < threshold] = 0.0
        active_now = np.flatnonzero(next_activation > 0).tolist()

        step_edges: list[tuple[int, int]] = []
        for target in active_now:
            incoming = transmitted[:, target]
            source = int(np.argmax(incoming))
            if incoming[source] >= threshold and self.adjacency[source, target]:
                step_edges.append(tuple(sorted((source, target))))

        return next_activation, active_now, step_edges

    def propagate(
        self,
        source_nodes: Iterable[int],
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
        context_nodes: Iterable[int] | None = None,
    ) -> SignalResult:
        sources = [int(node) for node in source_nodes]
        if not sources:
            raise ValueError("source_nodes must not be empty")
        if any(node < 0 or node >= self.node_count for node in sources):
            raise ValueError("source node is outside Core")

        activation = np.zeros(self.node_count, dtype=float)
        for index, node in enumerate(sources):
            activation[node] = max(activation[node], 1.0 - index * 0.08)

        if context_nodes:
            for node in context_nodes:
                node = int(node)
                if 0 <= node < self.node_count:
                    activation[node] = max(activation[node], 0.42)

        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]

        for _ in range(steps):
            activation, active_now, step_edges = self._propagation_step(
                activation=activation,
                threshold=threshold,
                noise=noise,
            )
            history.append(active_now)
            activated_nodes.update(active_now)
            traversed_edges.update(step_edges)
            if not active_now:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        return SignalResult(
            source_nodes=sources,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=history,
            final_activation=activation.copy(),
        )

    def replay_trace(
        self,
        activation_history: Iterable[Iterable[int]],
        replay_strength: float = 0.28,
        replay_decay: float = 0.72,
        threshold: float = 0.12,
        noise: float = 0.012,
        settle_steps: int = 6,
        learn: bool = True,
    ) -> SignalResult:
        """過去のTraceを時間順に弱く再刺激し、現在のCoreで再体験する。"""
        steps = [[int(node) for node in step] for step in activation_history]
        steps = [step for step in steps if step]
        if not steps:
            raise ValueError("activation_history must contain active nodes")
        if any(node < 0 or node >= self.node_count for step in steps for node in step):
            raise ValueError("trace node is outside Core")
        if not 0.0 < replay_strength <= 1.0:
            raise ValueError("replay_strength must be between 0 and 1")
        if not 0.0 <= replay_decay <= 1.0:
            raise ValueError("replay_decay must be between 0 and 1")

        activation = np.zeros(self.node_count, dtype=float)
        source_nodes = list(dict.fromkeys(steps[0]))
        activated_nodes: set[int] = set()
        traversed_edges: set[tuple[int, int]] = set()
        replay_history: list[list[int]] = []

        for trace_step in steps:
            activation *= replay_decay
            for index, node in enumerate(trace_step):
                strength = replay_strength * max(0.45, 1.0 - index * 0.03)
                activation[node] = max(activation[node], strength)

            injected = np.flatnonzero(activation > 0).tolist()
            activated_nodes.update(injected)
            replay_history.append(injected)

            activation, active_now, step_edges = self._propagation_step(
                activation=activation,
                threshold=threshold,
                noise=noise,
                carry=0.34,
            )
            activated_nodes.update(active_now)
            traversed_edges.update(step_edges)
            replay_history.append(active_now)

        for _ in range(settle_steps):
            activation, active_now, step_edges = self._propagation_step(
                activation=activation,
                threshold=threshold,
                noise=noise,
                carry=0.18,
            )
            activated_nodes.update(active_now)
            traversed_edges.update(step_edges)
            replay_history.append(active_now)
            if not active_now:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        return SignalResult(
            source_nodes=source_nodes,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=replay_history,
            final_activation=activation.copy(),
        )

    def _reinforce(self, edges: Iterable[tuple[int, int]], nodes: Iterable[int]) -> None:
        self.weights[self.adjacency] *= 1.0 - self.decay_rate

        for a, b in edges:
            current = self.weights[a, b]
            new = current + self.learning_rate * (1.0 - current)
            self.weights[a, b] = new
            self.weights[b, a] = new
            self.usage[a, b] += 1
            self.usage[b, a] += 1

        for node in nodes:
            self.node_usage[node] += 1

        self.weights = np.clip(self.weights, 0.0, 1.0)

    def strongest_edges(self, limit: int = 40) -> list[dict]:
        upper = np.triu_indices(self.node_count, k=1)
        mask = self.adjacency[upper]
        pairs = list(zip(upper[0][mask], upper[1][mask]))
        pairs.sort(key=lambda e: self.weights[e[0], e[1]], reverse=True)
        return [
            {
                "a": int(a),
                "b": int(b),
                "weight": float(self.weights[a, b]),
                "usage": int(self.usage[a, b]),
            }
            for a, b in pairs[:limit]
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "node_count": self.node_count,
            "neighbors_per_node": self.neighbors_per_node,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "positions": self.positions.tolist(),
            "adjacency": self.adjacency.astype(int).tolist(),
            "weights": self.weights.tolist(),
            "usage": self.usage.tolist(),
            "node_usage": self.node_usage.tolist(),
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SphereBrain":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        brain = cls(
            node_count=data["node_count"],
            neighbors_per_node=data["neighbors_per_node"],
            seed=data["seed"],
            learning_rate=data["learning_rate"],
            decay_rate=data["decay_rate"],
        )
        brain.positions = np.asarray(data["positions"], dtype=float)
        brain.adjacency = np.asarray(data["adjacency"], dtype=bool)
        brain.weights = np.asarray(data["weights"], dtype=float)
        brain.usage = np.asarray(data["usage"], dtype=int)
        brain.node_usage = np.asarray(data.get("node_usage", [0] * brain.node_count), dtype=int)
        return brain
