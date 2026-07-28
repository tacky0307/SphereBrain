from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import re

import numpy as np


@dataclass
class SignalResult:
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    final_activation: np.ndarray


class SphereBrain:
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
        self._rebuild_sparse_cache()

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

    def _rebuild_sparse_cache(self) -> None:
        """Build compact neighbour lists used by propagation."""
        self._neighbors = [
            np.flatnonzero(self.adjacency[node]).astype(np.int32)
            for node in range(self.node_count)
        ]
        upper = np.triu_indices(self.node_count, k=1)
        mask = self.adjacency[upper]
        self._edge_a = upper[0][mask]
        self._edge_b = upper[1][mask]

    @staticmethod
    def text_units(text: str, min_size: int = 2, max_size: int = 4) -> list[str]:
        """Split text into reusable short units without a language dictionary."""
        clean = " ".join(text.strip().split())
        if not clean:
            return []

        parts = re.findall(r"[一-龥々〆ヵヶぁ-んァ-ヶー]+|[A-Za-z0-9]+", clean)
        units: list[str] = []
        for part in parts:
            if re.fullmatch(r"[A-Za-z0-9]+", part):
                units.append(part.lower())
                continue
            if len(part) <= min_size:
                units.append(part)
                continue
            for size in range(min_size, min(max_size, len(part)) + 1):
                units.extend(part[i : i + size] for i in range(len(part) - size + 1))

        return list(dict.fromkeys(units)) or [clean]

    def text_to_sources(self, text: str, count: int = 3) -> list[int]:
        units = self.text_units(text)
        if not units:
            raise ValueError("入力が空です。")
        if count <= 0:
            return []

        sources: list[int] = []
        selected_indexes = np.linspace(
            0, len(units) - 1, num=min(len(units), max(count * 2, count)), dtype=int
        )
        for index in selected_indexes:
            digest = hashlib.blake2b(units[int(index)].encode("utf-8"), digest_size=8).digest()
            node = int.from_bytes(digest, "big") % self.node_count
            if node not in sources:
                sources.append(node)
            if len(sources) >= count:
                break

        salt = 0
        clean = " ".join(text.strip().split())
        while len(sources) < min(count, self.node_count):
            digest = hashlib.blake2b(
                f"{clean}\0{salt}".encode("utf-8"), digest_size=8
            ).digest()
            node = int.from_bytes(digest, "big") % self.node_count
            if node not in sources:
                sources.append(node)
            salt += 1
        return sources

    def propagate(
        self,
        source_nodes: Iterable[int],
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
        context_nodes: Iterable[int] | None = None,
    ) -> SignalResult:
        sources = list(dict.fromkeys(int(node) for node in source_nodes))
        activation = np.zeros(self.node_count, dtype=float)

        for index, node in enumerate(sources):
            activation[node] = max(activation[node], 1.0 - index * 0.08)

        if context_nodes:
            for node in context_nodes:
                activation[int(node)] = max(activation[int(node)], 0.42)

        active = np.flatnonzero(activation > 0)
        activated_nodes = set(active.tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]

        for _ in range(steps):
            best_source = np.full(self.node_count, -1, dtype=np.int32)
            best_signal = np.zeros(self.node_count, dtype=float)

            for source in active:
                neighbours = self._neighbors[int(source)]
                if neighbours.size == 0:
                    continue
                signals = activation[int(source)] * self.weights[int(source), neighbours]
                improved = signals > best_signal[neighbours]
                if np.any(improved):
                    targets = neighbours[improved]
                    best_signal[targets] = signals[improved]
                    best_source[targets] = int(source)

            next_activation = best_signal * 0.82
            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)
            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0

            active = np.flatnonzero(next_activation > 0)
            active_now = active.tolist()
            history.append(active_now)
            activated_nodes.update(active_now)

            for target in active_now:
                source = int(best_source[target])
                if source >= 0 and best_signal[target] >= threshold:
                    traversed_edges.add(tuple(sorted((source, target))))

            activation = next_activation
            if active.size == 0:
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

    def _reinforce(self, edges: Iterable[tuple[int, int]], nodes: Iterable[int]) -> None:
        if self._edge_a.size:
            decayed = self.weights[self._edge_a, self._edge_b] * (1.0 - self.decay_rate)
            self.weights[self._edge_a, self._edge_b] = decayed
            self.weights[self._edge_b, self._edge_a] = decayed

        for a, b in edges:
            current = self.weights[a, b]
            new = current + self.learning_rate * (1.0 - current)
            self.weights[a, b] = new
            self.weights[b, a] = new
            self.usage[a, b] += 1
            self.usage[b, a] += 1

        for node in nodes:
            self.node_usage[int(node)] += 1

        np.clip(self.weights, 0.0, 1.0, out=self.weights)

    def idle_cycle(self, remembered_nodes: Iterable[int]) -> SignalResult | None:
        nodes = list(remembered_nodes)
        if not nodes:
            return None
        source_count = min(2, len(nodes))
        sources = self.rng.choice(nodes, size=source_count, replace=False).tolist()
        return self.propagate(
            sources,
            steps=10,
            threshold=0.19,
            noise=0.025,
            learn=True,
        )

    def strongest_edges(self, limit: int = 40) -> list[dict]:
        pairs = list(zip(self._edge_a.tolist(), self._edge_b.tolist()))
        pairs.sort(key=lambda edge: self.weights[edge[0], edge[1]], reverse=True)
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
            "adjacency": self.adjacency.astype(np.uint8).tolist(),
            "weights": self.weights.tolist(),
            "usage": self.usage.tolist(),
            "node_usage": self.node_usage.tolist(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

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
        brain.node_usage = np.asarray(
            data.get("node_usage", [0] * brain.node_count), dtype=int
        )
        brain._rebuild_sparse_cache()
        return brain
