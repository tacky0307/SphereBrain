from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json

import numpy as np

from reflection import ReflectionEngine, ReflectionResult
from scheduler import Scheduler, SchedulerConfig, SchedulerPhase
from trace import TraceFrame, TraceRecorder


@dataclass
class SignalResult:
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    final_activation: np.ndarray


class SphereBrain:
    """SphereBrain core with the v27 experience/reflection life cycle.

    Existing propagation and persistence APIs remain available. The v27
    additions observe each completed activity, store it as Trace, and allow
    recorded whole-brain activity to return as internal experience.
    """

    def __init__(
        self,
        node_count: int = 240,
        neighbors_per_node: int = 7,
        seed: int = 42,
        learning_rate: float = 0.07,
        decay_rate: float = 0.0008,
        *,
        max_trace_frames: int | None = None,
        reflections_per_experience: int = 1,
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

        self.activity = np.zeros(node_count, dtype=float)
        self.previous_activity = np.zeros(node_count, dtype=float)
        self.fatigue = np.zeros(node_count, dtype=float)

        self.trace = TraceRecorder(max_frames=max_trace_frames)
        self.reflection = ReflectionEngine(rng=self.rng)
        self.scheduler = Scheduler(
            SchedulerConfig(
                reflections_per_experience=reflections_per_experience,
            )
        )

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

    def text_to_sources(self, text: str, count: int = 3) -> list[int]:
        clean = text.strip()
        if not clean:
            raise ValueError("入力が空です。")

        digest = hashlib.sha256(clean.encode("utf-8")).digest()
        sources: list[int] = []
        offset = 0
        while len(sources) < count:
            value = int.from_bytes(digest[offset:offset + 4], "big")
            node = value % self.node_count
            if node not in sources:
                sources.append(node)
            offset += 4
            if offset + 4 > len(digest):
                digest = hashlib.sha256(digest).digest()
                offset = 0
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
        sources = list(source_nodes)
        initial_activation = np.zeros(self.node_count, dtype=float)

        for index, node in enumerate(sources):
            initial_activation[node] = max(
                initial_activation[node],
                1.0 - index * 0.08,
            )

        if context_nodes:
            for node in context_nodes:
                initial_activation[node] = max(initial_activation[node], 0.42)

        return self._propagate_activation(
            initial_activation,
            source_nodes=sources,
            steps=steps,
            threshold=threshold,
            noise=noise,
            learn=learn,
        )

    def propagate_signal(
        self,
        signal: Iterable[float] | np.ndarray,
        *,
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
    ) -> SignalResult:
        """Propagate an arbitrary whole-brain signal.

        This v27 entry point is used by Reflection. Unlike ``propagate``, it
        preserves the distributed activity pattern stored in a Trace frame.
        """

        activation = np.asarray(signal, dtype=float)
        if activation.shape != (self.node_count,):
            raise ValueError(
                f"signal must have shape ({self.node_count},), "
                f"received {activation.shape}"
            )
        if not np.all(np.isfinite(activation)):
            raise ValueError("signal must contain only finite values")

        activation = np.clip(activation, 0.0, 1.0)
        sources = np.flatnonzero(activation > 0.0).tolist()
        return self._propagate_activation(
            activation,
            source_nodes=sources,
            steps=steps,
            threshold=threshold,
            noise=noise,
            learn=learn,
        )

    def _propagate_activation(
        self,
        initial_activation: np.ndarray,
        *,
        source_nodes: list[int],
        steps: int,
        threshold: float,
        noise: float,
        learn: bool,
    ) -> SignalResult:
        activation = initial_activation.copy()
        self.previous_activity = self.activity.copy()

        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]

        for _ in range(steps):
            transmitted = activation[:, None] * self.weights
            next_activation = transmitted.max(axis=0) * 0.82

            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)

            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)
            activated_nodes.update(active_now)

            for target in active_now:
                incoming = transmitted[:, target]
                source = int(np.argmax(incoming))
                if incoming[source] >= threshold and self.adjacency[source, target]:
                    traversed_edges.add(tuple(sorted((source, target))))

            activation = next_activation
            if not active_now:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        self.activity = activation.copy()

        return SignalResult(
            source_nodes=source_nodes,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=history,
            final_activation=activation.copy(),
        )

    def experience(
        self,
        stimulus: str | Iterable[int],
        *,
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
        metadata: dict | None = None,
    ) -> tuple[SignalResult, TraceFrame]:
        """Run one external experience and record what happened."""

        if not self.scheduler.is_experience:
            raise RuntimeError(
                "scheduler is in REFLECTION phase; finish scheduled reflections "
                "before starting another experience"
            )

        if isinstance(stimulus, str):
            source_nodes = self.text_to_sources(stimulus)
            trace_metadata = {"text": stimulus}
            if metadata:
                trace_metadata.update(metadata)
        else:
            source_nodes = list(stimulus)
            trace_metadata = {} if metadata is None else dict(metadata)

        stimulus_signal = np.zeros(self.node_count, dtype=float)
        for index, node in enumerate(source_nodes):
            stimulus_signal[node] = max(
                stimulus_signal[node],
                1.0 - index * 0.08,
            )

        result = self.propagate(
            source_nodes,
            steps=steps,
            threshold=threshold,
            noise=noise,
            learn=learn,
        )
        frame = self.trace.record_core(
            self,
            source="experience",
            stimulus=stimulus_signal,
            metadata=trace_metadata,
        )
        self.scheduler.finish_experience()
        return result, frame

    def reflect(
        self,
        *,
        selection: str = "latest",
        frame_index: int | None = None,
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
    ) -> tuple[SignalResult, TraceFrame, ReflectionResult]:
        """Replay one Trace frame and record the resulting internal experience."""

        if not self.scheduler.is_reflection:
            raise RuntimeError("scheduler is not in REFLECTION phase")

        if selection == "latest":
            replay = self.reflection.latest(self.trace)
        elif selection == "random":
            replay = self.reflection.random(self.trace)
        elif selection == "index":
            if frame_index is None:
                raise ValueError("frame_index is required when selection='index'")
            replay = self.reflection.index(self.trace, frame_index)
        else:
            raise ValueError("selection must be 'latest', 'random', or 'index'")

        result = self.propagate_signal(
            replay.signal,
            steps=steps,
            threshold=threshold,
            noise=noise,
            learn=learn,
        )
        frame = self.trace.record_core(
            self,
            source="reflection",
            stimulus=replay.signal,
            metadata={
                "reflected_frame_index": replay.frame_index,
                "reflected_time_index": replay.time_index,
                "reflected_source": replay.source,
            },
        )
        self.scheduler.finish_reflection()
        return result, frame, replay

    def complete_reflections(
        self,
        *,
        selection: str = "latest",
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
    ) -> list[tuple[SignalResult, TraceFrame, ReflectionResult]]:
        """Run reflections until the Scheduler returns to EXPERIENCE."""

        completed = []
        while self.scheduler.phase is SchedulerPhase.REFLECTION:
            completed.append(
                self.reflect(
                    selection=selection,
                    steps=steps,
                    threshold=threshold,
                    noise=noise,
                    learn=learn,
                )
            )
        return completed

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
            "activity": self.activity.tolist(),
            "previous_activity": self.previous_activity.tolist(),
            "fatigue": self.fatigue.tolist(),
            "scheduler_time_index": self.scheduler.time_index,
            "scheduler_phase": self.scheduler.phase.name,
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
        brain.node_usage = np.asarray(
            data.get("node_usage", [0] * brain.node_count),
            dtype=int,
        )
        brain.activity = np.asarray(
            data.get("activity", [0.0] * brain.node_count),
            dtype=float,
        )
        brain.previous_activity = np.asarray(
            data.get("previous_activity", [0.0] * brain.node_count),
            dtype=float,
        )
        brain.fatigue = np.asarray(
            data.get("fatigue", [0.0] * brain.node_count),
            dtype=float,
        )
        brain.scheduler.time_index = int(data.get("scheduler_time_index", 0))
        phase_name = data.get("scheduler_phase", "EXPERIENCE")
        brain.scheduler.phase = SchedulerPhase[phase_name]
        return brain
