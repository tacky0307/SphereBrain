from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

import numpy as np


@dataclass(frozen=True)
class WaveConfig:
    """Parameters for the first observable wave experiment.

    The values are deliberately conservative.  Wave Core v0 should settle
    naturally and reveal small learned biases before it attempts autonomous
    replay or long-lived internal activity.
    """

    node_count: int = 240
    neighbors_per_node: int = 8
    propagation_rate: float = 0.22
    persistence: float = 0.34
    decay: float = 0.94
    fatigue_gain: float = 0.055
    fatigue_recovery: float = 0.86
    fire_threshold: float = 0.72
    fire_gain: float = 0.10
    learning_rate: float = 0.006
    conductivity_min: float = 0.05
    conductivity_max: float = 1.80
    quiet_threshold: float = 0.015
    max_steps: int = 90
    seed: int = 27


@dataclass
class WaveSnapshot:
    step: int
    activity: np.ndarray
    fatigue: np.ndarray
    fired_nodes: tuple[int, ...]
    total_activity: float
    center: tuple[float, float, float]


@dataclass
class ExperimentTrace:
    name: str
    snapshots: list[WaveSnapshot] = field(default_factory=list)
    changed_edges: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.snapshots)

    @property
    def peak_total_activity(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(item.total_activity for item in self.snapshots)

    def max_activity_for(self, node_ids: Iterable[int]) -> float:
        ids = np.asarray(tuple(node_ids), dtype=int)
        if ids.size == 0 or not self.snapshots:
            return 0.0
        return float(max(np.max(item.activity[ids]) for item in self.snapshots))


class SphereWaveCore:
    """A discrete observation mesh for a continuous-field-like experiment.

    Nodes are not symbols and edges are not prescribed answers.  Nodes sample
    local state; edges represent local conductivity.  Activity is updated
    synchronously so the internal motion behaves as a distributed wave rather
    than a token walking from node to node.
    """

    def __init__(self, config: WaveConfig | None = None) -> None:
        self.config = config or WaveConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.positions = self._fibonacci_sphere(self.config.node_count)
        self.adjacency = self._build_local_adjacency(
            self.positions,
            self.config.neighbors_per_node,
        )
        self.conductivity = self._initial_conductivity(self.adjacency)

        self.activity = np.zeros(self.config.node_count, dtype=float)
        self.previous_activity = np.zeros(self.config.node_count, dtype=float)
        self.fatigue = np.zeros(self.config.node_count, dtype=float)
        self.step_index = 0

    @staticmethod
    def _fibonacci_sphere(count: int) -> np.ndarray:
        index = np.arange(count, dtype=float)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        y = 1.0 - (2.0 * index / max(count - 1, 1))
        radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        theta = golden_angle * index
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.column_stack((x, y, z))

    @staticmethod
    def _build_local_adjacency(positions: np.ndarray, neighbors: int) -> np.ndarray:
        delta = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distances, np.inf)

        adjacency = np.zeros((len(positions), len(positions)), dtype=bool)
        for node_id in range(len(positions)):
            nearest = np.argpartition(distances[node_id], neighbors)[:neighbors]
            adjacency[node_id, nearest] = True

        # A local field should not depend on which endpoint selected the edge.
        return np.logical_or(adjacency, adjacency.T)

    def _initial_conductivity(self, adjacency: np.ndarray) -> np.ndarray:
        conductivity = np.zeros(adjacency.shape, dtype=float)
        conductivity[adjacency] = 0.42
        return conductivity

    def reset_activity(self) -> None:
        """Clear short-term state without erasing learned terrain."""

        self.activity.fill(0.0)
        self.previous_activity.fill(0.0)
        self.fatigue.fill(0.0)
        self.step_index = 0

    def reset_terrain(self) -> None:
        """Return conductivity to its unexperienced state."""

        self.conductivity = self._initial_conductivity(self.adjacency)

    def stimulus_region(self, anchor: int, radius: int = 2) -> tuple[int, ...]:
        """Return an anchor and nearby observation points as one stimulus."""

        if not 0 <= anchor < self.config.node_count:
            raise ValueError(f"anchor out of range: {anchor}")

        distances = np.linalg.norm(self.positions - self.positions[anchor], axis=1)
        selected = np.argsort(distances)[: max(1, radius + 1)]
        return tuple(int(value) for value in selected)

    def stimulate(self, node_ids: Iterable[int], strength: float = 1.0) -> None:
        ids = np.asarray(tuple(node_ids), dtype=int)
        if ids.size == 0:
            raise ValueError("stimulus must contain at least one node")
        if np.any(ids < 0) or np.any(ids >= self.config.node_count):
            raise ValueError("stimulus contains an invalid node id")

        # A distributed input is normalized so larger regions do not receive
        # more total energy merely because they contain more observation nodes.
        self.activity[ids] += float(strength) / math.sqrt(float(ids.size))

    def _activity_center(self) -> tuple[float, float, float]:
        total = float(np.sum(self.activity))
        if total <= 1e-12:
            return (0.0, 0.0, 0.0)
        center = np.sum(self.positions * self.activity[:, None], axis=0) / total
        return tuple(float(value) for value in center)

    def _plasticity_update(self) -> list[tuple[int, int, float]]:
        """Change terrain from actual temporal overlap, not target labels.

        Previous source activity multiplied by current target activity creates
        a weak directional tendency.  No desired successor is supplied.
        """

        source = self.previous_activity[:, None]
        target = self.activity[None, :]
        delta = self.config.learning_rate * source * target
        delta *= self.adjacency

        changed = np.argwhere(delta > 1e-8)
        self.conductivity += delta
        np.clip(
            self.conductivity,
            self.config.conductivity_min,
            self.config.conductivity_max,
            out=self.conductivity,
        )
        self.conductivity[~self.adjacency] = 0.0

        return [
            (int(a), int(b), float(delta[a, b]))
            for a, b in changed
        ]

    def step(self, learn: bool = False) -> tuple[WaveSnapshot, list[tuple[int, int, float]]]:
        current = self.activity.copy()

        # Normalize by each receiver's incoming conductivity.  This prevents
        # graph degree from becoming an accidental energy source.
        weighted = current[:, None] * self.conductivity
        incoming = np.sum(weighted, axis=0)
        normalizer = np.sum(self.conductivity, axis=0)
        incoming = np.divide(
            incoming,
            normalizer,
            out=np.zeros_like(incoming),
            where=normalizer > 0,
        )

        next_activity = (
            current * self.config.persistence
            + incoming * self.config.propagation_rate
            - self.fatigue
        )
        next_activity = np.maximum(next_activity, 0.0)
        next_activity *= self.config.decay

        fired = np.flatnonzero(next_activity >= self.config.fire_threshold)
        if fired.size:
            # Localized events remain part of the model, but only feed a small
            # amount back into the field.  They do not dictate the destination.
            next_activity[fired] += self.config.fire_gain

        next_fatigue = (
            self.fatigue * self.config.fatigue_recovery
            + current * self.config.fatigue_gain
        )

        self.previous_activity = current
        self.activity = next_activity
        self.fatigue = next_fatigue
        self.step_index += 1

        changes = self._plasticity_update() if learn else []
        snapshot = WaveSnapshot(
            step=self.step_index,
            activity=self.activity.copy(),
            fatigue=self.fatigue.copy(),
            fired_nodes=tuple(int(value) for value in fired),
            total_activity=float(np.sum(self.activity)),
            center=self._activity_center(),
        )
        return snapshot, changes

    def run_until_quiet(
        self,
        name: str,
        learn: bool = False,
        minimum_steps: int = 3,
    ) -> ExperimentTrace:
        trace = ExperimentTrace(name=name)

        for _ in range(self.config.max_steps):
            snapshot, changes = self.step(learn=learn)
            trace.snapshots.append(snapshot)
            trace.changed_edges.extend(changes)

            if (
                snapshot.step >= minimum_steps
                and snapshot.total_activity < self.config.quiet_threshold
            ):
                break

        return trace

    def advance(self, steps: int, learn: bool = False, name: str = "advance") -> ExperimentTrace:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        trace = ExperimentTrace(name=name)
        for _ in range(steps):
            snapshot, changes = self.step(learn=learn)
            trace.snapshots.append(snapshot)
            trace.changed_edges.extend(changes)
        return trace
