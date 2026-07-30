from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

import numpy as np

from .core import SphereWaveCore, WaveConfig


@dataclass(frozen=True)
class AttractorConfig:
    """Dynamics for experience-shaped spontaneous state formation.

    Direction and capacity remain separate. Stable states must arise from local
    recurrent excitation balanced by fatigue, global inhibition, and a
    graph-distance surround-inhibition ring. No target, winner, replay path, or
    prescribed answer is supplied.
    """

    node_count: int = 240
    neighbors_per_node: int = 8
    seed: int = 27

    persistence: float = 0.42
    propagation_gain: float = 1.10
    recurrent_gain: float = 0.56
    local_inhibition_gain: float = 0.05
    surround_inhibition_gain: float = 0.32
    global_inhibition_gain: float = 0.10
    fatigue_gain: float = 0.045
    fatigue_recovery: float = 0.90
    decay: float = 0.985

    surround_min_hops: int = 2
    surround_max_hops: int = 3

    response_gain: float = 2.2
    activity_cap: float = 1.0
    quiet_threshold: float = 0.01
    meaningful_threshold: float = 0.025
    max_steps: int = 120

    direction_learning_rate: float = 0.0020
    capacity_learning_rate: float = 0.0012
    homeostasis_rate: float = 0.0008
    direction_min: float = 0.05
    direction_max: float = 2.0
    capacity_min: float = 0.20
    capacity_max: float = 2.4


@dataclass
class AttractorSnapshot:
    step: int
    activity: np.ndarray
    fatigue: np.ndarray
    excitation: np.ndarray
    inhibition: np.ndarray
    total_activity: float
    active_count: int
    center: tuple[float, float, float]


@dataclass
class AttractorTrace:
    name: str
    snapshots: list[AttractorSnapshot] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.snapshots)

    @property
    def lifetime(self) -> int:
        if not self.snapshots:
            return 0
        return max(
            (snapshot.step for snapshot in self.snapshots if snapshot.total_activity > 0.01),
            default=0,
        )

    @property
    def final_pattern(self) -> np.ndarray:
        if not self.snapshots:
            return np.empty(0, dtype=float)
        tail = self.snapshots[-min(5, len(self.snapshots)) :]
        return np.mean(np.stack([item.activity for item in tail], axis=0), axis=0)


class AttractorSphereCore:
    """A recurrent sphere whose stable states can be shaped by experience."""

    def __init__(self, config: AttractorConfig | None = None) -> None:
        self.config = config or AttractorConfig()

        geometry = SphereWaveCore(
            WaveConfig(
                node_count=self.config.node_count,
                neighbors_per_node=self.config.neighbors_per_node,
                seed=self.config.seed,
            )
        )
        self.positions = geometry.positions.copy()
        self.adjacency = geometry.adjacency.copy()
        self.surround_adjacency = self._build_surround_adjacency()

        self.direction = np.zeros_like(geometry.conductivity)
        self.direction[self.adjacency] = 1.0
        self.capacity = np.zeros_like(geometry.conductivity)
        self.capacity[self.adjacency] = 0.42

        self.activity = np.zeros(self.config.node_count, dtype=float)
        self.previous_activity = np.zeros(self.config.node_count, dtype=float)
        self.fatigue = np.zeros(self.config.node_count, dtype=float)
        self.step_index = 0

    def _build_surround_adjacency(self) -> np.ndarray:
        """Return nodes lying in the configured graph-distance inhibition ring."""
        node_count = self.config.node_count
        direct = self.adjacency.astype(bool)
        visited = np.eye(node_count, dtype=bool)
        frontier = np.eye(node_count, dtype=bool)
        ring = np.zeros((node_count, node_count), dtype=bool)

        for hop in range(1, self.config.surround_max_hops + 1):
            frontier = (frontier.astype(np.int8) @ direct.astype(np.int8)) > 0
            frontier &= ~visited
            visited |= frontier
            if hop >= self.config.surround_min_hops:
                ring |= frontier

        ring &= ~direct
        np.fill_diagonal(ring, False)
        return ring

    def clone(self) -> "AttractorSphereCore":
        other = AttractorSphereCore(self.config)
        other.direction = self.direction.copy()
        other.capacity = self.capacity.copy()
        return other

    def reset_activity(self) -> None:
        self.activity.fill(0.0)
        self.previous_activity.fill(0.0)
        self.fatigue.fill(0.0)
        self.step_index = 0

    def stimulus_region(self, anchor: int, radius: int = 2) -> tuple[int, ...]:
        distances = np.linalg.norm(self.positions - self.positions[int(anchor)], axis=1)
        selected = np.argsort(distances)[: max(1, radius + 1)]
        return tuple(int(value) for value in selected)

    def stimulate(self, node_ids: Iterable[int], strength: float = 1.0) -> None:
        ids = np.asarray(tuple(node_ids), dtype=int)
        if ids.size == 0:
            raise ValueError("stimulus must contain at least one node")
        self.activity[ids] += float(strength) / math.sqrt(float(ids.size))
        np.clip(self.activity, 0.0, self.config.activity_cap, out=self.activity)

    @staticmethod
    def _saturate(values: np.ndarray, gain: float) -> np.ndarray:
        positive = np.maximum(values, 0.0)
        return 1.0 - np.exp(-gain * positive)

    def _activity_center(self) -> tuple[float, float, float]:
        total = float(np.sum(self.activity))
        if total <= 1e-12:
            return (0.0, 0.0, 0.0)
        center = np.sum(self.positions * self.activity[:, None], axis=0) / total
        return tuple(float(value) for value in center)

    def _direction_probabilities(self) -> np.ndarray:
        weighted = self.direction * self.adjacency
        totals = np.sum(weighted, axis=1, keepdims=True)
        return np.divide(weighted, totals, out=np.zeros_like(weighted), where=totals > 0)

    @staticmethod
    def _mean_connected_activity(current: np.ndarray, mask: np.ndarray) -> np.ndarray:
        degree = np.sum(mask, axis=0)
        connected_activity = current @ mask
        return np.divide(
            connected_activity,
            degree,
            out=np.zeros_like(connected_activity),
            where=degree > 0,
        )

    def _propagation_bias(self) -> np.ndarray:
        """Return an optional node-wise multiplier for propagated activity.

        The base attractor has no experience guidance and therefore returns
        ones. Subclasses may expose ``experience_bias()`` to shape propagation
        without modifying direction or capacity themselves.
        """

        provider = getattr(self, "experience_bias", None)
        if provider is None:
            return np.ones(self.config.node_count, dtype=float)

        bias = np.asarray(provider(), dtype=float)
        expected_shape = (self.config.node_count,)
        if bias.shape != expected_shape:
            raise ValueError(
                "propagation bias must have shape "
                f"{expected_shape}, received {bias.shape}"
            )
        if not np.all(np.isfinite(bias)):
            raise ValueError("propagation bias must contain only finite values")
        if np.any(bias < 0.0):
            raise ValueError("propagation bias must not contain negative values")

        return bias

    def _plasticity_update(self) -> None:
        cfg = self.config
        temporal = self.previous_activity[:, None] * self.activity[None, :]
        coactive = np.sqrt(
            np.maximum(self.previous_activity[:, None], 0.0)
            * np.maximum(self.activity[None, :], 0.0)
        )

        self.direction += cfg.direction_learning_rate * temporal * self.adjacency
        self.capacity += cfg.capacity_learning_rate * coactive * self.adjacency

        incoming_capacity = np.sum(self.capacity, axis=0)
        baseline = 0.42 * np.sum(self.adjacency, axis=0)
        excess = np.maximum(incoming_capacity - baseline, 0.0)
        penalty = cfg.homeostasis_rate * excess[None, :] * self.adjacency
        unused = 1.0 - np.clip(temporal, 0.0, 1.0)
        self.capacity -= penalty * unused

        np.clip(self.direction, cfg.direction_min, cfg.direction_max, out=self.direction)
        np.clip(self.capacity, cfg.capacity_min, cfg.capacity_max, out=self.capacity)
        self.direction[~self.adjacency] = 0.0
        self.capacity[~self.adjacency] = 0.0

    def step(self, learn: bool = False) -> AttractorSnapshot:
        cfg = self.config
        current = self.activity.copy()

        # Experience guidance changes how strongly activity leaves each node,
        # while preserving learned direction probabilities and capacities.
        # An all-ones bias reproduces the original dynamics exactly.
        propagation_bias = self._propagation_bias()
        effective_current = current * propagation_bias
        effective_previous = self.previous_activity * propagation_bias

        direction = self._direction_probabilities()
        transmitted = effective_current[:, None] * direction * self.capacity
        feedforward = np.sum(transmitted, axis=0)

        recurrent = np.sum(
            effective_previous[:, None] * direction * self.capacity,
            axis=0,
        )
        excitation_raw = (
            cfg.propagation_gain * feedforward
            + cfg.recurrent_gain * recurrent
            + cfg.persistence * current
        )
        excitation = self._saturate(excitation_raw, cfg.response_gain)

        direct_activity = self._mean_connected_activity(current, self.adjacency)
        surround_activity = self._mean_connected_activity(current, self.surround_adjacency)
        local_inhibition = cfg.local_inhibition_gain * direct_activity
        surround_inhibition = cfg.surround_inhibition_gain * surround_activity
        global_inhibition = cfg.global_inhibition_gain * float(np.mean(current))
        inhibition = local_inhibition + surround_inhibition + global_inhibition

        next_activity = excitation - inhibition - self.fatigue
        next_activity = np.maximum(next_activity, 0.0)
        next_activity *= cfg.decay
        np.clip(next_activity, 0.0, cfg.activity_cap, out=next_activity)

        next_fatigue = self.fatigue * cfg.fatigue_recovery + current * cfg.fatigue_gain

        self.previous_activity = current
        self.activity = next_activity
        self.fatigue = next_fatigue
        self.step_index += 1

        if learn:
            self._plasticity_update()

        return AttractorSnapshot(
            step=self.step_index,
            activity=self.activity.copy(),
            fatigue=self.fatigue.copy(),
            excitation=excitation.copy(),
            inhibition=np.asarray(inhibition, dtype=float).copy(),
            total_activity=float(np.sum(self.activity)),
            active_count=int(np.sum(self.activity >= cfg.meaningful_threshold)),
            center=self._activity_center(),
        )

    def advance(self, steps: int, learn: bool = False, name: str = "advance") -> AttractorTrace:
        trace = AttractorTrace(name=name)
        for _ in range(int(steps)):
            trace.snapshots.append(self.step(learn=learn))
        return trace

    def run_until_settled(
        self,
        name: str,
        learn: bool = False,
        minimum_steps: int = 12,
        stability_window: int = 8,
        stability_tolerance: float = 0.002,
    ) -> AttractorTrace:
        trace = AttractorTrace(name=name)
        recent: list[np.ndarray] = []

        for _ in range(self.config.max_steps):
            snapshot = self.step(learn=learn)
            trace.snapshots.append(snapshot)
            recent.append(snapshot.activity)
            if len(recent) > stability_window:
                recent.pop(0)

            if snapshot.step < minimum_steps:
                continue
            if snapshot.total_activity < self.config.quiet_threshold:
                break
            if len(recent) == stability_window:
                movement = float(np.mean(np.abs(recent[-1] - recent[0])))
                if movement < stability_tolerance:
                    break

        return trace
