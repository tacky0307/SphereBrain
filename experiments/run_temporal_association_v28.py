from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalAssociationConfig:
    """Configuration for the first v28 temporal-association experiment."""

    node_count: int = 24
    trace_decay: float = 0.72
    learning_rate: float = 0.18
    weight_decay: float = 0.0005
    activation_threshold: float = 0.05
    propagation_gain: float = 1.0
    max_weight: float = 1.0


class TemporalAssociationCore:
    """Minimal directed core where fading activity traces shape future flow.

    This experiment deliberately keeps Trace outside the architecture. The only
    persistent memory is the directed weight matrix inside the Core itself.
    """

    def __init__(
        self,
        config: TemporalAssociationConfig | None = None,
        seed: int = 28,
    ) -> None:
        self.config = config or TemporalAssociationConfig()
        self.rng = np.random.default_rng(seed)
        n = self.config.node_count

        self.activity = np.zeros(n, dtype=float)
        self.temporal_trace = np.zeros(n, dtype=float)
        self.weights = np.zeros((n, n), dtype=float)
        self.usage = np.zeros((n, n), dtype=int)
        self.step_index = 0

    def reset_activity(self, *, keep_trace: bool = True) -> None:
        self.activity.fill(0.0)
        if not keep_trace:
            self.temporal_trace.fill(0.0)

    def stimulate(
        self,
        nodes: Iterable[int],
        strength: float = 1.0,
        *,
        learn: bool = True,
    ) -> np.ndarray:
        """Apply an external experience and optionally learn from prior traces."""

        current = np.zeros_like(self.activity)
        for node in nodes:
            self._validate_node(node)
            current[node] = max(current[node], float(strength))

        if learn:
            self._learn_from_temporal_trace(current)

        self.activity = current
        self.temporal_trace = np.maximum(
            self.temporal_trace * self.config.trace_decay,
            current,
        )
        self.step_index += 1
        return self.activity.copy()

    def quiet_step(self) -> None:
        """Advance time without an external stimulus while preserving its echo."""

        self.activity.fill(0.0)
        self.temporal_trace *= self.config.trace_decay
        self.step_index += 1

    def recall(self, source_nodes: Iterable[int], steps: int = 3) -> np.ndarray:
        """Propagate through learned directed paths without changing the Core."""

        activation = np.zeros_like(self.activity)
        for node in source_nodes:
            self._validate_node(node)
            activation[node] = 1.0

        for _ in range(steps):
            transmitted = activation @ self.weights
            next_activation = np.clip(
                transmitted * self.config.propagation_gain,
                0.0,
                1.0,
            )
            next_activation[next_activation < self.config.activation_threshold] = 0.0
            activation = np.maximum(activation * 0.18, next_activation)

        return activation

    def association_strength(self, source: int, target: int) -> float:
        self._validate_node(source)
        self._validate_node(target)
        return float(self.weights[source, target])

    def _learn_from_temporal_trace(self, current: np.ndarray) -> None:
        past = self.temporal_trace.copy()
        if not np.any(past) or not np.any(current):
            return

        delta = self.config.learning_rate * np.outer(past, current)
        np.fill_diagonal(delta, 0.0)

        self.weights *= 1.0 - self.config.weight_decay
        self.weights += delta * (self.config.max_weight - self.weights)
        self.weights = np.clip(self.weights, 0.0, self.config.max_weight)
        self.usage += (delta > 0.0).astype(int)

    def _validate_node(self, node: int) -> None:
        if not 0 <= int(node) < self.config.node_count:
            raise IndexError(f"node {node} is outside the Core")


def train_sequence(
    core: TemporalAssociationCore,
    source: int,
    target: int,
    *,
    gap_steps: int,
    repetitions: int,
) -> None:
    for _ in range(repetitions):
        core.reset_activity(keep_trace=False)
        core.stimulate([source], learn=True)
        for _ in range(gap_steps):
            core.quiet_step()
        core.stimulate([target], learn=True)


def run_experiment() -> dict[str, float]:
    a, b, c = 2, 11, 17
    core = TemporalAssociationCore()

    before = core.recall([a], steps=2)
    train_sequence(core, a, b, gap_steps=2, repetitions=24)
    train_sequence(core, a, c, gap_steps=5, repetitions=24)
    after = core.recall([a], steps=2)

    result = {
        "before_b": float(before[b]),
        "after_b": float(after[b]),
        "after_c": float(after[c]),
        "weight_a_to_b": core.association_strength(a, b),
        "weight_b_to_a": core.association_strength(b, a),
        "weight_a_to_c": core.association_strength(a, c),
    }

    print("SphereBrain v28 — Temporal Association")
    print(f"A -> B weight: {result['weight_a_to_b']:.6f}")
    print(f"B -> A weight: {result['weight_b_to_a']:.6f}")
    print(f"A -> C weight: {result['weight_a_to_c']:.6f}")
    print(f"Recall B before learning: {result['before_b']:.6f}")
    print(f"Recall B after learning:  {result['after_b']:.6f}")
    print(f"Recall C after learning:  {result['after_c']:.6f}")

    assert result["after_b"] > result["before_b"]
    assert result["weight_a_to_b"] > result["weight_b_to_a"]
    assert result["weight_a_to_b"] > result["weight_a_to_c"]

    print("PASS: separated activity formed a directed, time-sensitive path.")
    return result


if __name__ == "__main__":
    run_experiment()
