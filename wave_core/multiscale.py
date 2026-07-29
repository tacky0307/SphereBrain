from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SphereWaveCore


@dataclass(frozen=True)
class MultiScaleConfig:
    """Apply the same local growth rule to larger temporal activity units."""

    learning_rate: float = 0.00028
    window_steps: int = 4
    bridge_passes: int = 24
    diffusion_decay: float = 0.84
    interval_span: int = 2
    directional_weight: float = 0.55
    corridor_weight: float = 0.45
    minimum_activity: float = 1e-10
    minimum_delta: float = 1e-10


@dataclass
class MultiScaleReflection:
    changed_edges: list[tuple[int, int, float]] = field(default_factory=list)
    interval_count: int = 0
    total_change: float = 0.0
    max_change: float = 0.0
    corridor_mass: float = 0.0
    corridor_peak: float = 0.0


class MultiScaleExperienceReflector:
    """Grow local terrain from transitions between larger activity windows.

    The experience is divided into temporal windows. Each window becomes one
    macro activity unit. Consecutive macro units exert the same kind of
    before/after growth pressure as the Core's local plasticity, but the
    pressure is diffused through existing local adjacency before it is applied.
    No destination label, prescribed route, direct A-to-C edge, or new edge is
    supplied.
    """

    def __init__(self, config: MultiScaleConfig | None = None) -> None:
        self.config = config or MultiScaleConfig()

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        maximum = float(np.max(values)) if values.size else 0.0
        if maximum <= 1e-15:
            return np.zeros_like(values)
        return values / maximum

    def _window_fields(self, activity_history: np.ndarray) -> list[np.ndarray]:
        history = np.asarray(activity_history, dtype=float)
        if history.ndim != 2:
            raise ValueError("activity_history must have shape (steps, nodes)")

        fields: list[np.ndarray] = []
        size = max(1, self.config.window_steps)
        for start in range(0, history.shape[0], size):
            field = np.sum(history[start : start + size], axis=0)
            if float(np.sum(field)) > self.config.minimum_activity:
                fields.append(self._normalize(field))
        return fields

    def _reach(self, core: SphereWaveCore, seed: np.ndarray) -> np.ndarray:
        current = self._normalize(seed)
        accumulated = current.copy()
        degree = np.sum(core.adjacency, axis=1)

        for pass_index in range(1, self.config.bridge_passes + 1):
            neighbor_sum = core.adjacency @ current
            current = np.divide(
                neighbor_sum,
                degree,
                out=np.zeros_like(neighbor_sum),
                where=degree > 0,
            )
            accumulated += (self.config.diffusion_decay ** pass_index) * current
        return self._normalize(accumulated)

    def reflect(
        self,
        core: SphereWaveCore,
        activity_history: np.ndarray,
    ) -> MultiScaleReflection:
        fields = self._window_fields(activity_history)
        if len(fields) < 2:
            return MultiScaleReflection()
        if activity_history.shape[1] != core.config.node_count:
            raise ValueError("activity_history node count does not match core")

        delta = np.zeros_like(core.conductivity)
        interval_count = 0
        corridor_mass = 0.0
        corridor_peak = 0.0

        for source_index, source_field in enumerate(fields[:-1]):
            maximum_gap = min(self.config.interval_span, len(fields) - source_index - 1)
            for gap in range(1, maximum_gap + 1):
                target_field = fields[source_index + gap]
                source_reach = self._reach(core, source_field)
                target_reach = self._reach(core, target_field)

                corridor = self._normalize(np.sqrt(source_reach * target_reach))
                local_corridor = np.sqrt(corridor[:, None] * corridor[None, :])
                directional = source_reach[:, None] * target_reach[None, :]

                pair_delta = (
                    self.config.directional_weight * directional
                    + self.config.corridor_weight * local_corridor
                )
                pair_delta *= core.adjacency
                pair_delta *= self.config.learning_rate / float(gap)
                delta += pair_delta

                interval_count += 1
                corridor_mass += float(np.sum(corridor))
                corridor_peak = max(corridor_peak, float(np.max(corridor)))

        if interval_count:
            delta /= float(interval_count)

        old = core.conductivity.copy()
        core.conductivity += delta
        np.clip(
            core.conductivity,
            core.config.conductivity_min,
            core.config.conductivity_max,
            out=core.conductivity,
        )
        core.conductivity[~core.adjacency] = 0.0
        applied = core.conductivity - old

        changed = np.argwhere(np.abs(applied) > self.config.minimum_delta)
        changes = [
            (int(source), int(target), float(applied[source, target]))
            for source, target in changed
        ]
        return MultiScaleReflection(
            changed_edges=changes,
            interval_count=interval_count,
            total_change=float(np.sum(np.abs(applied))),
            max_change=float(np.max(np.abs(applied))) if applied.size else 0.0,
            corridor_mass=corridor_mass,
            corridor_peak=corridor_peak,
        )
