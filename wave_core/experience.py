from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .core import SphereWaveCore, WaveSnapshot


@dataclass(frozen=True)
class ExperienceConfig:
    """Slow learning parameters applied after one complete experience.

    This layer does not replace the step-level plasticity in ``SphereWaveCore``.
    It adds a longer time scale by reflecting a complete activity history back
    onto existing local edges. No target edge or prescribed route is supplied.
    """

    learning_rate: float = 0.00035
    temporal_decay: float = 0.94
    temporal_weight: float = 0.65
    terrain_weight: float = 0.35
    spatial_passes: int = 8
    spatial_decay: float = 0.72
    minimum_delta: float = 1e-10


@dataclass(frozen=True)
class StimulusEvent:
    label: str
    step: int
    strength: float
    node_ids: tuple[int, ...]


@dataclass
class ExperienceSummary:
    name: str
    steps: int
    stimulus_events: tuple[StimulusEvent, ...]
    cumulative_activity: np.ndarray
    peak_activity: np.ndarray
    first_active_step: np.ndarray
    last_active_step: np.ndarray
    temporal_credit: np.ndarray
    activity_integral: float


@dataclass
class ExperienceReflection:
    changed_edges: list[tuple[int, int, float]] = field(default_factory=list)
    total_change: float = 0.0
    max_change: float = 0.0
    temporal_component: float = 0.0
    terrain_component: float = 0.0


class ExperienceBuffer:
    """Collect one continuous experience without changing the core API."""

    def __init__(self, name: str, node_count: int, active_threshold: float = 1e-8) -> None:
        self.name = name
        self.node_count = int(node_count)
        self.active_threshold = float(active_threshold)
        self._activities: list[np.ndarray] = []
        self._events: list[StimulusEvent] = []

    @property
    def steps(self) -> int:
        return len(self._activities)

    def mark_stimulus(
        self,
        label: str,
        node_ids: Iterable[int],
        strength: float,
        step: int | None = None,
    ) -> None:
        event_step = self.steps if step is None else int(step)
        self._events.append(
            StimulusEvent(
                label=str(label),
                step=event_step,
                strength=float(strength),
                node_ids=tuple(int(value) for value in node_ids),
            )
        )

    def record(self, snapshot: WaveSnapshot) -> None:
        activity = np.asarray(snapshot.activity, dtype=float)
        if activity.shape != (self.node_count,):
            raise ValueError(
                f"snapshot activity shape {activity.shape} does not match node_count {self.node_count}"
            )
        self._activities.append(activity.copy())

    def extend(self, snapshots: Iterable[WaveSnapshot]) -> None:
        for snapshot in snapshots:
            self.record(snapshot)

    def summarize(self, config: ExperienceConfig | None = None) -> ExperienceSummary:
        cfg = config or ExperienceConfig()
        if not self._activities:
            zeros = np.zeros(self.node_count, dtype=float)
            return ExperienceSummary(
                name=self.name,
                steps=0,
                stimulus_events=tuple(self._events),
                cumulative_activity=zeros.copy(),
                peak_activity=zeros.copy(),
                first_active_step=np.full(self.node_count, -1, dtype=int),
                last_active_step=np.full(self.node_count, -1, dtype=int),
                temporal_credit=np.zeros((self.node_count, self.node_count), dtype=float),
                activity_integral=0.0,
            )

        activity = np.stack(self._activities, axis=0)
        cumulative = np.sum(activity, axis=0)
        peak = np.max(activity, axis=0)
        active = activity > self.active_threshold

        first = np.full(self.node_count, -1, dtype=int)
        last = np.full(self.node_count, -1, dtype=int)
        ever_active = np.any(active, axis=0)
        first[ever_active] = np.argmax(active[:, ever_active], axis=0)
        last[ever_active] = activity.shape[0] - 1 - np.argmax(
            active[::-1, ever_active], axis=0
        )

        # Eligibility carries earlier activity forward through the experience.
        # The matrix is not yet an edge update; adjacency is applied later.
        eligibility = np.zeros(self.node_count, dtype=float)
        temporal_credit = np.zeros((self.node_count, self.node_count), dtype=float)
        for current in activity:
            eligibility = eligibility * cfg.temporal_decay + current
            temporal_credit += eligibility[:, None] * current[None, :]

        return ExperienceSummary(
            name=self.name,
            steps=activity.shape[0],
            stimulus_events=tuple(self._events),
            cumulative_activity=cumulative,
            peak_activity=peak,
            first_active_step=first,
            last_active_step=last,
            temporal_credit=temporal_credit,
            activity_integral=float(np.sum(activity)),
        )


class ExperienceReflector:
    """Reflect a complete experience onto the existing conductivity terrain.

    Only existing local edges can change. The reflector never creates a direct
    A-to-C edge and never receives a desired route. A softly diffused activity
    field allows one experience to leave a broad terrain trace, while temporal
    eligibility preserves ordering information from the original wave motion.
    """

    def __init__(self, config: ExperienceConfig | None = None) -> None:
        self.config = config or ExperienceConfig()

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        maximum = float(np.max(values)) if values.size else 0.0
        if maximum <= 1e-15:
            return np.zeros_like(values)
        return values / maximum

    def _diffused_field(self, core: SphereWaveCore, values: np.ndarray) -> np.ndarray:
        field = self._normalize(np.asarray(values, dtype=float))
        accumulated = field.copy()
        current = field.copy()
        degree = np.sum(core.adjacency, axis=1)

        for pass_index in range(1, self.config.spatial_passes + 1):
            neighbor_sum = core.adjacency @ current
            current = np.divide(
                neighbor_sum,
                degree,
                out=np.zeros_like(neighbor_sum),
                where=degree > 0,
            )
            accumulated += (self.config.spatial_decay ** pass_index) * current

        return self._normalize(accumulated)

    def reflect(
        self,
        core: SphereWaveCore,
        summary: ExperienceSummary,
    ) -> ExperienceReflection:
        if summary.steps == 0:
            return ExperienceReflection()

        adjacency = core.adjacency
        temporal = self._normalize(summary.temporal_credit)
        temporal *= adjacency

        field = self._diffused_field(core, summary.cumulative_activity)
        terrain = np.sqrt(field[:, None] * field[None, :])
        terrain *= adjacency

        delta = self.config.learning_rate * (
            self.config.temporal_weight * temporal
            + self.config.terrain_weight * terrain
        )
        delta *= adjacency

        old = core.conductivity.copy()
        core.conductivity += delta
        np.clip(
            core.conductivity,
            core.config.conductivity_min,
            core.config.conductivity_max,
            out=core.conductivity,
        )
        core.conductivity[~adjacency] = 0.0
        applied = core.conductivity - old

        changed = np.argwhere(np.abs(applied) > self.config.minimum_delta)
        changes = [
            (int(source), int(target), float(applied[source, target]))
            for source, target in changed
        ]

        return ExperienceReflection(
            changed_edges=changes,
            total_change=float(np.sum(np.abs(applied))),
            max_change=float(np.max(np.abs(applied))) if applied.size else 0.0,
            temporal_component=float(np.sum(temporal)),
            terrain_component=float(np.sum(terrain)),
        )
