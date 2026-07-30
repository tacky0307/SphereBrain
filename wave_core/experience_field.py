from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attractor import AttractorConfig, AttractorSphereCore


@dataclass(frozen=True)
class ExperienceFieldConfig:
    """Whole-experience eligibility and experience-guided propagation.

    ``experience_trace`` is the short-lived field active during one explicit
    experience. At the end of the experience, part of that field is deposited
    into ``experience_memory``. The persistent memory does not prescribe a
    target or replay a stored answer; it only makes previously experienced
    regions slightly easier to activate during later propagation.
    """

    trace_decay: float = 0.92
    trace_gain: float = 1.0
    directional_learning_rate: float = 0.0010
    capacity_learning_rate: float = 0.0006
    trace_cap: float = 3.0

    memory_decay: float = 0.995
    memory_gain: float = 0.05
    memory_cap: float = 10.0

    propagation_bias_gain: float = 0.30
    propagation_bias_max: float = 1.50


class ExperienceFieldAttractorCore(AttractorSphereCore):
    """AttractorSphereCore with transient and persistent experience fields."""

    def __init__(
        self,
        config: AttractorConfig | None = None,
        field_config: ExperienceFieldConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.field_config = field_config or ExperienceFieldConfig()
        self.experience_trace = np.zeros(self.config.node_count, dtype=float)
        self.experience_memory = np.zeros(self.config.node_count, dtype=float)
        self.experience_open = False
        self.experience_step = 0

    def clone(self) -> "ExperienceFieldAttractorCore":
        other = ExperienceFieldAttractorCore(self.config, self.field_config)
        other.direction = self.direction.copy()
        other.capacity = self.capacity.copy()
        other.experience_trace = self.experience_trace.copy()
        other.experience_memory = self.experience_memory.copy()
        other.experience_open = self.experience_open
        other.experience_step = self.experience_step
        return other

    def reset_activity(self) -> None:
        """Clear dynamic state without erasing learned experience memory."""

        super().reset_activity()
        self.experience_trace.fill(0.0)
        self.experience_open = False
        self.experience_step = 0

    def reset_experience_memory(self) -> None:
        """Erase persistent experience guidance without changing edge terrain."""

        self.experience_trace.fill(0.0)
        self.experience_memory.fill(0.0)
        self.experience_open = False
        self.experience_step = 0

    def begin_experience(self) -> None:
        """Open a new unlabeled experience and clear its transient field."""

        self.experience_trace.fill(0.0)
        self.experience_open = True
        self.experience_step = 0

    def end_experience(self) -> np.ndarray:
        """Close the experience and deposit its field into persistent memory."""

        field = self.experience_trace.copy()
        cfg = self.field_config
        self.experience_memory *= cfg.memory_decay
        self.experience_memory += cfg.memory_gain * field
        np.clip(
            self.experience_memory,
            0.0,
            cfg.memory_cap,
            out=self.experience_memory,
        )
        self.experience_open = False
        return field

    def _update_experience_trace(self) -> None:
        if not self.experience_open:
            return

        cfg = self.field_config
        self.experience_trace *= cfg.trace_decay
        self.experience_trace += (
            cfg.trace_gain * np.maximum(self.previous_activity, 0.0)
        )
        np.clip(
            self.experience_trace,
            0.0,
            cfg.trace_cap,
            out=self.experience_trace,
        )
        self.experience_step += 1

    def experience_bias(self) -> np.ndarray:
        """Return a bounded node-wise propagation multiplier."""

        memory = np.maximum(self.experience_memory, 0.0)
        maximum = float(np.max(memory))

        if maximum <= 1e-12:
            return np.ones(self.config.node_count, dtype=float)

        normalized = memory / maximum
        bias = 1.0 + self.field_config.propagation_bias_gain * normalized
        np.clip(
            bias,
            1.0,
            self.field_config.propagation_bias_max,
            out=bias,
        )
        return bias

    def _plasticity_update(self) -> None:
        super()._plasticity_update()

        if not self.experience_open:
            return

        self._update_experience_trace()
        cfg = self.field_config

        forward = self.experience_trace[:, None] * self.activity[None, :]
        coactive = np.sqrt(
            np.maximum(self.experience_trace[:, None], 0.0)
            * np.maximum(self.activity[None, :], 0.0)
        )

        self.direction += (
            cfg.directional_learning_rate * forward * self.adjacency
        )
        self.capacity += (
            cfg.capacity_learning_rate * coactive * self.adjacency
        )

        np.clip(
            self.direction,
            self.config.direction_min,
            self.config.direction_max,
            out=self.direction,
        )
        np.clip(
            self.capacity,
            self.config.capacity_min,
            self.config.capacity_max,
            out=self.capacity,
        )
        self.direction[~self.adjacency] = 0.0
        self.capacity[~self.adjacency] = 0.0
