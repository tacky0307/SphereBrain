from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attractor import AttractorConfig, AttractorSphereCore


@dataclass(frozen=True)
class ExperienceFieldConfig:
    """Short-lived whole-experience eligibility for the attractor core.

    The field contains no label, target, prescribed path, or answer. It only
    carries recent internal activity forward so that activity separated in time
    can alter existing local edges while one experience is open.
    """

    trace_decay: float = 0.92
    trace_gain: float = 1.0
    directional_learning_rate: float = 0.0010
    capacity_learning_rate: float = 0.0006
    trace_cap: float = 3.0


class ExperienceFieldAttractorCore(AttractorSphereCore):
    """AttractorSphereCore with an optional transient experience field."""

    def __init__(
        self,
        config: AttractorConfig | None = None,
        field_config: ExperienceFieldConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.field_config = field_config or ExperienceFieldConfig()
        self.experience_trace = np.zeros(self.config.node_count, dtype=float)
        self.experience_open = False
        self.experience_step = 0

    def clone(self) -> "ExperienceFieldAttractorCore":
        other = ExperienceFieldAttractorCore(self.config, self.field_config)
        other.direction = self.direction.copy()
        other.capacity = self.capacity.copy()
        other.experience_trace = self.experience_trace.copy()
        other.experience_open = self.experience_open
        other.experience_step = self.experience_step
        return other

    def reset_activity(self) -> None:
        super().reset_activity()
        self.experience_trace.fill(0.0)
        self.experience_open = False
        self.experience_step = 0

    def begin_experience(self) -> None:
        """Open a new unlabeled experience and clear only its transient field."""
        self.experience_trace.fill(0.0)
        self.experience_open = True
        self.experience_step = 0

    def end_experience(self) -> np.ndarray:
        """Close the experience and return a copy of its final transient field."""
        field = self.experience_trace.copy()
        self.experience_open = False
        return field

    def _update_experience_trace(self) -> None:
        if not self.experience_open:
            return
        cfg = self.field_config
        self.experience_trace *= cfg.trace_decay
        self.experience_trace += cfg.trace_gain * np.maximum(self.previous_activity, 0.0)
        np.clip(self.experience_trace, 0.0, cfg.trace_cap, out=self.experience_trace)
        self.experience_step += 1

    def _plasticity_update(self) -> None:
        # Keep the original immediate local plasticity and homeostasis.
        super()._plasticity_update()
        if not self.experience_open:
            return

        self._update_experience_trace()
        cfg = self.field_config

        # Earlier activity carried by the field can influence currently active
        # neighboring nodes. Existing adjacency still strictly limits learning.
        forward = self.experience_trace[:, None] * self.activity[None, :]
        coactive = np.sqrt(
            np.maximum(self.experience_trace[:, None], 0.0)
            * np.maximum(self.activity[None, :], 0.0)
        )

        self.direction += cfg.directional_learning_rate * forward * self.adjacency
        self.capacity += cfg.capacity_learning_rate * coactive * self.adjacency

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
