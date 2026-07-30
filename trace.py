"""Trace recording for SphereBrain v27 whole-brain formation.

Trace is an observer-side record of what actually happened inside the Core.
It does not decide what is important, alter learning, or interpret meaning.
Reflection can later replay selected frames as new internal experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


Array = np.ndarray


def _copy_array(value: Any, *, name: str, ndim: int | None = None) -> Array:
    """Return an owned finite float array suitable for immutable trace storage."""

    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, received {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True, slots=True)
class TraceFrame:
    """One immutable snapshot of whole-brain activity at an internal time.

    ``activity`` stores the whole experience pattern used by Reflection. For a
    compatible v27 Core this is the maximum activity reached by every node
    during propagation, not merely the final state after activity has decayed.
    """

    time_index: int
    source: str
    activity: Array
    previous_activity: Array
    fatigue: Array
    stimulus: Array | None = None
    direction: Array | None = None
    capacity: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative")
        if not self.source:
            raise ValueError("source must not be empty")

        activity = _copy_array(self.activity, name="activity", ndim=1)
        previous = _copy_array(
            self.previous_activity,
            name="previous_activity",
            ndim=1,
        )
        fatigue = _copy_array(self.fatigue, name="fatigue", ndim=1)

        if previous.shape != activity.shape:
            raise ValueError("previous_activity must match activity shape")
        if fatigue.shape != activity.shape:
            raise ValueError("fatigue must match activity shape")

        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "previous_activity", previous)
        object.__setattr__(self, "fatigue", fatigue)

        if self.stimulus is not None:
            stimulus = _copy_array(self.stimulus, name="stimulus", ndim=1)
            if stimulus.shape != activity.shape:
                raise ValueError("stimulus must match activity shape")
            object.__setattr__(self, "stimulus", stimulus)

        expected_matrix_shape = (activity.shape[0], activity.shape[0])

        if self.direction is not None:
            direction = _copy_array(self.direction, name="direction", ndim=2)
            if direction.shape != expected_matrix_shape:
                raise ValueError(
                    "direction must have shape "
                    f"{expected_matrix_shape}, received {direction.shape}"
                )
            object.__setattr__(self, "direction", direction)

        if self.capacity is not None:
            capacity = _copy_array(self.capacity, name="capacity", ndim=2)
            if capacity.shape != expected_matrix_shape:
                raise ValueError(
                    "capacity must have shape "
                    f"{expected_matrix_shape}, received {capacity.shape}"
                )
            object.__setattr__(self, "capacity", capacity)

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def node_count(self) -> int:
        return int(self.activity.shape[0])

    @property
    def total_activity(self) -> float:
        return float(np.sum(self.activity))

    def replay_signal(self, gain: float = 1.0, threshold: float = 0.0) -> Array:
        """Convert the recorded experience pattern into a reflection stimulus."""

        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")

        signal = self.activity.copy()
        signal[signal < threshold] = 0.0
        signal *= gain
        return signal


class TraceRecorder:
    """Append-only in-memory recorder for v27 internal time."""

    def __init__(self, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive or None")
        self.max_frames = max_frames
        self._frames: list[TraceFrame] = []
        self._next_time_index = 0

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterable[TraceFrame]:
        return iter(self._frames)

    @property
    def next_time_index(self) -> int:
        return self._next_time_index

    def clear(self, *, reset_time: bool = False) -> None:
        self._frames.clear()
        if reset_time:
            self._next_time_index = 0

    def frames(self) -> tuple[TraceFrame, ...]:
        return tuple(self._frames)

    def latest(self) -> TraceFrame:
        if not self._frames:
            raise LookupError("trace is empty")
        return self._frames[-1]

    def get(self, index: int) -> TraceFrame:
        return self._frames[index]

    def record(
        self,
        *,
        source: str,
        activity: Any,
        previous_activity: Any,
        fatigue: Any,
        stimulus: Any | None = None,
        direction: Any | None = None,
        capacity: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceFrame:
        frame = TraceFrame(
            time_index=self._next_time_index,
            source=source,
            activity=activity,
            previous_activity=previous_activity,
            fatigue=fatigue,
            stimulus=stimulus,
            direction=direction,
            capacity=capacity,
            metadata={} if metadata is None else metadata,
        )
        self._frames.append(frame)
        self._next_time_index += 1

        if self.max_frames is not None:
            overflow = len(self._frames) - self.max_frames
            if overflow > 0:
                del self._frames[:overflow]

        return frame

    def record_core(
        self,
        core: Any,
        *,
        source: str,
        stimulus: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_structure: bool = False,
    ) -> TraceFrame:
        """Capture a compatible Core without importing a specific Core class.

        v27 Cores expose ``peak_activity``. It is preferred because Reflection
        must replay the experienced whole-brain pattern even when final Core
        activity has already decayed to zero. Older Cores remain compatible and
        fall back to ``activity``.
        """

        required = ("activity", "previous_activity", "fatigue")
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise AttributeError(
                "core is missing required trace attributes: " + ", ".join(missing)
            )

        experience_pattern = getattr(core, "peak_activity", core.activity)

        direction = None
        capacity = None
        if include_structure:
            direction_provider = getattr(core, "_direction_probabilities", None)
            if callable(direction_provider):
                direction = direction_provider()
            elif hasattr(core, "direction"):
                direction = core.direction

            if hasattr(core, "capacity"):
                capacity = core.capacity

        return self.record(
            source=source,
            activity=experience_pattern,
            previous_activity=core.previous_activity,
            fatigue=core.fatigue,
            stimulus=stimulus,
            direction=direction,
            capacity=capacity,
            metadata=metadata,
        )
