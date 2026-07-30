"""Reflection engine for SphereBrain v27 whole-brain formation.

Reflection does not learn and does not interpret memories.

Its only responsibility is to replay previously recorded Trace frames as new
internal stimuli for the Core.

The Core remains the only component that changes through experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .trace import TraceFrame, TraceRecorder


Array = np.ndarray


@dataclass(slots=True)
class ReflectionConfig:
    """Configuration for replaying Trace frames."""

    replay_gain: float = 1.0
    replay_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Result of one reflection replay."""

    frame_index: int
    time_index: int
    source: str
    signal: Array


class ReflectionEngine:
    """Replay previously recorded Trace frames.

    Reflection itself contains no learning rule.

    It simply converts stored whole-brain activity back into an internal
    stimulus which can be injected into the Core.
    """

    def __init__(
        self,
        config: ReflectionConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or ReflectionConfig()
        self.rng = rng or np.random.default_rng()

    def latest(
        self,
        recorder: TraceRecorder,
    ) -> ReflectionResult:
        """Replay the newest Trace frame."""

        frame = recorder.latest()
        return self._build_result(len(recorder) - 1, frame)

    def index(
        self,
        recorder: TraceRecorder,
        frame_index: int,
    ) -> ReflectionResult:
        """Replay a Trace frame by index."""

        frame = recorder.get(frame_index)
        return self._build_result(frame_index, frame)

    def random(
        self,
        recorder: TraceRecorder,
    ) -> ReflectionResult:
        """Replay one randomly selected Trace frame."""

        if len(recorder) == 0:
            raise LookupError("trace is empty")

        frame_index = int(self.rng.integers(len(recorder)))
        frame = recorder.get(frame_index)

        return self._build_result(frame_index, frame)

    def replay_signal(
        self,
        frame: TraceFrame,
    ) -> Array:
        """Convert a Trace frame into an internal stimulus."""

        return frame.replay_signal(
            gain=self.config.replay_gain,
            threshold=self.config.replay_threshold,
        )

    def _build_result(
        self,
        frame_index: int,
        frame: TraceFrame,
    ) -> ReflectionResult:

        signal = self.replay_signal(frame)

        return ReflectionResult(
            frame_index=frame_index,
            time_index=frame.time_index,
            source=frame.source,
            signal=signal,
        )