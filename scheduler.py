"""Internal scheduler for SphereBrain v27 whole-brain formation.

The Scheduler owns the internal flow of time.

It does not perform learning itself.
It only determines which phase of life the brain is currently in.

Typical sequence::

    Experience
        ↓
    Trace
        ↓
    Reflection
        ↓
    Reflection
        ↓
    Experience
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SchedulerPhase(Enum):
    """Current phase of the brain."""

    EXPERIENCE = auto()
    REFLECTION = auto()


@dataclass(slots=True)
class SchedulerConfig:
    """Configuration of internal timing."""

    reflections_per_experience: int = 1

    def __post_init__(self) -> None:
        if self.reflections_per_experience < 0:
            raise ValueError(
                "reflections_per_experience must be non-negative"
            )


class Scheduler:
    """Controls the internal life cycle of SphereBrain.

    The scheduler owns only time.

    It does not know how the Core works.
    It does not know how Reflection works.
    It simply decides what happens next.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
    ) -> None:

        self.config = config or SchedulerConfig()

        self.time_index = 0
        self.phase = SchedulerPhase.EXPERIENCE

        self._reflection_count = 0

    @property
    def is_experience(self) -> bool:
        return self.phase is SchedulerPhase.EXPERIENCE

    @property
    def is_reflection(self) -> bool:
        return self.phase is SchedulerPhase.REFLECTION

    def reset(self) -> None:
        """Reset internal time."""

        self.time_index = 0
        self.phase = SchedulerPhase.EXPERIENCE
        self._reflection_count = 0

    def begin_experience(self) -> None:
        """Begin an external experience."""

        self.phase = SchedulerPhase.EXPERIENCE
        self._reflection_count = 0

    def finish_experience(self) -> None:
        """Finish one experience.

        Reflection starts automatically if configured.
        """

        self.time_index += 1

        if self.config.reflections_per_experience == 0:
            self.phase = SchedulerPhase.EXPERIENCE
        else:
            self.phase = SchedulerPhase.REFLECTION

    def begin_reflection(self) -> None:
        """Begin one reflection cycle."""

        self.phase = SchedulerPhase.REFLECTION

    def finish_reflection(self) -> None:
        """Finish one reflection cycle."""

        self.time_index += 1
        self._reflection_count += 1

        if (
            self._reflection_count
            >= self.config.reflections_per_experience
        ):
            self.phase = SchedulerPhase.EXPERIENCE
            self._reflection_count = 0
        else:
            self.phase = SchedulerPhase.REFLECTION

    def step(self) -> SchedulerPhase:
        """Advance one internal step.

        Returns the phase that should be executed next.
        """

        if self.phase is SchedulerPhase.EXPERIENCE:
            self.finish_experience()
        else:
            self.finish_reflection()

        return self.phase