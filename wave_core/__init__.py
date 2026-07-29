"""SphereBrain Wave Core experimental package.

The package keeps short-time wave dynamics in ``core`` and adds optional
experience-scale reflection in ``experience``. Existing experiments can keep
using ``SphereWaveCore`` without enabling the longer learning time scale.
"""

from .core import (
    SphereWaveCore,
    WaveConfig,
    WaveSnapshot,
    ExperimentTrace,
)
from .experience import (
    ExperienceBuffer,
    ExperienceConfig,
    ExperienceReflection,
    ExperienceReflector,
    ExperienceSummary,
    StimulusEvent,
)

__all__ = [
    "SphereWaveCore",
    "WaveConfig",
    "WaveSnapshot",
    "ExperimentTrace",
    "ExperienceBuffer",
    "ExperienceConfig",
    "ExperienceReflection",
    "ExperienceReflector",
    "ExperienceSummary",
    "StimulusEvent",
]
