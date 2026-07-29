"""SphereBrain Wave Core experimental package.

The package keeps short-time wave dynamics in ``core`` and adds optional
experience-scale reflection and observation layers. Existing experiments can
keep using ``SphereWaveCore`` without enabling those layers.
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
from .recall import RecallConfig, RecallPathDiagnostics

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
    "RecallConfig",
    "RecallPathDiagnostics",
]
