"""SphereBrain experimental package.

The package keeps the original wave dynamics and adds optional experience,
recall-diagnostic, and attractor-state layers. Existing experiments can keep
using ``SphereWaveCore`` without enabling the newer dynamics.
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
from .attractor import (
    AttractorConfig,
    AttractorSnapshot,
    AttractorSphereCore,
    AttractorTrace,
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
    "RecallConfig",
    "RecallPathDiagnostics",
    "AttractorConfig",
    "AttractorSnapshot",
    "AttractorSphereCore",
    "AttractorTrace",
]
