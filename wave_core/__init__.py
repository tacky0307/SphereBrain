"""SphereBrain Wave Core v0 experimental package.

This package is intentionally isolated from the existing path-replay prototype.
It studies how experience changes the propagation of distributed activity.
"""

from .core import (
    SphereWaveCore,
    WaveConfig,
    WaveSnapshot,
    ExperimentTrace,
)

__all__ = [
    "SphereWaveCore",
    "WaveConfig",
    "WaveSnapshot",
    "ExperimentTrace",
]
