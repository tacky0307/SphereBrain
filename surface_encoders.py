from __future__ import annotations

from hashlib import sha256

import numpy as np

from surface_flow import SurfacePattern


def ordered_surface_nodes(positions: np.ndarray, nodes: list[int]) -> list[int]:
    """Create a stable one-dimensional coordinate over a surface region."""
    return sorted(
        nodes,
        key=lambda node: (
            float(positions[node, 1]),
            float(positions[node, 2]),
            float(positions[node, 0]),
        ),
    )


class TextSurfaceEncoder:
    """Experimental text adapter kept strictly outside SphereBrain core.

    This encoder is intentionally simple and does not claim to represent meaning.
    It exists only so older word-association tests can use the new numeric core API.
    """

    def __init__(self, candidate_nodes: list[int], width: int = 4) -> None:
        if not candidate_nodes:
            raise ValueError("candidate_nodes must not be empty")
        if width < 1 or width > len(candidate_nodes):
            raise ValueError("invalid width")
        self.candidate_nodes = list(candidate_nodes)
        self.width = width

    def encode(self, text: str) -> dict[int, float]:
        clean = text.strip()
        if not clean:
            raise ValueError("text is empty")
        digest = sha256(clean.encode("utf-8")).digest()
        selected: list[int] = []
        offset = 0
        while len(selected) < self.width:
            value = int.from_bytes(digest[offset : offset + 4], "big")
            node = self.candidate_nodes[value % len(self.candidate_nodes)]
            if node not in selected:
                selected.append(node)
            offset += 4
            if offset + 4 > len(digest):
                digest = sha256(digest).digest()
                offset = 0
        return {
            node: max(0.1, 1.0 - index * 0.12)
            for index, node in enumerate(selected)
        }


class ScalarSurfaceEncoder:
    """Encode a scalar in [0, 1] as an overlapping population pattern.

    Nearby values activate nearby, overlapping surface nodes. This preserves
    numerical neighborhood information instead of hashing each value separately.
    """

    def __init__(
        self,
        ordered_nodes: list[int],
        width: int = 5,
        sigma: float | None = None,
    ) -> None:
        if not ordered_nodes:
            raise ValueError("ordered_nodes must not be empty")
        if width < 1 or width > len(ordered_nodes):
            raise ValueError("invalid width")
        self.ordered_nodes = list(ordered_nodes)
        self.width = width
        self.sigma = sigma if sigma is not None else max(1.0, width / 2.0)

    def encode(self, value: float) -> dict[int, float]:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {value}")

        exact_center = value * (len(self.ordered_nodes) - 1)
        center = int(round(exact_center))
        half = self.width // 2
        start = max(0, min(center - half, len(self.ordered_nodes) - self.width))
        indices = range(start, start + self.width)

        pattern: dict[int, float] = {}
        for index in indices:
            distance = (index - exact_center) / self.sigma
            activity = float(np.exp(-0.5 * distance * distance))
            if activity > 0.0:
                pattern[self.ordered_nodes[index]] = activity

        peak = max(pattern.values())
        return {node: value / peak for node, value in pattern.items()}


def pattern_nodes(pattern: SurfacePattern) -> list[int]:
    return list(pattern.keys())
