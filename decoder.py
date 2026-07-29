from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from brain import SignalResult


@dataclass(frozen=True)
class DecodedState:
    """Coreの収束状態を観測可能な数値表現へ変換したもの。"""

    top_nodes: list[tuple[int, float]]
    active_count: int
    traversed_edge_count: int
    steps: int
    peak_activation: float
    settled: bool


class NumericDecoder:
    def __init__(self, top_k: int = 8, settle_threshold: float = 0.15) -> None:
        self.top_k = top_k
        self.settle_threshold = settle_threshold

    def decode(self, result: SignalResult) -> DecodedState:
        activation = np.asarray(result.final_activation, dtype=float)
        ranked = np.argsort(activation)[::-1]
        top_nodes = [
            (int(node), float(activation[node]))
            for node in ranked[: self.top_k]
            if activation[node] > 0
        ]
        peak = float(activation.max()) if activation.size else 0.0
        active_count = int(np.count_nonzero(activation > 0))
        return DecodedState(
            top_nodes=top_nodes,
            active_count=active_count,
            traversed_edge_count=len(result.traversed_edges),
            steps=max(0, len(result.activation_history) - 1),
            peak_activation=peak,
            settled=peak < self.settle_threshold or active_count == 0,
        )
