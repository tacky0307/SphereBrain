from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowBiasStats:
    mean: float
    maximum: float
    active_edges: int
    total_energy: float


class FlowBias:
    """
    Short-term pathway memory.

    Unlike weights, FlowBias is temporary.
    Frequently traversed edges become easier to traverse again,
    but the effect naturally decays over time.

    Long-term memory  -> weights
    Short-term memory -> flow_bias
    """

    def __init__(
        self,
        node_count: int,
        *,
        initial_bias: float = 1.0,
        reinforce_gain: float = 0.25,
        decay: float = 0.92,
        maximum_bias: float = 3.0,
    ) -> None:

        self.node_count = int(node_count)

        self.initial_bias = float(initial_bias)
        self.reinforce_gain = float(reinforce_gain)
        self.decay_rate = float(decay)
        self.maximum_bias = float(maximum_bias)

        self.bias = np.full(
            (self.node_count, self.node_count),
            self.initial_bias,
            dtype=float,
        )

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    def reset(self) -> None:
        """Return every edge to its neutral state."""
        self.bias.fill(self.initial_bias)

    def decay(self) -> None:
        """
        Bias slowly returns toward 1.0.

        3.0
          ↓
        2.84
          ↓
        2.69
          ↓
        ...
          ↓
        1.0
        """

        self.bias = (
            self.initial_bias
            + (self.bias - self.initial_bias) * self.decay_rate
        )

    # ---------------------------------------------------------
    # learning
    # ---------------------------------------------------------

    def reinforce_edge(
        self,
        source: int,
        target: int,
        amount: float | None = None,
    ) -> None:

        gain = self.reinforce_gain if amount is None else float(amount)

        self.bias[source, target] = min(
            self.maximum_bias,
            self.bias[source, target] + gain,
        )

    def reinforce(
        self,
        edges,
        amount: float | None = None,
    ) -> None:

        gain = self.reinforce_gain if amount is None else float(amount)

        for source, target in edges:
            self.bias[source, target] = min(
                self.maximum_bias,
                self.bias[source, target] + gain,
            )

    # ---------------------------------------------------------
    # access
    # ---------------------------------------------------------

    def multiplier(self) -> np.ndarray:
        """
        Matrix used directly inside propagation.

        effective_weight =
            weight
            * flow_bias
        """

        return self.bias

    def edge(
        self,
        source: int,
        target: int,
    ) -> float:

        return float(self.bias[source, target])

    # ---------------------------------------------------------
    # analysis
    # ---------------------------------------------------------

    def strongest_edges(
        self,
        top_n: int = 20,
    ):

        flat = np.argpartition(
            self.bias.ravel(),
            -top_n,
        )[-top_n:]

        values = self.bias.ravel()[flat]

        order = np.argsort(values)[::-1]

        result = []

        for index in flat[order]:
            source, target = np.unravel_index(
                index,
                self.bias.shape,
            )

            result.append(
                (
                    int(source),
                    int(target),
                    float(self.bias[source, target]),
                )
            )

        return result

    def stats(self) -> FlowBiasStats:

        active = self.bias > (self.initial_bias + 1e-6)

        return FlowBiasStats(
            mean=float(np.mean(self.bias)),
            maximum=float(np.max(self.bias)),
            active_edges=int(np.count_nonzero(active)),
            total_energy=float(np.sum(self.bias - self.initial_bias)),
        )