from __future__ import annotations

from collections import defaultdict

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT
from dormant_surface_flow_v9 import SameRegionDistinctInputRecoveryBrain


class CandidateWidthOverlapMeasurementBrain(SameRegionDistinctInputRecoveryBrain):
    """v9 recovery behavior plus shadow overlap measurements at wider top-N cuts.

    The actual recovery behavior remains v9 and still uses
    ``max_candidates_per_experience`` (normally 20). In parallel, this class
    records which distinct input values would select each dormant pathway at
    top 20, 50, 100, and 200. These shadow measurements never promote or alter
    a pathway.
    """

    def __init__(
        self,
        *args,
        candidate_widths: tuple[int, ...] = (20, 50, 100, 200),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cleaned = tuple(sorted({int(width) for width in candidate_widths}))
        if not cleaned or cleaned[0] < 1:
            raise ValueError("candidate_widths must contain positive integers")
        self.candidate_widths = cleaned
        self.shadow_inputs_by_width: dict[
            int, dict[tuple[int, int], set[float]]
        ] = {width: defaultdict(set) for width in cleaned}

    def _record_width_overlap(
        self, peak_contributions: np.ndarray, input_key: float
    ) -> None:
        dormant = self.edge_enabled & (self.pathway_state == PATHWAY_DORMANT)
        baseline = self.prelesion_mean_activity
        threshold = np.maximum(
            self.strong_contribution_threshold,
            baseline + self.activity_increase_margin,
        )
        candidates = np.argwhere(dormant & (peak_contributions >= threshold))
        if candidates.size == 0:
            return

        ranked = sorted(
            ((int(source), int(target)) for source, target in candidates),
            key=lambda edge: (
                float(peak_contributions[edge] - baseline[edge]),
                float(peak_contributions[edge]),
            ),
            reverse=True,
        )

        for width in self.candidate_widths:
            histories = self.shadow_inputs_by_width[width]
            for edge in ranked[:width]:
                histories[edge].add(input_key)

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        input_key = self.current_experience_input_key
        if self.recovery_mode and input_key is not None:
            self._record_width_overlap(peak_contributions, input_key)
        super()._process_recovery_candidates(peak_contributions)

    def candidate_width_overlap_stats(self) -> dict[int, dict[str, int]]:
        result: dict[int, dict[str, int]] = {}
        for width in self.candidate_widths:
            counts = [
                len(inputs)
                for inputs in self.shadow_inputs_by_width[width].values()
                if inputs
            ]
            result[width] = {
                "unique_paths": len(counts),
                "exactly_1_input": sum(count == 1 for count in counts),
                "at_least_2_inputs": sum(count >= 2 for count in counts),
                "at_least_3_inputs": sum(count >= 3 for count in counts),
                "at_least_4_inputs": sum(count >= 4 for count in counts),
                "max_distinct_inputs": max(counts, default=0),
            }
        return result
