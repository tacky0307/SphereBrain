from __future__ import annotations

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT, PATHWAY_PROTECTED
from dormant_surface_flow_v7 import DiverseRegionCappedRecoveryBrain


class SameRegionDistinctInputRecoveryBrain(DiverseRegionCappedRecoveryBrain):
    """Promote dormant pathways reused by distinct inputs in the same region.

    Recovery order:
        dormant -> candidate on distinct inputs in one region -> protected
        -> teacher reinforcement

    Repeated presentations of the same input value are ignored. Evidence from
    different regions is kept separately and cannot be combined for promotion.
    Promotion caps and dormant-teacher blocking are inherited from v7.
    """

    def __init__(
        self,
        *args,
        distinct_inputs_required: int = 2,
        input_key_decimals: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if distinct_inputs_required < 2:
            raise ValueError("distinct_inputs_required must be at least 2")
        if input_key_decimals < 0:
            raise ValueError("input_key_decimals must be non-negative")

        self.distinct_inputs_required = int(distinct_inputs_required)
        self.input_key_decimals = int(input_key_decimals)
        self.current_experience_input_key: float | None = None
        self.candidate_inputs_by_edge_region: dict[
            tuple[int, int], dict[int, set[float]]
        ] = {}
        self.all_candidate_inputs_by_edge_region: dict[
            tuple[int, int], dict[int, set[float]]
        ] = {}
        self.duplicate_input_selections_ignored = 0
        self.promoted_from_distinct_inputs_total = 0

    def set_experience(self, region_id: int | None, input_value: float | None) -> None:
        self.set_experience_region(region_id)
        self.current_experience_input_key = (
            None
            if input_value is None
            else round(float(input_value), self.input_key_decimals)
        )

    def _record_history(self, edge: tuple[int, int], region: int, input_key: float) -> None:
        regions = self.all_candidate_inputs_by_edge_region.setdefault(edge, {})
        regions.setdefault(region, set()).add(input_key)

    def distinct_input_distribution_stats(self) -> dict[str, float]:
        one_input = 0
        two_or_more = 0
        maximum = 0
        region_counts = {0: 0, 1: 0, 2: 0}
        for regions in self.all_candidate_inputs_by_edge_region.values():
            best = max((len(inputs) for inputs in regions.values()), default=0)
            maximum = max(maximum, best)
            if best == 1:
                one_input += 1
            elif best >= 2:
                two_or_more += 1
            for region in region_counts:
                if len(regions.get(region, set())) >= self.distinct_inputs_required:
                    region_counts[region] += 1

        return {
            "candidate_paths_one_distinct_input": float(one_input),
            "candidate_paths_two_or_more_distinct_inputs": float(two_or_more),
            "max_distinct_inputs_same_region": float(maximum),
            "low_region_promotion_eligible": float(region_counts[0]),
            "middle_region_promotion_eligible": float(region_counts[1]),
            "high_region_promotion_eligible": float(region_counts[2]),
            "duplicate_input_selections_ignored": float(
                self.duplicate_input_selections_ignored
            ),
            "promoted_from_distinct_inputs_total": float(
                self.promoted_from_distinct_inputs_total
            ),
        }

    def recovery_measurement_stats(self) -> dict[str, float]:
        stats = super().recovery_measurement_stats()
        stats.update(self.distinct_input_distribution_stats())
        return stats

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        self.last_candidate_edges = set()
        self.last_promoted_edges = set()
        region = self.current_experience_region
        input_key = self.current_experience_input_key
        if not self.recovery_mode or region is None or input_key is None:
            return

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
        )[: self.max_candidates_per_experience]

        for edge in ranked:
            regions = self.candidate_inputs_by_edge_region.setdefault(edge, {})
            inputs = regions.setdefault(region, set())
            self._record_history(edge, region, input_key)

            if input_key in inputs:
                self.duplicate_input_selections_ignored += 1
                continue

            inputs.add(input_key)
            self.last_candidate_edges.add(edge)
            distinct_count = len(inputs)
            self.candidate_experience_count[edge] = distinct_count
            self.candidate_selected_total[edge] += 1
            self.candidate_selection_events_total += 1
            self.candidate_unique_edges_seen.add(edge)

            if distinct_count < self.distinct_inputs_required:
                continue
            if self.selective_promotions_total >= self.max_promotions_total:
                self.total_promotion_cap_blocks += 1
                continue
            if self.promotions_this_epoch >= self.max_promotions_per_epoch:
                self.epoch_promotion_cap_blocks += 1
                continue

            source, target = edge
            self._reactivate_pathway(source, target, automatic=True)
            if self.pathway_state[edge] == PATHWAY_PROTECTED:
                self.last_promoted_edges.add(edge)
                self.selective_promotions_total += 1
                self.selective_promoted_edges.add(edge)
                self.promotions_this_epoch += 1
                self.promoted_from_distinct_inputs_total += 1
                self.candidate_experience_count[edge] = 0
                self.candidate_inputs_by_edge_region.pop(edge, None)
