from __future__ import annotations

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT, PATHWAY_PROTECTED
from dormant_surface_flow_v6 import MeanBaselineStagedRecoveryBrain


class DiverseRegionCappedRecoveryBrain(MeanBaselineStagedRecoveryBrain):
    """Selective recovery requiring evidence from distinct input regions.

    Recovery order remains:
        dormant -> candidate in distinct regions -> protected -> teacher reinforcement

    Repeated presentations from the same input region count only once for each edge.
    Promotions are capped per epoch and across the complete recovery run.
    """

    def __init__(
        self,
        *args,
        max_promotions_per_epoch: int = 10,
        max_promotions_total: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if max_promotions_per_epoch < 1:
            raise ValueError("max_promotions_per_epoch must be at least 1")
        if max_promotions_total < 1:
            raise ValueError("max_promotions_total must be at least 1")

        self.max_promotions_per_epoch = int(max_promotions_per_epoch)
        self.max_promotions_total = int(max_promotions_total)
        self.current_experience_region: int | None = None
        self.candidate_regions_by_edge: dict[tuple[int, int], set[int]] = {}
        self.promotions_this_epoch = 0
        self.duplicate_region_selections_ignored = 0
        self.epoch_promotion_cap_blocks = 0
        self.total_promotion_cap_blocks = 0

    def begin_recovery_epoch(self) -> None:
        self.promotions_this_epoch = 0

    def set_experience_region(self, region_id: int | None) -> None:
        self.current_experience_region = None if region_id is None else int(region_id)

    def recovery_measurement_stats(self) -> dict[str, float]:
        stats = super().recovery_measurement_stats()
        pending_region_counts = [
            len(regions)
            for edge, regions in self.candidate_regions_by_edge.items()
            if self.edge_enabled[edge]
            and self.pathway_state[edge] == PATHWAY_DORMANT
            and regions
        ]
        stats.update(
            {
                "promotions_this_epoch": float(self.promotions_this_epoch),
                "promotion_cap_per_epoch": float(self.max_promotions_per_epoch),
                "promotion_cap_total": float(self.max_promotions_total),
                "duplicate_region_selections_ignored": float(
                    self.duplicate_region_selections_ignored
                ),
                "epoch_promotion_cap_blocks": float(self.epoch_promotion_cap_blocks),
                "total_promotion_cap_blocks": float(self.total_promotion_cap_blocks),
                "pending_distinct_region_evidence": float(sum(pending_region_counts)),
                "max_distinct_regions_pending": float(
                    max(pending_region_counts) if pending_region_counts else 0
                ),
            }
        )
        return stats

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        self.last_candidate_edges = set()
        self.last_promoted_edges = set()
        if not self.recovery_mode or self.current_experience_region is None:
            return

        dormant = self.edge_enabled & (self.pathway_state == PATHWAY_DORMANT)
        baseline = self.prelesion_mean_activity
        threshold = np.maximum(
            self.strong_contribution_threshold,
            baseline + self.activity_increase_margin,
        )
        eligible = dormant & (peak_contributions >= threshold)
        candidates = np.argwhere(eligible)
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
            regions = self.candidate_regions_by_edge.setdefault(edge, set())
            if self.current_experience_region in regions:
                self.duplicate_region_selections_ignored += 1
                continue

            regions.add(self.current_experience_region)
            self.last_candidate_edges.add(edge)
            distinct_count = len(regions)
            self.candidate_experience_count[edge] = distinct_count
            self.candidate_selected_total[edge] += 1
            self.candidate_selection_events_total += 1
            self.candidate_unique_edges_seen.add(edge)

            if distinct_count < self.candidate_required_experiences:
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
                self.candidate_experience_count[edge] = 0
                self.candidate_regions_by_edge.pop(edge, None)
