from __future__ import annotations

from collections import Counter

from dormant_surface_flow_v7 import DiverseRegionCappedRecoveryBrain


class RegionDistributionMeasuredRecoveryBrain(DiverseRegionCappedRecoveryBrain):
    """v7 recovery behavior plus persistent input-region distribution metrics.

    Recovery behavior is unchanged. This class only preserves the complete
    region history for every candidate pathway, including pathways that later
    become promoted and are removed from the v7 pending-candidate dictionary.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.all_candidate_regions_by_edge: dict[tuple[int, int], set[int]] = {}

    def _merge_region_history(self) -> None:
        for edge, regions in self.candidate_regions_by_edge.items():
            self.all_candidate_regions_by_edge.setdefault(edge, set()).update(regions)

    def _process_recovery_candidates(self, peak_contributions) -> None:
        # Preserve evidence accumulated before this experience. This matters
        # when super() promotes an edge and removes it from the pending map.
        self._merge_region_history()
        super()._process_recovery_candidates(peak_contributions)
        self._merge_region_history()

        # A promoted edge may already have been popped by v7. Its final region
        # is still recoverable from the current experience and promoted set.
        if self.current_experience_region is not None:
            for edge in self.last_promoted_edges:
                self.all_candidate_regions_by_edge.setdefault(edge, set()).add(
                    self.current_experience_region
                )

    def region_distribution_stats(self) -> dict[str, int]:
        """Return unique pathway counts by distinct-region reach and region."""
        self._merge_region_history()
        histories = {
            edge: set(regions)
            for edge, regions in self.all_candidate_regions_by_edge.items()
            if regions
        }
        reach = Counter(len(regions) for regions in histories.values())

        region_unique = {
            region_id: sum(region_id in regions for regions in histories.values())
            for region_id in range(3)
        }
        exact_masks = Counter(
            "".join(str(region_id) for region_id in sorted(regions))
            for regions in histories.values()
        )

        return {
            "candidate_paths_total": len(histories),
            "reached_1_region": reach.get(1, 0),
            "reached_2_regions": reach.get(2, 0),
            "reached_3_regions": reach.get(3, 0),
            "region_low_candidates": region_unique[0],
            "region_middle_candidates": region_unique[1],
            "region_high_candidates": region_unique[2],
            "low_only": exact_masks.get("0", 0),
            "middle_only": exact_masks.get("1", 0),
            "high_only": exact_masks.get("2", 0),
            "low_middle": exact_masks.get("01", 0),
            "low_high": exact_masks.get("02", 0),
            "middle_high": exact_masks.get("12", 0),
            "low_middle_high": exact_masks.get("012", 0),
        }

    def recovery_measurement_stats(self) -> dict[str, float]:
        stats = super().recovery_measurement_stats()
        stats.update(
            {key: float(value) for key, value in self.region_distribution_stats().items()}
        )
        return stats
