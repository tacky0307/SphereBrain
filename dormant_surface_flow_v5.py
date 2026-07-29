from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT, PATHWAY_PROTECTED
from dormant_surface_flow_v4 import SelectiveRecoveryDormantBrain


class MeasuredSelectiveRecoveryBrain(SelectiveRecoveryDormantBrain):
    """Selective dormant recovery with persistent, separated recovery metrics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_selection_events_total = 0
        self.candidate_unique_edges_seen: set[tuple[int, int]] = set()
        self.selective_promotions_total = 0
        self.selective_promoted_edges: set[tuple[int, int]] = set()
        self.teacher_direct_reactivations_total = 0
        self.teacher_direct_reactivated_edges: set[tuple[int, int]] = set()

    def recovery_measurement_stats(self) -> dict[str, float]:
        connected = self.adjacency
        pending_mask = (
            connected
            & (self.pathway_state == PATHWAY_DORMANT)
            & (self.candidate_experience_count > 0)
        )
        return {
            "candidate_selection_events_total": float(self.candidate_selection_events_total),
            "candidate_unique_edges_total": float(len(self.candidate_unique_edges_seen)),
            "selective_promotions_total": float(self.selective_promotions_total),
            "selective_promoted_unique_edges": float(len(self.selective_promoted_edges)),
            "teacher_direct_reactivations_total": float(
                self.teacher_direct_reactivations_total
            ),
            "teacher_direct_reactivated_unique_edges": float(
                len(self.teacher_direct_reactivated_edges)
            ),
            "pending_candidate_edges": float(np.count_nonzero(pending_mask)),
            "pending_candidate_selection_sum": float(
                np.sum(self.candidate_experience_count[pending_mask])
            ),
        }

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        self.last_candidate_edges = set()
        self.last_promoted_edges = set()
        if not self.recovery_mode:
            return

        dormant = self.edge_enabled & (self.pathway_state == PATHWAY_DORMANT)
        baseline = self.prelesion_peak_activity
        increased = peak_contributions >= np.maximum(
            self.strong_contribution_threshold,
            baseline * self.activity_increase_ratio,
        )
        eligible = dormant & increased
        candidates = np.argwhere(eligible)
        if candidates.size == 0:
            return

        ranked = sorted(
            ((int(source), int(target)) for source, target in candidates),
            key=lambda edge: (
                float(peak_contributions[edge]),
                float(peak_contributions[edge] - baseline[edge]),
            ),
            reverse=True,
        )[: self.max_candidates_per_experience]

        for source, target in ranked:
            edge = (source, target)
            self.last_candidate_edges.add(edge)
            self.candidate_experience_count[edge] += 1
            self.candidate_selected_total[edge] += 1
            self.candidate_selection_events_total += 1
            self.candidate_unique_edges_seen.add(edge)

            if self.candidate_experience_count[edge] < self.candidate_required_experiences:
                continue

            self._reactivate_pathway(source, target, automatic=True)
            if self.pathway_state[edge] == PATHWAY_PROTECTED:
                self.last_promoted_edges.add(edge)
                self.selective_promotions_total += 1
                self.selective_promoted_edges.add(edge)
                self.candidate_experience_count[edge] = 0

    def _update_pathway_states(self, reinforced: set[tuple[int, int]]) -> None:
        direct_teacher_edges = {
            (source, target)
            for source, target in reinforced
            if self.edge_enabled[source, target]
            and self.pathway_state[source, target] == PATHWAY_DORMANT
        }
        super()._update_pathway_states(reinforced)
        if self.recovery_mode and direct_teacher_edges:
            self.teacher_direct_reactivations_total += len(direct_teacher_edges)
            self.teacher_direct_reactivated_edges.update(direct_teacher_edges)
