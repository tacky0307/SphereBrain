from __future__ import annotations

from collections import defaultdict

import numpy as np

from dormant_surface_flow_v11 import Top50SharedExperienceRecoveryBrain


class PromotedContributionTrackingBrain(Top50SharedExperienceRecoveryBrain):
    """Top50 shared-experience recovery with promoted-path contribution tracking.

    Contribution values are the peak edge contributions already calculated by
    the flow engine for each observation.  The tracker records:

    * the candidate contribution immediately before an edge is promoted;
    * the first contribution seen on a later observation after promotion;
    * all contributions during the remainder of recovery training;
    * contributions during a dedicated recovery-end measurement pass; and
    * contributions during final evaluation.

    Measurement never changes candidate selection or promotion behaviour.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contribution_phase = "idle"
        self.promotion_candidate_contribution: dict[tuple[int, int], float] = {}
        self.first_post_promotion_contribution: dict[tuple[int, int], float] = {}
        self.recovery_contributions: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.recovery_end_contributions: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.final_eval_contributions: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.promotion_observation_index: dict[tuple[int, int], int] = {}
        self.observation_index = 0

    def set_contribution_phase(self, phase: str) -> None:
        allowed = {"idle", "recovery", "recovery_end", "final_eval"}
        if phase not in allowed:
            raise ValueError(f"unknown contribution phase: {phase}")
        self.contribution_phase = phase

    def _record_existing_promoted(self, peak_contributions: np.ndarray) -> None:
        for edge in self.selective_promoted_edges:
            value = float(peak_contributions[edge])
            promoted_at = self.promotion_observation_index.get(edge, self.observation_index)
            if (
                edge not in self.first_post_promotion_contribution
                and self.observation_index > promoted_at
            ):
                self.first_post_promotion_contribution[edge] = value

            if self.contribution_phase == "recovery":
                self.recovery_contributions[edge].append(value)
            elif self.contribution_phase == "recovery_end":
                self.recovery_end_contributions[edge].append(value)
            elif self.contribution_phase == "final_eval":
                self.final_eval_contributions[edge].append(value)

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        self.observation_index += 1
        self._record_existing_promoted(peak_contributions)
        promoted_before = set(self.selective_promoted_edges)
        super()._process_recovery_candidates(peak_contributions)
        newly_promoted = set(self.selective_promoted_edges) - promoted_before
        for edge in newly_promoted:
            self.promotion_candidate_contribution[edge] = float(peak_contributions[edge])
            self.promotion_observation_index[edge] = self.observation_index
            if self.contribution_phase == "recovery":
                self.recovery_contributions[edge].append(float(peak_contributions[edge]))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def promoted_contribution_stats(self) -> dict[str, float]:
        promoted = set(self.selective_promoted_edges)
        candidate_values = [
            self.promotion_candidate_contribution[edge]
            for edge in promoted
            if edge in self.promotion_candidate_contribution
        ]
        first_post_values = [
            self.first_post_promotion_contribution[edge]
            for edge in promoted
            if edge in self.first_post_promotion_contribution
        ]
        recovery_values = [
            value
            for edge in promoted
            for value in self.recovery_contributions.get(edge, [])
        ]
        recovery_end_values = [
            value
            for edge in promoted
            for value in self.recovery_end_contributions.get(edge, [])
        ]
        final_values = [
            value
            for edge in promoted
            for value in self.final_eval_contributions.get(edge, [])
        ]

        promoted_with_final = sum(
            bool(self.final_eval_contributions.get(edge)) for edge in promoted
        )
        promoted_with_recovery_end = sum(
            bool(self.recovery_end_contributions.get(edge)) for edge in promoted
        )
        return {
            "promoted_edges": float(len(promoted)),
            "candidate_mean": self._mean(candidate_values),
            "first_post_mean": self._mean(first_post_values),
            "recovery_mean": self._mean(recovery_values),
            "recovery_end_mean": self._mean(recovery_end_values),
            "final_eval_mean": self._mean(final_values),
            "candidate_samples": float(len(candidate_values)),
            "first_post_samples": float(len(first_post_values)),
            "recovery_samples": float(len(recovery_values)),
            "recovery_end_samples": float(len(recovery_end_values)),
            "final_eval_samples": float(len(final_values)),
            "promoted_seen_recovery_end": float(promoted_with_recovery_end),
            "promoted_seen_final_eval": float(promoted_with_final),
        }

    def promoted_edge_contribution_rows(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for source, target in sorted(self.selective_promoted_edges):
            edge = (source, target)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "candidate": self.promotion_candidate_contribution.get(edge, 0.0),
                    "first_post": self.first_post_promotion_contribution.get(edge, 0.0),
                    "recovery_mean": self._mean(self.recovery_contributions.get(edge, [])),
                    "recovery_end_mean": self._mean(
                        self.recovery_end_contributions.get(edge, [])
                    ),
                    "final_eval_mean": self._mean(
                        self.final_eval_contributions.get(edge, [])
                    ),
                    "final_samples": len(self.final_eval_contributions.get(edge, [])),
                }
            )
        return rows
