from __future__ import annotations

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT, PATHWAY_PROTECTED
from dormant_surface_flow_v5 import MeasuredSelectiveRecoveryBrain
from surface_flow import SurfaceFlowResult, SurfacePattern


class MeanBaselineStagedRecoveryBrain(MeasuredSelectiveRecoveryBrain):
    """Selective recovery using a pre-lesion mean and strict stage ordering.

    Recovery order:
        dormant -> candidate -> protected -> teacher reinforcement

    During recovery, teacher reinforcement cannot directly wake dormant edges.
    Only edges already promoted by repeated signal-driven candidate selection may
    receive ordinary teacher-guided reinforcement.
    """

    def __init__(
        self,
        *args,
        activity_increase_margin: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if activity_increase_margin < 0.0:
            raise ValueError("activity_increase_margin must be non-negative")
        self.activity_increase_margin = float(activity_increase_margin)

        shape = (self.node_count, self.node_count)
        self.prelesion_activity_sum = np.zeros(shape, dtype=float)
        self.prelesion_activity_count = np.zeros(shape, dtype=np.int32)
        self.prelesion_mean_activity = np.zeros(shape, dtype=float)
        self.teacher_blocked_dormant_total = 0
        self.teacher_blocked_dormant_edges: set[tuple[int, int]] = set()

    def begin_prelesion_baseline_collection(self) -> None:
        self.prelesion_activity_sum.fill(0.0)
        self.prelesion_activity_count.fill(0)
        self.prelesion_mean_activity.fill(0.0)
        self.prelesion_peak_activity.fill(0.0)
        self._collect_prelesion_baseline = True

    def end_prelesion_baseline_collection(self) -> None:
        observed = self.prelesion_activity_count > 0
        self.prelesion_mean_activity[observed] = (
            self.prelesion_activity_sum[observed]
            / self.prelesion_activity_count[observed]
        )
        # Keep compatibility with inherited reporting.
        self.prelesion_peak_activity = self.prelesion_mean_activity.copy()
        self._collect_prelesion_baseline = False

    def recovery_measurement_stats(self) -> dict[str, float]:
        stats = super().recovery_measurement_stats()
        connected = self.adjacency
        observed = connected & (self.prelesion_activity_count > 0)
        stats.update(
            {
                "baseline_mean_active_edges": float(np.count_nonzero(observed)),
                "baseline_mean_contribution": float(
                    np.mean(self.prelesion_mean_activity[observed])
                    if np.any(observed)
                    else 0.0
                ),
                "teacher_blocked_dormant_total": float(
                    self.teacher_blocked_dormant_total
                ),
                "teacher_blocked_dormant_unique_edges": float(
                    len(self.teacher_blocked_dormant_edges)
                ),
            }
        )
        return stats

    def _process_recovery_candidates(self, peak_contributions: np.ndarray) -> None:
        self.last_candidate_edges = set()
        self.last_promoted_edges = set()
        if not self.recovery_mode:
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
            self.last_candidate_edges.add(edge)
            self.candidate_experience_count[edge] += 1
            self.candidate_selected_total[edge] += 1
            self.candidate_selection_events_total += 1
            self.candidate_unique_edges_seen.add(edge)

            if self.candidate_experience_count[edge] < self.candidate_required_experiences:
                continue

            source, target = edge
            self._reactivate_pathway(source, target, automatic=True)
            if self.pathway_state[edge] == PATHWAY_PROTECTED:
                self.last_promoted_edges.add(edge)
                self.selective_promotions_total += 1
                self.selective_promoted_edges.add(edge)
                self.candidate_experience_count[edge] = 0

    def _update_pathway_states(self, reinforced: set[tuple[int, int]]) -> None:
        if not self.recovery_mode:
            super()._update_pathway_states(reinforced)
            return

        blocked = {
            edge
            for edge in reinforced
            if self.edge_enabled[edge]
            and self.pathway_state[edge] == PATHWAY_DORMANT
        }
        allowed = reinforced - blocked
        if blocked:
            self.teacher_blocked_dormant_total += len(blocked)
            self.teacher_blocked_dormant_edges.update(blocked)

        # Dormant edges cannot be directly reactivated by the teacher. Edges that
        # were promoted during the preceding propagation are protected already and
        # therefore remain in `allowed` and can now be strengthened.
        super()._update_pathway_states(allowed)

    def propagate(
        self,
        input_pattern: SurfacePattern,
        steps: int = 24,
        threshold: float = 0.08,
        noise: float = 0.006,
        use_activation_field: bool | None = None,
        update_activation_field: bool = False,
    ) -> SurfaceFlowResult:
        sources = self._validate_pattern(input_pattern, self.input_nodes, "input_pattern")
        activation = np.zeros(self.node_count, dtype=float)
        fatigue = np.zeros(self.node_count, dtype=float)
        for node, value in sources.items():
            activation[node] = value

        if use_activation_field is None:
            use_activation_field = self.activation_field_enabled
        if use_activation_field and self.activation_field_enabled:
            field_activity = np.clip(
                self.activation_field * self.activation_field_influence, 0.0, 1.0
            )
            activation = 1.0 - (1.0 - activation) * (1.0 - field_activity)

        trace = activation.copy()
        output_mask = np.zeros(self.node_count, dtype=bool)
        output_mask[self.output_nodes] = True
        output_history: list[dict[int, float]] = []
        history: list[list[int]] = [np.flatnonzero(activation > 0).tolist()]
        traversed: list[tuple[int, int]] = []
        peak_contributions = np.zeros_like(self.weights)
        effective_weights = self._effective_weight_matrix()

        for _ in range(steps):
            effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
            contributions = effective[:, None] * effective_weights * self.transmission_gain
            contributions[~self.edge_enabled] = 0.0
            contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)
            peak_contributions = np.maximum(peak_contributions, contributions)

            next_activation = 1.0 - np.prod(1.0 - contributions, axis=0)
            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)
            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0
            trace = np.maximum(trace, next_activation)

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)
            edge_threshold = threshold * self.edge_activity_ratio
            active_edges = np.argwhere(contributions >= edge_threshold)
            traversed.extend((int(source), int(target)) for source, target in active_edges)

            visible = np.flatnonzero((next_activation > 0) & output_mask)
            output_history.append({int(n): float(next_activation[n]) for n in visible})

            fatigue = fatigue * self.fatigue_decay
            fatigue += next_activation * self.fatigue_gain
            fatigue = np.clip(fatigue, 0.0, 0.95)
            activation = next_activation
            if not active_now:
                break

        if update_activation_field:
            self._update_activation_field(trace)

        if self._collect_prelesion_baseline:
            connected = self.adjacency
            self.prelesion_activity_sum[connected] += peak_contributions[connected]
            self.prelesion_activity_count[connected] += 1
        self._process_recovery_candidates(peak_contributions)

        return SurfaceFlowResult(
            source_nodes=sorted(sources),
            output_history=output_history,
            activation_history=history,
            traversed_edges=traversed,
            final_activation=activation.copy(),
        )
