from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dormant_surface_flow import PATHWAY_DORMANT, PATHWAY_PROTECTED
from dormant_surface_flow_v3 import RecoveryGatedDormantBrain
from surface_flow import SurfaceFlowResult, SurfacePattern


class SelectiveRecoveryDormantBrain(RecoveryGatedDormantBrain):
    """Dormant brain with staged, selective recovery-path recruitment.

    Recovery follows:

        dormant -> awakening candidate -> protected

    A dormant pathway becomes a candidate only when all conditions are met:
    - recovery mode is enabled;
    - its peak contribution is strong;
    - its contribution increased relative to the pre-lesion baseline;
    - it ranks in the top N eligible pathways for the current experience.

    A candidate is reactivated only after being selected across multiple
    experiences. This prevents a single widespread propagation from waking the
    entire graph at once.
    """

    def __init__(
        self,
        *args,
        strong_contribution_threshold: float = 0.08,
        activity_increase_ratio: float = 2.0,
        candidate_required_experiences: int = 3,
        max_candidates_per_experience: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= strong_contribution_threshold <= 1.0:
            raise ValueError("strong_contribution_threshold must be in [0, 1]")
        if activity_increase_ratio <= 1.0:
            raise ValueError("activity_increase_ratio must be greater than 1")
        if candidate_required_experiences < 1:
            raise ValueError("candidate_required_experiences must be at least 1")
        if max_candidates_per_experience < 1:
            raise ValueError("max_candidates_per_experience must be at least 1")

        self.strong_contribution_threshold = float(strong_contribution_threshold)
        self.activity_increase_ratio = float(activity_increase_ratio)
        self.candidate_required_experiences = int(candidate_required_experiences)
        self.max_candidates_per_experience = int(max_candidates_per_experience)

        shape = (self.node_count, self.node_count)
        self.prelesion_peak_activity = np.zeros(shape, dtype=float)
        self.candidate_experience_count = np.zeros(shape, dtype=np.int32)
        self.candidate_selected_total = np.zeros(shape, dtype=np.int32)
        self.last_candidate_edges: set[tuple[int, int]] = set()
        self.last_promoted_edges: set[tuple[int, int]] = set()
        self._collect_prelesion_baseline = False

    def begin_prelesion_baseline_collection(self) -> None:
        self.prelesion_peak_activity.fill(0.0)
        self._collect_prelesion_baseline = True

    def end_prelesion_baseline_collection(self) -> None:
        self._collect_prelesion_baseline = False

    def candidate_stats(self) -> dict[str, float]:
        connected = self.adjacency
        candidate_mask = connected & (self.candidate_experience_count > 0)
        return {
            "candidate_edges": float(np.count_nonzero(candidate_mask)),
            "candidate_selections": float(np.sum(self.candidate_selected_total[connected])),
            "max_candidate_experiences": float(
                np.max(self.candidate_experience_count[connected])
                if np.any(connected)
                else 0.0
            ),
            "baseline_active_edges": float(
                np.count_nonzero(connected & (self.prelesion_peak_activity > 0.0))
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
            if self.candidate_experience_count[edge] < self.candidate_required_experiences:
                continue
            self._reactivate_pathway(source, target, automatic=True)
            if self.pathway_state[edge] == PATHWAY_PROTECTED:
                self.last_promoted_edges.add(edge)
                self.candidate_experience_count[edge] = 0

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
            self.prelesion_peak_activity = np.maximum(
                self.prelesion_peak_activity, peak_contributions
            )
        self._process_recovery_candidates(peak_contributions)

        return SurfaceFlowResult(
            source_nodes=sorted(sources),
            output_history=output_history,
            activation_history=history,
            traversed_edges=traversed,
            final_activation=activation.copy(),
        )
