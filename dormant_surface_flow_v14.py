from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from dormant_surface_flow_v13 import PromotedContributionTrackingBrain


Edge = tuple[int, int]
Pair = tuple[Edge, Edge]


class PromotedCooccurrenceTrackingBrain(PromotedContributionTrackingBrain):
    """Measure coordination among promoted pathways without changing flow.

    v14a is observation-only.  It records promoted pathways that contribute in
    the same recovery observation, grouped by input region and distinct input
    value.  No bridge, bias, weight change, or teacher signal is introduced.
    """

    def __init__(
        self,
        *args,
        cooccurrence_contribution_threshold: float = 1e-12,
        stable_pair_distinct_inputs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if stable_pair_distinct_inputs < 2:
            raise ValueError("stable_pair_distinct_inputs must be at least 2")
        self.cooccurrence_contribution_threshold = float(
            cooccurrence_contribution_threshold
        )
        self.stable_pair_distinct_inputs = int(stable_pair_distinct_inputs)
        self.cooccurrence_epoch = -1
        self.cooccurrence_events_total = 0
        self.cooccurrence_duplicate_events_ignored = 0
        self.promoted_use_inputs_by_edge_region: dict[
            Edge, dict[int, set[float]]
        ] = defaultdict(lambda: defaultdict(set))
        self.pair_inputs_by_region: dict[
            Pair, dict[int, set[float]]
        ] = defaultdict(lambda: defaultdict(set))
        self.pair_event_counts: dict[Pair, int] = defaultdict(int)
        self.pair_joint_contributions: dict[Pair, list[float]] = defaultdict(list)
        self.pair_product_contributions: dict[Pair, list[float]] = defaultdict(list)
        self._seen_epoch_region_input_pair: set[tuple[int, int, float, Pair]] = set()

    def begin_recovery_epoch(self) -> None:
        super().begin_recovery_epoch()
        self.cooccurrence_epoch += 1

    @staticmethod
    def _canonical_pair(edge_a: Edge, edge_b: Edge) -> Pair:
        return (edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)

    def _record_promoted_cooccurrence(self, peak_contributions: np.ndarray) -> None:
        if self.contribution_phase != "recovery":
            return
        region = self.current_experience_region
        input_key = self.current_experience_input_key
        if region is None or input_key is None:
            return

        used: list[tuple[Edge, float]] = []
        for edge in self.selective_promoted_edges:
            contribution = float(peak_contributions[edge])
            if contribution <= self.cooccurrence_contribution_threshold:
                continue
            used.append((edge, contribution))
            self.promoted_use_inputs_by_edge_region[edge][region].add(input_key)

        for (edge_a, value_a), (edge_b, value_b) in combinations(sorted(used), 2):
            pair = self._canonical_pair(edge_a, edge_b)
            dedupe_key = (self.cooccurrence_epoch, region, input_key, pair)
            if dedupe_key in self._seen_epoch_region_input_pair:
                self.cooccurrence_duplicate_events_ignored += 1
                continue
            self._seen_epoch_region_input_pair.add(dedupe_key)
            self.cooccurrence_events_total += 1
            self.pair_event_counts[pair] += 1
            self.pair_inputs_by_region[pair][region].add(input_key)
            self.pair_joint_contributions[pair].append(min(value_a, value_b))
            self.pair_product_contributions[pair].append(value_a * value_b)

    def _record_existing_promoted(self, peak_contributions: np.ndarray) -> None:
        super()._record_existing_promoted(peak_contributions)
        self._record_promoted_cooccurrence(peak_contributions)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _pair_best_region(self, pair: Pair) -> tuple[int, int]:
        regions = self.pair_inputs_by_region.get(pair, {})
        if not regions:
            return -1, 0
        region, inputs = max(regions.items(), key=lambda item: len(item[1]))
        return int(region), len(inputs)

    def _pair_stability(self, pair: Pair, region: int) -> float:
        if region < 0:
            return 0.0
        edge_a, edge_b = pair
        pair_count = len(self.pair_inputs_by_region[pair].get(region, set()))
        a_count = len(self.promoted_use_inputs_by_edge_region[edge_a].get(region, set()))
        b_count = len(self.promoted_use_inputs_by_edge_region[edge_b].get(region, set()))
        denominator = min(a_count, b_count)
        return 0.0 if denominator == 0 else pair_count / denominator

    def cooccurrence_rows(self) -> list[dict[str, float | int | Edge]]:
        rows: list[dict[str, float | int | Edge]] = []
        for pair in self.pair_event_counts:
            region, distinct_inputs = self._pair_best_region(pair)
            rows.append(
                {
                    "edge_a": pair[0],
                    "edge_b": pair[1],
                    "events": self.pair_event_counts[pair],
                    "best_region": region,
                    "distinct_inputs": distinct_inputs,
                    "stability": self._pair_stability(pair, region),
                    "joint_mean": self._mean(self.pair_joint_contributions[pair]),
                    "product_mean": self._mean(self.pair_product_contributions[pair]),
                    "stable": int(distinct_inputs >= self.stable_pair_distinct_inputs),
                }
            )
        rows.sort(
            key=lambda row: (
                int(row["stable"]),
                int(row["distinct_inputs"]),
                float(row["stability"]),
                float(row["joint_mean"]),
                int(row["events"]),
            ),
            reverse=True,
        )
        return rows

    def cooccurrence_stats(self) -> dict[str, float]:
        rows = self.cooccurrence_rows()
        stable_rows = [row for row in rows if int(row["stable"]) == 1]
        stable_pairs = {
            self._canonical_pair(row["edge_a"], row["edge_b"])
            for row in stable_rows
        }
        adjacency: dict[Edge, set[Edge]] = defaultdict(set)
        for edge_a, edge_b in stable_pairs:
            adjacency[edge_a].add(edge_b)
            adjacency[edge_b].add(edge_a)

        visited: set[Edge] = set()
        component_sizes: list[int] = []
        for start in adjacency:
            if start in visited:
                continue
            stack = [start]
            size = 0
            while stack:
                edge = stack.pop()
                if edge in visited:
                    continue
                visited.add(edge)
                size += 1
                stack.extend(adjacency[edge] - visited)
            component_sizes.append(size)

        return {
            "cooccurrence_events_total": float(self.cooccurrence_events_total),
            "cooccurrence_unique_pairs": float(len(rows)),
            "stable_pairs": float(len(stable_rows)),
            "stable_pair_mean_stability": self._mean(
                [float(row["stability"]) for row in stable_rows]
            ),
            "stable_pair_mean_joint_contribution": self._mean(
                [float(row["joint_mean"]) for row in stable_rows]
            ),
            "stable_pair_mean_distinct_inputs": self._mean(
                [float(row["distinct_inputs"]) for row in stable_rows]
            ),
            "stable_clusters": float(len(component_sizes)),
            "largest_stable_cluster": float(max(component_sizes, default=0)),
            "duplicate_events_ignored": float(
                self.cooccurrence_duplicate_events_ignored
            ),
        }
