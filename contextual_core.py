from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from brain import SignalResult, SphereBrain


class ContextualSphereBrain(SphereBrain):
    """Backward-compatible Core extension for persistent contextual propagation.

    Existing ``SphereBrain.propagate`` remains unchanged.  This class adds a
    separate contextual path so experiments and saved brains can opt in without
    changing historical behavior or file formats.
    """

    @classmethod
    def from_brain(cls, source: SphereBrain) -> "ContextualSphereBrain":
        brain = cls(
            node_count=source.node_count,
            neighbors_per_node=source.neighbors_per_node,
            seed=source.seed,
            learning_rate=source.learning_rate,
            decay_rate=source.decay_rate,
            propagation_mode=source.propagation_mode,
            signal_decay=source.signal_decay,
            max_branches=source.max_branches,
            max_active_per_step=source.max_active_per_step,
            max_total_active_nodes=source.max_total_active_nodes,
        )
        brain.positions = source.positions.copy()
        brain.adjacency = source.adjacency.copy()
        brain.weights = source.weights.copy()
        brain.usage = source.usage.copy()
        brain.node_usage = source.node_usage.copy()
        return brain

    def propagate_contextual(
        self,
        source_nodes: Iterable[int],
        context_activation: Mapping[int, float],
        *,
        steps: int = 18,
        threshold: float = 0.18,
        noise: float = 0.0,
        learn: bool = False,
        context_anchor: float = 0.58,
        context_decay: float = 0.94,
        resonance: bool = True,
    ) -> SignalResult:
        """Propagate while preserving a weighted context through every step.

        ``context_activation`` carries route/history information as weighted
        nodes.  Values are normalized to ``context_anchor`` and then retained
        with ``context_decay``.  With resonance enabled, weak contributions are
        combined before thresholding instead of being discarded independently.
        """
        threshold = max(float(threshold), 0.18)
        context_anchor = float(np.clip(context_anchor, threshold, 1.0))
        context_decay = float(np.clip(context_decay, 0.0, 1.0))
        sources = [int(node) for node in source_nodes]

        raw_context = {
            int(node): max(0.0, float(value))
            for node, value in context_activation.items()
            if 0 <= int(node) < self.node_count and float(value) > 0
        }
        peak = max(raw_context.values(), default=0.0)
        context = {
            node: min(1.0, value / peak * context_anchor)
            for node, value in raw_context.items()
        } if peak else {}

        activation = np.zeros(self.node_count, dtype=float)
        for index, node in enumerate(sources):
            activation[node] = max(activation[node], 1.0 - index * 0.08)
        for node, value in context.items():
            activation[node] = max(activation[node], value)

        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]
        retained_context = dict(context)

        for _ in range(max(0, int(steps))):
            active_sources = np.flatnonzero(activation > 0)
            if active_sources.size == 0:
                break

            contributions: dict[int, list[tuple[float, int]]] = {}
            for source in active_sources:
                source = int(source)
                neighbors = np.flatnonzero(self.adjacency[source])
                if neighbors.size == 0:
                    continue
                scores = activation[source] * self.weights[source, neighbors]
                branch_count = min(self.max_branches, neighbors.size)
                best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
                for local_index in best_indices:
                    target = int(neighbors[local_index])
                    value = float(scores[local_index]) * self.signal_decay
                    # Resonance must combine weak signals before thresholding.
                    if not resonance and value < threshold:
                        continue
                    if value > 0:
                        contributions.setdefault(target, []).append((value, source))

            candidates: list[tuple[int, float, list[tuple[float, int]]]] = []
            for target, items in contributions.items():
                if resonance:
                    value = min(1.0, sum(item[0] for item in items))
                    used = items
                else:
                    strongest = max(items, key=lambda item: item[0])
                    value = strongest[0]
                    used = [strongest]
                if value >= threshold:
                    candidates.append((target, value, used))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[1], reverse=True)
            remaining_capacity = max(0, self.max_total_active_nodes - len(activated_nodes))
            step_limit = min(self.max_active_per_step, len(candidates))
            selected: list[tuple[int, float, list[tuple[float, int]]]] = []
            new_count = 0
            for item in candidates:
                target = item[0]
                is_new = target not in activated_nodes
                if is_new and new_count >= remaining_capacity:
                    continue
                selected.append(item)
                if is_new:
                    new_count += 1
                if len(selected) >= step_limit:
                    break

            if not selected:
                break

            next_activation = np.zeros(self.node_count, dtype=float)
            for target, value, used in selected:
                if noise:
                    value += float(self.rng.normal(0.0, min(noise, 0.006)))
                value = float(np.clip(value, 0.0, 1.0))
                if value < threshold:
                    continue
                next_activation[target] = max(next_activation[target], value)
                for _, source in used:
                    traversed_edges.add(tuple(sorted((int(source), int(target)))))

            # The context remains present while the new component is processed.
            next_retained: dict[int, float] = {}
            for node, value in retained_context.items():
                retained = value * context_decay
                if retained >= threshold:
                    next_activation[node] = max(next_activation[node], retained)
                    next_retained[node] = retained
            retained_context = next_retained

            active_now = np.flatnonzero(next_activation > 0).tolist()
            if not active_now:
                break
            activated_nodes.update(active_now)
            history.append(active_now)
            activation = next_activation
            if len(activated_nodes) >= self.max_total_active_nodes:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        return SignalResult(
            source_nodes=sources,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=history,
            final_activation=activation.copy(),
        )
