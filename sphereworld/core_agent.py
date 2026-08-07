from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from brain import SphereBrain


ACTION_ORDER = ("N", "E", "S", "W", "STAY")
OUTCOMES = ("good", "neutral", "bad")


@dataclass(frozen=True)
class Decision:
    action: str
    scores: dict[str, float]
    good_overlap: dict[str, float]
    bad_overlap: dict[str, float]


class CoreAgent:
    """Thin numeric encoder/decoder around the real SphereBrain Core.

    The adapter does not contain a rule such as "move toward food".
    It turns sensory channels/actions/outcomes into fixed Core node anchors,
    lets SphereBrain form pathways, and ranks actions by current Core structure.
    """

    def __init__(
        self,
        brain: SphereBrain,
        brain_path: str | Path,
        seed: int = 17,
    ) -> None:
        self.brain = brain
        self.brain_path = Path(brain_path)
        self.rng = np.random.default_rng(seed)
        self._anchor_cache: dict[str, int] = {}
        self.last_decision: Decision | None = None

    @classmethod
    def load_or_create(
        cls,
        brain_path: str | Path,
        *,
        node_count: int = 240,
        seed: int = 42,
        reset: bool = False,
    ) -> "CoreAgent":
        path = Path(brain_path)
        if path.exists() and not reset:
            brain = SphereBrain.load(path)
        else:
            brain = SphereBrain(
                node_count=node_count,
                neighbors_per_node=7,
                seed=seed,
                propagation_mode="focused",
                structural_assist_enabled=True,
            )
        return cls(brain=brain, brain_path=path, seed=seed + 1000)

    def _anchor(self, label: str) -> int:
        cached = self._anchor_cache.get(label)
        if cached is not None:
            return cached

        digest = hashlib.sha256(("sphereworld:v0.1:" + label).encode("utf-8")).digest()
        candidate = int.from_bytes(digest[:8], "big") % self.brain.node_count
        used = set(self._anchor_cache.values())
        while candidate in used:
            candidate = (candidate + 1) % self.brain.node_count
        self._anchor_cache[label] = candidate
        return candidate

    def sensor_nodes(self, senses: dict[str, str]) -> list[int]:
        # Numeric identity only: the Core never receives the label meaning itself.
        labels = [f"sense:{channel}:{value}" for channel, value in sorted(senses.items())]
        return [self._anchor(label) for label in labels]

    def _action_node(self, action: str) -> int:
        return self._anchor(f"action:{action}")

    def _outcome_node(self, outcome: str) -> int:
        return self._anchor(f"outcome:{outcome}")

    @staticmethod
    def _overlap(a: set[int], b: set[int]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def choose_action(self, senses: dict[str, str]) -> Decision:
        sensors = self.sensor_nodes(senses)

        good_probe = self.brain.propagate(
            [self._outcome_node("good")],
            steps=12,
            noise=0.0,
            learn=False,
        )
        bad_probe = self.brain.propagate(
            [self._outcome_node("bad")],
            steps=12,
            noise=0.0,
            learn=False,
        )
        good_nodes = set(good_probe.activated_nodes)
        bad_nodes = set(bad_probe.activated_nodes)

        scores: dict[str, float] = {}
        good_overlap: dict[str, float] = {}
        bad_overlap: dict[str, float] = {}

        for action in ACTION_ORDER:
            probe = self.brain.propagate(
                sensors + [self._action_node(action)],
                steps=12,
                noise=0.0,
                learn=False,
            )
            active = set(probe.activated_nodes)
            g = self._overlap(active, good_nodes)
            b = self._overlap(active, bad_nodes)

            if probe.traversed_edges:
                familiarity = float(np.mean([
                    self.brain.usage[a, edge_b]
                    for a, edge_b in probe.traversed_edges
                ]))
            else:
                familiarity = 0.0

            # The score is derived only from the current Core state.
            scores[action] = (2.0 * g) - (2.0 * b) + (0.02 * familiarity)
            good_overlap[action] = g
            bad_overlap[action] = b

        best = max(scores.values())
        candidates = [a for a in ACTION_ORDER if abs(scores[a] - best) < 1e-12]
        action = str(self.rng.choice(candidates))
        decision = Decision(
            action=action,
            scores=scores,
            good_overlap=good_overlap,
            bad_overlap=bad_overlap,
        )
        self.last_decision = decision
        return decision

    def experience(
        self,
        senses: dict[str, str],
        action: str,
        outcome: str,
    ) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome: {outcome}")

        sources = self.sensor_nodes(senses)
        sources.extend([self._action_node(action), self._outcome_node(outcome)])

        # Good/bad events are salient, so they are experienced twice.
        # This changes exposure strength, not the choice of a "correct" action.
        repeats = 2 if outcome in {"good", "bad"} else 1
        for _ in range(repeats):
            self.brain.propagate(
                sources,
                steps=16,
                noise=0.004,
                learn=True,
            )

    def save(self) -> None:
        self.brain_path.parent.mkdir(parents=True, exist_ok=True)
        self.brain.save(self.brain_path)

    def core_stats(self) -> dict[str, float | int]:
        used_edges = int(np.count_nonzero(np.triu(self.brain.usage, k=1)))
        total_usage = int(np.triu(self.brain.usage, k=1).sum())
        active_nodes = int(np.count_nonzero(self.brain.node_usage))
        return {
            "used_edges": used_edges,
            "total_edge_usage": total_usage,
            "experienced_nodes": active_nodes,
            "max_edge_weight": float(self.brain.weights.max()),
        }
