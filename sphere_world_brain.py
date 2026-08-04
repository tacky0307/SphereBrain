from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput, component_nodes
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    merge_contexts,
    result_to_context,
)
from sphere_world import ACTIONS, SphereWorld


@dataclass
class ActionCandidate:
    action: str
    score: float
    node_score: float
    edge_score: float
    common_nodes: int
    common_edges: int

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": self.score,
            "node_score": self.node_score,
            "edge_score": self.edge_score,
            "common_nodes": self.common_nodes,
            "common_edges": self.common_edges,
        }


def _jaccard(left, right) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class SphereWorldBrain:
    """Dedicated in-memory SphereBrain copy for SphereWorld 0.1."""

    def __init__(self, repeats: int = 12) -> None:
        self.brain = load_contextual_brain()
        self.prototypes: dict[str, list] = {action: [] for action in ACTIONS}
        self.repeats = max(1, int(repeats))
        self._train_minimal_policy()

    @staticmethod
    def action_for_positions(player_position: int, enemy_position: int) -> str:
        if player_position < enemy_position:
            return "右へ移動"
        if player_position > enemy_position:
            return "左へ移動"
        return "停止"

    @staticmethod
    def state_subject(player_position: int, enemy_position: int) -> str:
        labels = ("左", "中央", "右")
        return f"Player{labels[player_position]}_Enemy{labels[enemy_position]}"

    def _train_minimal_policy(self) -> None:
        """Train the disposable Core, then record fair Raw Output prototypes.

        The previous implementation compared a no-content Raw Output probe with
        content-stage results that already contained an explicit action stimulus.
        That made the comparison asymmetric.  Here all nine world states are
        trained first, then each state is probed again without an action/content
        stimulus.  Decision-time Raw Output is therefore compared only with
        Raw Output produced under the same conditions.
        """
        states: list[tuple[int, int, str]] = []

        # Phase 1: form state -> action experiences in the disposable Core.
        for player_position in range(3):
            for enemy_position in range(3):
                action = self.action_for_positions(player_position, enemy_position)
                states.append((player_position, enemy_position, action))
                item = StructuredInput(
                    self.state_subject(player_position, enemy_position),
                    "次行動",
                    action,
                )
                for _ in range(self.repeats):
                    encode_and_experience_contextual(self.brain, item, learn=True)

        # Phase 2: create prototypes from no-content Raw Output, exactly like
        # the state presented at decision time.
        for player_position, enemy_position, action in states:
            world = SphereWorld(player_position, enemy_position)
            _, _, raw_result = self._probe(world)
            self.prototypes[action].append(raw_result)

    def _probe(self, world: SphereWorld):
        subject = self.state_subject(world.player.position, world.enemy.position)
        subject_sources = (
            component_nodes(self.brain, "role:subject", "subject", 2)
            + component_nodes(self.brain, "entity", subject, 3)
        )
        subject_result = self.brain.propagate(
            subject_sources,
            steps=8,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
        subject_context = result_to_context(subject_result)

        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "次行動", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            subject_context,
            steps=8,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
        relation_context = result_to_context(relation_result)
        context = merge_contexts((subject_context, 0.72), (relation_context, 1.0))

        # No content/action stimulus is supplied. This is the Core's raw
        # continuation from the current world state plus the next-action cue.
        raw_result = self.brain.propagate_contextual(
            [],
            context,
            steps=10,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
        return subject_result, relation_result, raw_result

    def decide(self, world: SphereWorld) -> dict:
        subject_result, relation_result, raw_result = self._probe(world)
        raw_nodes = set(raw_result.activated_nodes)
        raw_edges = set(tuple(edge) for edge in raw_result.traversed_edges)

        candidates: list[ActionCandidate] = []
        for action, prototypes in self.prototypes.items():
            best = ActionCandidate(action, 0.0, 0.0, 0.0, 0, 0)
            for prototype in prototypes:
                prototype_nodes = set(prototype.activated_nodes)
                prototype_edges = set(tuple(edge) for edge in prototype.traversed_edges)
                node_score = _jaccard(raw_nodes, prototype_nodes)
                edge_score = _jaccard(raw_edges, prototype_edges)
                score = 0.35 * node_score + 0.65 * edge_score
                if score > best.score:
                    best = ActionCandidate(
                        action,
                        score,
                        node_score,
                        edge_score,
                        len(raw_nodes & prototype_nodes),
                        len(raw_edges & prototype_edges),
                    )
            candidates.append(best)

        candidates.sort(key=lambda item: (-item.score, item.action))
        selected = candidates[0].action if candidates else "停止"
        return {
            "selected_action": selected,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "subject_nodes": len(subject_result.activated_nodes),
            "relation_nodes": len(relation_result.activated_nodes),
            "raw_nodes": len(raw_result.activated_nodes),
            "raw_edges": len(raw_result.traversed_edges),
            "raw_top_nodes": sorted(
                ((index, float(value)) for index, value in enumerate(raw_result.final_activation) if value > 0),
                key=lambda item: (-item[1], item[0]),
            )[:12],
        }
