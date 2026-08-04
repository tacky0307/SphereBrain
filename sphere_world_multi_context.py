from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput, component_nodes
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    merge_contexts,
    result_to_context,
)
from sphere_world import ACTIONS, POSITIONS, SphereWorld
from sphere_world_brain import ActionCandidate, _jaccard


TRAIN_STATES = {
    (0, 2): "右へ移動",
    (2, 0): "左へ移動",
    (1, 1): "停止",
}


@dataclass(frozen=True)
class WorldFact:
    subject: str
    relation: str
    content: str

    def as_input(self) -> StructuredInput:
        return StructuredInput(self.subject, self.relation, self.content)

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


def relative_label(player_position: int, enemy_position: int) -> str:
    if player_position < enemy_position:
        return "Enemyより左"
    if player_position > enemy_position:
        return "Enemyより右"
    return "Enemyと同じ"


def facts_for_positions(player_position: int, enemy_position: int) -> list[WorldFact]:
    touching = player_position == enemy_position
    return [
        WorldFact("Player", "位置", POSITIONS[player_position]),
        WorldFact("Enemy", "位置", POSITIONS[enemy_position]),
        WorldFact("Player", "相対位置", relative_label(player_position, enemy_position)),
        WorldFact("Player", "接触", "している" if touching else "していない"),
    ]


class MultiContextSphereWorldBrain:
    """SphereWorld 0.2 brain that receives a scene as multiple contextual facts."""

    def __init__(self, repeats: int = 12) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.prototypes: dict[str, list] = {action: [] for action in ACTIONS}
        self._train_sparse_policy()

    @staticmethod
    def action_for_positions(player_position: int, enemy_position: int) -> str:
        if player_position < enemy_position:
            return "右へ移動"
        if player_position > enemy_position:
            return "左へ移動"
        return "停止"

    def _world_context(self, player_position: int, enemy_position: int, *, learn: bool) -> tuple[dict[int, float], list]:
        fact_results = []
        contexts = []
        for fact in facts_for_positions(player_position, enemy_position):
            experience = encode_and_experience_contextual(
                self.brain,
                fact.as_input(),
                learn=learn,
            )
            fact_results.append(experience.content_result)
            contexts.append((result_to_context(experience.content_result), 1.0))

        # Max-based merging preserves the strongest evidence from every object/fact.
        world_context = merge_contexts(*contexts)
        return world_context, fact_results

    def _continue_to_action(self, world_context: dict[int, float], *, action: str | None, learn: bool):
        noise = 0.004 if learn else 0.0
        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "次行動", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            world_context,
            steps=8,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        relation_context = result_to_context(relation_result)
        decision_context = merge_contexts((world_context, 0.78), (relation_context, 1.0))

        if action is None:
            return relation_result, self.brain.propagate_contextual(
                [],
                decision_context,
                steps=10,
                threshold=0.18,
                noise=0.0,
                learn=False,
            )

        content_sources = (
            component_nodes(self.brain, "role:content", "content", 2)
            + component_nodes(self.brain, "content", action, 3)
        )
        action_result = self.brain.propagate_contextual(
            content_sources,
            decision_context,
            steps=10,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        return relation_result, action_result

    def _train_sparse_policy(self) -> None:
        for (player_position, enemy_position), action in TRAIN_STATES.items():
            for _ in range(self.repeats):
                world_context, _ = self._world_context(player_position, enemy_position, learn=True)
                self._continue_to_action(world_context, action=action, learn=True)

        # Fair prototypes: content is omitted, exactly as at decision time.
        for (player_position, enemy_position), action in TRAIN_STATES.items():
            world_context, _ = self._world_context(player_position, enemy_position, learn=False)
            _, raw_result = self._continue_to_action(world_context, action=None, learn=False)
            self.prototypes[action].append(raw_result)

    def probe(self, player_position: int, enemy_position: int) -> dict:
        world_context, fact_results = self._world_context(player_position, enemy_position, learn=False)
        relation_result, raw_result = self._continue_to_action(world_context, action=None, learn=False)
        return {
            "facts": [fact.label for fact in facts_for_positions(player_position, enemy_position)],
            "fact_node_counts": [len(result.activated_nodes) for result in fact_results],
            "world_context_nodes": len(world_context),
            "relation_result": relation_result,
            "raw_result": raw_result,
        }

    def decide_positions(self, player_position: int, enemy_position: int) -> dict:
        probe = self.probe(player_position, enemy_position)
        raw_result = probe["raw_result"]
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
        return {
            "selected_action": candidates[0].action if candidates else "停止",
            "candidates": [candidate.to_dict() for candidate in candidates],
            "facts": probe["facts"],
            "world_context_nodes": probe["world_context_nodes"],
            "raw_nodes": len(raw_result.activated_nodes),
            "raw_edges": len(raw_result.traversed_edges),
        }


def evaluate_multi_context(repeats: int = 12) -> dict:
    brain = MultiContextSphereWorldBrain(repeats=repeats)
    rows = []
    correct_seen = correct_unseen = seen_count = unseen_count = 0

    for player_position in range(3):
        for enemy_position in range(3):
            decision = brain.decide_positions(player_position, enemy_position)
            expected = brain.action_for_positions(player_position, enemy_position)
            trained = (player_position, enemy_position) in TRAIN_STATES
            candidates = decision["candidates"]
            margin = candidates[0]["score"] - candidates[1]["score"] if len(candidates) > 1 else 0.0
            correct = decision["selected_action"] == expected
            if trained:
                seen_count += 1
                correct_seen += int(correct)
            else:
                unseen_count += 1
                correct_unseen += int(correct)

            rows.append({
                "player": POSITIONS[player_position],
                "enemy": POSITIONS[enemy_position],
                "trained": trained,
                "expected": expected,
                "selected": decision["selected_action"],
                "correct": correct,
                "margin": margin,
                "candidates": candidates,
                "facts": decision["facts"],
                "world_context_nodes": decision["world_context_nodes"],
            })

    return {
        "rows": rows,
        "seen_accuracy": correct_seen / seen_count if seen_count else 0.0,
        "unseen_accuracy": correct_unseen / unseen_count if unseen_count else 0.0,
        "overall_accuracy": (correct_seen + correct_unseen) / 9.0,
        "train_states": [
            {
                "player": POSITIONS[p],
                "enemy": POSITIONS[e],
                "action": action,
                "facts": [fact.label for fact in facts_for_positions(p, e)],
            }
            for (p, e), action in TRAIN_STATES.items()
        ],
    }
