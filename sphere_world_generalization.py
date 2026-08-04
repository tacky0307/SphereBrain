from __future__ import annotations

from sphere_world import ACTIONS, SphereWorld
from sphere_world_brain import ActionCandidate, SphereWorldBrain, _jaccard
from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import encode_and_experience_contextual


TRAIN_STATES = {
    (0, 2): "右へ移動",
    (2, 0): "左へ移動",
    (1, 1): "停止",
}


class SparseSphereWorldBrain(SphereWorldBrain):
    """SphereWorld brain trained on only three representative states."""

    def _train_minimal_policy(self) -> None:
        for (player_position, enemy_position), action in TRAIN_STATES.items():
            item = StructuredInput(
                self.state_subject(player_position, enemy_position),
                "次行動",
                action,
            )
            for _ in range(self.repeats):
                encode_and_experience_contextual(self.brain, item, learn=True)

        # Build fair Raw-Output prototypes from the same stage used at decision time.
        self.prototypes = {action: [] for action in ACTIONS}
        for (player_position, enemy_position), action in TRAIN_STATES.items():
            world = SphereWorld(player_position, enemy_position)
            _, _, raw_result = self._probe(world)
            self.prototypes[action].append(raw_result)


def evaluate_all_states(repeats: int = 12) -> dict:
    brain = SparseSphereWorldBrain(repeats=repeats)
    rows = []
    correct_seen = 0
    correct_unseen = 0
    seen_count = 0
    unseen_count = 0

    for player_position in range(3):
        for enemy_position in range(3):
            world = SphereWorld(player_position, enemy_position)
            decision = brain.decide(world)
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
                "player": world.player.position_label,
                "enemy": world.enemy.position_label,
                "trained": trained,
                "expected": expected,
                "selected": decision["selected_action"],
                "correct": correct,
                "margin": margin,
                "candidates": candidates,
            })

    return {
        "rows": rows,
        "train_states": [
            {"player": SphereWorld(p, e).player.position_label,
             "enemy": SphereWorld(p, e).enemy.position_label,
             "action": action}
            for (p, e), action in TRAIN_STATES.items()
        ],
        "seen_accuracy": correct_seen / seen_count if seen_count else 0.0,
        "unseen_accuracy": correct_unseen / unseen_count if unseen_count else 0.0,
        "overall_accuracy": (correct_seen + correct_unseen) / 9.0,
    }
