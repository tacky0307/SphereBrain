from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import encode_and_experience_contextual, load_contextual_brain
from sphere_world_brain import _jaccard

GRID_SIZE = 5
TARGET_ROW = 3
TARGET_COL = 3
TARGET_POSITION = f"{TARGET_ROW},{TARGET_COL}"
ACTIONS = ["上へ移動", "下へ移動", "左へ移動", "右へ移動", "停止"]
POSITIONS = [f"{row},{col}" for row in range(1, GRID_SIZE + 1) for col in range(1, GRID_SIZE + 1)]


def action_toward_target(position: str) -> str:
    row, col = (int(value) for value in position.split(","))
    if row < TARGET_ROW:
        return "下へ移動"
    if row > TARGET_ROW:
        return "上へ移動"
    if col < TARGET_COL:
        return "右へ移動"
    if col > TARGET_COL:
        return "左へ移動"
    return "停止"


TRAINING = {position: action_toward_target(position) for position in POSITIONS}


@dataclass(frozen=True)
class BallExperience:
    position: str
    action: str
    result: object


class BallCenterBrain:
    """Move a ball one cell at a time toward the center of a 5x5 world."""

    def __init__(self, repeats: int = 8) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.experiences: list[BallExperience] = []
        self._train()

    def _run(self, position: str, *, learn: bool):
        item = StructuredInput("ボール", "位置", position)
        return encode_and_experience_contextual(self.brain, item, learn=learn).content_result

    def _train(self) -> None:
        # First finish all learning. Prototypes made during learning become stale
        # because later experiences continue changing the same disposable Core.
        for position in POSITIONS:
            for _ in range(self.repeats):
                self._run(position, learn=True)

        # Build every comparison prototype from the same final Core state.
        self.experiences = [
            BallExperience(position, TRAINING[position], self._run(position, learn=False))
            for position in POSITIONS
        ]

    @staticmethod
    def _next_position(position: str, action: str) -> str:
        row, col = (int(value) for value in position.split(","))
        if action == "上へ移動":
            row = max(1, row - 1)
        elif action == "下へ移動":
            row = min(GRID_SIZE, row + 1)
        elif action == "左へ移動":
            col = max(1, col - 1)
        elif action == "右へ移動":
            col = min(GRID_SIZE, col + 1)
        return f"{row},{col}"

    def decide(self, position: str) -> dict:
        if position not in POSITIONS:
            raise ValueError("ボールの位置を選び直してください。")

        current = self._run(position, learn=False)
        current_nodes = set(int(value) for value in current.activated_nodes)
        current_edges = {
            tuple(sorted((int(a), int(b))))
            for a, b in current.traversed_edges
        }

        route_rows = []
        for item in self.experiences:
            prototype_nodes = set(int(value) for value in item.result.activated_nodes)
            prototype_edges = {
                tuple(sorted((int(a), int(b))))
                for a, b in item.result.traversed_edges
            }
            node_score = _jaccard(current_nodes, prototype_nodes)
            edge_score = _jaccard(current_edges, prototype_edges)
            score = 0.30 * node_score + 0.70 * edge_score
            route_rows.append({
                "position": item.position,
                "action": item.action,
                "score": score,
                "node_score": node_score,
                "edge_score": edge_score,
                "common_nodes": len(current_nodes & prototype_nodes),
                "common_edges": len(current_edges & prototype_edges),
            })

        route_rows.sort(key=lambda row: (-row["score"], row["position"]))
        selected_route = route_rows[0] if route_rows else None
        selected_action = selected_route["action"] if selected_route else "停止"

        # Present action-level scores as the strongest matching position route for
        # each motor action. The selected route itself remains visible separately.
        action_rows = []
        for action in ACTIONS:
            matching = [row for row in route_rows if row["action"] == action]
            if not matching:
                continue
            best = max(matching, key=lambda row: row["score"])
            action_rows.append({
                "action": action,
                "score": best["score"],
                "matched_position": best["position"],
                "node_score": best["node_score"],
                "edge_score": best["edge_score"],
            })
        action_rows.sort(key=lambda row: (-row["score"], row["action"]))
        maximum = max((row["score"] for row in action_rows), default=1.0) or 1.0
        for row in action_rows:
            row["normalized"] = row["score"] / maximum

        next_position = self._next_position(position, selected_action)
        expected_action = TRAINING[position]
        speech = (
            "中央の目標マスなので停止します。"
            if selected_action == "停止"
            else f"{selected_action}します。"
        )
        return {
            "position": position,
            "target_position": TARGET_POSITION,
            "selected_action": selected_action,
            "selected_route_position": selected_route["position"] if selected_route else None,
            "selected_route_score": selected_route["score"] if selected_route else 0.0,
            "next_position": next_position,
            "speech": speech,
            "correct": selected_action == expected_action,
            "expected_action": expected_action,
            "candidates": action_rows,
            "facts": [
                f"ボール｜位置｜{position}",
                f"目標｜位置｜{TARGET_POSITION}",
                "要求｜行動｜中央へ1マス寄せる",
            ],
            "raw_nodes": len(current_nodes),
            "raw_edges": len(current_edges),
            "repeats": self.repeats,
            "grid_size": GRID_SIZE,
        }
