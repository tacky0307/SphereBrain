from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import encode_and_experience_contextual, load_contextual_brain
from sphere_world_brain import _jaccard

POSITIONS = ["左", "中央", "右"]
ACTIONS = ["左へ移動", "停止", "右へ移動"]
TRAINING = {
    "左": "右へ移動",
    "中央": "停止",
    "右": "左へ移動",
}


@dataclass(frozen=True)
class BallExperience:
    position: str
    action: str
    result: object


class BallCenterBrain:
    """Minimal route-choice experiment: move a ball toward the center."""

    def __init__(self, repeats: int = 8) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.experiences: list[BallExperience] = []
        self._train()

    def _run(self, position: str, *, learn: bool):
        item = StructuredInput("ボール", "位置", position)
        return encode_and_experience_contextual(self.brain, item, learn=learn).content_result

    def _train(self) -> None:
        for position, action in TRAINING.items():
            for _ in range(self.repeats):
                self._run(position, learn=True)
            prototype = self._run(position, learn=False)
            self.experiences.append(BallExperience(position, action, prototype))

    def decide(self, position: str) -> dict:
        if position not in POSITIONS:
            raise ValueError("ボールの位置を選び直してください。")

        current = self._run(position, learn=False)
        current_nodes = set(int(v) for v in current.activated_nodes)
        current_edges = {tuple(sorted((int(a), int(b)))) for a, b in current.traversed_edges}

        candidates = []
        for item in self.experiences:
            p_nodes = set(int(v) for v in item.result.activated_nodes)
            p_edges = {tuple(sorted((int(a), int(b)))) for a, b in item.result.traversed_edges}
            node_score = _jaccard(current_nodes, p_nodes)
            edge_score = _jaccard(current_edges, p_edges)
            score = 0.30 * node_score + 0.70 * edge_score
            candidates.append({
                "position": item.position,
                "action": item.action,
                "score": score,
                "node_score": node_score,
                "edge_score": edge_score,
                "common_nodes": len(current_nodes & p_nodes),
                "common_edges": len(current_edges & p_edges),
            })

        candidates.sort(key=lambda row: (-row["score"], row["action"]))
        maximum = max((row["score"] for row in candidates), default=1.0) or 1.0
        for row in candidates:
            row["normalized"] = row["score"] / maximum

        selected = candidates[0]["action"] if candidates else "停止"
        next_position = {
            ("左", "右へ移動"): "中央",
            ("右", "左へ移動"): "中央",
            ("中央", "停止"): "中央",
        }.get((position, selected), position)

        return {
            "position": position,
            "selected_action": selected,
            "next_position": next_position,
            "speech": f"{selected}します。" if selected != "停止" else "中央なので停止します。",
            "correct": selected == TRAINING[position],
            "expected_action": TRAINING[position],
            "candidates": candidates,
            "facts": [
                f"ボール｜位置｜{position}",
                "目標｜位置｜中央",
                "要求｜行動｜中央へ寄せる",
            ],
            "raw_nodes": len(current_nodes),
            "raw_edges": len(current_edges),
            "repeats": self.repeats,
        }
