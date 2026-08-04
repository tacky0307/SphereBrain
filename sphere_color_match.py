from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
)
from sphere_world_brain import _jaccard

COLORS = ["赤", "青", "緑"]


@dataclass(frozen=True)
class ColorPrototype:
    color: str
    result: object


class SphereColorMatchBrain:
    """Small route-matching experiment for three colors.

    A disposable copy of the existing contextual Core is used. The saved Core and
    semantic database are never modified.
    """

    def __init__(self, repeats: int = 8) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.prototypes: list[ColorPrototype] = []
        self._train()

    @staticmethod
    def _item(color: str) -> StructuredInput:
        return StructuredInput("見本", "色", color)

    def _run(self, color: str, *, learn: bool):
        return encode_and_experience_contextual(
            self.brain,
            self._item(color),
            learn=learn,
        ).content_result

    def _train(self) -> None:
        for color in COLORS:
            for _ in range(self.repeats):
                self._run(color, learn=True)
            self.prototypes.append(
                ColorPrototype(color=color, result=self._run(color, learn=False))
            )

    def decide(self, color: str) -> dict:
        if color not in COLORS:
            raise ValueError("色を選び直してください。")

        current = self._run(color, learn=False)
        current_nodes = set(int(node) for node in current.activated_nodes)
        current_edges = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in current.traversed_edges
        }

        candidates = []
        for prototype in self.prototypes:
            prototype_nodes = set(int(node) for node in prototype.result.activated_nodes)
            prototype_edges = {
                tuple(sorted((int(edge[0]), int(edge[1]))))
                for edge in prototype.result.traversed_edges
            }
            node_score = _jaccard(current_nodes, prototype_nodes)
            edge_score = _jaccard(current_edges, prototype_edges)
            score = 0.30 * node_score + 0.70 * edge_score
            candidates.append(
                {
                    "color": prototype.color,
                    "score": score,
                    "node_score": node_score,
                    "edge_score": edge_score,
                    "common_nodes": len(current_nodes & prototype_nodes),
                    "common_edges": len(current_edges & prototype_edges),
                }
            )

        candidates.sort(key=lambda item: (-item["score"], item["color"]))
        selected = candidates[0]["color"] if candidates else "不明"
        maximum = max((item["score"] for item in candidates), default=1.0) or 1.0
        for item in candidates:
            item["normalized"] = item["score"] / maximum

        return {
            "input_color": color,
            "selected_color": selected,
            "correct": selected == color,
            "speech": f"{selected}を選びます。",
            "facts": [
                "見本｜役割｜選ぶ色",
                f"見本｜色｜{color}",
                "課題｜種類｜色合わせ",
            ],
            "candidates": candidates,
            "raw_nodes": len(current_nodes),
            "raw_edges": len(current_edges),
            "training_repeats": self.repeats,
            "training_experiences": len(COLORS),
            "decoder": "Color Route Match Decoder",
        }
