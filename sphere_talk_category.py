from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
)
from sphere_world_brain import _jaccard

CATEGORIES = ["動物", "人工物", "自然", "植物"]

# 一つの主分類だけを持つ、小さな実験世界。
KNOWN_CLASSIFICATIONS = {
    "犬": "動物",
    "猫": "動物",
    "鳥": "動物",
    "車": "人工物",
    "船": "人工物",
    "機械": "人工物",
    "山": "自然",
    "海": "自然",
    "川": "自然",
    "森": "自然",
    "花": "植物",
    "木": "植物",
    "草": "植物",
}

# 未経験主体。既知概念へ似ていても断定せず、不明を返すために残す。
UNKNOWN_SUBJECTS = ["カエル", "ロボット", "雲"]
SUBJECTS = list(KNOWN_CLASSIFICATIONS) + UNKNOWN_SUBJECTS
ANSWERS = ["一致", "不一致", "不明"]

# 各既知主体について、正しい分類を1件、誤分類を同数だけ教育する。
# これにより一致・不一致の経験数を完全に対称化する。
TRAINING_FACTS = [
    (subject, category, "一致" if category == correct else "不一致")
    for subject, correct in KNOWN_CLASSIFICATIONS.items()
    for category in CATEGORIES
]


@dataclass(frozen=True)
class CategoryExperience:
    subject: str
    category: str
    answer: str
    result: object


def question_item(subject: str, category: str) -> StructuredInput:
    return StructuredInput(subject, "種類", category)


class SphereTalkCategoryBrain:
    """Classify a subject/category relation from experienced Core routes."""

    def __init__(self, repeats: int = 6) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.experiences: list[CategoryExperience] = []
        self._train()

    def _run(self, subject: str, category: str, *, learn: bool):
        return encode_and_experience_contextual(
            self.brain,
            question_item(subject, category),
            learn=learn,
        ).content_result

    def _train(self) -> None:
        for subject, category, answer in TRAINING_FACTS:
            for _ in range(self.repeats):
                self._run(subject, category, learn=True)
            prototype = self._run(subject, category, learn=False)
            self.experiences.append(CategoryExperience(subject, category, answer, prototype))

    @staticmethod
    def verbalize(answer: str, subject: str, category: str) -> str:
        if answer == "一致":
            return f"はい。{subject}は{category}です。"
        if answer == "不一致":
            return f"いいえ。{subject}は{category}ではありません。"
        return f"{subject}が{category}かどうかは、まだ分かりません。"

    def answer(self, subject: str, category: str) -> dict:
        if subject not in SUBJECTS or category not in CATEGORIES:
            raise ValueError("主体または分類を選び直してください。")

        current = self._run(subject, category, learn=False)
        current_nodes = set(current.activated_nodes)
        current_edges = {tuple(sorted(edge)) for edge in current.traversed_edges}

        trained_subject = subject in KNOWN_CLASSIFICATIONS
        exact = [
            item
            for item in self.experiences
            if item.subject == subject and item.category == category
        ]

        answer_scores = {answer: 0.0 for answer in ANSWERS}
        answer_details = {
            answer: {"node": 0.0, "edge": 0.0, "matches": 0}
            for answer in ANSWERS
        }

        for item in self.experiences:
            prototype_nodes = set(item.result.activated_nodes)
            prototype_edges = {tuple(sorted(edge)) for edge in item.result.traversed_edges}
            node_score = _jaccard(current_nodes, prototype_nodes)
            edge_score = _jaccard(current_edges, prototype_edges)
            score = 0.30 * node_score + 0.70 * edge_score

            # 主体固有経路を最優先し、分類全体の共通経路への偏りを抑える。
            if item.subject == subject:
                score *= 1.55
                if item.category == category:
                    score *= 1.20
            else:
                score *= 0.42

            if score > answer_scores[item.answer]:
                answer_scores[item.answer] = score
                answer_details[item.answer] = {
                    "node": node_score,
                    "edge": edge_score,
                    "matches": len(current_edges & prototype_edges),
                }

        # 未経験主体は、既知の似た経路だけで分類しない。
        if not trained_subject:
            answer_scores["不明"] = max(answer_scores.values(), default=0.0) + 0.25
        elif not exact:
            answer_scores["不明"] = max(answer_scores.values(), default=0.0) + 0.10

        maximum = max(answer_scores.values(), default=1.0) or 1.0
        candidates = [
            {
                "answer": answer,
                "score": answer_scores[answer] / maximum,
                "raw_score": answer_scores[answer],
                **answer_details[answer],
            }
            for answer in ANSWERS
        ]
        candidates.sort(key=lambda item: (-item["score"], item["answer"]))
        selected = candidates[0]["answer"] if candidates else "不明"

        return {
            "subject": subject,
            "category": category,
            "question": f"{subject}は{category}ですか？",
            "selected_answer": selected,
            "speech": self.verbalize(selected, subject, category),
            "trained_subject": trained_subject,
            "trained_pair": bool(exact),
            "facts": [
                f"質問主体｜対象｜{subject}",
                f"分類候補｜内容｜{category}",
                "分類関係｜種類｜所属するか",
                f"質問｜組合せ｜{subject}対{category}",
            ],
            "candidates": candidates,
            "raw_nodes": len(current_nodes),
            "raw_edges": len(current_edges),
            "decoder": "Category Route Decoder — Four Categories",
        }
