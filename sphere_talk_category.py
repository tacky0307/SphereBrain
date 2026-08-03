from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
)
from sphere_world_brain import _jaccard

SUBJECTS = ["犬", "猫", "鳥", "魚", "車", "船", "カエル"]
CATEGORIES = ["動物", "人工物"]
ANSWERS = ["一致", "不一致", "不明"]

# 研究用の明示経験。カエルは未経験として残す。
TRAINING_FACTS = [
    ("犬", "動物", "一致"),
    ("犬", "人工物", "不一致"),
    ("猫", "動物", "一致"),
    ("猫", "人工物", "不一致"),
    ("鳥", "動物", "一致"),
    ("鳥", "人工物", "不一致"),
    ("魚", "動物", "一致"),
    ("魚", "人工物", "不一致"),
    ("車", "人工物", "一致"),
    ("車", "動物", "不一致"),
    ("船", "人工物", "一致"),
    ("船", "動物", "不一致"),
]


@dataclass(frozen=True)
class CategoryExperience:
    subject: str
    category: str
    answer: str
    result: object


def question_item(subject: str, category: str) -> StructuredInput:
    # 主体・関係・内容をそのまま使い、分類関係の経路を形成する。
    return StructuredInput(subject, "種類", category)


class SphereTalkCategoryBrain:
    """Classify a subject/category relation from experienced Core routes."""

    def __init__(self, repeats: int = 8) -> None:
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
        # 正例と負例を同数用意し、一方の答えへ偏らないようにする。
        for subject, category, answer in TRAINING_FACTS:
            latest = None
            for _ in range(self.repeats):
                latest = self._run(subject, category, learn=True)
            assert latest is not None
            # 推論と同条件のノイズなし経路を基準経路として保存する。
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

        trained_subject = any(item.subject == subject for item in self.experiences)
        exact = [
            item for item in self.experiences
            if item.subject == subject and item.category == category
        ]

        answer_scores = {answer: 0.0 for answer in ANSWERS}
        answer_details = {answer: {"node": 0.0, "edge": 0.0, "matches": 0} for answer in ANSWERS}

        for item in self.experiences:
            prototype_nodes = set(item.result.activated_nodes)
            prototype_edges = {tuple(sorted(edge)) for edge in item.result.traversed_edges}
            node_score = _jaccard(current_nodes, prototype_nodes)
            edge_score = _jaccard(current_edges, prototype_edges)
            score = 0.30 * node_score + 0.70 * edge_score

            # 同一主体の経験を優先し、他主体の共通処理に支配されないようにする。
            if item.subject == subject:
                score *= 1.35
            else:
                score *= 0.55

            if score > answer_scores[item.answer]:
                answer_scores[item.answer] = score
                answer_details[item.answer] = {
                    "node": node_score,
                    "edge": edge_score,
                    "matches": len(current_edges & prototype_edges),
                }

        # 未経験主体では、既知主体の似た経路だけで断定しない。
        if not trained_subject:
            answer_scores["不明"] = max(answer_scores.values(), default=0.0) + 0.20
        elif not exact:
            answer_scores["不明"] = max(answer_scores.values(), default=0.0) + 0.08

        maximum = max(answer_scores.values(), default=1.0) or 1.0
        candidates = []
        for answer in ANSWERS:
            candidates.append({
                "answer": answer,
                "score": answer_scores[answer] / maximum,
                "raw_score": answer_scores[answer],
                **answer_details[answer],
            })
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
            "decoder": "Category Route Decoder",
        }
