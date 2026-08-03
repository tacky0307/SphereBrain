from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    merge_contexts,
    result_to_context,
)
from sphere_world_brain import _jaccard

CHARACTERS = ["魔王", "勇者", "騎士", "スライム"]
ANSWER_PORTS = ["肯定", "否定", "同等", "不明"]

# 研究用に与える比較経験。推論時に答え表としては参照せず、
# 各経験から生じたCore経路を「上位・下位・同等」の関係別に保持する。
TRAINING_COMPARISONS = [
    ("魔王", "勇者", "肯定"),
    ("勇者", "魔王", "否定"),
    ("勇者", "騎士", "肯定"),
    ("騎士", "勇者", "否定"),
    ("騎士", "スライム", "肯定"),
    ("スライム", "騎士", "否定"),
    ("魔王", "スライム", "肯定"),
    ("スライム", "魔王", "否定"),
]


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"

    def as_input(self) -> StructuredInput:
        return StructuredInput(self.subject, self.relation, self.content)


def question_facts(subject: str, target: str) -> list[Fact]:
    """Encode the direction of the comparison as the central world structure."""
    return [
        Fact("質問主体", "対象", subject),
        Fact("比較対象", "対象", target),
        Fact("比較方向", "始点", subject),
        Fact("比較方向", "終点", target),
        Fact("比較方向", "組合せ", f"{subject}→{target}"),
        Fact("比較関係", "種類", "強さ"),
        Fact("質問形式", "判定", "より強いか"),
    ]


class SphereTalkStrengthBrain:
    """Two-stage strength question experiment.

    Stage 1: recall the learned relation route (higher/lower/equal/unknown).
    Stage 2: Language Decoder turns that relation into a Japanese answer.
    """

    def __init__(self, repeats: int = 8) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.relation_prototypes: dict[str, list] = {
            "肯定": [],
            "否定": [],
            "同等": [],
        }
        self.trained_pairs: set[tuple[str, str]] = set()
        self._train()

    def _question_context(
        self,
        subject: str,
        target: str,
        *,
        learn: bool,
    ) -> tuple[dict[int, float], list[str]]:
        contexts = []
        labels = []
        for fact in question_facts(subject, target):
            exp = encode_and_experience_contextual(
                self.brain,
                fact.as_input(),
                learn=learn,
            )
            # The directed pair is the most important distinguishing information.
            if fact.subject == "比較方向" and fact.relation == "組合せ":
                scale = 1.65
            elif fact.subject == "比較方向":
                scale = 1.30
            else:
                scale = 0.82
            contexts.append((result_to_context(exp.content_result), scale))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _relation_probe(self, question_context: dict[int, float], *, learn: bool):
        """Ask Core for the strength relation without supplying an answer label."""
        relation_input = StructuredInput("比較質問", "想起", "強さ関係")
        exp = encode_and_experience_contextual(
            self.brain,
            relation_input,
            learn=learn,
        )
        relation_context = result_to_context(exp.content_result)
        decision_context = merge_contexts(
            (question_context, 1.0),
            (relation_context, 0.72),
        )
        return self.brain.propagate_contextual(
            [],
            decision_context,
            steps=12,
            threshold=0.18,
            noise=0.003 if learn else 0.0,
            learn=learn,
            context_anchor=0.62,
            context_decay=0.95,
            resonance=True,
        )

    def _experience_pair(self, subject: str, target: str, answer: str) -> None:
        for _ in range(self.repeats):
            context, _ = self._question_context(subject, target, learn=True)
            self._relation_probe(context, learn=True)
        # Build the prototype at exactly the same answer-free stage used for questions.
        context, _ = self._question_context(subject, target, learn=False)
        self.relation_prototypes[answer].append(self._relation_probe(context, learn=False))
        self.trained_pairs.add((subject, target))

    def _train(self) -> None:
        for subject, target, answer in TRAINING_COMPARISONS:
            self._experience_pair(subject, target, answer)

        for character in CHARACTERS:
            self._experience_pair(character, character, "同等")

    @staticmethod
    def verbalize(answer: str, subject: str, target: str) -> str:
        if answer == "肯定":
            return f"はい。{subject}は{target}より強いです。"
        if answer == "否定":
            return f"いいえ。{subject}は{target}より強くありません。"
        if answer == "同等":
            return f"{subject}と{target}は同じくらいです。"
        return f"{subject}と{target}の強さの関係は、まだ分かりません。"

    def answer(self, subject: str, target: str) -> dict:
        if subject not in CHARACTERS or target not in CHARACTERS:
            raise ValueError("登場人物を選び直してください。")

        question_context, labels = self._question_context(subject, target, learn=False)
        raw = self._relation_probe(question_context, learn=False)
        raw_nodes = set(raw.activated_nodes)
        raw_edges = {tuple(sorted((int(a), int(b)))) for a, b in raw.traversed_edges}

        trained_pair = (subject, target) in self.trained_pairs
        scores: dict[str, float] = {answer: 0.0 for answer in ANSWER_PORTS}
        details: dict[str, dict] = {}

        for answer in ("肯定", "否定", "同等"):
            best_score = 0.0
            best_nodes = 0.0
            best_edges = 0.0
            common_nodes = 0
            common_edges = 0
            for prototype in self.relation_prototypes[answer]:
                p_nodes = set(prototype.activated_nodes)
                p_edges = {
                    tuple(sorted((int(a), int(b))))
                    for a, b in prototype.traversed_edges
                }
                node_score = _jaccard(raw_nodes, p_nodes)
                edge_score = _jaccard(raw_edges, p_edges)
                score = 0.30 * node_score + 0.70 * edge_score
                if score > best_score:
                    best_score = score
                    best_nodes = node_score
                    best_edges = edge_score
                    common_nodes = len(raw_nodes & p_nodes)
                    common_edges = len(raw_edges & p_edges)
            scores[answer] = best_score
            details[answer] = {
                "node_similarity": best_nodes,
                "edge_similarity": best_edges,
                "common_nodes": common_nodes,
                "common_edges": common_edges,
            }

        # No stored directed-pair experience means the system should admit uncertainty.
        # This checks memory availability, not the correct semantic answer.
        scores["不明"] = 1.0 if not trained_pair else 0.0
        details["不明"] = {
            "node_similarity": 0.0,
            "edge_similarity": 0.0,
            "common_nodes": 0,
            "common_edges": 0,
        }

        maximum = max(scores.values(), default=1.0) or 1.0
        candidates = [
            {
                "answer": answer,
                "score": scores[answer] / maximum,
                "raw_score": scores[answer],
                **details[answer],
            }
            for answer in ANSWER_PORTS
        ]
        candidates.sort(key=lambda item: (-item["score"], item["answer"]))
        selected = candidates[0]["answer"] if candidates else "不明"

        return {
            "subject": subject,
            "target": target,
            "question": f"{subject}は{target}より強いですか？",
            "selected_answer": selected,
            "speech": self.verbalize(selected, subject, target),
            "facts": labels,
            "candidates": candidates,
            "trained_pair": trained_pair,
            "raw_nodes": len(raw_nodes),
            "raw_edges": len(raw_edges),
            "decoder": "Two-Stage Relation Recall → Language Decoder",
        }
