from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_encoder_v2 import StructuredInput, component_nodes
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    merge_contexts,
    result_to_context,
)

CHARACTERS = ["魔王", "勇者", "騎士", "スライム"]
ANSWER_PORTS = ["肯定", "否定", "同等", "不明"]

# 研究用に与える比較経験。推論時にこの表を直接参照しない。
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
    return [
        Fact("質問主体", "対象", subject),
        Fact("比較対象", "対象", target),
        Fact("比較関係", "種類", "強さ"),
        Fact("質問形式", "判定", "より強いか"),
        Fact("質問", "組合せ", f"{subject}対{target}"),
    ]


class SphereTalkStrengthBrain:
    def __init__(self, repeats: int = 8) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.answer_ports = {
            answer: component_nodes(self.brain, "answer_port_strength", answer, 4)
            for answer in ANSWER_PORTS
        }
        self._train()

    def _question_context(self, subject: str, target: str, *, learn: bool) -> tuple[dict[int, float], list[str]]:
        contexts = []
        labels = []
        for fact in question_facts(subject, target):
            exp = encode_and_experience_contextual(self.brain, fact.as_input(), learn=learn)
            scale = 1.25 if fact.subject in {"質問主体", "比較対象", "質問"} else 1.0
            contexts.append((result_to_context(exp.content_result), scale))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _decision_context(self, question_context: dict[int, float], *, learn: bool) -> dict[int, float]:
        noise = 0.004 if learn else 0.0
        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "回答", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            question_context,
            steps=8,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        return merge_contexts(
            (question_context, 0.78),
            (result_to_context(relation_result), 1.0),
        )

    def _train_answer(self, decision_context: dict[int, float], answer: str) -> None:
        sources = self.answer_ports[answer]
        self.brain.propagate_contextual(
            sources,
            decision_context,
            steps=12,
            threshold=0.18,
            noise=0.004,
            learn=True,
            context_anchor=0.72,
            context_decay=0.96,
            resonance=True,
        )

    def _train(self) -> None:
        for subject, target, answer in TRAINING_COMPARISONS:
            for _ in range(self.repeats):
                q_context, _ = self._question_context(subject, target, learn=True)
                d_context = self._decision_context(q_context, learn=True)
                self._train_answer(d_context, answer)

        # 同一人物同士は同等として教育。
        for character in CHARACTERS:
            for _ in range(max(2, self.repeats // 2)):
                q_context, _ = self._question_context(character, character, learn=True)
                d_context = self._decision_context(q_context, learn=True)
                self._train_answer(d_context, "同等")

    def _read_ports(self, decision_context: dict[int, float]):
        result = self.brain.propagate_contextual(
            [],
            decision_context,
            steps=16,
            threshold=0.16,
            noise=0.0,
            learn=False,
            context_anchor=0.64,
            context_decay=0.96,
            resonance=True,
        )
        final = np.asarray(result.final_activation, dtype=float)
        history = list(result.activation_history or [])
        recent = history[-5:] if history else []
        raw_scores = {}
        details = {}
        for answer, nodes in self.answer_ports.items():
            final_strength = max((float(final[node]) for node in nodes), default=0.0)
            recent_hits = sum(1 for step in recent for node in nodes if node in step)
            active_count = len(set(nodes) & set(result.activated_nodes))
            incoming = sum(1 for a, b in result.traversed_edges if int(a) in nodes or int(b) in nodes)
            score = final_strength + 0.12 * recent_hits + 0.06 * active_count + 0.02 * incoming
            raw_scores[answer] = score
            details[answer] = {
                "port_nodes": nodes,
                "final_strength": final_strength,
                "recent_hits": recent_hits,
                "active_port_nodes": active_count,
                "incoming_edges": incoming,
            }
        return result, raw_scores, details

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
        q_context, labels = self._question_context(subject, target, learn=False)
        d_context = self._decision_context(q_context, learn=False)
        raw, raw_scores, details = self._read_ports(d_context)

        # 未教育ペアでは、最大発火が弱い場合は不明へ寄せる。
        trained_pair = any(a == subject and b == target for a, b, _ in TRAINING_COMPARISONS) or subject == target
        if not trained_pair:
            raw_scores["不明"] += 0.18

        maximum = max(raw_scores.values(), default=1.0) or 1.0
        candidates = []
        for answer in ANSWER_PORTS:
            item = {
                "answer": answer,
                "score": raw_scores[answer] / maximum,
                "raw_score": raw_scores[answer],
                **details[answer],
            }
            candidates.append(item)
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
            "raw_nodes": len(set(raw.activated_nodes)),
            "raw_edges": len({tuple(sorted(edge)) for edge in raw.traversed_edges}),
            "decoder": "Strength Answer Port Decoder",
        }
