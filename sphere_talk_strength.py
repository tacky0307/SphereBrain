from __future__ import annotations

from collections import deque
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
    """Strength-question experiment with fixed answer ports.

    Training grows real Core routes in the direction used at inference:

        question context -> Core route -> answer port

    The comparison table is used only to create experiences. It is never read
    when answering a question.
    """

    def __init__(self, repeats: int = 9) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.answer_ports = self._allocate_distinct_ports()
        self._train()

    def _allocate_distinct_ports(self) -> dict[str, list[int]]:
        used: set[int] = set()
        ports: dict[str, list[int]] = {}
        for answer in ANSWER_PORTS:
            candidates = component_nodes(
                self.brain,
                "answer_port_strength_v2",
                answer,
                14,
            )
            selected: list[int] = []
            for node in candidates:
                node = int(node)
                if node in used:
                    continue
                selected.append(node)
                used.add(node)
                if len(selected) >= 4:
                    break
            if len(selected) < 4:
                for node in range(self.brain.node_count):
                    if node not in used:
                        selected.append(node)
                        used.add(node)
                    if len(selected) >= 4:
                        break
            ports[answer] = selected
        return ports

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
            scale = 1.35 if fact.subject in {"質問主体", "比較対象", "質問"} else 1.0
            contexts.append((result_to_context(exp.content_result), scale))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _decision_context(
        self,
        question_context: dict[int, float],
        *,
        learn: bool,
    ) -> dict[int, float]:
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

    def _shortest_core_path(
        self,
        start: int,
        goals: set[int],
        max_depth: int = 16,
    ) -> list[int]:
        queue = deque([int(start)])
        previous: dict[int, int | None] = {int(start): None}
        depth: dict[int, int] = {int(start): 0}
        found: int | None = None

        while queue:
            node = queue.popleft()
            if node in goals:
                found = node
                break
            if depth[node] >= max_depth:
                continue
            neighbors = np.flatnonzero(self.brain.adjacency[node]).tolist()
            neighbors.sort(
                key=lambda other: float(self.brain.weights[node, int(other)]),
                reverse=True,
            )
            for other in neighbors:
                other = int(other)
                if other in previous:
                    continue
                previous[other] = node
                depth[other] = depth[node] + 1
                queue.append(other)

        if found is None:
            return []

        path: list[int] = []
        cursor: int | None = found
        while cursor is not None:
            path.append(cursor)
            cursor = previous[cursor]
        path.reverse()
        return path

    def _grow_answer_route(
        self,
        decision_context: dict[int, float],
        answer: str,
    ) -> None:
        ranked = sorted(
            decision_context.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        starts = [int(node) for node, value in ranked[:20] if float(value) > 0]
        goals = {int(node) for node in self.answer_ports[answer]}

        for start in starts[:7]:
            path = self._shortest_core_path(start, goals)
            if len(path) < 2:
                continue
            for a, b in zip(path, path[1:]):
                a, b = int(a), int(b)
                new_weight = min(0.997, float(self.brain.weights[a, b]) + 0.13)
                self.brain.weights[a, b] = new_weight
                self.brain.weights[b, a] = new_weight
                self.brain.usage[a, b] += 1
                self.brain.usage[b, a] += 1
                self.brain.node_usage[a] += 1
                self.brain.node_usage[b] += 1

    def _rehearse_question_route(
        self,
        decision_context: dict[int, float],
    ) -> None:
        ranked = sorted(
            decision_context.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        sources = [int(node) for node, value in ranked[:10] if float(value) > 0]
        self.brain.propagate_contextual(
            sources,
            decision_context,
            steps=20,
            threshold=0.13,
            noise=0.003,
            learn=True,
            context_anchor=0.60,
            context_decay=0.95,
            resonance=True,
        )

    def _train_one(self, subject: str, target: str, answer: str, repeats: int) -> None:
        for _ in range(repeats):
            q_context, _ = self._question_context(subject, target, learn=True)
            d_context = self._decision_context(q_context, learn=True)
            self._grow_answer_route(d_context, answer)
            self._rehearse_question_route(d_context)

    def _train(self) -> None:
        for subject, target, answer in TRAINING_COMPARISONS:
            self._train_one(subject, target, answer, self.repeats)

        for character in CHARACTERS:
            self._train_one(
                character,
                character,
                "同等",
                max(4, self.repeats // 2),
            )

        # 不明PORTにも実際の出口経路を持たせる。これらは明示的な強弱を
        # 与えていない組み合わせであり、未知質問の代表経験として使う。
        unknown_examples = [
            ("魔王", "騎士"),
            ("勇者", "スライム"),
        ]
        for subject, target in unknown_examples:
            self._train_one(subject, target, "不明", max(3, self.repeats // 3))

    def _read_ports(self, decision_context: dict[int, float]):
        ranked = sorted(
            decision_context.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        sources = [int(node) for node, value in ranked[:10] if float(value) > 0]
        result = self.brain.propagate_contextual(
            sources,
            decision_context,
            steps=22,
            threshold=0.12,
            noise=0.0,
            learn=False,
            context_anchor=0.58,
            context_decay=0.95,
            resonance=True,
        )

        final = np.asarray(result.final_activation, dtype=float)
        history = list(result.activation_history or [])
        recent = history[-8:] if history else []
        active_nodes = {int(node) for node in result.activated_nodes}
        traversed = {
            tuple(sorted((int(a), int(b))))
            for a, b in result.traversed_edges
        }

        raw_scores: dict[str, float] = {}
        details: dict[str, dict] = {}
        for answer, nodes in self.answer_ports.items():
            node_set = {int(node) for node in nodes}
            final_sum = sum(float(final[node]) for node in node_set)
            recent_hits = sum(
                sum(1 for node in step if int(node) in node_set)
                for step in recent
            )
            active_count = len(node_set & active_nodes)
            incoming = sum(
                1 for a, b in traversed
                if a in node_set or b in node_set
            )
            score = (
                final_sum
                + 0.20 * recent_hits
                + 0.11 * active_count
                + 0.04 * incoming
            )
            raw_scores[answer] = score
            details[answer] = {
                "port_nodes": list(nodes),
                "final_strength": final_sum,
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

        trained_pair = (
            any(a == subject and b == target for a, b, _ in TRAINING_COMPARISONS)
            or subject == target
        )
        maximum = max(raw_scores.values(), default=1.0) or 1.0
        candidates = []
        for answer in ANSWER_PORTS:
            candidates.append(
                {
                    "answer": answer,
                    "score": raw_scores[answer] / maximum,
                    "raw_score": raw_scores[answer],
                    **details[answer],
                }
            )
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
            "decoder": "Strength Answer Port Decoder v2 — Question-to-Port Routes",
        }
