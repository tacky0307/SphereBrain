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

TALK_PORTS = ["同意", "否定", "質問", "保留", "警告"]


@dataclass(frozen=True)
class TalkFact:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"

    def as_input(self) -> StructuredInput:
        return StructuredInput(self.subject, self.relation, self.content)


SCENES = {
    "agree": {
        "name": "一致する証言",
        "speaker": "案内人",
        "utterance": "青い扉の先に鍵があります。",
        "facts": [
            TalkFact("案内人", "主張", "鍵は青い扉の先"),
            TalkFact("地図", "表示", "鍵は青い扉の先"),
            TalkFact("会話", "整合状態", "一致"),
            TalkFact("危険", "状態", "なし"),
        ],
        "expected": "同意",
    },
    "deny": {
        "name": "矛盾する証言",
        "speaker": "旅人",
        "utterance": "赤い扉が出口です。",
        "facts": [
            TalkFact("旅人", "主張", "赤い扉が出口"),
            TalkFact("地図", "表示", "青い扉が出口"),
            TalkFact("会話", "整合状態", "矛盾"),
            TalkFact("危険", "状態", "なし"),
        ],
        "expected": "否定",
    },
    "ask": {
        "name": "情報が足りない",
        "speaker": "門番",
        "utterance": "宝箱を開けてもよいです。",
        "facts": [
            TalkFact("門番", "許可", "宝箱を開けてよい"),
            TalkFact("宝箱", "鍵", "状態不明"),
            TalkFact("宝箱", "罠", "状態不明"),
            TalkFact("会話", "情報量", "不足"),
        ],
        "expected": "質問",
    },
    "hold": {
        "name": "判断を保留",
        "speaker": "研究者",
        "utterance": "右の装置が故障の原因かもしれません。",
        "facts": [
            TalkFact("研究者", "推測", "右の装置が原因"),
            TalkFact("計測", "結果", "左右とも異常なし"),
            TalkFact("証拠", "確度", "低い"),
            TalkFact("会話", "判断可能性", "未確定"),
        ],
        "expected": "保留",
    },
    "warn": {
        "name": "危険を知らせる",
        "speaker": "作業員",
        "utterance": "このまま装置を動かします。",
        "facts": [
            TalkFact("作業員", "予定", "装置を動かす"),
            TalkFact("装置", "温度", "危険域"),
            TalkFact("装置", "警報", "作動中"),
            TalkFact("危険", "状態", "高い"),
        ],
        "expected": "警告",
    },
}


TRAINING_SCENES = {
    "同意": [
        [TalkFact("発言", "整合状態", "一致"), TalkFact("証拠", "確度", "高い"), TalkFact("危険", "状態", "なし")],
        [TalkFact("提案", "検証結果", "正しい"), TalkFact("追加確認", "必要性", "低い")],
    ],
    "否定": [
        [TalkFact("発言", "整合状態", "矛盾"), TalkFact("証拠", "確度", "高い")],
        [TalkFact("主張", "検証結果", "誤り"), TalkFact("訂正", "必要性", "高い")],
    ],
    "質問": [
        [TalkFact("会話", "情報量", "不足"), TalkFact("重要項目", "状態", "不明")],
        [TalkFact("指示", "条件", "不足"), TalkFact("追加確認", "必要性", "高い")],
    ],
    "保留": [
        [TalkFact("証拠", "確度", "低い"), TalkFact("会話", "判断可能性", "未確定")],
        [TalkFact("候補", "数", "複数"), TalkFact("決定材料", "状態", "不足")],
    ],
    "警告": [
        [TalkFact("危険", "状態", "高い"), TalkFact("行動", "予定", "継続")],
        [TalkFact("警報", "状態", "作動中"), TalkFact("停止", "必要性", "高い")],
    ],
}


LANGUAGE_DECODER = {
    "同意": {
        "agree": "その説明で合っています。青い扉を調べます。",
        "default": "その判断に同意します。",
    },
    "否定": {
        "deny": "その説明は地図と矛盾しています。赤い扉ではありません。",
        "default": "その説明は違うと思います。",
    },
    "質問": {
        "ask": "判断する前に、鍵と罠の状態を教えてください。",
        "default": "もう少し情報を教えてください。",
    },
    "保留": {
        "hold": "今の情報だけでは決められません。判断を保留します。",
        "default": "まだ判断できません。",
    },
    "警告": {
        "warn": "危険です。装置を動かさず、まず停止してください。",
        "default": "危険かもしれません。いったん止めてください。",
    },
}


class SphereTalkBrain:
    """Experimental conversation agent using fixed semantic output ports."""

    def __init__(self, repeats: int = 6) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.talk_ports = {
            port: component_nodes(self.brain, "talk_port", port, 4)
            for port in TALK_PORTS
        }
        self._train()

    def _facts_context(self, facts: list[TalkFact], *, learn: bool) -> tuple[dict[int, float], list[str]]:
        contexts = []
        labels = []
        for fact in facts:
            experience = encode_and_experience_contextual(self.brain, fact.as_input(), learn=learn)
            scale = 1.35 if fact.subject in {"危険", "会話", "証拠"} else 1.0
            contexts.append((result_to_context(experience.content_result), scale))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _decision_context(self, facts_context: dict[int, float], *, learn: bool) -> dict[int, float]:
        noise = 0.004 if learn else 0.0
        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "会話判断", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            facts_context,
            steps=8,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        return merge_contexts(
            (facts_context, 0.78),
            (result_to_context(relation_result), 1.0),
        )

    def _shortest_path(self, starts: list[int], goals: set[int], max_depth: int = 16) -> list[int]:
        queue = deque()
        previous: dict[int, int | None] = {}
        depth: dict[int, int] = {}
        for start in starts:
            start = int(start)
            queue.append(start)
            previous[start] = None
            depth[start] = 0

        found: int | None = None
        while queue:
            node = queue.popleft()
            if node in goals:
                found = node
                break
            if depth[node] >= max_depth:
                continue
            neighbors = np.flatnonzero(self.brain.adjacency[node]).tolist()
            neighbors.sort(key=lambda other: float(self.brain.weights[node, other]), reverse=True)
            for other in neighbors:
                other = int(other)
                if other in previous:
                    continue
                previous[other] = node
                depth[other] = depth[node] + 1
                queue.append(other)

        if found is None:
            return []
        path = []
        cursor: int | None = found
        while cursor is not None:
            path.append(cursor)
            cursor = previous[cursor]
        return list(reversed(path))

    def _grow_route(self, context: dict[int, float], port: str) -> None:
        ranked = sorted(context.items(), key=lambda item: item[1], reverse=True)
        starts = [int(node) for node, value in ranked[:16] if float(value) > 0]
        goals = {int(node) for node in self.talk_ports[port]}
        for start in starts[:5]:
            path = self._shortest_path([start], goals)
            for a, b in zip(path, path[1:]):
                value = min(0.995, float(self.brain.weights[a, b]) + 0.105)
                self.brain.weights[a, b] = value
                self.brain.weights[b, a] = value
                self.brain.usage[a, b] += 1
                self.brain.usage[b, a] += 1
                self.brain.node_usage[a] += 1
                self.brain.node_usage[b] += 1

    def _train(self) -> None:
        for port, examples in TRAINING_SCENES.items():
            for facts in examples:
                for _ in range(self.repeats):
                    facts_context, _ = self._facts_context(facts, learn=True)
                    decision_context = self._decision_context(facts_context, learn=True)
                    self._grow_route(decision_context, port)

    def _read_ports(self, decision_context: dict[int, float]):
        ranked = sorted(decision_context.items(), key=lambda item: item[1], reverse=True)
        sources = [int(node) for node, value in ranked[:10] if float(value) > 0]
        result = self.brain.propagate_contextual(
            sources,
            decision_context,
            steps=20,
            threshold=0.13,
            noise=0.0,
            learn=False,
            context_anchor=0.58,
            context_decay=0.95,
            resonance=True,
        )
        final = np.asarray(result.final_activation, dtype=float)
        history = list(result.activation_history or [])
        recent = history[-7:] if history else []
        traversed = {tuple(sorted((int(a), int(b)))) for a, b in result.traversed_edges}
        activated = {int(node) for node in result.activated_nodes}

        scores = {}
        details = {}
        for port, nodes in self.talk_ports.items():
            node_set = {int(node) for node in nodes}
            final_sum = sum(float(final[node]) for node in node_set)
            recent_hits = sum(sum(1 for node in step if int(node) in node_set) for step in recent)
            active_nodes = len(node_set & activated)
            incoming = sum(1 for a, b in traversed if a in node_set or b in node_set)
            score = final_sum + 0.18 * recent_hits + 0.10 * active_nodes + 0.035 * incoming
            scores[port] = score
            details[port] = {
                "port_nodes": list(nodes),
                "final_strength": final_sum,
                "recent_arrivals": recent_hits,
                "activated_port_nodes": active_nodes,
                "incoming_edges": incoming,
            }
        return result, scores, details

    def respond(self, scene_key: str) -> dict:
        scene = SCENES.get(scene_key, SCENES["agree"])
        facts_context, labels = self._facts_context(scene["facts"], learn=False)
        decision_context = self._decision_context(facts_context, learn=False)
        raw, raw_scores, details = self._read_ports(decision_context)

        maximum = max(raw_scores.values(), default=1.0) or 1.0
        candidates = []
        for port in TALK_PORTS:
            candidates.append({
                "port": port,
                "score": raw_scores[port] / maximum,
                "strength": raw_scores[port],
                **details[port],
            })
        candidates.sort(key=lambda item: (-item["score"], item["port"]))
        selected = candidates[0]["port"] if candidates else "保留"
        speech = LANGUAGE_DECODER[selected].get(scene_key, LANGUAGE_DECODER[selected]["default"])

        return {
            "scene_key": scene_key,
            "scene_name": scene["name"],
            "speaker": scene["speaker"],
            "utterance": scene["utterance"],
            "expected": scene["expected"],
            "selected_port": selected,
            "speech": speech,
            "correct": selected == scene["expected"],
            "facts": labels,
            "candidates": candidates,
            "raw_nodes": len(set(raw.activated_nodes)),
            "raw_edges": len({tuple(sorted(edge)) for edge in raw.traversed_edges}),
            "decoder": "Talk Port → Language Decoder",
        }
