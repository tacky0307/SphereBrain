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

ACTIONS = ["上へ移動", "下へ移動", "左へ移動", "右へ移動", "停止"]
DELTAS = {
    "上へ移動": (-1, 0),
    "下へ移動": (1, 0),
    "左へ移動": (0, -1),
    "右へ移動": (0, 1),
    "停止": (0, 0),
}

PUZZLES = {
    "straight": {"name": "一本道", "grid": ["P..", "##.", "..G"]},
    "turn": {"name": "曲がり道", "grid": ["P#.", ".#.", "..G"]},
    "unseen": {"name": "未経験配置", "grid": ["..G", ".#.", "P.."]},
}


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


class PuzzleWorld:
    def __init__(self, key: str = "straight") -> None:
        if key not in PUZZLES:
            key = "straight"
        self.key = key
        self.name = PUZZLES[key]["name"]
        rows = [list(row) for row in PUZZLES[key]["grid"]]
        self.rows = len(rows)
        self.cols = len(rows[0])
        self.walls: set[tuple[int, int]] = set()
        self.player = (0, 0)
        self.goal = (0, 0)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value == "#":
                    self.walls.add((r, c))
                elif value == "P":
                    self.player = (r, c)
                elif value == "G":
                    self.goal = (r, c)
        self.turn = 0
        self.history: list[dict] = []

    @property
    def solved(self) -> bool:
        return self.player == self.goal

    def can_move(self, action: str) -> bool:
        dr, dc = DELTAS[action]
        r, c = self.player
        target = (r + dr, c + dc)
        return (
            0 <= target[0] < self.rows
            and 0 <= target[1] < self.cols
            and target not in self.walls
        )

    def move(self, action: str) -> bool:
        if action == "停止":
            return self.solved
        if not self.can_move(action):
            return False
        dr, dc = DELTAS[action]
        self.player = (self.player[0] + dr, self.player[1] + dc)
        self.turn += 1
        return True

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "player": list(self.player),
            "goal": list(self.goal),
            "walls": [list(item) for item in sorted(self.walls)],
            "turn": self.turn,
            "solved": self.solved,
        }


def _direction_label(delta: int, negative: str, positive: str) -> str:
    if delta < 0:
        return negative
    if delta > 0:
        return positive
    return "同じ"


def _goal_passage_facts(world: PuzzleWorld) -> list[Fact]:
    """Bind goal direction and passability into reusable world relations."""
    pr, pc = world.player
    gr, gc = world.goal
    facts: list[Fact] = []

    vertical = _direction_label(gr - pr, "上", "下")
    horizontal = _direction_label(gc - pc, "左", "右")

    if vertical != "同じ":
        action = f"{vertical}へ移動"
        state = "進める" if world.can_move(action) else "障害物"
        facts.append(Fact("Goal方向", "上下通行", f"{vertical}へ{state}"))

    if horizontal != "同じ":
        action = f"{horizontal}へ移動"
        state = "進める" if world.can_move(action) else "障害物"
        facts.append(Fact("Goal方向", "左右通行", f"{horizontal}へ{state}"))

    if world.solved:
        facts.append(Fact("Goal方向", "到達状態", "到着済み"))

    return facts


def facts_for_world(world: PuzzleWorld) -> list[Fact]:
    pr, pc = world.player
    gr, gc = world.goal
    facts = [
        Fact("Player", "位置", f"{pr},{pc}"),
        Fact("Goal", "位置", f"{gr},{gc}"),
        Fact("Goal", "上下関係", _direction_label(gr - pr, "上", "下")),
        Fact("Goal", "左右関係", _direction_label(gc - pc, "左", "右")),
    ]
    for action, relation in [
        ("上へ移動", "上"),
        ("下へ移動", "下"),
        ("左へ移動", "左"),
        ("右へ移動", "右"),
    ]:
        facts.append(
            Fact(
                "Player",
                f"{relation}方向",
                "移動可能" if world.can_move(action) else "障害物",
            )
        )
    facts.extend(_goal_passage_facts(world))
    facts.append(Fact("Player", "Goal接触", "している" if world.solved else "していない"))
    return facts


def shortest_action(world: PuzzleWorld) -> str:
    """Teacher used only while creating training experiences."""
    if world.solved:
        return "停止"
    queue = deque([(world.player, [])])
    visited = {world.player}
    order = ["右へ移動", "下へ移動", "左へ移動", "上へ移動"]
    while queue:
        position, path = queue.popleft()
        if position == world.goal:
            return path[0] if path else "停止"
        for action in order:
            dr, dc = DELTAS[action]
            nxt = (position[0] + dr, position[1] + dc)
            if not (0 <= nxt[0] < world.rows and 0 <= nxt[1] < world.cols):
                continue
            if nxt in world.walls or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [action]))
    return "停止"


def representative_world(player: tuple[int, int], goal: tuple[int, int], name: str) -> PuzzleWorld:
    world = PuzzleWorld("straight")
    world.name = name
    world.walls = set()
    world.player = player
    world.goal = goal
    world.turn = 0
    return world


class PuzzleSphereBrain:
    """Puzzle brain with fixed motor output nodes (Action Ports)."""

    def __init__(self, repeats: int = 7) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.training_examples: list[dict] = []
        self.action_ports: dict[str, list[int]] = {
            action: component_nodes(self.brain, "action_port", action, 5)
            for action in ACTIONS
        }
        self._train()

    def _world_context(self, world: PuzzleWorld, *, learn: bool) -> tuple[dict[int, float], list[str]]:
        contexts = []
        labels = []
        for fact in facts_for_world(world):
            exp = encode_and_experience_contextual(self.brain, fact.as_input(), learn=learn)
            scale = 1.35 if fact.subject == "Goal方向" else 1.0
            contexts.append((result_to_context(exp.content_result), scale))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _decision_context(self, world_context: dict[int, float], *, learn: bool) -> dict[int, float]:
        noise = 0.004 if learn else 0.0
        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "次行動", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            world_context,
            steps=8,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        return merge_contexts(
            (world_context, 0.78),
            (result_to_context(relation_result), 1.0),
        )

    def _train_port(self, decision_context: dict[int, float], action: str) -> None:
        """Co-activate the correct motor port and the current world context."""
        port_sources = self.action_ports[action]
        self.brain.propagate_contextual(
            port_sources,
            decision_context,
            steps=12,
            threshold=0.18,
            noise=0.004,
            learn=True,
            context_anchor=0.72,
            context_decay=0.96,
            resonance=True,
        )

    def _read_ports(self, decision_context: dict[int, float]):
        """Let activity propagate freely, then measure arrival at each motor port."""
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
        history = list(result.activation_history or [])
        final = np.asarray(result.final_activation, dtype=float)
        recent = history[-5:] if history else []

        raw_scores: dict[str, float] = {}
        details: dict[str, dict] = {}
        for action, nodes in self.action_ports.items():
            final_strength = max((float(final[node]) for node in nodes), default=0.0)
            arrival_count = sum(1 for step in recent for node in nodes if node in step)
            ever_count = sum(1 for node in nodes if node in set(result.activated_nodes))
            incoming = sum(
                1
                for a, b in result.traversed_edges
                if int(a) in nodes or int(b) in nodes
            )
            score = final_strength + 0.10 * arrival_count + 0.04 * ever_count + 0.015 * incoming
            raw_scores[action] = score
            details[action] = {
                "port_nodes": nodes,
                "final_strength": final_strength,
                "recent_arrivals": arrival_count,
                "activated_port_nodes": ever_count,
                "incoming_edges": incoming,
            }
        return result, raw_scores, details

    def _training_worlds(self) -> list[PuzzleWorld]:
        worlds = [
            representative_world((1, 1), (0, 1), "基本経験・上"),
            representative_world((1, 1), (2, 1), "基本経験・下"),
            representative_world((1, 1), (1, 0), "基本経験・左"),
            representative_world((1, 1), (1, 2), "基本経験・右"),
            representative_world((1, 1), (1, 1), "基本経験・停止"),
        ]
        for key in ("straight", "turn"):
            world = PuzzleWorld(key)
            safety = 0
            while not world.solved and safety < 12:
                copy = PuzzleWorld(key)
                copy.player = world.player
                worlds.append(copy)
                world.move(shortest_action(world))
                safety += 1
            copy = PuzzleWorld(key)
            copy.player = copy.goal
            worlds.append(copy)
        return worlds

    def _train(self) -> None:
        for world in self._training_worlds():
            action = shortest_action(world)
            self.training_examples.append(
                {
                    "puzzle": world.name,
                    "player": list(world.player),
                    "goal": list(world.goal),
                    "action": action,
                    "facts": [fact.label for fact in facts_for_world(world)],
                    "port_nodes": self.action_ports[action],
                }
            )
            for _ in range(self.repeats):
                world_context, _ = self._world_context(world, learn=True)
                decision_context = self._decision_context(world_context, learn=True)
                self._train_port(decision_context, action)

    def decide(self, world: PuzzleWorld) -> dict:
        if world.solved:
            return {
                "selected_action": "停止",
                "speech": "ゴールに到着しました。停止します。",
                "facts": [fact.label for fact in facts_for_world(world)],
                "candidates": [],
                "raw_nodes": 0,
                "raw_edges": 0,
                "decoder": "Action Port Decoder",
                "action_ports": self.action_ports,
            }

        world_context, labels = self._world_context(world, learn=False)
        decision_context = self._decision_context(world_context, learn=False)
        raw, raw_scores, port_details = self._read_ports(decision_context)

        adjusted: dict[str, float] = {}
        for action, score in raw_scores.items():
            if action != "停止" and not world.can_move(action):
                score *= 0.08
            if action == "停止" and not world.solved:
                score *= 0.22
            adjusted[action] = max(0.0, score)

        maximum = max(adjusted.values(), default=1.0) or 1.0
        candidates = []
        for action in ACTIONS:
            normalized = adjusted[action] / maximum
            details = port_details[action]
            candidates.append(
                {
                    "action": action,
                    "score": normalized,
                    "port_strength": adjusted[action],
                    **details,
                }
            )
        candidates.sort(key=lambda item: (-item["score"], item["action"]))
        selected = candidates[0]["action"] if candidates else "停止"
        speech = f"{selected}します。" if selected != "停止" else "停止します。"

        return {
            "selected_action": selected,
            "speech": speech,
            "facts": labels,
            "candidates": candidates,
            "raw_nodes": len(set(raw.activated_nodes)),
            "raw_edges": len({tuple(sorted(edge)) for edge in raw.traversed_edges}),
            "decoder": "Action Port Decoder",
            "action_ports": self.action_ports,
        }
