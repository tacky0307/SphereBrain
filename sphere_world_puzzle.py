from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from semantic_encoder_v2 import StructuredInput, component_nodes
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    merge_contexts,
    result_to_context,
)
from sphere_world_brain import ActionCandidate, _jaccard

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
    facts.append(Fact("Player", "Goal接触", "している" if world.solved else "していない"))
    return facts


def shortest_action(world: PuzzleWorld) -> str:
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
    def __init__(self, repeats: int = 5) -> None:
        self.brain = load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.prototypes: dict[str, list] = {action: [] for action in ACTIONS}
        self.distinctive_edges: dict[str, set[tuple[int, int]]] = {action: set() for action in ACTIONS}
        self.training_examples: list[dict] = []
        self._train()

    def _world_context(self, world: PuzzleWorld, *, learn: bool) -> tuple[dict[int, float], list[str]]:
        contexts, labels = [], []
        for fact in facts_for_world(world):
            exp = encode_and_experience_contextual(self.brain, fact.as_input(), learn=learn)
            contexts.append((result_to_context(exp.content_result), 1.0))
            labels.append(fact.label)
        return merge_contexts(*contexts), labels

    def _continue(self, context: dict[int, float], action: str | None, *, learn: bool):
        noise = 0.004 if learn else 0.0
        relation_sources = (
            component_nodes(self.brain, "role:relation", "relation", 2)
            + component_nodes(self.brain, "relation", "次行動", 3)
        )
        relation_result = self.brain.propagate_contextual(
            relation_sources,
            context,
            steps=8,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        decision_context = merge_contexts(
            (context, 0.78),
            (result_to_context(relation_result), 1.0),
        )
        if action is None:
            return self.brain.propagate_contextual(
                [],
                decision_context,
                steps=10,
                threshold=0.18,
                noise=0.0,
                learn=False,
            )
        content_sources = (
            component_nodes(self.brain, "role:content", "content", 2)
            + component_nodes(self.brain, "content", action, 3)
        )
        return self.brain.propagate_contextual(
            content_sources,
            decision_context,
            steps=10,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )

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

    def _build_distinctive_signatures(self) -> None:
        action_edges: dict[str, set[tuple[int, int]]] = {}
        for action, prototypes in self.prototypes.items():
            merged: set[tuple[int, int]] = set()
            for prototype in prototypes:
                merged.update(tuple(edge) for edge in prototype.traversed_edges)
            action_edges[action] = merged

        for action in ACTIONS:
            other_edges: set[tuple[int, int]] = set()
            for other_action, edges in action_edges.items():
                if other_action != action:
                    other_edges.update(edges)
            self.distinctive_edges[action] = action_edges[action] - other_edges

    def _train(self) -> None:
        worlds = self._training_worlds()
        for world in worlds:
            action = shortest_action(world)
            self.training_examples.append(
                {
                    "puzzle": world.name,
                    "player": list(world.player),
                    "goal": list(world.goal),
                    "action": action,
                    "facts": [fact.label for fact in facts_for_world(world)],
                }
            )
            for _ in range(self.repeats):
                context, _ = self._world_context(world, learn=True)
                self._continue(context, action, learn=True)

        for world in worlds:
            action = shortest_action(world)
            context, _ = self._world_context(world, learn=False)
            self.prototypes[action].append(self._continue(context, None, learn=False))

        self._build_distinctive_signatures()

    def decide(self, world: PuzzleWorld) -> dict:
        if world.solved:
            return {
                "selected_action": "停止",
                "speech": "ゴールに到着しました。停止します。",
                "facts": [fact.label for fact in facts_for_world(world)],
                "candidates": [],
                "raw_nodes": 0,
                "raw_edges": 0,
            }

        context, labels = self._world_context(world, learn=False)
        raw = self._continue(context, None, learn=False)
        raw_nodes = set(raw.activated_nodes)
        raw_edges = {tuple(edge) for edge in raw.traversed_edges}
        candidates: list[ActionCandidate] = []

        for action, prototypes in self.prototypes.items():
            best = ActionCandidate(action, 0.0, 0.0, 0.0, 0, 0)
            distinctive = self.distinctive_edges[action]
            distinctive_score = (
                len(raw_edges & distinctive) / len(distinctive)
                if distinctive
                else 0.0
            )

            for prototype in prototypes:
                p_nodes = set(prototype.activated_nodes)
                p_edges = {tuple(edge) for edge in prototype.traversed_edges}
                node_score = _jaccard(raw_nodes, p_nodes)
                edge_score = _jaccard(raw_edges, p_edges)
                score = (
                    0.20 * node_score
                    + 0.45 * edge_score
                    + 0.35 * distinctive_score
                )
                if score > best.score:
                    best = ActionCandidate(
                        action,
                        score,
                        node_score,
                        edge_score,
                        len(raw_nodes & p_nodes),
                        len(raw_edges & p_edges),
                    )

            if action != "停止" and not world.can_move(action):
                best = ActionCandidate(
                    action,
                    best.score * 0.30,
                    best.node_score,
                    best.edge_score,
                    best.common_nodes,
                    best.common_edges,
                )
            candidates.append(best)

        candidates.sort(key=lambda item: (-item.score, item.action))
        selected = candidates[0].action if candidates else "停止"
        speech = f"{selected}します。" if selected != "停止" else "停止します。"
        return {
            "selected_action": selected,
            "speech": speech,
            "facts": labels,
            "candidates": [item.to_dict() for item in candidates],
            "raw_nodes": len(raw_nodes),
            "raw_edges": len(raw_edges),
        }
