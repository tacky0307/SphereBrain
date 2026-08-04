from __future__ import annotations

from collections import Counter

from sphere_world_puzzle import ACTIONS, DELTAS, PuzzleWorld
from sphere_world_puzzle_ports_v2 import ActionPortPuzzleBrainV2


OPPOSITE = {
    "上へ移動": "下へ移動",
    "下へ移動": "上へ移動",
    "左へ移動": "右へ移動",
    "右へ移動": "左へ移動",
    "停止": "停止",
}


class TemporalActionPortPuzzleBrain(ActionPortPuzzleBrainV2):
    """Action Port v2 with a small, per-world working memory.

    The Core still produces the motor-port activations.  Working memory does not
    know the maze solution; it only suppresses immediate reversals, repeatedly
    visited positions and actions whose last outcome moved away from the goal.
    """

    def __init__(self, repeats: int = 7) -> None:
        self._temporal: dict[int, dict] = {}
        super().__init__(repeats=repeats)

    @staticmethod
    def _distance(position: tuple[int, int], goal: tuple[int, int]) -> int:
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def _memory_for(self, world: PuzzleWorld) -> dict:
        key = id(world)
        memory = self._temporal.get(key)
        if memory is None:
            memory = {
                "positions": [tuple(world.player)],
                "visits": Counter({tuple(world.player): 1}),
                "last_position": None,
                "last_action": None,
                "last_distance": self._distance(tuple(world.player), tuple(world.goal)),
                "last_outcome": "開始",
            }
            self._temporal[key] = memory
            return memory

        current = tuple(world.player)
        if current != memory["positions"][-1]:
            previous = memory["positions"][-1]
            old_distance = int(memory["last_distance"])
            new_distance = self._distance(current, tuple(world.goal))
            if new_distance < old_distance:
                outcome = "Goalに近づいた"
            elif new_distance > old_distance:
                outcome = "Goalから遠ざかった"
            else:
                outcome = "距離は同じ"
            memory["last_position"] = previous
            memory["positions"].append(current)
            memory["visits"][current] += 1
            memory["last_distance"] = new_distance
            memory["last_outcome"] = outcome
        return memory

    @staticmethod
    def _target(world: PuzzleWorld, action: str) -> tuple[int, int]:
        dr, dc = DELTAS[action]
        return world.player[0] + dr, world.player[1] + dc

    def decide(self, world: PuzzleWorld) -> dict:
        memory = self._memory_for(world)
        result = super().decide(world)
        if world.solved or not result.get("candidates"):
            result["decoder"] = "Action Port v3 — Temporal Context"
            result["temporal_context"] = {
                "last_action": memory["last_action"],
                "last_outcome": memory["last_outcome"],
                "visit_count": int(memory["visits"][tuple(world.player)]),
            }
            return result

        adjusted = []
        current = tuple(world.player)
        last_position = memory["last_position"]
        last_action = memory["last_action"]
        last_outcome = memory["last_outcome"]

        for candidate in result["candidates"]:
            item = dict(candidate)
            action = str(item["action"])
            score = float(item.get("port_strength", item.get("score", 0.0)))
            factor = 1.0
            reasons: list[str] = []

            if action != "停止":
                target = self._target(world, action)
                visits = int(memory["visits"][target])
                if last_position is not None and target == last_position:
                    factor *= 0.30
                    reasons.append("直前位置へ戻る")
                if visits:
                    factor *= 1.0 / (1.0 + 0.42 * visits)
                    reasons.append(f"訪問済み{visits}回")

                if last_action:
                    if action == OPPOSITE.get(last_action):
                        factor *= 0.52
                        reasons.append("直前行動の逆方向")
                    elif action == last_action and last_outcome == "Goalに近づいた":
                        factor *= 1.14
                        reasons.append("直前行動が有効")
                    elif action == last_action and last_outcome == "Goalから遠ざかった":
                        factor *= 0.48
                        reasons.append("直前行動で遠ざかった")
            elif not world.solved:
                factor *= 0.45
                reasons.append("Goal未到着")

            item["temporal_factor"] = factor
            item["temporal_reasons"] = reasons
            item["port_strength"] = score * factor
            adjusted.append(item)

        maximum = max((float(item["port_strength"]) for item in adjusted), default=1.0) or 1.0
        for item in adjusted:
            item["score"] = float(item["port_strength"]) / maximum
        adjusted.sort(key=lambda item: (-float(item["score"]), str(item["action"])))

        selected = str(adjusted[0]["action"])
        memory["last_action"] = selected
        result["selected_action"] = selected
        result["speech"] = f"{selected}します。" if selected != "停止" else "停止します。"
        result["candidates"] = adjusted
        result["decoder"] = "Action Port v3 — Temporal Context"
        result["temporal_context"] = {
            "current_position": list(current),
            "last_position": list(last_position) if last_position is not None else None,
            "last_action": last_action,
            "last_outcome": last_outcome,
            "visit_count": int(memory["visits"][current]),
            "position_history": [list(position) for position in memory["positions"][-8:]],
        }
        return result
