from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from sphere_world import POSITIONS, SphereWorld
from sphere_world_multi_context import MultiContextSphereWorldBrain, TRAIN_STATES, facts_for_positions


@dataclass
class DemoState:
    player_position: int = 0
    enemy_position: int = 2
    turn: int = 0

    def clamp(self) -> None:
        self.player_position = max(0, min(2, int(self.player_position)))
        self.enemy_position = max(0, min(2, int(self.enemy_position)))
        self.turn = max(0, int(self.turn))

    def snapshot(self) -> dict:
        self.clamp()
        cells: list[list[str]] = [[], [], []]
        cells[self.player_position].append("P")
        cells[self.enemy_position].append("E")
        return {
            "player_position": self.player_position,
            "enemy_position": self.enemy_position,
            "player_label": POSITIONS[self.player_position],
            "enemy_label": POSITIONS[self.enemy_position],
            "touching": self.player_position == self.enemy_position,
            "turn": self.turn,
            "cells": [" / ".join(cell) if cell else "" for cell in cells],
        }

    def apply(self, action: str) -> dict:
        before = self.snapshot()
        if action == "左へ移動":
            self.player_position = max(0, self.player_position - 1)
        elif action == "右へ移動":
            self.player_position = min(2, self.player_position + 1)
        elif action != "停止":
            raise ValueError(f"未知の行動です: {action}")
        self.turn += 1
        return {"before": before, "after": self.snapshot(), "action": action}


class SphereWorldShowcase:
    """Presentation-facing wrapper around the actual SphereWorld 0.2 brain."""

    def __init__(self, repeats: int = 12) -> None:
        self._lock = Lock()
        self._brain: MultiContextSphereWorldBrain | None = None
        self.repeats = max(1, int(repeats))

    def _get_brain(self) -> MultiContextSphereWorldBrain:
        # The disposable Core is expensive enough to build that the showcase keeps
        # one in memory, but it never writes to the saved semantic Core or DB.
        with self._lock:
            if self._brain is None:
                self._brain = MultiContextSphereWorldBrain(repeats=self.repeats)
            return self._brain

    def decide(self, player_position: int, enemy_position: int) -> dict:
        player_position = max(0, min(2, int(player_position)))
        enemy_position = max(0, min(2, int(enemy_position)))
        brain = self._get_brain()
        decision = brain.decide_positions(player_position, enemy_position)
        candidates = decision["candidates"]
        margin = candidates[0]["score"] - candidates[1]["score"] if len(candidates) > 1 else 0.0
        expected = brain.action_for_positions(player_position, enemy_position)
        trained = (player_position, enemy_position) in TRAIN_STATES
        return {
            **decision,
            "player_position": player_position,
            "enemy_position": enemy_position,
            "player_label": POSITIONS[player_position],
            "enemy_label": POSITIONS[enemy_position],
            "expected_action": expected,
            "trained_state": trained,
            "margin": margin,
            "correct": decision["selected_action"] == expected,
        }

    @staticmethod
    def training_experiences() -> list[dict]:
        return [
            {
                "player": POSITIONS[player],
                "enemy": POSITIONS[enemy],
                "action": action,
                "facts": [fact.label for fact in facts_for_positions(player, enemy)],
            }
            for (player, enemy), action in TRAIN_STATES.items()
        ]
