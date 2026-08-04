from __future__ import annotations

from dataclasses import asdict, dataclass


POSITIONS = ("左", "中央", "右")
ACTIONS = ("左へ移動", "右へ移動", "停止")


@dataclass
class WorldObject:
    object_id: str
    kind: str
    position: int
    state: str = "停止"
    hp: int = 1

    @property
    def position_label(self) -> str:
        return POSITIONS[self.position]


class SphereWorld:
    """Three-cell world used only to observe SphereBrain decisions."""

    def __init__(self, player_position: int = 0, enemy_position: int = 2) -> None:
        self.player = WorldObject("player", "Player", self._clip(player_position), hp=3)
        self.enemy = WorldObject("enemy", "Enemy", self._clip(enemy_position), hp=1)
        self.turn = 0
        self.history: list[dict] = []

    @staticmethod
    def _clip(position: int) -> int:
        return max(0, min(2, int(position)))

    @property
    def touching(self) -> bool:
        return self.player.position == self.enemy.position

    @property
    def state_key(self) -> str:
        return f"Player位置:{self.player.position_label}|Enemy位置:{self.enemy.position_label}"

    def facts(self) -> list[tuple[str, str, str]]:
        return [
            ("Player", "位置", self.player.position_label),
            ("Enemy", "位置", self.enemy.position_label),
            ("Player", "接触", "している" if self.touching else "していない"),
        ]

    def apply_action(self, action: str) -> dict:
        before = self.snapshot()
        if action == "左へ移動":
            self.player.position = self._clip(self.player.position - 1)
        elif action == "右へ移動":
            self.player.position = self._clip(self.player.position + 1)
        elif action != "停止":
            raise ValueError(f"未知の行動です: {action}")

        self.player.state = action
        self.turn += 1
        after = self.snapshot()
        event = {"turn": self.turn, "action": action, "before": before, "after": after}
        self.history.append(event)
        return event

    def reset(self, player_position: int = 0, enemy_position: int = 2) -> None:
        self.__init__(player_position, enemy_position)

    def snapshot(self) -> dict:
        cells: list[list[str]] = [[], [], []]
        cells[self.player.position].append("P")
        cells[self.enemy.position].append("E")
        return {
            "turn": self.turn,
            "player": asdict(self.player),
            "enemy": asdict(self.enemy),
            "touching": self.touching,
            "state_key": self.state_key,
            "cells": [" / ".join(cell) if cell else "" for cell in cells],
        }
