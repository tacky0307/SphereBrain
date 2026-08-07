from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


class Tile(str, Enum):
    EMPTY = "."
    FOOD = "F"
    DANGER = "!"
    WALL = "#"


ACTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
    "STAY": (0, 0),
}


@dataclass(frozen=True)
class StepResult:
    action: str
    tile: Tile
    outcome: str
    reward: int
    energy_delta: int
    alive: bool


class SphereWorld:
    """The environment only exposes consequences; it contains no movement policy."""

    def __init__(self, size: int = 7, seed: int = 7, max_energy: int = 30) -> None:
        if size < 5:
            raise ValueError("size must be at least 5")
        self.size = size
        self.seed = seed
        self.max_energy = max_energy
        self.rng = random.Random(seed)
        self.turn = 0
        self.food_eaten = 0
        self.energy = max_energy
        self.agent = (size // 2, size // 2)
        self.tiles: dict[tuple[int, int], Tile] = {}
        self._generate()

    def _generate(self) -> None:
        self.tiles.clear()
        cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) != self.agent
        ]
        self.rng.shuffle(cells)
        wall_count = max(4, self.size)
        danger_count = max(3, self.size // 2)
        food_count = 3

        cursor = 0
        for pos in cells[cursor:cursor + wall_count]:
            self.tiles[pos] = Tile.WALL
        cursor += wall_count
        for pos in cells[cursor:cursor + danger_count]:
            self.tiles[pos] = Tile.DANGER
        cursor += danger_count
        for pos in cells[cursor:cursor + food_count]:
            self.tiles[pos] = Tile.FOOD

    def tile_at(self, pos: tuple[int, int]) -> Tile:
        r, c = pos
        if r < 0 or c < 0 or r >= self.size or c >= self.size:
            return Tile.WALL
        return self.tiles.get(pos, Tile.EMPTY)

    def sense(self) -> dict[str, str]:
        r, c = self.agent
        observations: dict[str, str] = {}
        for action, (dr, dc) in ACTIONS.items():
            if action == "STAY":
                continue
            observations[action] = self.tile_at((r + dr, c + dc)).name.lower()

        ratio = self.energy / self.max_energy
        if ratio <= 0.30:
            energy_band = "low"
        elif ratio <= 0.65:
            energy_band = "mid"
        else:
            energy_band = "high"
        observations["ENERGY"] = energy_band
        return observations

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")

        self.turn += 1
        dr, dc = ACTIONS[action]
        target = (self.agent[0] + dr, self.agent[1] + dc)
        tile = self.tile_at(target)

        # Existing itself costs energy. This is physics, not an action policy.
        delta = -1
        reward = 0
        outcome = "neutral"

        if tile is Tile.WALL:
            outcome = "bad"
            reward = -1
            delta -= 1
        else:
            self.agent = target
            if tile is Tile.FOOD:
                outcome = "good"
                reward = 1
                delta += 11
                self.food_eaten += 1
                self.tiles.pop(target, None)
                self._respawn_food()
            elif tile is Tile.DANGER:
                outcome = "bad"
                reward = -1
                delta -= 6
            elif action == "STAY":
                outcome = "neutral"

        old_energy = self.energy
        self.energy = max(0, min(self.max_energy, self.energy + delta))
        actual_delta = self.energy - old_energy
        return StepResult(
            action=action,
            tile=tile,
            outcome=outcome,
            reward=reward,
            energy_delta=actual_delta,
            alive=self.energy > 0,
        )

    def _respawn_food(self) -> None:
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) != self.agent and self.tile_at((r, c)) is Tile.EMPTY
        ]
        if candidates:
            self.tiles[self.rng.choice(candidates)] = Tile.FOOD

    def render(self) -> str:
        rows: list[str] = []
        for r in range(self.size):
            row: list[str] = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent:
                    row.append("O")
                else:
                    row.append(self.tile_at(pos).value)
            rows.append(" ".join(row))
        return "\n".join(rows)
