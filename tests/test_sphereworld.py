from pathlib import Path
import tempfile
import unittest

from brain import SphereBrain
from sphereworld.core_agent import ANCHOR_LABELS, CoreAgent
from sphereworld.world import SphereWorld, Tile


class SphereWorldTests(unittest.TestCase):
    def test_world_sense_has_no_policy_output(self) -> None:
        world = SphereWorld(size=7, seed=7)
        senses = world.sense()
        self.assertEqual(set(senses), {"N", "E", "S", "W", "ENERGY"})
        self.assertNotIn("action", senses)

    def test_wall_is_consequence_not_forced_choice(self) -> None:
        world = SphereWorld(size=7, seed=7)
        world.agent = (0, 0)
        before = world.agent
        result = world.step("N")
        self.assertEqual(result.tile, Tile.WALL)
        self.assertEqual(result.outcome, "bad")
        self.assertEqual(world.agent, before)

    def test_anchor_map_is_stable_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brain.json"
            left = CoreAgent(SphereBrain(seed=42), path, seed=1)
            right = CoreAgent(SphereBrain(seed=42), path, seed=999)
            self.assertEqual(left._anchor_map, right._anchor_map)
            self.assertEqual(len(left._anchor_map), len(ANCHOR_LABELS))
            self.assertEqual(len(set(left._anchor_map.values())), len(ANCHOR_LABELS))


if __name__ == "__main__":
    unittest.main()
