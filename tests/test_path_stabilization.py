from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_store import ResearchStore


class PathStabilizationTest(unittest.TestCase):
    def test_detects_persistent_edges_and_latest_stabilization_ratio(self):
        with TemporaryDirectory() as tmp:
            store = ResearchStore(Path(tmp) / "research.db")
            experiment = store.create_experiment("test", "stabilization")
            session = store.start_session(experiment, "sphere-test", "config-test", random_seed=1)

            paths = [
                [(1, 2), (2, 3)],
                [(1, 2), (2, 4)],
                [(1, 2), (2, 3)],
                [(1, 2), (2, 3)],
            ]
            for sequence_no, path in enumerate(paths, start=1):
                input_id = store.add_input("こんにちは", source="input")
                trial = store.start_trial(session, sequence_no, input_id, [1])
                for step_no, (a, b) in enumerate(path, start=1):
                    store.add_path_step(trial.trial_id, step_no, a, b)
                store.finish_trial(trial.trial_id, [path[-1][1]])

            result = store.path_stabilization(
                "こんにちは",
                stable_ratio=0.75,
                minimum_trials=3,
            )

            self.assertEqual(result["trial_count"], 4)
            self.assertEqual(result["stable_edge_count"], 2)
            self.assertIn((1, 2), result["stable_edges"])
            self.assertIn((2, 3), result["stable_edges"])
            self.assertAlmostEqual(result["stabilization_ratio"], 1.0)

            stats = {tuple(item["edge"]): item for item in result["edge_stats"]}
            self.assertEqual(stats[(1, 2)]["occurrence_count"], 4)
            self.assertEqual(stats[(1, 2)]["current_streak"], 4)
            self.assertEqual(stats[(2, 3)]["occurrence_count"], 3)
            self.assertEqual(stats[(2, 3)]["current_streak"], 2)
            self.assertFalse(stats[(2, 4)]["stable"])


if __name__ == "__main__":
    unittest.main()
