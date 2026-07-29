from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_store import ResearchStore


class RepeatedInputComparisonTest(unittest.TestCase):
    def test_reports_shared_new_and_lost_edges(self):
        with TemporaryDirectory() as tmp:
            store = ResearchStore(Path(tmp) / "research.db")
            experiment = store.create_experiment("test", "comparison")
            session = store.start_session(experiment, "sphere-test", "config-test", random_seed=1)

            first_input = store.add_input("こんにちは", source="input")
            first = store.start_trial(session, 1, first_input, [1])
            store.add_path_step(first.trial_id, 1, 1, 2)
            store.add_path_step(first.trial_id, 2, 2, 3)
            store.finish_trial(first.trial_id, [3])

            second_input = store.add_input("こんにちは", source="input")
            second = store.start_trial(session, 2, second_input, [1])
            store.add_path_step(second.trial_id, 1, 1, 2)
            store.add_path_step(second.trial_id, 2, 2, 4)
            store.finish_trial(second.trial_id, [4])

            result = store.repeated_input_comparison("こんにちは")
            latest = result["latest"]

            self.assertEqual(result["trial_count"], 2)
            self.assertEqual(latest["shared_edges"], 1)
            self.assertEqual(latest["new_edges"], 1)
            self.assertEqual(latest["lost_edges"], 1)
            self.assertAlmostEqual(latest["edge_similarity"], 1 / 3)
            self.assertEqual(latest["matching_prefix_steps"], 1)
            self.assertAlmostEqual(latest["ordered_similarity"], 0.5)


if __name__ == "__main__":
    unittest.main()
