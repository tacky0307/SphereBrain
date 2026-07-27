from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from brain import SphereBrain
from research_store import ResearchStore


class ResearchPipelineTest(unittest.TestCase):
    def test_signal_contains_ordered_steps_and_weight_changes(self) -> None:
        brain = SphereBrain(node_count=48, neighbors_per_node=5, seed=7)
        sources = brain.text_to_sources("こんにちは", count=3)
        result = brain.propagate(sources, steps=8, noise=0.0, learn=True)

        self.assertEqual(result.source_nodes, sources)
        self.assertGreater(len(result.activated_nodes), 0)
        self.assertGreater(len(result.path_steps), 0)
        self.assertTrue(all(step.step_no >= 1 for step in result.path_steps))
        self.assertTrue(all(step.weight_after >= step.weight_before for step in result.path_steps))

    def test_research_store_records_complete_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir) / "research.db")
            experiment_id = store.create_experiment(
                name="test experiment",
                purpose="verify persistence",
                hypothesis="paths are stored",
            )
            session_id = store.start_session(
                experiment_id,
                structure_version="test-structure",
                config_version="test-config",
                random_seed=1,
            )
            input_id = store.add_input("hello", source="test")
            trial = store.start_trial(session_id, 1, input_id, [1, 2])
            store.add_path_step(trial.trial_id, 1, 1, 3, 0.8, 0.4, 0.45)
            store.add_snapshot(trial.trial_id, 0, "initial", {"nodes": [1, 2]})
            store.add_snapshot(trial.trial_id, 1, "final", {"nodes": [3]})
            store.add_output(trial.trial_id, {"nodes": [3]})
            store.add_metric(trial.trial_id, "path_step_count", 1)
            store.finish_trial(trial.trial_id, [3])
            store.finish_session(session_id)

            summary = store.summary()
            self.assertEqual(summary["experiments"], 1)
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(summary["inputs"], 1)
            self.assertEqual(summary["trials"], 1)
            self.assertEqual(summary["path_steps"], 1)
            self.assertEqual(summary["snapshots"], 2)
            self.assertEqual(summary["outputs"], 1)
            self.assertEqual(summary["metrics"], 1)


if __name__ == "__main__":
    unittest.main()
