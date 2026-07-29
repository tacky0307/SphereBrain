from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brain import SphereBrain
from research_store import ResearchStore
from visualization import build_html


class ComparisonVisualizationTest(unittest.TestCase):
    def test_renders_comparison_legend_and_metrics(self):
        with TemporaryDirectory() as tmp:
            data = Path(tmp)
            store = ResearchStore(data / "research.db")
            experiment = store.create_experiment("test", "visual comparison")
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

            brain = SphereBrain(node_count=12, neighbors_per_node=3, seed=1)
            output = data / "brain_view.html"
            build_html(
                brain,
                output,
                highlighted_edges=[(1, 2), (2, 4)],
                highlighted_nodes=[1, 2, 4],
                title="Sphere Brain：こんにちは",
            )

            html = output.read_text(encoding="utf-8")
            self.assertIn("共通経路", html)
            self.assertIn("今回の新規経路", html)
            self.assertIn("前回のみの経路", html)
            self.assertIn("一致率 33.3%", html)
            self.assertIn("序盤一致 50.0%", html)


if __name__ == "__main__":
    unittest.main()
