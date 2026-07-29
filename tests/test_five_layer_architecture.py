from pathlib import Path

from brain import SphereBrain
from decoder import NumericDecoder
from encoder import TextEncoder
from reflection import ReflectionEngine
from trace_store import TraceStore


def test_encoder_is_deterministic():
    encoder = TextEncoder(node_count=120, source_count=4)
    first = encoder.encode("空は青い")
    second = encoder.encode("空は青い")
    assert first == second
    assert len(first.source_nodes) == 4


def test_core_accepts_numeric_sources_only():
    core = SphereBrain(node_count=80, neighbors_per_node=5)
    result = core.propagate([1, 2, 3], steps=3, noise=0.0, learn=False)
    assert result.source_nodes == [1, 2, 3]
    assert result.activation_history


def test_trace_reflection_cycle(tmp_path: Path):
    core = SphereBrain(node_count=100, neighbors_per_node=6)
    encoder = TextEncoder(core.node_count)
    decoder = NumericDecoder()
    store = TraceStore(tmp_path / "trace.db")

    stimulus = encoder.encode("雨が降っている")
    event_id = store.add_external_event("text", "雨が降っている", stimulus.encoder, 1.0)
    result = core.propagate(stimulus.source_nodes, steps=5, noise=0.0, learn=True)
    trace_id = store.add_trace(result, "experience", True, event_id=event_id)

    saved = store.get_trace(trace_id)
    assert saved is not None
    assert saved["activation_history"] == result.activation_history
    assert len(saved["final_activation"]) == core.node_count

    reflection = ReflectionEngine(source_limit=3)
    reflection_input = reflection.build_input(saved, learn=False)
    reflected = reflection.replay(core, reflection_input)
    assert reflected is not None
    assert decoder.decode(reflected).steps >= 1
