from experiments.spherebrain_v27 import Encoder, SphereBrainV27, SphereCore, TraceStore


def test_encoder_is_stable_and_numeric_only():
    encoder = Encoder(dimensions=16)
    first = encoder.encode("空は青い")
    second = encoder.encode("空は青い")
    assert first.values == second.values
    assert len(first.values) == 16
    assert all(-1.0 <= value <= 1.0 for value in first.values)


def test_experience_appends_trace_and_reinforces_route():
    brain = SphereBrainV27(core=SphereCore(seed=27), traces=TraceStore())
    brain.experience("今日は楽しい")
    first = brain.traces.records[-1]
    brain.experience("今日は楽しい")
    second = brain.traces.records[-1]
    assert len(brain.traces.records) == 2
    assert first.path == second.path
    assert sum(second.edge_strengths_after) > sum(first.edge_strengths_after)


def test_reflection_creates_new_trace_without_mutating_original():
    brain = SphereBrainV27(core=SphereCore(seed=27), traces=TraceStore())
    brain.experience("雨が降っている")
    original = brain.traces.records[0]
    original_path = list(original.path)
    outputs = brain.reflect(limit=1, steps=6)
    assert outputs
    assert len(brain.traces.records) == 2
    assert brain.traces.records[1].replayed is True
    assert original.path == original_path
