from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from brain import SphereBrain
from decoder import NumericDecoder
from encoder import TextEncoder
from reflection import ReflectionEngine
from trace_store import TraceStore


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"


def run_experience(text: str, learn: bool = True) -> dict:
    core = SphereBrain.load(BRAIN_FILE) if BRAIN_FILE.exists() else SphereBrain(
        node_count=600,
        neighbors_per_node=8,
    )
    encoder = TextEncoder(core.node_count, source_count=4)
    decoder = NumericDecoder()
    traces = TraceStore(DB_FILE)

    stimulus = encoder.encode(text)
    event_id = traces.add_external_event(
        kind="text",
        external_text=text,
        encoder_name=stimulus.encoder,
        importance=1.0,
    )
    context = traces.recent_context_nodes(trace_limit=7, node_limit=18)
    result = core.propagate(
        source_nodes=stimulus.source_nodes,
        context_nodes=context,
        steps=20,
        learn=learn,
    )
    trace_id = traces.add_trace(
        result=result,
        mode="experience",
        learn_enabled=learn,
        event_id=event_id,
    )
    decoded = decoder.decode(result)
    core.save(BRAIN_FILE)

    return {
        "layer_flow": ["Encoder", "Core", "Trace", "Decoder"],
        "trace_id": trace_id,
        "stimulus": asdict(stimulus),
        "decoded": asdict(decoded),
    }


def run_reflection(trace_id: int | None = None, learn: bool = True) -> dict:
    core = SphereBrain.load(BRAIN_FILE)
    traces = TraceStore(DB_FILE)
    reflection = ReflectionEngine()
    decoder = NumericDecoder()

    trace = traces.get_trace(trace_id) if trace_id is not None else traces.latest_trace("experience")
    if trace is None:
        raise ValueError("Reflection対象のTraceがありません。")

    reflection_input = reflection.build_input(trace, learn=learn)
    result = reflection.replay(core, reflection_input)
    if result is None:
        raise ValueError("Traceから数値刺激を生成できませんでした。")

    reflected_trace_id = traces.add_trace(
        result=result,
        mode="reflection",
        learn_enabled=learn,
        parent_trace_id=reflection_input.trace_id,
    )
    decoded = decoder.decode(result)
    core.save(BRAIN_FILE)

    return {
        "layer_flow": ["Trace", "Reflection", "Core", "Trace", "Decoder"],
        "source_trace_id": reflection_input.trace_id,
        "trace_id": reflected_trace_id,
        "reflection_input": asdict(reflection_input),
        "decoded": asdict(decoded),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sphere Brain five-layer research cycle")
    sub = parser.add_subparsers(dest="command", required=True)

    experience = sub.add_parser("experience")
    experience.add_argument("text")
    experience.add_argument("--no-learn", action="store_true")

    reflection = sub.add_parser("reflect")
    reflection.add_argument("--trace-id", type=int)
    reflection.add_argument("--no-learn", action="store_true")

    args = parser.parse_args()
    if args.command == "experience":
        output = run_experience(args.text, learn=not args.no_learn)
    else:
        output = run_reflection(args.trace_id, learn=not args.no_learn)
    print(json.dumps(output, ensure_ascii=False, indent=2))
