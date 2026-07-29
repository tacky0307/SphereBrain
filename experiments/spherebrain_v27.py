from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import log2
from pathlib import Path
from typing import Iterable
import json
import random


@dataclass(frozen=True)
class Stimulus:
    values: tuple[float, ...]
    source_text: str = ""


@dataclass
class TraceRecord:
    stimulus: tuple[float, ...]
    path: list[int]
    edge_strengths_before: list[float]
    edge_strengths_after: list[float]
    final_state: tuple[float, ...]
    replayed: bool = False


@dataclass
class CoreResult:
    path: list[int]
    final_state: tuple[float, ...]
    confidence: float
    entropy: float


class Encoder:
    """Convert external text into a stable numeric stimulus.

    The Core never sees words. Text is retained only as optional external
    metadata so that the numeric route can be inspected without contaminating
    the internal learning rule.
    """

    def __init__(self, dimensions: int = 16) -> None:
        if dimensions < 4:
            raise ValueError("dimensions must be at least 4")
        self.dimensions = dimensions

    def encode(self, text: str) -> Stimulus:
        normalized = " ".join(text.strip().split())
        if not normalized:
            raise ValueError("text must not be empty")
        digest = sha256(normalized.encode("utf-8")).digest()
        values = tuple((digest[i % len(digest)] / 127.5) - 1.0 for i in range(self.dimensions))
        return Stimulus(values=values, source_text=normalized)


class TraceStore:
    """Append-only storage of routes actually traversed by the Core."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[TraceRecord] = []
        if path and path.exists():
            self._load()

    def append(self, record: TraceRecord) -> None:
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 10) -> list[TraceRecord]:
        return self.records[-limit:]

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            self.records.append(
                TraceRecord(
                    stimulus=tuple(raw["stimulus"]),
                    path=list(raw["path"]),
                    edge_strengths_before=list(raw["edge_strengths_before"]),
                    edge_strengths_after=list(raw["edge_strengths_after"]),
                    final_state=tuple(raw["final_state"]),
                    replayed=bool(raw.get("replayed", False)),
                )
            )


class SphereCore:
    """Numeric-only route-forming core.

    Nodes are arranged as an abstract sphere index. Learning is stored on
    directed edges. Frequently traversed routes become easier to select, while
    weak unused routes slowly decay. Activation is conserved at every step.
    """

    def __init__(
        self,
        node_count: int = 256,
        fan_out: int = 8,
        learning_rate: float = 0.08,
        decay: float = 0.0005,
        seed: int = 27,
    ) -> None:
        if node_count < 32:
            raise ValueError("node_count must be at least 32")
        if not 2 <= fan_out < node_count:
            raise ValueError("fan_out must be between 2 and node_count - 1")
        self.node_count = node_count
        self.fan_out = fan_out
        self.learning_rate = learning_rate
        self.decay = decay
        self.rng = random.Random(seed)
        self.edges: dict[int, dict[int, float]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for node in range(self.node_count):
            targets: dict[int, float] = {}
            offset = 1
            while len(targets) < self.fan_out:
                target = (node + offset * offset + 3 * offset) % self.node_count
                if target != node:
                    targets[target] = 1.0
                offset += 1
            self.edges[node] = targets

    def _source_node(self, stimulus: tuple[float, ...]) -> int:
        weighted = sum((index + 1) * value for index, value in enumerate(stimulus))
        return int(abs(weighted) * 10_000) % self.node_count

    def propagate(self, stimulus: Stimulus, steps: int = 18, learn: bool = True) -> tuple[CoreResult, TraceRecord]:
        if steps < 1:
            raise ValueError("steps must be positive")
        state = list(stimulus.values)
        node = self._source_node(stimulus.values)
        path = [node]
        before: list[float] = []
        after: list[float] = []
        choice_probabilities: list[float] = []

        for step in range(steps):
            candidates = self.edges[node]
            scored: list[tuple[int, float]] = []
            signal = state[step % len(state)]
            for target, strength in candidates.items():
                geometry = 1.0 - (abs(target - node) / self.node_count)
                phase = (((target + 1) * (step + 3)) % 17) / 17.0
                score = max(1e-9, strength * (1.0 + 0.20 * signal + 0.08 * geometry + 0.04 * phase))
                scored.append((target, score))

            total = sum(score for _, score in scored)
            probabilities = [(target, score / total) for target, score in scored]
            target, probability = max(probabilities, key=lambda item: item[1])
            old_strength = candidates[target]
            new_strength = old_strength
            if learn:
                new_strength = old_strength + self.learning_rate * (1.0 - old_strength / 8.0)
                candidates[target] = new_strength
                for other in candidates:
                    if other != target:
                        candidates[other] = max(0.05, candidates[other] * (1.0 - self.decay))

            before.append(old_strength)
            after.append(new_strength)
            choice_probabilities.append(probability)
            node = target
            path.append(node)
            shift = (target % len(state))
            state = state[shift:] + state[:shift]

        entropy = -sum(p * log2(p) for p in choice_probabilities if p > 0) / len(choice_probabilities)
        confidence = sum(choice_probabilities) / len(choice_probabilities)
        final_state = tuple(state)
        result = CoreResult(path=path, final_state=final_state, confidence=confidence, entropy=entropy)
        trace = TraceRecord(
            stimulus=stimulus.values,
            path=path,
            edge_strengths_before=before,
            edge_strengths_after=after,
            final_state=final_state,
        )
        return result, trace


class Decoder:
    """Translate a converged numeric state into an external description."""

    def decode(self, result: CoreResult) -> str:
        dominant = max(range(len(result.final_state)), key=lambda i: abs(result.final_state[i]))
        route_signature = "-".join(str(node) for node in result.path[-5:])
        return (
            f"state:{dominant} route:{route_signature} "
            f"confidence:{result.confidence:.4f} entropy:{result.entropy:.4f}"
        )


class Reflection:
    """Re-enter recorded experience without directly editing its original trace."""

    def __init__(self, core: SphereCore, traces: TraceStore) -> None:
        self.core = core
        self.traces = traces

    def replay(self, records: Iterable[TraceRecord], steps: int = 10) -> list[CoreResult]:
        results: list[CoreResult] = []
        for record in records:
            stimulus = Stimulus(values=record.final_state)
            result, new_trace = self.core.propagate(stimulus, steps=steps, learn=True)
            new_trace.replayed = True
            self.traces.append(new_trace)
            results.append(result)
        return results


@dataclass
class SphereBrainV27:
    encoder: Encoder = field(default_factory=Encoder)
    core: SphereCore = field(default_factory=SphereCore)
    traces: TraceStore = field(default_factory=TraceStore)
    decoder: Decoder = field(default_factory=Decoder)

    def experience(self, text: str, steps: int = 18) -> str:
        stimulus = self.encoder.encode(text)
        result, trace = self.core.propagate(stimulus, steps=steps, learn=True)
        self.traces.append(trace)
        return self.decoder.decode(result)

    def reflect(self, limit: int = 5, steps: int = 10) -> list[str]:
        reflection = Reflection(self.core, self.traces)
        results = reflection.replay(self.traces.recent(limit), steps=steps)
        return [self.decoder.decode(result) for result in results]


if __name__ == "__main__":
    brain = SphereBrainV27(traces=TraceStore(Path("data/v27_traces.jsonl")))
    print("SphereBrain v27 — layered numeric-path architecture")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/reflect":
            for line in brain.reflect():
                print(line)
            continue
        print(brain.experience(text))
