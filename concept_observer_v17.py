from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass
class ConceptState:
    """A recurring internal activation pattern.

    The identifier is deliberately meaningless. A human-readable label is
    optional and may be attached later, after recurrence has been observed.
    """

    state_id: int
    prototype: dict[int, float]
    occurrences: int = 1
    first_seen: str = field(default_factory=lambda: _utc_now())
    last_seen: str = field(default_factory=lambda: _utc_now())
    label: str | None = None
    contexts: dict[str, int] = field(default_factory=dict)
    recent_similarities: list[float] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.label or f"unknown-{self.state_id}"


@dataclass(frozen=True)
class Observation:
    state_id: int
    is_new: bool
    similarity: float
    occurrences: int
    label: str | None


class UnknownStateObserver:
    """Store unnamed activation patterns and detect their recurrence.

    This class does not alter the brain and does not invent semantic labels.
    It watches the output of SphereBrain, normalises it into a sparse numeric
    pattern, and matches it against previously observed patterns.

    Philosophy:
        phenomenon first -> recurrence second -> name later
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        prototype_learning_rate: float = 0.15,
        max_recent_similarities: int = 20,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not 0.0 < prototype_learning_rate <= 1.0:
            raise ValueError("prototype_learning_rate must be in (0, 1]")
        self.similarity_threshold = float(similarity_threshold)
        self.prototype_learning_rate = float(prototype_learning_rate)
        self.max_recent_similarities = int(max_recent_similarities)
        self.states: dict[int, ConceptState] = {}
        self._next_state_id = 1

    def observe(
        self,
        activation: Mapping[int, float] | Iterable[int],
        context: Sequence[str] | None = None,
    ) -> Observation:
        pattern = _normalise_activation(activation)
        if not pattern:
            raise ValueError("activation must contain at least one active node")

        state, similarity = self._best_match(pattern)
        if state is None or similarity < self.similarity_threshold:
            state = ConceptState(
                state_id=self._next_state_id,
                prototype=pattern,
            )
            self.states[state.state_id] = state
            self._next_state_id += 1
            self._record_context(state, context)
            return Observation(
                state_id=state.state_id,
                is_new=True,
                similarity=1.0,
                occurrences=state.occurrences,
                label=state.label,
            )

        self._update_state(state, pattern, similarity, context)
        return Observation(
            state_id=state.state_id,
            is_new=False,
            similarity=similarity,
            occurrences=state.occurrences,
            label=state.label,
        )

    def assign_label(self, state_id: int, label: str) -> None:
        clean = label.strip()
        if not clean:
            raise ValueError("label must not be empty")
        self.states[state_id].label = clean

    def recurrent_states(self, minimum_occurrences: int = 2) -> list[ConceptState]:
        return sorted(
            (state for state in self.states.values() if state.occurrences >= minimum_occurrences),
            key=lambda state: (-state.occurrences, state.state_id),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "similarity_threshold": self.similarity_threshold,
            "prototype_learning_rate": self.prototype_learning_rate,
            "max_recent_similarities": self.max_recent_similarities,
            "next_state_id": self._next_state_id,
            "states": [asdict(state) for state in self.states.values()],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnknownStateObserver":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        observer = cls(
            similarity_threshold=float(payload["similarity_threshold"]),
            prototype_learning_rate=float(payload["prototype_learning_rate"]),
            max_recent_similarities=int(payload["max_recent_similarities"]),
        )
        observer._next_state_id = int(payload["next_state_id"])
        observer.states = {
            int(raw["state_id"]): ConceptState(
                state_id=int(raw["state_id"]),
                prototype={int(k): float(v) for k, v in raw["prototype"].items()},
                occurrences=int(raw["occurrences"]),
                first_seen=str(raw["first_seen"]),
                last_seen=str(raw["last_seen"]),
                label=raw.get("label"),
                contexts={str(k): int(v) for k, v in raw.get("contexts", {}).items()},
                recent_similarities=[float(v) for v in raw.get("recent_similarities", [])],
            )
            for raw in payload["states"]
        }
        return observer

    def _best_match(self, pattern: Mapping[int, float]) -> tuple[ConceptState | None, float]:
        best_state: ConceptState | None = None
        best_similarity = -1.0
        for state in self.states.values():
            similarity = _cosine_similarity(pattern, state.prototype)
            if similarity > best_similarity:
                best_state = state
                best_similarity = similarity
        return best_state, max(best_similarity, 0.0)

    def _update_state(
        self,
        state: ConceptState,
        pattern: Mapping[int, float],
        similarity: float,
        context: Sequence[str] | None,
    ) -> None:
        rate = self.prototype_learning_rate
        all_nodes = set(state.prototype) | set(pattern)
        updated: dict[int, float] = {}
        for node in all_nodes:
            value = (1.0 - rate) * state.prototype.get(node, 0.0) + rate * pattern.get(node, 0.0)
            if value > 1e-9:
                updated[node] = value
        state.prototype = _normalise_activation(updated)
        state.occurrences += 1
        state.last_seen = _utc_now()
        state.recent_similarities.append(float(similarity))
        if len(state.recent_similarities) > self.max_recent_similarities:
            del state.recent_similarities[:-self.max_recent_similarities]
        self._record_context(state, context)

    @staticmethod
    def _record_context(state: ConceptState, context: Sequence[str] | None) -> None:
        if not context:
            return
        for item in context:
            clean = item.strip()
            if clean:
                state.contexts[clean] = state.contexts.get(clean, 0) + 1


def _normalise_activation(
    activation: Mapping[int, float] | Iterable[int],
) -> dict[int, float]:
    if isinstance(activation, Mapping):
        raw = {int(node): max(float(value), 0.0) for node, value in activation.items()}
    else:
        raw: dict[int, float] = {}
        for node in activation:
            node_id = int(node)
            raw[node_id] = raw.get(node_id, 0.0) + 1.0

    norm = math.sqrt(sum(value * value for value in raw.values()))
    if norm <= 0.0:
        return {}
    return {node: value / norm for node, value in raw.items() if value > 0.0}


def _cosine_similarity(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(node, 0.0) for node, value in left.items())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
