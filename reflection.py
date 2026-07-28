from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from brain import SignalResult, SphereBrain


@dataclass(frozen=True)
class ReflectionInput:
    trace_id: int
    source_nodes: list[int]
    learn: bool


class ReflectionEngine:
    """Traceを文章へ戻さず、数値活動としてCoreへ再入力する。"""

    def __init__(self, source_limit: int = 4) -> None:
        self.source_limit = source_limit

    def build_input(self, trace: dict, learn: bool = True) -> ReflectionInput:
        candidates: list[int] = []

        for step in reversed(trace.get("activation_history", [])):
            for node in step:
                node = int(node)
                if node not in candidates:
                    candidates.append(node)
                if len(candidates) >= self.source_limit:
                    break
            if len(candidates) >= self.source_limit:
                break

        if not candidates:
            candidates = [int(node) for node in trace.get("source_nodes", [])[: self.source_limit]]

        return ReflectionInput(
            trace_id=int(trace["id"]),
            source_nodes=candidates,
            learn=learn,
        )

    def replay(
        self,
        brain: SphereBrain,
        reflection_input: ReflectionInput,
        context_nodes: Iterable[int] | None = None,
    ) -> SignalResult | None:
        if not reflection_input.source_nodes:
            return None
        return brain.propagate(
            source_nodes=reflection_input.source_nodes,
            context_nodes=context_nodes,
            steps=12,
            threshold=0.19,
            noise=0.025,
            learn=reflection_input.learn,
        )
