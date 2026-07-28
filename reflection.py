from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from brain import SignalResult, SphereBrain


@dataclass(frozen=True)
class ReflectionInput:
    trace_id: int
    source_nodes: list[int]
    trace_steps: list[list[int]]
    replay_mode: str
    replay_strength: float
    learn: bool


class ReflectionEngine:
    """Traceを文章へ戻さず、時間的な数値活動としてCoreへ再入力する。"""

    def __init__(self, source_limit: int = 4, replay_strength: float = 0.28) -> None:
        self.source_limit = source_limit
        self.replay_strength = replay_strength

    def build_input(
        self,
        trace: dict,
        learn: bool = True,
        legacy: bool = False,
    ) -> ReflectionInput:
        trace_steps = [
            [int(node) for node in step]
            for step in trace.get("activation_history", [])
            if step
        ]

        candidates: list[int] = []
        for step in reversed(trace_steps):
            for node in step:
                if node not in candidates:
                    candidates.append(node)
                if len(candidates) >= self.source_limit:
                    break
            if len(candidates) >= self.source_limit:
                break

        if not candidates:
            candidates = [int(node) for node in trace.get("source_nodes", [])[: self.source_limit]]

        replay_mode = "legacy-nodes" if legacy else "temporal-trace"
        return ReflectionInput(
            trace_id=int(trace["id"]),
            source_nodes=candidates,
            trace_steps=trace_steps,
            replay_mode=replay_mode,
            replay_strength=self.replay_strength,
            learn=learn,
        )

    def replay(
        self,
        brain: SphereBrain,
        reflection_input: ReflectionInput,
        context_nodes: Iterable[int] | None = None,
    ) -> SignalResult | None:
        if reflection_input.replay_mode == "legacy-nodes":
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

        if not reflection_input.trace_steps:
            return None
        return brain.replay_trace(
            activation_history=reflection_input.trace_steps,
            replay_strength=reflection_input.replay_strength,
            learn=reflection_input.learn,
        )
