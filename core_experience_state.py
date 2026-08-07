from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CoreExperienceState:
    """Native, serializable experience-derived state owned by SphereBrain.

    This state is deliberately behavior-neutral. It can accumulate and expose
    experience evidence, confidence, surprise and drift, but it must not alter
    propagation, thresholds, weights, topology, usage, learning or Decoder output.
    """

    schema_version: int = 1
    kind: str = "motif_stability_profile"
    motif: str | None = None
    stability_class: str | None = None
    baseline_present: bool | None = None
    echo_resistant: bool | None = None
    position_resistant: bool | None = None
    common_resistant: bool | None = None
    evidence_conditions: int = 0
    evidence_experiences: int = 0
    confidence: float = 0.0
    surprise_ewma: float = 0.0
    drift_suspected: bool = False
    adaptive_forgetting: bool = False
    forgetting_decay: float | None = None
    condition_evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    def replace_from_mapping(self, value: dict[str, Any] | None) -> None:
        if not value:
            self.clear()
            return
        allowed = set(asdict(self))
        for key in allowed:
            if key in value:
                setattr(self, key, deepcopy(value[key]))

    def update_profile(
        self,
        *,
        motif: str | None,
        stability_class: str | None,
        baseline_present: bool | None,
        echo_resistant: bool | None,
        position_resistant: bool | None,
        common_resistant: bool | None,
        evidence_conditions: int,
        evidence_experiences: int,
        confidence: float,
        surprise_ewma: float = 0.0,
        drift_suspected: bool = False,
        adaptive_forgetting: bool = False,
        forgetting_decay: float | None = None,
        condition_evidence: dict[str, dict[str, float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.kind = "motif_stability_profile"
        self.motif = motif
        self.stability_class = stability_class
        self.baseline_present = baseline_present
        self.echo_resistant = echo_resistant
        self.position_resistant = position_resistant
        self.common_resistant = common_resistant
        self.evidence_conditions = int(evidence_conditions)
        self.evidence_experiences = int(evidence_experiences)
        self.confidence = float(confidence)
        self.surprise_ewma = float(surprise_ewma)
        self.drift_suspected = bool(drift_suspected)
        self.adaptive_forgetting = bool(adaptive_forgetting)
        self.forgetting_decay = None if forgetting_decay is None else float(forgetting_decay)
        if condition_evidence is not None:
            self.condition_evidence = deepcopy(condition_evidence)
        if metadata is not None:
            self.metadata = deepcopy(metadata)

    def clear(self) -> None:
        fresh = CoreExperienceState()
        for key, value in asdict(fresh).items():
            setattr(self, key, deepcopy(value))

    def contains_forbidden_position_label(self) -> bool:
        import json

        text = json.dumps(self.snapshot(), ensure_ascii=False)
        return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])
