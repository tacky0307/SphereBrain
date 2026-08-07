from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain import SphereBrain
from core_native_learning import CoreNativeLearningState


class NativeLearningSphereBrain(SphereBrain):
    """SphereBrain candidate with Temporal Credit + Homeostatic Consolidation owned by Core."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learning_state = CoreNativeLearningState()

    def clear_learning_state(self) -> None:
        self.learning_state.clear()

    def snapshot_learning_state(self) -> dict:
        return self.learning_state.snapshot()

    def observe_learning_episode(
        self,
        transitions,
        *,
        success: bool,
        expected_conditions,
        motif: str = "native_learning",
    ) -> dict:
        return self.learning_state.observe_episode(
            self,
            transitions,
            success=success,
            expected_conditions=expected_conditions,
            motif=motif,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "node_count": self.node_count,
            "neighbors_per_node": self.neighbors_per_node,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "propagation_mode": self.propagation_mode,
            "signal_decay": self.signal_decay,
            "max_branches": self.max_branches,
            "max_active_per_step": self.max_active_per_step,
            "max_total_active_nodes": self.max_total_active_nodes,
            "structural_assist_enabled": self.structural_assist_enabled,
            "structural_gain": self.structural_gain,
            "structural_tie_margin": self.structural_tie_margin,
            "structural_near_zero_margin": self.structural_near_zero_margin,
            "structural_relative_cap_ratio": self.structural_relative_cap_ratio,
            "structural_absolute_cap": self.structural_absolute_cap,
            "experience_state": self.experience_state.snapshot(),
            "learning_state": self.learning_state.snapshot(),
            "positions": self.positions.tolist(),
            "adjacency": self.adjacency.astype(int).tolist(),
            "weights": self.weights.tolist(),
            "usage": self.usage.tolist(),
            "node_usage": self.node_usage.tolist(),
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NativeLearningSphereBrain":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        brain = cls(
            node_count=data["node_count"],
            neighbors_per_node=data["neighbors_per_node"],
            seed=data["seed"],
            learning_rate=data["learning_rate"],
            decay_rate=data["decay_rate"],
            propagation_mode=data.get("propagation_mode", "focused"),
            signal_decay=data.get("signal_decay", 0.78),
            max_branches=data.get("max_branches", 2),
            max_active_per_step=data.get("max_active_per_step", 72),
            max_total_active_nodes=data.get("max_total_active_nodes", 100),
            structural_assist_enabled=data.get("structural_assist_enabled", False),
            structural_gain=data.get("structural_gain", 0.02),
            structural_tie_margin=data.get("structural_tie_margin", 0.0025),
            structural_near_zero_margin=data.get("structural_near_zero_margin", 1e-8),
            structural_relative_cap_ratio=data.get("structural_relative_cap_ratio", 0.35),
            structural_absolute_cap=data.get("structural_absolute_cap", 5e-5),
        )
        brain.positions = np.asarray(data["positions"], dtype=float)
        brain.adjacency = np.asarray(data["adjacency"], dtype=bool)
        brain.weights = np.asarray(data["weights"], dtype=float)
        # Preserve the current persisted dtype for exact backward compatibility.
        brain.usage = np.asarray(data["usage"])
        brain.node_usage = np.asarray(data.get("node_usage", [0] * brain.node_count), dtype=int)
        brain.experience_state.replace_from_mapping(data.get("experience_state"))
        brain.learning_state.replace_from_mapping(data.get("learning_state"))
        return brain

    @classmethod
    def from_sphere_brain(cls, source: SphereBrain) -> "NativeLearningSphereBrain":
        brain = cls(
            node_count=source.node_count,
            neighbors_per_node=source.neighbors_per_node,
            seed=source.seed,
            learning_rate=source.learning_rate,
            decay_rate=source.decay_rate,
            propagation_mode=source.propagation_mode,
            signal_decay=source.signal_decay,
            max_branches=source.max_branches,
            max_active_per_step=source.max_active_per_step,
            max_total_active_nodes=source.max_total_active_nodes,
            structural_assist_enabled=source.structural_assist_enabled,
            structural_gain=source.structural_gain,
            structural_tie_margin=source.structural_tie_margin,
            structural_near_zero_margin=source.structural_near_zero_margin,
            structural_relative_cap_ratio=source.structural_relative_cap_ratio,
            structural_absolute_cap=source.structural_absolute_cap,
        )
        brain.positions = source.positions.copy()
        brain.adjacency = source.adjacency.copy()
        brain.weights = source.weights.copy()
        brain.usage = source.usage.copy()
        brain.node_usage = source.node_usage.copy()
        brain.experience_state.replace_from_mapping(source.experience_state.snapshot())
        return brain
