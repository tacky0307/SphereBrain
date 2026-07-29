from __future__ import annotations

from dormant_surface_flow import DormantSurfaceFlowBrain
from surface_flow import SurfaceFlowResult


class RecoveryGatedDormantBrain(DormantSurfaceFlowBrain):
    """Dormant brain whose signal-driven waking is allowed only in recovery.

    Ordinary observation must not rewrite pathway state. During recovery mode,
    repeated strong propagation can wake standby routes before teacher-guided
    reinforcement selects and protects useful alternatives.
    """

    def _auto_reactivate_from_result(self, result: SurfaceFlowResult) -> None:
        if not self.recovery_mode:
            self.last_auto_reactivated = set()
            return
        super()._auto_reactivate_from_result(result)
