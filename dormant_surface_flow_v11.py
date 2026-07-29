from __future__ import annotations

from dormant_surface_flow_v9 import SameRegionDistinctInputRecoveryBrain


class Top50SharedExperienceRecoveryBrain(SameRegionDistinctInputRecoveryBrain):
    """Promote dormant pathways reused by distinct inputs inside one region.

    v11 widens actual candidate observation from top 20 to top 50. A dormant
    pathway becomes eligible only after it appears for at least two distinct
    input values in the same input region. Same-input repetitions are ignored.
    Promotion caps and teacher blocking are inherited unchanged from v9/v7.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_candidates_per_experience"] = 50
        kwargs.setdefault("distinct_inputs_required", 2)
        super().__init__(*args, **kwargs)
