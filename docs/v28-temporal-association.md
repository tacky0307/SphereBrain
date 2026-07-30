# SphereBrain v28 — Temporal Association

## Purpose

v28 tests whether activity separated in time can change the Core itself and later guide recall.

The minimum sequence is:

```text
A
↓
quiet interval
↓
B
```

After this sequence is repeated, stimulating only A should produce activity at B through a directed path formed inside the Core.

## Design rule

No Memory module is introduced.

- `activity` is the present state.
- `temporal_trace` is a fading internal echo of recent activity.
- `weights` are the persistent shape of the Core.
- learning changes `weights` from past trace to present activity.

Conceptually:

```text
weight change = learning rate × past trace × current activity
```

The weight matrix is directed, so `A → B` and `B → A` are different paths.

## Why this is an experiment first

The repository currently contains the earlier spherical prototype rather than the complete v27 lifecycle described in the research notes. To avoid mixing incompatible architectures, v28 begins as an isolated experiment.

Once the behavior is confirmed, the same mechanism can be integrated into the active v27 Core and applied during both Experience and Reflection.

## Files

- `experiments/run_temporal_association_v28.py`
- `tests/test_temporal_association_v28.py`

## Run

```bash
python experiments/run_temporal_association_v28.py
python -m pytest tests/test_temporal_association_v28.py
```

## Success conditions

1. `A → B` becomes stronger than `B → A`.
2. A shorter gap forms a stronger association than a longer gap.
3. Recall from A activates B more strongly after learning than before learning.

## Next integration step

When the current v27 code is committed, move the mechanism into Core as three responsibilities:

1. update and decay temporal traces at every Core step;
2. apply directed plasticity from previous traces to current activity;
3. use the same Core path for Experience and Reflection, while keeping Trace as observation only.
