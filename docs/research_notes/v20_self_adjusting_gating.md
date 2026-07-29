# v20 — Self-Adjusting Gating

## Question

Can SphereBrain decide when to narrow its own propagation without using a fixed step schedule?

## Principle

The gate reads only consecutive internal activation patterns.

```text
activity change = 1 - cosine(previous activity, current activity)
```

It never reads:

- words or labels
- expected successors
- decoder rankings
- probe identity

Selectivity is monotonic:

```text
0.80 -> 0.90 -> 0.95
```

A transition occurs when the internal activity change falls below the configured boundary. Different inputs may therefore narrow at different propagation steps.

## Controls

The v19c `boundary1` schedule remains as the external-time control:

```text
step 1:      0.80
steps 2-5:   0.90
steps 6-24:  0.95
```

## Experimental conditions

- `change_loose`: medium at change <= 0.20, narrow at change <= 0.08
- `change_mid`: medium at change <= 0.15, narrow at change <= 0.05
- `change_strict`: medium at change <= 0.10, narrow at change <= 0.03
- `change_mid2`: same as `change_mid`, but requires two consecutive stable observations

## Evaluation

- learned top-1 recall
- successor margin
- final pairwise similarity
- identity preservation
- transition step to 0.90 and 0.95
- per-probe transition differences

## Interpretation guardrail

This experiment does not yet modify the core `SurfaceFlowBrain`. It tests whether an internal-state-driven gate can replace the externally imposed temporal schedule before the rule is promoted into the core propagation model.

## Run

```powershell
python experiments/run_self_adjusting_gating_v20.py
```
