# Experiment 014.1: Experience Field Gain Sweep

## Purpose

Experiment 014 showed that the transient Experience Field can strengthen the relationship between activity separated in time. At 100 repetitions, both A and F recruited P more strongly than local plasticity alone. At larger repetition counts, that relationship disappeared while P itself became a strong attractor.

This experiment keeps the architecture and training sequence unchanged and varies only `trace_gain`. The goal is to find a gain that forms the temporal association early and preserves it through prolonged repetition.

## Hypothesis

A gain below the Experiment 014 default may retain the useful A/F-to-P association while reducing overgrowth of P-centered self-reinforcement.

The experiment does not assume that the largest P activity is best. The preferred condition is the one that:

1. produces a positive field advantage over local plasticity for A and F,
2. maintains that advantage across checkpoints,
3. avoids replacing the association with an isolated P attractor.

## Fixed conditions

Training remains identical to Experiment 014:

- A -> P
- F -> P
- 5 source steps
- 7 destination steps
- 8 settling/learning steps
- alternating A/F training order
- identical attractor seed, topology, learning rates, recall procedure, and checkpoints

Default checkpoints:

- 0
- 100
- 200
- 300
- 400
- 500

## Independent variable

Default `trace_gain` values:

- 0.25
- 0.50
- 0.75
- 1.00

The original immediate local plasticity remains active in every trained condition. The gain changes only how strongly recent activity is accumulated into the transient Experience Field.

## Controls

- `untrained`: no learning
- `local`: original immediate local plasticity only

Each field-gain condition is compared against the same local baseline at every checkpoint.

## Measurements

For cues A, F, and P:

- `P_peak_activity`
- `P_cumulative_activity`
- `P_final_activity`
- `P_peak_gain_vs_local`
- `P_cumulative_gain_vs_local`
- `experience_similarity`
- `trial_similarity`

The principal measurements for temporal association are the A-cue and F-cue gains over local plasticity.

## Run

```powershell
python experiments/run_experience_field_gain_sweep.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

Optional custom gains:

```powershell
python experiments/run_experience_field_gain_sweep.py --gains 0.10,0.25,0.50,0.75,1.00
```

## Output

Results are written to:

```text
results/experiment_014_1_gain_sweep/
```

Main curve:

```text
results/experiment_014_1_gain_sweep/experience_field_gain_sweep_curve.csv
```

PowerShell view:

```powershell
Import-Csv .\results\experiment_014_1_gain_sweep\experience_field_gain_sweep_curve.csv |
Where-Object { $_.mode -eq 'field' } |
Format-Table gain,checkpoint,cue,P_peak_activity,P_cumulative_activity,P_final_activity,P_peak_gain_vs_local,P_cumulative_gain_vs_local,experience_similarity,trial_similarity -AutoSize
```

## Success criterion

A promising gain should show positive A/F recruitment of P near 100 repetitions and preserve meaningful recruitment at later checkpoints, ideally through 500 repetitions.

The result should be interpreted as a stability curve, not as a single final score. A condition that peaks early and collapses later is less suitable than a smaller but persistent association.
