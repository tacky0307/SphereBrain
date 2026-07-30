# v28 / Experiment 014: Experience Field Association

## Purpose

Experiments 012 and 013 showed that repeated sequential stimulation strengthened individual regions but did not preserve the sequence as one experience.

v28 introduces a transient **Experience Field**. It is not a label or memory ID. While an experience is open, recent internal activity remains as a decaying eligibility field and can influence learning on existing local edges when later activity occurs.

## New core

`ExperienceFieldAttractorCore` extends `AttractorSphereCore` with:

- `begin_experience()`
- a per-node decaying `experience_trace`
- field-assisted direction and capacity plasticity
- `end_experience()`

The field:

- contains no target;
- contains no prescribed route;
- creates no new edge;
- is cleared at the beginning of each experience;
- affects learning only while the experience is open.

## Experiment

Both trained brains receive the same alternating experiences:

```text
A -> P
F -> P
```

Conditions:

- `untrained`: no learning
- `local`: original attractor plasticity only
- `field`: original plasticity plus the transient Experience Field

Recall supplies only:

- A
- F
- P

The main comparison is whether A and F recruit P more strongly and persistently in the `field` condition than in `local`.

## Metrics

- `P_peak_activity`
- `P_cumulative_activity`
- `P_final_activity`
- `trial_similarity`
- `experience_similarity`
- `P_peak_gain_vs_local`
- `P_cumulative_gain_vs_local`

A useful first signal is a positive P gain for both A and F at later checkpoints. Full convergence is not required on the first run.

## Run

```powershell
git pull
python experiments/run_experience_field_association.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

## Inspect

```powershell
Import-Csv .\results\experiment_014\experience_field_association_curve.csv |
Format-Table checkpoint,mode,cue,P_peak_activity,P_cumulative_activity,P_final_activity,P_peak_gain_vs_local,P_cumulative_gain_vs_local,experience_similarity,trial_similarity -AutoSize
```

## Parameter probes

Only after the baseline run, the field can be tested conservatively:

```powershell
python experiments/run_experience_field_association.py --trace-decay 0.95
```

```powershell
python experiments/run_experience_field_association.py --field-direction-rate 0.0005 --field-capacity-rate 0.0003
```

Interpret failure rather than tuning toward a desired answer. If the field condition still behaves like local learning, inspect whether the source trace remains when P begins. If it globally merges A and F, the field is too long-lived or too strong.
