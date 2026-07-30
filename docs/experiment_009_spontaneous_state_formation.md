# Experiment 009 — Spontaneous State Formation

## Research question

Does repeated experience change how the whole sphere naturally settles after a partial stimulus?

This experiment does **not** ask whether activity reaches a prescribed node C. C is only one part of the repeated experience. The observable is the complete post-stimulus activity pattern.

## v30 design

`AttractorSphereCore` is added beside the original `SphereWaveCore`. The original wave experiments remain unchanged.

The v30 dynamics separate two meanings that were previously combined in conductivity:

- `direction`: relative preference for where activity travels
- `capacity`: absolute ability of a local pathway to support activity

Activity is governed by:

- bounded recurrent excitation
- pathway capacity
- persistence
- local inhibition
- global inhibition
- fatigue
- decay

No target, winner, replay route, or answer edge is supplied.

## Experience

One experience is:

1. stimulate region A
2. allow five learning steps
3. stimulate region C
4. allow seven learning steps
5. allow eight additional learning steps

Only actual local activity changes direction and capacity.

## Recall observation

After each checkpoint:

1. clear short-term state without erasing terrain
2. stimulate only region A
3. stop external input
4. allow the sphere to become quiet or stable
5. repeat several trials

## Metrics

- `mean_lifetime`: how long meaningful activity remains
- `mean_final_total_activity`: total activity in the final pattern
- `mean_active_cluster_size`: number of meaningfully active nodes near settlement
- `trial_to_trial_similarity`: whether repeated A-only trials settle into similar patterns
- `experience_pattern_similarity`: whether the spontaneous pattern resembles the experienced whole-state pattern
- `mean_direction`: average local directional weight
- `mean_capacity`: average local pathway capacity

The experience pattern is used only for offline measurement. It is never injected during recall.

## First success signals

The first meaningful result does not need to be full pattern completion.

Possible progressive signals are:

1. trained activity lasts longer than control
2. trained trials settle more consistently than control
3. trained activity forms a bounded cluster rather than disappearing or activating the whole sphere
4. the trained settled pattern increasingly resembles the experienced activity pattern

## Failure modes

- **extinction:** trained and control both disappear immediately
- **global saturation:** nearly all nodes remain active
- **frozen input:** only the original A region remains active
- **unstable oscillation:** activity never settles and patterns vary strongly
- **terrain blindness:** trained and control metrics remain identical

Each failure is diagnostic and should be handled by changing only one dynamic component at a time.

## Run

```powershell
python experiments/run_spontaneous_state_formation.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

Inspect the curve:

```powershell
Import-Csv .\results\experiment_009\state_formation_curve.csv |
Format-Table checkpoint,mode,mean_lifetime,mean_final_total_activity,mean_active_cluster_size,trial_to_trial_similarity,experience_pattern_similarity,mean_capacity -AutoSize
```

Inspect one final trained trajectory:

```powershell
Import-Csv .\results\experiment_009\brain_01\checkpoint_0500_trained_trial_01_steps.csv |
Format-Table step,total_activity,active_count,mean_excitation,mean_inhibition,mean_fatigue -AutoSize
```

## Interpretation rule

Do not tune the model toward C. Tune only for a biologically and dynamically plausible middle region between extinction and global saturation, then observe whether experience creates repeatable spontaneous states.
