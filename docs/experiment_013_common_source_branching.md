# Experiment 013: Common Source Branching

## Question

Can one shared starting state form two different downstream experiences?

Shared-source training:

- P -> A
- P -> F

Disconnected comparison:

- Q -> A
- R -> F

This reverses Experiment 012. Experiment 012 asked whether A and F could converge into P. Experiment 013 asks whether P can become a shared source from which A and F remain available as distinct branches.

## Why this experiment matters

Experiment 012 showed that P became a strong state when directly stimulated, but A and F did not reach P after training. One possibility is that the current local learning dynamics are directionally asymmetric: a strong source state may remain active long enough to overlap with the following destination, even when the reverse ordering fails.

The experiment therefore tests both temporal directionality and branch formation.

## Training conditions

The shared-source brain alternates:

1. P -> A
2. P -> F

The disconnected brain alternates:

1. Q -> A
2. R -> F

The untrained clone is retained as a baseline.

P, Q, and R use radius 3. A and F use radius 2.

## Recall cues

At every checkpoint the experiment recalls with:

- P
- P + weak A
- P + weak F
- A
- F

The weak contextual stimulus defaults to strength 0.15 and can be changed with `--context-strength`.

P alone tests spontaneous branch preference. The two weak-context cues test whether a small contextual bias can select a branch from the common source.

## Recorded metrics

For A, F, P, Q, and R, the experiment records:

- peak activity
- cumulative activity
- peak step
- final activity

It also records:

- A peak selectivity = A peak - F peak
- A cumulative selectivity = A cumulative - F cumulative
- A final selectivity = A final - F final
- trial similarity
- similarity to the latest P -> A destination experience
- similarity to the latest P -> F destination experience
- experience preference
- context selectivity
- context pattern separation

Context selectivity is:

`A-selectivity(P + weak A) - A-selectivity(P + weak F)`

A positive value means the weak A cue moves recall toward A more than the weak F cue does.

Context pattern separation is:

`1 - cosine_similarity(P + weak A pattern, P + weak F pattern)`

A larger positive value means the two weak contexts lead to more distinct whole-brain states.

## Interpretation guide

### Shared source but no branching

Likely signs:

- P alone produces similar A and F activity
- P + weak A and P + weak F remain nearly identical
- context selectivity is near zero
- context pattern separation is near zero

### One branch dominates

Likely signs:

- P alone consistently favors A or F
- trial similarity remains near 1
- weak context cannot reverse the preference

### Context-sensitive branching

Strongest evidence:

- P + weak A favors A
- P + weak F favors F
- context selectivity is positive
- context pattern separation rises above the untrained state
- both branches retain similarity to their corresponding trained experiences

### Stochastic branching

Possible signs:

- P-alone trials divide between A-like and F-like outcomes
- trial similarity falls

The current deterministic core may make this outcome unlikely unless internal perturbations already exist.

## Run

```powershell
git pull
python experiments/run_common_source_branching.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

To change the weak context:

```powershell
python experiments/run_common_source_branching.py --context-strength 0.10
```

## Output

Main summary:

```text
results/experiment_013/common_source_branching_curve.csv
```

First-trial step traces and final patterns are written under:

```text
results/experiment_013/brain_01/
```

## PowerShell view

```powershell
Import-Csv .\results\experiment_013\common_source_branching_curve.csv |
Format-Table checkpoint,mode,cue,A_peak_activity,F_peak_activity,A_peak_selectivity,A_cumulative_selectivity,A_final_selectivity,trial_similarity,context_selectivity,context_pattern_separation -AutoSize
```
