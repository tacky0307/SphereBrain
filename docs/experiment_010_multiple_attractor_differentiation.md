# Experiment 010 — Multiple Attractor Differentiation

## Question

Can one SphereBrain form more than one experience-shaped spontaneous state?

Experiment009 showed that experience changes the sphere's natural dynamics, but a single stable regime is not yet evidence of differentiated memory. Experiment010 therefore places two experiences in the same Core:

- Experience AC: stimulate A, then C
- Experience FM: stimulate F, then M

The two experiences alternate during training so neither has a systematic recency advantage.

## Principle

No target node is supplied during recall. After training:

- A alone is stimulated and the sphere evolves freely.
- F alone is stimulated and the sphere evolves freely.

The experiment observes whether the two partial cues settle into distinguishable whole-sphere activity patterns.

## Main measures

### A_vs_F_pattern_similarity

Cosine similarity between the final patterns reached from A and F.

- Near 1: both cues settle into nearly the same state.
- Lower than control: training has differentiated the two settling states.

### separation_gain_over_control

`control A-vs-F similarity - trained A-vs-F similarity`

A positive value means training separated the two cue-dependent states beyond the separation caused by geometry alone.

### own_experience_similarity

Similarity between the recall pattern and the whole activity pattern observed during the matching experience.

### other_experience_similarity

Similarity between the recall pattern and the other experience.

### cue_selectivity

`own_experience_similarity - other_experience_similarity`

Positive values for both A and F are the strongest early sign of differentiated experience-dependent attractors.

Additional diagnostics:

- mean_lifetime
- mean_active_fraction
- mean_concentration
- trial_similarity
- mean_total_activity
- mean_direction
- mean_capacity

## First success criterion

Experiment010 does not require a specific destination to win. A promising result is:

1. A and F recall patterns are less similar after training than in control.
2. `separation_gain_over_control` is positive.
3. Cue A has positive selectivity toward AC.
4. Cue F has positive selectivity toward FM.
5. Activity remains bounded and does not collapse to zero or saturate the whole sphere.

## Run

```powershell
python experiments/run_multiple_attractor_differentiation.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

## Summary table

```powershell
Import-Csv .\results\experiment_010\multiple_attractor_curve.csv |
Format-Table checkpoint,mode,cue,mean_active_fraction,mean_concentration,trial_similarity,own_experience_similarity,other_experience_similarity,cue_selectivity,A_vs_F_pattern_similarity,separation_gain_over_control -AutoSize
```

## Final trace examples

```powershell
Import-Csv .\results\experiment_010\brain_01\checkpoint_0500_trained_A_trial_01_steps.csv |
Format-Table step,total_activity,active_count,active_fraction,pattern_concentration -AutoSize
```

```powershell
Import-Csv .\results\experiment_010\brain_01\checkpoint_0500_trained_F_trial_01_steps.csv |
Format-Table step,total_activity,active_count,active_fraction,pattern_concentration -AutoSize
```

## Interpretation guardrail

A lower A-vs-F similarity alone is not enough. The untrained control may already produce different states because A and F occupy different regions of the sphere. The primary comparison is therefore the separation gain over control, together with positive cue selectivity for both experiences.
