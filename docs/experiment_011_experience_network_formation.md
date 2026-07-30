# Experiment 011: Experience Network Formation

## Purpose

Experiment010 showed that two independent experiences could produce different final states, but the states did not reliably correspond to the intended experience labels.

Experiment011 asks a more SphereBrain-like question:

> Can overlapping experiences form a shared internal structure rather than two isolated paths?

The shared experience network is:

```text
A -> P -> C
F -> P -> M
```

`P` is deliberately repeated as a common intermediate experience.

## Comparison condition

A disconnected network is trained with the same number of stages and repetitions:

```text
A -> Q -> C
F -> R -> M
```

`Q` and `R` are separate intermediates. This condition helps distinguish a real shared-hub effect from ordinary learning or global activation.

An untrained clone is also measured.

## Recall cues

Recall supplies only one local cue:

- A
- F
- P

No destination is supplied during recall.

## Main questions

1. Does cue A or F recruit the shared P region more strongly than the disconnected comparison?
2. Does cue A favor the C branch and cue F favor the M branch?
3. Does cue P produce a blended or common state?
4. Does the shared experience network alter A-vs-F separation compared with disconnected learning?
5. Which temporal part of the experience resembles the recalled state?

## Main metrics

### `region_P_activity`

Mean final activity in the shared P region.

### `shared_hub_gain_vs_disconnected`

```text
P activity in shared network
-
P activity in disconnected comparison
```

A positive value suggests that repeated passage through P created a shared internal hub.

### `branch_selectivity`

For cue A:

```text
C-region activity - M-region activity
```

For cue F:

```text
M-region activity - C-region activity
```

Positive values mean the cue favors its expected branch.

### `experience_selectivity`

Similarity to the cue's own settled experience pattern minus similarity to the other experience's settled pattern.

This is retained for continuity with Experiment010, but it is no longer the only criterion because a whole-experience average may not be the correct identity of an attractor.

### `A_vs_F_pattern_similarity`

Cosine similarity between the final whole-sphere states produced by cue A and cue F.

### `shared_separation_change_vs_disconnected`

```text
A-vs-F similarity in disconnected network
-
A-vs-F similarity in shared network
```

Positive means the shared network separated the two branch states more strongly. Negative means the common P experience made them more alike.

Neither sign is automatically success or failure: a shared concept may reasonably make two memories more similar while preserving branch selectivity.

## Initial success signals

The strongest first evidence of an experience network would be:

- positive P hub gain for A and F cues;
- positive branch selectivity for both A and F;
- a characteristic P-cued state that is not merely identical to A or F;
- stable changes across checkpoints rather than a one-checkpoint accident.

## Run

```powershell
python experiments/run_experience_network_formation.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

## Inspect

```powershell
Import-Csv .\results\experiment_011\experience_network_curve.csv |
Format-Table checkpoint,mode,cue,region_P_activity,shared_hub_gain_vs_disconnected,region_C_activity,region_M_activity,branch_selectivity,experience_selectivity,A_vs_F_pattern_similarity,shared_separation_change_vs_disconnected -AutoSize
```

## Interpretation caution

The letters are external labels only. SphereBrain is not told that P is a concept, C is a correct answer, or M is an alternative answer. The experiment tests whether repeated overlapping experience changes the sphere's natural dynamics in a way consistent with a shared structure.
