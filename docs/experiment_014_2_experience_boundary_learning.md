# Experiment 014.2: Experience Boundary Learning

## Purpose

Experiment 014 showed that a transient Experience Field can bind activity separated in time. Experiment 014.1 showed that reducing `trace_gain` alone does not prevent the learned A/F-to-P relation from collapsing after extended repetition.

Experiment 014.2 tests a different hypothesis:

> The collapse is caused by learning continuing during the post-stimulus settling period, after the meaningful experience has already ended.

The experiment therefore defines an explicit boundary around the experience and disables learning during settling.

## Training experience

Two experiences are alternated:

- A -> P
- F -> P

Each experience is trained as:

1. `begin_experience()` for the Experience Field condition.
2. Stimulate the source region.
3. Run 5 learning steps.
4. Stimulate P.
5. Run 7 learning steps.
6. `end_experience()` immediately after the destination phase.
7. Run 8 settling steps with `learn=False`.

The local-plasticity control uses the same 5 + 7 learning steps and the same 8 non-learning settling steps. This keeps the temporal learning boundary identical across conditions.

## What changed from Experiment 014

Experiment 014 used 8 additional `learn=True` steps after the destination stimulus. Experiment 014.2 changes those steps to `learn=False` and closes the Experience Field before settling.

No change is made to the Experience Field equations or default gain.

## Conditions

- Untrained control
- Local plasticity
- Experience Field with explicit experience boundary

Default Experience Field parameters:

- `trace_decay = 0.92`
- `trace_gain = 1.0`
- `directional_learning_rate = 0.0010`
- `capacity_learning_rate = 0.0006`

## Checkpoints

Default checkpoints:

- 0
- 100
- 200
- 300
- 400
- 500

## Main measurements

For A, F, and P recall cues:

- `P_peak_activity`
- `P_cumulative_activity`
- `P_final_activity`
- `P_peak_gain_vs_local`
- `P_cumulative_gain_vs_local`
- `experience_similarity`
- `trial_similarity`

The field condition also records the Experience Field at the exact end of each experience:

- `experience_strength_sum`
- `experience_strength_mean`

A per-repetition log is written so that field strength can be inspected independently of recall performance.

## Success criterion

The intended result is:

1. A and F recruit P after early training.
2. Their P recruitment remains measurable at later checkpoints.
3. The relation does not collapse to zero by 200-500 repetitions.

The central criterion is persistence of temporal association, not maximum P self-activity.

## Run

```powershell
python experiments/run_experience_boundary_learning.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

## Output

```text
results/experiment_014_2_boundary_learning/
```

Main files:

- `experience_boundary_learning_curve.csv`
- `experience_strength_by_repetition.csv`
- Per-checkpoint recall step traces under `brain_XX/`

## Interpretation

If A/F-to-P recruitment persists, the result supports the hypothesis that SphereBrain needs a meaningful learning boundary: activity may continue to settle after an experience, but that settling should not automatically continue modifying the pathway.

If the relation still collapses, the remaining cause lies elsewhere, such as homeostasis, competition, pathway saturation, or asymmetric reinforcement. In that case Experiment 014.2 will still have isolated experience-boundary timing from Experience Field strength.
