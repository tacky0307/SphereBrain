# Experiment 012: Common Hub Convergence

## Purpose

Experiment011 combined two questions:

1. Can different experiences share a common internal state?
2. Can activity later branch from that shared state according to context?

That made failure difficult to interpret. Experiment012 removes the branch and tests only convergence.

## Experience conditions

### Shared destination

```text
A -> P
F -> P
```

A and F are different source cues. P is the same destination region for both experiences.

### Disconnected comparison

```text
A -> Q
F -> R
```

This comparison receives the same number of experiences and the same timing, but has no shared destination.

### Untrained comparison

An unchanged clone is recalled at every checkpoint to expose baseline geometry and spontaneous dynamics.

## Why P is broader

A and F use radius 2 stimulus regions. P, Q, and R use radius 3 regions.

The shared destination is treated as a distributed concept-like region rather than a single spatial point. Q and R use the same radius so the comparison remains balanced.

## Training

Each repetition contains both experiences. Their order alternates to avoid a systematic recency advantage:

```text
repetition 1: A -> P, then F -> P
repetition 2: F -> P, then A -> P
```

The disconnected sphere receives A -> Q and F -> R with the same alternating order.

No destination is supplied during recall.

## Recall cues

At each checkpoint, each sphere is recalled with:

- A only
- F only
- P only

The P cue provides a reference for the spontaneous state produced directly from the shared destination.

## Main measurements

### P trajectory measurements

Unlike Experiment011, Experiment012 does not inspect only the final state.

For every recall trial it records:

- `P_peak_activity`: maximum P-region activity during recall
- `P_cumulative_activity`: sum of P-region activity over all recall steps
- `P_mean_activity`: average P-region activity
- `P_peak_step`: step at which P activity is maximal
- `P_final_activity`: P-region activity in the final snapshot

This distinguishes:

- passing through P and later leaving it
- settling at P
- never activating P

### Convergence measurements

- `A_vs_F_pattern_similarity`: similarity between final recall patterns from A and F
- `convergence_gain_vs_disconnected`: shared A-vs-F similarity minus disconnected A-vs-F similarity
- `recall_to_P_cue_similarity`: similarity between A or F recall and direct P-cue recall
- `destination_experience_similarity`: similarity to the destination phase of the corresponding training experience
- `settled_experience_similarity`: similarity to the settled phase after the corresponding experience

### P gains over disconnected comparison

For the same cue:

- `P_peak_gain_vs_disconnected`
- `P_cumulative_gain_vs_disconnected`
- `P_final_gain_vs_disconnected`

These help separate learned shared convergence from accidental geometric activation of P.

## Initial success interpretation

The strongest evidence for a learned common destination would be:

1. A and F both show positive P peak or cumulative gain over the disconnected comparison.
2. A and F final patterns become more similar than in the disconnected comparison.
3. A and F recalls become similar to direct P-cue recall or to the learned P destination phase.
4. The effect persists across later checkpoints rather than appearing only briefly.

Final P activity does not have to remain high if P functions as a transient convergence state. Peak, cumulative, and timing measurements must therefore be interpreted together.

## Run

```powershell
python experiments/run_common_hub_convergence.py --brains 1 --repetitions 500 --trials 5 --checkpoints 0,100,200,300,400,500
```

## Inspect summary

```powershell
Import-Csv .\results\experiment_012\common_hub_convergence_curve.csv |
Format-Table checkpoint,mode,cue,P_peak_activity,P_cumulative_activity,P_peak_step,P_final_activity,P_peak_gain_vs_disconnected,P_cumulative_gain_vs_disconnected,recall_to_P_cue_similarity,A_vs_F_pattern_similarity,convergence_gain_vs_disconnected -AutoSize
```

## Detailed traces

The first trial at every checkpoint is written with region activity for A, F, P, Q, and R:

```text
results/experiment_012/brain_01/checkpoint_XXXX_<mode>_<cue>_trial_01_steps.csv
```

These files allow direct inspection of whether A and F activity reaches P during recall and when that happens.
