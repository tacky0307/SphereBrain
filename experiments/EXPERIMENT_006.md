# Experiment 006: Experience Growth

## Purpose

Experiment 005 showed that repeated A-then-C stimulation did not make C the strongest response. Experiment 006 does not try to force C to win. It asks a smaller and more important question:

> Does a complete repeated experience gradually change the C response more than step-level learning alone?

## Architecture

The existing `SphereWaveCore` remains unchanged.

```text
SphereWaveCore
  activity / propagation / persistence / fatigue / firing
  step-level temporal plasticity

ExperienceBuffer
  records all snapshots and stimulus events in one experience

ExperienceSummary
  compresses the full activity history
  cumulative activity / peaks / first and last activity / temporal credit

ExperienceReflector
  reflects the summary onto existing local conductivity edges
```

The experience layer is optional. Existing experiments that use only `SphereWaveCore` continue to work unchanged.

## Two learning time scales

### Local learning

The existing Core updates conductivity from adjacent step activity. It represents immediate local change.

### Experience reflection

After A stimulation, wave motion, C stimulation and settling are complete, the whole trace is summarized and weakly reflected onto the terrain.

The reflector:

- receives no correct route;
- creates no direct A-to-C edge;
- changes only existing local edges;
- preserves temporal order through a decaying eligibility trace;
- softly diffuses the experienced activity field so the experience can leave a broad terrain trace.

## Control comparison

Each brain creates two identical cores:

```text
control   = local learning only
reflected = local learning + experience reflection
```

Both receive:

- the same initial terrain;
- the same A and C regions;
- the same delay variation;
- the same stimulus-strength variation;
- the same number of repetitions.

This comparison separates ordinary step-level learning from the added experience-scale effect.

## Checkpoints

Default observations are made after:

```text
0, 10, 25, 50, 100, 200 experiences
```

The principal measurements are:

- C activity integral;
- C peak activity;
- first C response step;
- C selectivity relative to B and D;
- C change from the brain's own baseline;
- total terrain change;
- changed edge count;
- difference between reflected and control brains.

`winner` is recorded, but it is not the primary success criterion.

## Initial success criterion

Experiment 006 is promising when repeated experience produces a stable tendency such as:

```text
reflected C change > control C change
```

across checkpoints or across multiple brains.

C may remain below B or D. A small repeatable shift is more important than an immediate correct answer.

## Run

Smoke test:

```bash
python experiments/run_experience_growth.py --brains 1 --repetitions 10 --checkpoints 1,5,10
```

Default observation:

```bash
python experiments/run_experience_growth.py
```

Outputs:

```text
results/experiment_006/
  summary.json
  growth_curve.csv
  brain_01/
    growth_curve.csv
    training.csv
    result.json
```

## Interpretation discipline

Do not tune parameters merely to make C win.

First inspect:

1. whether C changes at all;
2. whether the reflected brain differs from its control;
3. whether the effect grows with experience count;
4. whether the tendency repeats across independently perturbed brains;
5. whether broader terrain change appears without one prescribed route.
