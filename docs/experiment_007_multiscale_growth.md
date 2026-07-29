# Experiment 007: Multi-scale Growth

## Hypothesis

The growth rule used for a small before/after transition can also operate on larger activity units.

A complete experience is divided into temporal windows:

```text
window 1 -> window 2 -> window 3 -> ...
```

Each window is treated as one macro activity point. The transition between neighboring windows is diffused through the existing local mesh and applied back to existing local conductivity.

The experiment does not create a direct A-to-C edge and does not prescribe intermediate nodes.

## Comparison

Two brains begin with identical terrain and receive identical A-then-C experiences:

- `control`: existing step-level local plasticity only
- `multiscale`: local plasticity plus multi-scale reflection after each experience

## Primary observation

C does not need to activate or win at first. The earliest signal may be a gradual improvement in the best existing A-to-C route.

- `ac_path_cost`: lower means the route became easier to traverse
- `ac_path_mean`: higher means the route's existing local edges became more conductive
- `c_integral`: later evidence that A-only activity actually reaches C

## First run

```powershell
python experiments/run_multiscale_growth.py --brains 1 --repetitions 10 --checkpoints 1,5,10
```

```powershell
Import-Csv .\results\experiment_007\growth_curve.csv |
Format-Table brain,checkpoint,mode,winner,c_integral,ac_path_cost_change,ac_path_mean_change,terrain_total_change -AutoSize
```

## Interpretation

A useful first result is:

```text
multiscale ac_path_cost_change < control ac_path_cost_change
```

Even while both modes still report `c_integral = 0`, that difference would indicate that a distributed bridge is beginning to form before activity can cross the whole sphere.
