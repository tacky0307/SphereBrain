# v29 / Experiment 008: Recall Path Diagnostics

## Question

Experience 007 showed that repeated A-then-C experience lowers the easiest A-to-C path cost, but A-only stimulation still produces no measurable C activity.

Experiment 008 asks:

> How does the A-only wave use the experienced terrain, and where does it stop?

## Principle

This experiment adds no prediction mechanism and does not force C to activate.

The diagnostic layer is read-only. It does not modify:

- activity;
- fatigue;
- conductivity;
- plasticity;
- the Wave Core update rule.

## Measurements

At each checkpoint, A alone is stimulated and the resulting trace is measured.

- `path_integral`: total activity observed on the current easiest A-to-C path
- `path_peak`: strongest single-step activity on that path
- `closest_active_distance_to_target`: closest any tiny activity comes to C
- `closest_meaningful_distance_to_target`: closest activity above the meaningful threshold comes to C
- `furthest_meaningful_path_index`: furthest path node reached above the meaningful threshold
- `path_progress`: fraction of the easiest path reached meaningfully
- `target_integral` / `target_peak`: direct C-region response
- `reached_target`: whether any activity reaches C
- `meaningfully_reached_target`: whether meaningful activity reaches C

## Run

```powershell
python experiments/run_recall_diagnostics.py --brains 1 --repetitions 500 --checkpoints 100,200,300,400,500
```

Inspect the checkpoint curve:

```powershell
Import-Csv .\results\experiment_008\diagnostic_curve.csv |
Format-Table checkpoint,mode,path_progress,closest_meaningful_distance_to_target,path_integral,target_integral,meaningfully_reached_target -AutoSize
```

Inspect the final wave step by step:

```powershell
Import-Csv .\results\experiment_008\brain_01\final_multiscale_steps.csv |
Format-Table step,total_activity,path_activity,closest_meaningful_distance_to_target,furthest_meaningful_path_index -AutoSize
```

## Interpretation

Possible outcomes:

1. `path_progress` increases and distance to C decreases: the wave is beginning to use the learned terrain.
2. Path activity rises but progress does not: activity enters the corridor but repeatedly dies near A.
3. Path cost improves while path activity is unchanged: the shortest-path metric and propagation dynamics are not aligned.
4. The wave reaches C but remains below the meaningful threshold: recall has begun weakly before C can win.
5. Control and multiscale remain similar: the observed terrain growth is not yet functionally used during recall.

The purpose is diagnosis, not parameter tuning to produce a desired answer.
