# Experiment 005 — Whole Brain Formation

## Question

Can different SphereBrain instances develop different complex conductivity terrains while still producing the same functional result?

The experiment does **not** prescribe or score an internal route. The route is free. The observable requirement is only:

```text
input A -> output C
```

## Idea

A learned brain is treated as the whole conductivity terrain, not as one selected path.

- Thick and thin connections may coexist.
- Different brains may form different terrains.
- Internal activity may travel differently each time.
- A brain is functionally successful when A stimulation makes C the strongest output region.

## Procedure

Each brain receives four fixed regions on the sphere: A, B, C and D.

1. Create a fresh Wave Core.
2. Add a very small terrain perturbation so brains can develop differently.
3. Save a baseline A-only probe.
4. Repeat the A-then-C experience 100 times by default.
5. Probe with A only; C is not externally stimulated during the probe.
6. Compare B, C and D by activity integral and peak.
7. Cut a small fraction of the strongest directed edges.
8. Probe again to test whether the whole terrain retains function.
9. Repeat across 10 brains by default.

## Success criteria

Primary:

- C wins after training more often than before training.

Whole-brain evidence:

- Multiple brains have measurably different terrain statistics.
- Different terrains can still return C.
- At least some brains retain A -> C after strong-edge lesions.

A failed result is still useful. It may show that the present plasticity rule changes terrain but does not yet create autonomous recall.

## Run

```bash
python experiments/run_whole_brain_formation.py
```

Smaller smoke run:

```bash
python experiments/run_whole_brain_formation.py --brains 2 --repetitions 10
```

Custom output:

```bash
python experiments/run_whole_brain_formation.py --output results/experiment_005_trial_01
```

## Outputs

```text
results/experiment_005/
  summary.json
  summary.csv
  brain_01/
    baseline_probe.csv
    trained_probe.csv
    lesioned_probe.csv
    training.csv
    terrain.csv
    result.json
  ...
```

`summary.csv` is the first file to inspect. It shows each brain's baseline winner, trained winner, lesion winner, output-region activity and terrain statistics.

## Interpretation rule

Do not judge whether an individual internal route was correct. Route traces are observation records only.

The object being evaluated is the learned brain as a whole:

```text
experience -> terrain formation -> functional output
```
