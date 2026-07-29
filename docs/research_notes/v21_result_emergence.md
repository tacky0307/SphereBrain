# v21 — Result Emergence

## Question

Can SphereBrain stop because a result has formed, rather than because a fixed time or activity-change threshold says to stop?

## Principle

v21 treats completion as an internal phenomenon:

```text
propagation
→ output activity appears
→ activity concentrates
→ the same output pattern persists
→ result formed
→ stop
```

The detector reads only numeric activity on the output surface:

- similarity between consecutive output patterns
- share of total energy present on the output surface
- share of output energy concentrated in the five strongest output nodes
- persistence across consecutive steps

It never reads:

- words or labels
- expected successors
- decoder rankings
- probe identity

The decoder is used only after propagation has already ended, for human observation.

## Why three signals?

Stability alone is ambiguous. An empty or globally collapsed state can also be stable.

A result therefore requires all three:

1. **Presence** — enough energy has reached the output surface.
2. **Concentration** — output activity has formed a localized candidate pattern.
3. **Persistence** — that pattern remains similar across consecutive steps.

## Conditions

- `emerge_loose`: similarity >= 0.970, top-5 share >= 0.22, output share >= 0.08, stable for 2 steps
- `emerge_mid`: similarity >= 0.985, top-5 share >= 0.28, output share >= 0.10, stable for 2 steps
- `emerge_strict`: similarity >= 0.995, top-5 share >= 0.34, output share >= 0.12, stable for 3 steps

The v19c best propagation schedule remains fixed:

```text
step 1:      0.80
steps 2-5:   0.90
steps 6+:    0.95
```

This isolates the research question: only the stopping mechanism changes.

## Control

`fixed24` always reads the state at step 24.

## Safety ceiling

The self-ending conditions may continue up to 48 steps. Step 48 is not treated as a result; it is only a safety ceiling. A run that has not formed a result is reported as `not formed`.

## Evaluation

- result formation rate
- mean, minimum, and maximum formation step
- learned top-1 recall after stopping
- successor margin
- identity preservation
- per-probe internal formation measurements

## Run

```powershell
python experiments/run_result_emergence_v21.py
```
