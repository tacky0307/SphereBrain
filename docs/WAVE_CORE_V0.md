# SphereBrain Wave Core v0

## Purpose

Wave Core v0 does not attempt to reproduce a memorized answer such as
`A → B → C`.

It asks one smaller question:

> After repeated `A → B` experience, does stimulation A produce a measurably
> different internal wave, especially around region B?

The result is observed as a tendency rather than graded as correct or wrong.
Unexpected motion is retained as research data.

## Design principles

1. **Input only stimulates.** It does not prescribe a route.
2. **Nodes are observation points.** They do not store words or meanings.
3. **Conductivity is terrain.** It describes how easily distributed activity
   crosses a local connection.
4. **All nodes update synchronously.** Activity behaves as a field-like wave,
   not a particle walking through a list of nodes.
5. **Localized firing remains.** Threshold events preserve the particle-like
   aspect, but they only add a small local pulse and never select the answer.
6. **Experience changes actual motion.** Plasticity uses previous source
   activity and current target activity; no expected successor is passed to
   the learning rule.
7. **The observer does not correct the core.** It records traces, terrain
   change, repeatability, and deviations from the initial expectation.

## What remains deliberately absent

- Replay commands
- Continuity Bridge
- FlowBias
- Top-k Competition
- A decoder that declares the answer
- Correct-path scoring

These mechanisms have not been deleted from the original SphereBrain work.
Wave Core v0 lives in a separate package and branch so both approaches can be
compared later.

## First experiment

The script performs the following sequence:

1. Stimulate distributed region A and record the baseline wave.
2. Repeat A, wait a few time steps, then stimulate B while A's activity remains.
3. Permit temporal overlap to change local conductivity.
4. Reset only short-term activity, preserving the changed terrain.
5. Stimulate A again and record the trained wave.
6. Compare B-region activity, center movement, and conductivity change.

Run from the repository root:

```bash
python experiments/run_wave_core_v0.py
```

Optional parameters:

```bash
python experiments/run_wave_core_v0.py --repetitions 40 --delay-steps 3
```

Output is written to `data/wave_core_v0/`:

- `baseline_a.csv`
- `trained_a.csv`
- `experience_summary.csv`
- `terrain_changes.csv`
- `result.json`

The data directory remains local research output and should not become the
brain itself. The lasting internal state in this experiment is the changed
conductivity matrix.

## How to interpret the first result

A useful initial observation may be very small:

- peak activity near B increases after experience;
- the activity center passes slightly closer to B;
- the wave lasts longer or shorter;
- another region becomes consistently favored instead of B.

The last case is not automatically a failure. It should be repeated and traced.
A stable unexpected tendency may reveal a property formed by the whole terrain.

## Next research steps

Only after the baseline experiment is stable:

1. Repeat runs with several seeds and measure reproducibility.
2. Add a simple trace visualizer for wave fronts and terrain differences.
3. Test competing experiences such as `A → B` and `A → D`.
4. Introduce phase or directional flow gradually.
5. Allow strong localized events to become new wave sources.

Each addition should be isolated so its effect remains observable.
