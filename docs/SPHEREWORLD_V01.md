# SphereWorld v0.1

SphereWorld is a tiny game world that uses the repository's real `SphereBrain` Core.

## First principle

The world must not contain the organism's intelligence.

SphereWorld may define physics and consequences:

- walking costs energy;
- food restores energy;
- danger and walls cost extra energy.

It must **not** contain rules such as:

- "if food is north, go north";
- "avoid danger";
- "use the shortest path to food".

The action is chosen from the structure that has formed inside `brain.py`.

## v0.1 loop

```text
7x7 World
    ↓
numeric sensory anchors
    ↓
SphereBrain Core (`brain.py`)
    ↓
candidate structural probes
    ↓
N / E / S / W / STAY
    ↓
world consequence
    ↓
experience: sensors + action + outcome
    ↓
Core pathways change
```

The 27 sensory/action/outcome channels are assigned to stable Core node anchors. The labels are only encoder-side identities; their linguistic meaning is never passed into Core.

`good`, `neutral`, and `bad` are fixed numeric outcome anchors. Positive and negative events are experienced twice because they are treated as salient events.

For an automatic decision, every candidate action is probed with `learn=False`. The candidate's active Core structure is compared with the current structures reached from the `good` and `bad` outcome anchors. The score is:

```text
2 * overlap(good) - 2 * overlap(bad) + small familiarity term
```

No state-action lookup table is stored outside Core.

## Files

- `sphereworld/world.py` — world physics and sensory state
- `sphereworld/core_agent.py` — thin numeric Core adapter
- `sphereworld/game.py` — teach/auto game loop
- `sphereworld/__main__.py` — module entry point
- `data/sphereworld_v01/brain.json` — generated Core state (ignored by git)

## Play

Teach the organism by moving it yourself:

```bash
python -m sphereworld --mode teach --reset-core
```

Keys:

- `W` north
- `A` west
- `S` south
- `D` east
- `X` stay
- `Q` quit

Then let the same Core act:

```bash
python -m sphereworld --mode auto
```

Try a different world while keeping the same Core:

```bash
python -m sphereworld --mode auto --seed 99
```

The important observation is not merely the score. Watch whether action scores, used edges, and survival behavior change as experience accumulates.

## What v0.1 is testing

v0.1 is deliberately small. It asks one question:

> Can a Core whose pathways were changed by experience make later action choices from that changed structure, without a game-side rule that tells it where food is?

It is not yet proof of intelligence or reinforcement learning. It is a playable structural-learning experiment.
