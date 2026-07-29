# v22 — Experience Bundles

## Question

Can SphereBrain begin to form concept-like structure when one learning event contains a bundle of co-occurring experience elements rather than a sentence-order pair?

## Shift in representation

Previous experiments mainly learned ordered relations:

```text
空 -> 青い
```

v22 treats one event as an unordered bundle:

```text
{空, 青い, 広がっている, 澄んでいる, 昼, 太陽, 風, 鳥, 見上げる}
```

Every element may evoke every other element in that same experience. The order written in the source file has no learning meaning.

## Why qualities and actions are included

A concept should not be built only from a noun and one property. Experiences therefore include mixed fragments such as:

- objects: 空, 海, 雲, 鳥
- qualities: 青い, 広い, 澄んでいる, 暗い
- actions: 見上げる, 歩く, 流れる
- context: 昼, 夜, 晴れ, 夕方
- sensations: 暖かい, 冷たい, 静か

These categories are explanations for human readers only. SphereBrain receives no part-of-speech or category labels.

## Experience set

The experiment includes clear sky, cloudy sky, night sky, sunset, blue sea, open field, and clear water experiences.

This intentionally creates both:

- repeated shared structure, such as 空 with 広がっている
- cross-object structure, such as 青い with 空 and 海
- contextual differences, such as daytime blue sky, grey cloudy sky, red sunset, and dark night sky

## Comparison

Two brains use identical topology and encoders:

1. `chain`: learns only adjacent elements in the written order
2. `bundle`: learns all directed pairs inside each experience

The comparison reports:

- number of learning events
- shared internal-context similarity
- final active-node count
- observer decoding for selected probes

## Important limitation

This is not sensory grounding yet. All elements still originate as externally encoded text tokens. The experiment only asks whether unordered co-occurrence produces a richer internal relationship structure than sentence-like order.

A future stage should let image, sound, spatial, temporal, and bodily-state encoders contribute numeric patterns to the same experience bundle.

## Run

```powershell
python experiments/run_experience_bundles_v22.py
```
