from experiments.run_temporal_association_v28 import (
    TemporalAssociationCore,
    train_sequence,
)


def test_temporally_separated_activity_forms_directed_path() -> None:
    core = TemporalAssociationCore()
    a, b = 2, 11

    train_sequence(core, a, b, gap_steps=2, repetitions=20)

    assert core.association_strength(a, b) > 0.0
    assert core.association_strength(a, b) > core.association_strength(b, a)


def test_shorter_gap_forms_stronger_association() -> None:
    core = TemporalAssociationCore()
    a, short_target, long_target = 2, 11, 17

    train_sequence(core, a, short_target, gap_steps=1, repetitions=20)
    train_sequence(core, a, long_target, gap_steps=5, repetitions=20)

    assert core.association_strength(a, short_target) > core.association_strength(
        a, long_target
    )


def test_learned_path_changes_recall_without_further_learning() -> None:
    core = TemporalAssociationCore()
    a, b = 2, 11

    before = core.recall([a], steps=2)[b]
    train_sequence(core, a, b, gap_steps=2, repetitions=20)
    after = core.recall([a], steps=2)[b]

    assert before == 0.0
    assert after > before
