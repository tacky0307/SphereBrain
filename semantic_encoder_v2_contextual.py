from __future__ import annotations

from typing import Mapping

from brain import SignalResult
from contextual_core import ContextualSphereBrain
from semantic_encoder_v2 import (
    BRAIN_FILE,
    StructuredExperience,
    StructuredInput,
    component_nodes,
    load_brain,
    save_experience,
)


def result_to_context(
    result: SignalResult,
    *,
    history_decay: float = 0.88,
    output_scale: float = 1.0,
) -> dict[int, float]:
    """Convert a propagation trace into a weighted, time-aware context.

    Recent activity remains stronger, but entry and route nodes are retained.
    Final activation values are also preserved, so the context carries both
    path history and the current state instead of only a flattened node tail.
    """
    history_decay = max(0.0, min(1.0, float(history_decay)))
    output_scale = max(0.0, float(output_scale))
    context: dict[int, float] = {}
    history = result.activation_history
    last_index = max(0, len(history) - 1)

    for index, nodes in enumerate(history):
        age = last_index - index
        strength = (history_decay ** age) * output_scale
        for node in nodes:
            node = int(node)
            context[node] = max(context.get(node, 0.0), strength)

    for node, value in enumerate(result.final_activation):
        value = float(value) * output_scale
        if value > 0:
            context[int(node)] = max(context.get(int(node), 0.0), value)

    return context


def merge_contexts(
    *contexts: tuple[Mapping[int, float], float],
) -> dict[int, float]:
    """Merge stage contexts while preserving their relative contribution."""
    merged: dict[int, float] = {}
    for context, scale in contexts:
        scale = max(0.0, float(scale))
        for node, value in context.items():
            weighted = max(0.0, float(value)) * scale
            if weighted > 0:
                merged[int(node)] = max(merged.get(int(node), 0.0), weighted)
    return merged


def encode_and_experience_contextual(
    brain: ContextualSphereBrain,
    item: StructuredInput,
    *,
    learn: bool = True,
    context_anchor: float = 0.58,
    context_decay: float = 0.94,
    resonance: bool = True,
) -> StructuredExperience:
    """Encode subject, relation and content without erasing earlier context."""
    noise = 0.004 if learn else 0.0

    subject_sources = (
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", item.subject, 3)
    )
    subject_result = brain.propagate(
        subject_sources,
        steps=8,
        threshold=0.18,
        noise=noise,
        learn=learn,
    )
    subject_context = result_to_context(subject_result)

    relation_sources = (
        component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", item.relation, 3)
    )
    relation_result = brain.propagate_contextual(
        relation_sources,
        subject_context,
        steps=8,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_anchor=context_anchor,
        context_decay=context_decay,
        resonance=resonance,
    )
    relation_context = result_to_context(relation_result)
    semantic_context = merge_contexts(
        (subject_context, 0.72),
        (relation_context, 1.0),
    )

    content_sources = (
        component_nodes(brain, "role:content", "content", 2)
        + component_nodes(brain, "content", item.content, 3)
    )
    content_result = brain.propagate_contextual(
        content_sources,
        semantic_context,
        steps=10,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_anchor=context_anchor,
        context_decay=context_decay,
        resonance=resonance,
    )

    return StructuredExperience(item, subject_result, relation_result, content_result)


def load_contextual_brain() -> ContextualSphereBrain:
    """Load the existing semantic-v2 Core without changing its file format."""
    return ContextualSphereBrain.from_brain(load_brain())


def train_contextual(
    subject: str,
    relation: str,
    content: str,
    repeats: int = 1,
    *,
    context_anchor: float = 0.58,
    context_decay: float = 0.94,
    resonance: bool = True,
) -> StructuredExperience:
    subject = subject.strip()
    relation = relation.strip()
    content = content.strip()
    if not subject or not relation or not content:
        raise ValueError("主体・関係・内容をすべて入力してください。")

    brain = load_contextual_brain()
    latest: StructuredExperience | None = None
    for _ in range(max(1, int(repeats))):
        latest = encode_and_experience_contextual(
            brain,
            StructuredInput(subject, relation, content),
            learn=True,
            context_anchor=context_anchor,
            context_decay=context_decay,
            resonance=resonance,
        )
        save_experience(latest)

    # ContextualSphereBrain uses the same JSON schema as SphereBrain.
    brain.save(BRAIN_FILE)
    assert latest is not None
    return latest


def observe_contextual(
    subject: str,
    relation: str,
    content: str,
    *,
    context_anchor: float = 0.58,
    context_decay: float = 0.94,
    resonance: bool = True,
) -> StructuredExperience:
    """Run the full contextual pipeline without learning or saving."""
    brain = load_contextual_brain()
    return encode_and_experience_contextual(
        brain,
        StructuredInput(subject.strip(), relation.strip(), content.strip()),
        learn=False,
        context_anchor=context_anchor,
        context_decay=context_decay,
        resonance=resonance,
    )
