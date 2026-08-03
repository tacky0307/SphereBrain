from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sphere_world_multi_context import MultiContextSphereWorldBrain, TRAIN_STATES, facts_for_positions
from sphere_world import POSITIONS


@dataclass(frozen=True)
class SphereVisualDataset:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    experiences: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "experiences": self.experiences,
            "summary": self.summary,
        }


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def build_sphere_visual_dataset(repeats: int = 12) -> SphereVisualDataset:
    """Build a visual dataset from the actual three SphereWorld training experiences.

    The displayed routes are the union of real activated nodes and traversed edges
    produced by the contextual Core when each trained world state is observed.
    """
    world_brain = MultiContextSphereWorldBrain(repeats=repeats)
    brain = world_brain.brain

    node_tags: dict[int, set[str]] = {}
    edge_tags: dict[tuple[int, int], set[str]] = {}
    experiences: list[dict[str, Any]] = []

    for index, ((player_position, enemy_position), action) in enumerate(TRAIN_STATES.items(), start=1):
        probe = world_brain.probe(player_position, enemy_position)
        raw = probe["raw_result"]
        route_nodes = sorted(set(int(node) for node in raw.activated_nodes))
        route_edges = sorted({_edge_key(a, b) for a, b in raw.traversed_edges})
        tag = f"experience-{index}"

        for node in route_nodes:
            node_tags.setdefault(node, set()).add(tag)
        for edge in route_edges:
            edge_tags.setdefault(edge, set()).add(tag)

        experiences.append({
            "id": tag,
            "number": index,
            "player": POSITIONS[player_position],
            "enemy": POSITIONS[enemy_position],
            "action": action,
            "facts": [fact.label for fact in facts_for_positions(player_position, enemy_position)],
            "node_count": len(route_nodes),
            "edge_count": len(route_edges),
        })

    used_nodes = sorted(node_tags)
    nodes = []
    for node in used_nodes:
        x, y, z = (float(value) for value in brain.positions[node])
        nodes.append({
            "id": node,
            "x": x,
            "y": y,
            "z": z,
            "usage": int(brain.node_usage[node]),
            "experiences": sorted(node_tags[node]),
            "shared": len(node_tags[node]),
        })

    edges = []
    for (a, b), tags in sorted(edge_tags.items()):
        edges.append({
            "a": a,
            "b": b,
            "weight": float(brain.weights[a, b]),
            "usage": int(brain.usage[a, b]),
            "experiences": sorted(tags),
            "shared": len(tags),
        })

    shared_nodes = sum(1 for tags in node_tags.values() if len(tags) >= 2)
    shared_edges = sum(1 for tags in edge_tags.values() if len(tags) >= 2)
    summary = {
        "core_node_count": int(brain.node_count),
        "displayed_node_count": len(nodes),
        "displayed_edge_count": len(edges),
        "shared_node_count": shared_nodes,
        "shared_edge_count": shared_edges,
        "experience_count": len(experiences),
        "input_fact_count": sum(len(item["facts"]) for item in experiences),
        "note": "球体には3経験を観測した際に実際に活動したNodeと通過したEdgeの和集合を表示しています。",
    }

    return SphereVisualDataset(nodes, edges, experiences, summary)
