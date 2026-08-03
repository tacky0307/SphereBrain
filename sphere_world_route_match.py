from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sphere_world import ACTIONS, POSITIONS
from sphere_world_multi_context import MultiContextSphereWorldBrain, TRAIN_STATES


@dataclass(frozen=True)
class RouteLane:
    action: str
    score: float
    node_score: float
    edge_score: float
    shared_nodes: int
    shared_edges: int
    current_nodes: int
    current_edges: int
    prototype_nodes: int
    prototype_edges: int
    current_route: list[dict]
    prototype_route: list[dict]

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": self.score,
            "node_score": self.node_score,
            "edge_score": self.edge_score,
            "shared_nodes": self.shared_nodes,
            "shared_edges": self.shared_edges,
            "current_nodes": self.current_nodes,
            "current_edges": self.current_edges,
            "prototype_nodes": self.prototype_nodes,
            "prototype_edges": self.prototype_edges,
            "current_route": self.current_route,
            "prototype_route": self.prototype_route,
        }


def _unique_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for raw in edges:
        edge = (int(raw[0]), int(raw[1]))
        if edge not in seen:
            seen.add(edge)
            ordered.append(edge)
    return ordered


def _route_sample(
    edges: list[tuple[int, int]],
    shared: set[tuple[int, int]],
    *,
    limit: int = 24,
) -> list[dict]:
    """Return a compact but honest route sample, prioritising shared edges."""
    if not edges:
        return []

    shared_ordered = [edge for edge in edges if edge in shared]
    other_ordered = [edge for edge in edges if edge not in shared]

    # Keep all shared evidence where possible, then fill with route-local context.
    selected = shared_ordered[:limit]
    remaining = max(0, limit - len(selected))
    if remaining:
        if len(other_ordered) <= remaining:
            selected.extend(other_ordered)
        else:
            step = max(1, len(other_ordered) // remaining)
            selected.extend(other_ordered[::step][:remaining])

    selected_set = set(selected)
    ordered_selected = [edge for edge in edges if edge in selected_set]
    return [
        {
            "from": edge[0],
            "to": edge[1],
            "shared": edge in shared,
        }
        for edge in ordered_selected[:limit]
    ]


class SphereWorldRouteMatch:
    """Presentation observer for real Raw-Output route matching."""

    def __init__(self, repeats: int = 12) -> None:
        self.brain = MultiContextSphereWorldBrain(repeats=repeats)

    @staticmethod
    def _best_prototype(raw_result, prototypes):
        raw_nodes = set(int(node) for node in raw_result.activated_nodes)
        raw_edges = set(_unique_edges(tuple(edge) for edge in raw_result.traversed_edges))
        best = None
        best_score = -1.0

        for prototype in prototypes:
            prototype_nodes = set(int(node) for node in prototype.activated_nodes)
            prototype_edges = set(_unique_edges(tuple(edge) for edge in prototype.traversed_edges))
            node_union = raw_nodes | prototype_nodes
            edge_union = raw_edges | prototype_edges
            node_score = len(raw_nodes & prototype_nodes) / len(node_union) if node_union else 1.0
            edge_score = len(raw_edges & prototype_edges) / len(edge_union) if edge_union else 1.0
            score = 0.35 * node_score + 0.65 * edge_score
            if score > best_score:
                best_score = score
                best = (prototype, node_score, edge_score)

        return best

    def inspect(self, player_position: int, enemy_position: int) -> dict:
        player_position = max(0, min(2, int(player_position)))
        enemy_position = max(0, min(2, int(enemy_position)))
        probe = self.brain.probe(player_position, enemy_position)
        raw_result = probe["raw_result"]
        raw_nodes = set(int(node) for node in raw_result.activated_nodes)
        raw_edges_ordered = _unique_edges(tuple(edge) for edge in raw_result.traversed_edges)
        raw_edges = set(raw_edges_ordered)

        lanes: list[RouteLane] = []
        for action in ACTIONS:
            match = self._best_prototype(raw_result, self.brain.prototypes[action])
            if match is None:
                continue
            prototype, node_score, edge_score = match
            prototype_nodes = set(int(node) for node in prototype.activated_nodes)
            prototype_edges_ordered = _unique_edges(tuple(edge) for edge in prototype.traversed_edges)
            prototype_edges = set(prototype_edges_ordered)
            shared_nodes = raw_nodes & prototype_nodes
            shared_edges = raw_edges & prototype_edges
            score = 0.35 * node_score + 0.65 * edge_score

            lanes.append(RouteLane(
                action=action,
                score=score,
                node_score=node_score,
                edge_score=edge_score,
                shared_nodes=len(shared_nodes),
                shared_edges=len(shared_edges),
                current_nodes=len(raw_nodes),
                current_edges=len(raw_edges),
                prototype_nodes=len(prototype_nodes),
                prototype_edges=len(prototype_edges),
                current_route=_route_sample(raw_edges_ordered, shared_edges),
                prototype_route=_route_sample(prototype_edges_ordered, shared_edges),
            ))

        lanes.sort(key=lambda lane: (-lane.score, lane.action))
        selected = lanes[0].action if lanes else "停止"
        expected = self.brain.action_for_positions(player_position, enemy_position)
        trained = (player_position, enemy_position) in TRAIN_STATES

        return {
            "player_position": player_position,
            "enemy_position": enemy_position,
            "player": POSITIONS[player_position],
            "enemy": POSITIONS[enemy_position],
            "trained": trained,
            "expected": expected,
            "selected": selected,
            "correct": selected == expected,
            "facts": probe["facts"],
            "world_context_nodes": probe["world_context_nodes"],
            "raw_nodes": len(raw_nodes),
            "raw_edges": len(raw_edges),
            "lanes": [lane.to_dict() for lane in lanes],
            "note": "表示経路は実際のtraversed_edgesから最大24本を抽出。共通Edgeを優先表示しています。",
        }
