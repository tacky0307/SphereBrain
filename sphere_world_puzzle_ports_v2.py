from __future__ import annotations

from collections import deque

import numpy as np

from sphere_world_puzzle import ACTIONS, PuzzleSphereBrain, PuzzleWorld, facts_for_world, shortest_action


class ActionPortPuzzleBrainV2(PuzzleSphereBrain):
    """Action Port decoder that explicitly grows routes from world context to ports.

    The original port experiment co-activated a port as a source.  This version
    reinforces existing Core paths in the required direction of use:

        strong world-context nodes -> existing Core edges -> motor port nodes

    No action rule is used at inference time.  The teacher is used only while
    forming those motor routes.
    """

    def __init__(self, repeats: int = 7) -> None:
        # Build the same Core and fixed port nodes, but train with route growth.
        self.brain = __import__("semantic_encoder_v2_contextual").load_contextual_brain()
        self.repeats = max(1, int(repeats))
        self.training_examples: list[dict] = []
        component_nodes = __import__("semantic_encoder_v2").component_nodes
        self.action_ports = {
            action: component_nodes(self.brain, "action_port_v2", action, 4)
            for action in ACTIONS
        }
        self._train()

    def _shortest_core_path(self, starts: list[int], goals: set[int], max_depth: int = 14) -> list[int]:
        """Find a short existing graph path from any active context node to a port."""
        queue = deque()
        previous: dict[int, int | None] = {}
        depth: dict[int, int] = {}

        for node in starts:
            node = int(node)
            queue.append(node)
            previous[node] = None
            depth[node] = 0

        found: int | None = None
        while queue:
            node = queue.popleft()
            if node in goals:
                found = node
                break
            if depth[node] >= max_depth:
                continue

            neighbors = np.flatnonzero(self.brain.adjacency[node]).tolist()
            # Prefer already-strong edges, while still staying inside the Core graph.
            neighbors.sort(key=lambda other: float(self.brain.weights[node, other]), reverse=True)
            for other in neighbors:
                other = int(other)
                if other in previous:
                    continue
                previous[other] = node
                depth[other] = depth[node] + 1
                queue.append(other)

        if found is None:
            return []

        path = []
        cursor: int | None = found
        while cursor is not None:
            path.append(cursor)
            cursor = previous[cursor]
        path.reverse()
        return path

    def _grow_motor_route(self, decision_context: dict[int, float], action: str) -> None:
        """Strengthen real Core edges from the current world context to a motor port."""
        ranked = sorted(decision_context.items(), key=lambda item: item[1], reverse=True)
        starts = [int(node) for node, value in ranked[:18] if float(value) > 0]
        goals = {int(node) for node in self.action_ports[action]}
        if not starts or not goals:
            return

        # Grow several converging routes, not just one brittle wire.
        used_starts = starts[:6]
        for start in used_starts:
            path = self._shortest_core_path([start], goals, max_depth=16)
            if len(path) < 2:
                continue
            for a, b in zip(path, path[1:]):
                a, b = int(a), int(b)
                boost = 0.11
                self.brain.weights[a, b] = min(0.995, float(self.brain.weights[a, b]) + boost)
                self.brain.weights[b, a] = self.brain.weights[a, b]
                self.brain.usage[a, b] += 1
                self.brain.usage[b, a] += 1
                self.brain.node_usage[a] += 1
                self.brain.node_usage[b] += 1

    def _train(self) -> None:
        for world in self._training_worlds():
            action = shortest_action(world)
            self.training_examples.append(
                {
                    "puzzle": world.name,
                    "player": list(world.player),
                    "goal": list(world.goal),
                    "action": action,
                    "facts": [fact.label for fact in facts_for_world(world)],
                    "port_nodes": self.action_ports[action],
                }
            )
            for _ in range(self.repeats):
                world_context, _ = self._world_context(world, learn=True)
                decision_context = self._decision_context(world_context, learn=True)
                self._grow_motor_route(decision_context, action)

                # Let activity actually travel from the world side through the
                # newly reinforced route; this further strengthens used edges.
                ranked = sorted(decision_context.items(), key=lambda item: item[1], reverse=True)
                sources = [int(node) for node, _ in ranked[:8]]
                self.brain.propagate_contextual(
                    sources,
                    decision_context,
                    steps=18,
                    threshold=0.14,
                    noise=0.003,
                    learn=True,
                    context_anchor=0.64,
                    context_decay=0.95,
                    resonance=True,
                )

    def _read_ports(self, decision_context: dict[int, float]):
        ranked = sorted(decision_context.items(), key=lambda item: item[1], reverse=True)
        sources = [int(node) for node, value in ranked[:10] if float(value) > 0]
        result = self.brain.propagate_contextual(
            sources,
            decision_context,
            steps=20,
            threshold=0.13,
            noise=0.0,
            learn=False,
            context_anchor=0.58,
            context_decay=0.95,
            resonance=True,
        )

        history = list(result.activation_history or [])
        final = np.asarray(result.final_activation, dtype=float)
        recent = history[-7:] if history else []
        traversed = {tuple(sorted((int(a), int(b)))) for a, b in result.traversed_edges}

        raw_scores: dict[str, float] = {}
        details: dict[str, dict] = {}
        for action, nodes in self.action_ports.items():
            node_set = {int(node) for node in nodes}
            final_sum = sum(float(final[node]) for node in node_set)
            recent_hits = sum(sum(1 for node in step if int(node) in node_set) for step in recent)
            activated = len(node_set & {int(node) for node in result.activated_nodes})
            incoming = sum(1 for a, b in traversed if a in node_set or b in node_set)
            score = final_sum + 0.18 * recent_hits + 0.10 * activated + 0.035 * incoming
            raw_scores[action] = score
            details[action] = {
                "port_nodes": list(nodes),
                "final_strength": final_sum,
                "recent_arrivals": recent_hits,
                "activated_port_nodes": activated,
                "incoming_edges": incoming,
            }
        return result, raw_scores, details

    def decide(self, world: PuzzleWorld) -> dict:
        result = super().decide(world)
        result["decoder"] = "Action Port Decoder v2 — World-to-Port Routes"
        return result
