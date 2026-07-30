from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .core import ExperimentTrace, SphereWaveCore


@dataclass(frozen=True)
class RecallConfig:
    """Observation thresholds for recall diagnostics.

    This layer does not alter activity, conductivity, or learning. It only
    measures how an A-only wave uses the terrain that experience has formed.
    """

    active_threshold: float = 1e-8
    meaningful_threshold: float = 1e-4


class RecallPathDiagnostics:
    def __init__(self, config: RecallConfig | None = None) -> None:
        self.config = config or RecallConfig()

    @staticmethod
    def easiest_path(
        core: SphereWaveCore,
        source_ids: tuple[int, ...],
        target_ids: tuple[int, ...],
    ) -> tuple[list[int], float]:
        targets = set(int(value) for value in target_ids)
        distances = np.full(core.config.node_count, np.inf)
        previous = np.full(core.config.node_count, -1, dtype=int)
        queue: list[tuple[float, int]] = []

        for node_id in source_ids:
            node = int(node_id)
            distances[node] = 0.0
            heapq.heappush(queue, (0.0, node))

        destination = -1
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances[node_id]:
                continue
            if node_id in targets:
                destination = node_id
                break
            for neighbor in np.flatnonzero(core.adjacency[node_id]):
                conductivity = max(float(core.conductivity[node_id, neighbor]), 1e-12)
                candidate = distance + 1.0 / conductivity
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = node_id
                    heapq.heappush(queue, (candidate, int(neighbor)))

        if destination < 0:
            return [], float("inf")

        path = [destination]
        current = destination
        while previous[current] >= 0:
            current = int(previous[current])
            path.append(current)
        path.reverse()
        return path, float(distances[destination])

    def analyze(
        self,
        core: SphereWaveCore,
        trace: ExperimentTrace,
        source_ids: tuple[int, ...],
        target_ids: tuple[int, ...],
    ) -> tuple[dict, list[dict]]:
        path, path_cost = self.easiest_path(core, source_ids, target_ids)
        path_ids = np.asarray(path, dtype=int)
        target = np.asarray(target_ids, dtype=int)

        target_integral = 0.0
        target_peak = 0.0
        path_integral = 0.0
        path_peak = 0.0
        closest_distance = float("inf")
        closest_step = -1
        closest_node = -1
        meaningful_closest_distance = float("inf")
        meaningful_closest_step = -1
        meaningful_closest_node = -1
        furthest_path_index = -1
        furthest_path_step = -1
        step_rows: list[dict] = []

        for step_number, snapshot in enumerate(trace.snapshots, start=1):
            activity = snapshot.activity
            target_value = float(np.mean(activity[target])) if target.size else 0.0
            path_value = float(np.sum(activity[path_ids])) if path_ids.size else 0.0
            target_integral += target_value
            target_peak = max(target_peak, target_value)
            path_integral += path_value
            path_peak = max(path_peak, path_value)

            active = np.flatnonzero(activity > self.config.active_threshold)
            meaningful = np.flatnonzero(activity > self.config.meaningful_threshold)

            step_closest = float("inf")
            step_closest_node = -1
            if active.size and target.size:
                distances = np.linalg.norm(
                    core.positions[active, None, :] - core.positions[target][None, :, :], axis=2
                )
                flat = int(np.argmin(distances))
                active_index, _ = np.unravel_index(flat, distances.shape)
                step_closest = float(np.min(distances))
                step_closest_node = int(active[active_index])
                if step_closest < closest_distance:
                    closest_distance = step_closest
                    closest_step = step_number
                    closest_node = step_closest_node

            step_meaningful = float("inf")
            step_meaningful_node = -1
            if meaningful.size and target.size:
                distances = np.linalg.norm(
                    core.positions[meaningful, None, :] - core.positions[target][None, :, :], axis=2
                )
                flat = int(np.argmin(distances))
                meaningful_index, _ = np.unravel_index(flat, distances.shape)
                step_meaningful = float(np.min(distances))
                step_meaningful_node = int(meaningful[meaningful_index])
                if step_meaningful < meaningful_closest_distance:
                    meaningful_closest_distance = step_meaningful
                    meaningful_closest_step = step_number
                    meaningful_closest_node = step_meaningful_node

            step_furthest = -1
            if path_ids.size:
                path_active = np.flatnonzero(activity[path_ids] > self.config.meaningful_threshold)
                if path_active.size:
                    step_furthest = int(np.max(path_active))
                    if step_furthest > furthest_path_index:
                        furthest_path_index = step_furthest
                        furthest_path_step = step_number

            step_rows.append({
                "step": step_number,
                "total_activity": snapshot.total_activity,
                "target_mean_activity": target_value,
                "path_activity": path_value,
                "closest_active_distance_to_target": step_closest,
                "closest_active_node": step_closest_node,
                "closest_meaningful_distance_to_target": step_meaningful,
                "closest_meaningful_node": step_meaningful_node,
                "furthest_meaningful_path_index": step_furthest,
            })

        path_progress = (
            float(furthest_path_index) / float(max(len(path) - 1, 1))
            if furthest_path_index >= 0 and path
            else 0.0
        )
        summary = {
            "steps": len(trace.snapshots),
            "path_cost": path_cost,
            "path_nodes": len(path),
            "path_integral": path_integral,
            "path_peak": path_peak,
            "target_integral": target_integral,
            "target_peak": target_peak,
            "closest_active_distance_to_target": closest_distance,
            "closest_active_step": closest_step,
            "closest_active_node": closest_node,
            "closest_meaningful_distance_to_target": meaningful_closest_distance,
            "closest_meaningful_step": meaningful_closest_step,
            "closest_meaningful_node": meaningful_closest_node,
            "furthest_meaningful_path_index": furthest_path_index,
            "furthest_meaningful_path_step": furthest_path_step,
            "path_progress": path_progress,
            "reached_target": bool(target_peak > self.config.active_threshold),
            "meaningfully_reached_target": bool(target_peak > self.config.meaningful_threshold),
        }
        return summary, step_rows
