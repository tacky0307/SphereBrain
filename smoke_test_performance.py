from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from brain import SphereBrain
from memory_store import MemoryStore


SAMPLES = [
    "空は青い",
    "今日は楽しい",
    "雨が降っている",
    "青い空が見える",
    "私はうれしい",
    "今日は雨です",
]


def main() -> None:
    brain = SphereBrain(node_count=600, neighbors_per_node=8)
    started = perf_counter()
    results = []

    for index in range(75):
        text = SAMPLES[index % len(SAMPLES)]
        sources = brain.text_to_sources(text, count=4)
        result = brain.propagate(sources, steps=20, learn=True)
        results.append((text, result))

    elapsed = perf_counter() - started
    assert elapsed > 0
    assert np.count_nonzero(brain.usage) > 0
    assert brain.text_to_sources("空は青い", 4) == brain.text_to_sources("空は青い", 4)
    assert set(brain.text_units("空は青い")) & set(brain.text_units("青い空が見える"))

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        brain_path = root / "brain.json"
        database_path = root / "memory.db"

        brain.save(brain_path)
        restored = SphereBrain.load(brain_path)
        assert np.array_equal(brain.adjacency, restored.adjacency)
        assert np.allclose(brain.weights, restored.weights)
        assert np.array_equal(brain.usage, restored.usage)

        memory = MemoryStore(database_path)
        memory.add_memories(
            [
                {
                    "kind": "test",
                    "input_text": text,
                    "source_nodes": result.source_nodes,
                    "activated_nodes": result.activated_nodes,
                    "traversed_edges": result.traversed_edges,
                    "importance": 1.0,
                }
                for text, result in results
            ]
        )
        assert memory.count() == 75
        assert len(memory.recent(6)) == 6
        memory.close()

    print(f"75件の経路計算: {elapsed:.3f}秒")
    print(f"1件平均: {elapsed / 75 * 1000:.2f}ミリ秒")
    print("保存・復元・SQLite一括登録: OK")


if __name__ == "__main__":
    main()
