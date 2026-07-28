from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class NumericStimulus:
    """Coreへ渡す、外部表現を含まない数値刺激。"""

    source_nodes: list[int]
    strength: list[float]
    encoder: str = "stable-hash-v1"


class TextEncoder:
    """日本語などの文字列を決定的な数値刺激へ変換する入口。"""

    def __init__(self, node_count: int, source_count: int = 4) -> None:
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        if source_count <= 0 or source_count > node_count:
            raise ValueError("source_count must be between 1 and node_count")
        self.node_count = node_count
        self.source_count = source_count

    def encode(self, text: str) -> NumericStimulus:
        clean = " ".join(text.strip().split())
        if not clean:
            raise ValueError("入力が空です。")

        digest = hashlib.sha256(clean.encode("utf-8")).digest()
        nodes: list[int] = []
        strengths: list[float] = []
        offset = 0

        while len(nodes) < self.source_count:
            if offset + 4 > len(digest):
                digest = hashlib.sha256(digest).digest()
                offset = 0
            value = int.from_bytes(digest[offset:offset + 4], "big")
            node = value % self.node_count
            offset += 4
            if node in nodes:
                continue
            nodes.append(node)
            strengths.append(max(0.55, 1.0 - len(nodes) * 0.08))

        return NumericStimulus(source_nodes=nodes, strength=strengths)
