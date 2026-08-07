from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from brain import SignalResult, SphereBrain
from llm_core_pipeline import OpenAIAdapter


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "sphereword_v01"
BRAIN_FILE = DATA / "brain.json"
PROJECTION_FILE = DATA / "projection.npy"

EMBEDDING_MODEL = os.getenv("SPHERE_EMBEDDING_MODEL", "text-embedding-3-small")
WORD_MODEL = os.getenv("SPHERE_WORD_MODEL", os.getenv("SPHERE_DECODER_MODEL", "gpt-5-mini"))
STIMULUS_DIM = int(os.getenv("SPHEREWORD_STIMULUS_DIM", "128"))
SOURCE_COUNT = int(os.getenv("SPHEREWORD_SOURCE_COUNT", "10"))
PROJECTION_SEED = int(os.getenv("SPHEREWORD_PROJECTION_SEED", "20260807"))


@dataclass
class CoreTrace:
    text: str
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]


@dataclass
class RoundChoice:
    prompt: str
    secret: str
    candidates: list[str]
    scores: dict[str, float]


def _jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class SphereWordEngine:
    """Thin game layer around the real SphereBrain Core.

    The LLM generates human-readable association candidates. It does not choose
    the secret word. Candidate ranking and guess distance are calculated from
    activity produced by the persisted SphereBrain Core.
    """

    def __init__(self, reset_core: bool = False) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        self.adapter = OpenAIAdapter()
        if reset_core or not BRAIN_FILE.exists():
            self.brain = SphereBrain(
                node_count=600,
                neighbors_per_node=7,
                seed=42,
                propagation_mode="focused",
                structural_assist_enabled=True,
            )
            self.brain.save(BRAIN_FILE)
        else:
            self.brain = SphereBrain.load(BRAIN_FILE)

    def _projection(self, input_dim: int) -> np.ndarray:
        if PROJECTION_FILE.exists():
            matrix = np.load(PROJECTION_FILE)
            if matrix.shape == (STIMULUS_DIM, input_dim):
                return matrix
        rng = np.random.default_rng(PROJECTION_SEED)
        matrix = rng.normal(
            0.0,
            1.0 / np.sqrt(STIMULUS_DIM),
            size=(STIMULUS_DIM, input_dim),
        )
        np.save(PROJECTION_FILE, matrix)
        return matrix

    def _sources(self, text: str) -> list[int]:
        embedding = np.asarray(self.adapter.embed(text), dtype=float)
        norm = float(np.linalg.norm(embedding))
        if norm:
            embedding /= norm
        stimulus = self._projection(embedding.size) @ embedding
        stimulus_norm = float(np.linalg.norm(stimulus))
        if stimulus_norm:
            stimulus /= stimulus_norm

        strongest = np.argsort(np.abs(stimulus))[-min(SOURCE_COUNT, stimulus.size):][::-1]
        nodes: list[int] = []
        import hashlib
        for dimension in strongest:
            sign = "positive" if stimulus[dimension] >= 0 else "negative"
            material = f"sphereword-v01|{int(dimension)}|{sign}".encode("utf-8")
            node = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % self.brain.node_count
            if node not in nodes:
                nodes.append(node)
        return nodes

    def trace(self, text: str, *, learn: bool = False, noise: float = 0.0) -> CoreTrace:
        clean = text.strip()
        if not clean:
            raise ValueError("言葉が空です。")
        sources = self._sources(clean)
        result: SignalResult = self.brain.propagate(
            sources,
            steps=14,
            threshold=0.18,
            noise=noise,
            learn=learn,
        )
        if learn:
            self.brain.save(BRAIN_FILE)
        return CoreTrace(
            text=clean,
            source_nodes=sources,
            activated_nodes=result.activated_nodes,
            traversed_edges=result.traversed_edges,
        )

    def _parse_candidates(self, raw: str) -> list[str]:
        text = raw.strip()
        candidates: list[str] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = parsed.get("candidates", [])
            if isinstance(parsed, list):
                candidates = [str(item).strip() for item in parsed]
        except json.JSONDecodeError:
            for line in text.splitlines():
                item = line.strip().lstrip("-・0123456789. ").strip()
                if item:
                    candidates.append(item)
        unique: list[str] = []
        for item in candidates:
            item = item.strip(" \t\r\n\"'「」『』")
            if item and item not in unique:
                unique.append(item)
        return unique[:8]

    def generate_candidates(self, prompt: str) -> list[str]:
        instructions = (
            "日本語の連想ゲーム用候補生成器です。"
            "入力語から自然に連想できる、互いに少し方向性の異なる短い日本語の単語を8個作ってください。"
            "答えを選ばず、候補だけをJSON配列で返してください。説明は禁止です。"
        )
        response = self.adapter.client.responses.create(
            model=WORD_MODEL,
            instructions=instructions,
            input=prompt.strip(),
        )
        candidates = self._parse_candidates(response.output_text)
        if len(candidates) < 3:
            raise RuntimeError("LLMから十分な連想候補を取得できませんでした。")
        return candidates

    def _familiarity(self, trace: CoreTrace) -> float:
        if not trace.traversed_edges:
            return 0.0
        usages = [float(self.brain.usage[a, b]) for a, b in trace.traversed_edges]
        return float(np.mean(usages)) if usages else 0.0

    def choose_secret(self, prompt: str, candidates: list[str]) -> RoundChoice:
        prompt_trace = self.trace(prompt, learn=False)
        scored: dict[str, float] = {}
        for candidate in candidates:
            joined = f"{prompt} ｜ {candidate}"
            candidate_trace = self.trace(joined, learn=False)
            node_overlap = _jaccard(prompt_trace.activated_nodes, candidate_trace.activated_nodes)
            edge_overlap = _jaccard(prompt_trace.traversed_edges, candidate_trace.traversed_edges)
            familiarity = self._familiarity(candidate_trace)
            scored[candidate] = (0.30 * node_overlap) + (0.60 * edge_overlap) + (0.01 * familiarity)

        best = max(scored.values())
        tied = [word for word, score in scored.items() if abs(score - best) < 1e-12]
        if len(tied) == 1:
            secret = tied[0]
        else:
            # Deterministic tie break derived from the prompt, not semantic rules.
            import hashlib
            digest = hashlib.sha256((prompt + "|" + "|".join(tied)).encode("utf-8")).digest()
            secret = tied[int.from_bytes(digest[:4], "big") % len(tied)]

        # The chosen experience changes only Core pathways; no external policy table is stored.
        for _ in range(2):
            self.trace(f"{prompt} ｜ {secret}", learn=True, noise=0.004)

        return RoundChoice(prompt=prompt, secret=secret, candidates=candidates, scores=scored)

    def new_round(self, prompt: str) -> RoundChoice:
        clean = prompt.strip()
        if not clean:
            raise ValueError("お題を入力してください。")
        candidates = self.generate_candidates(clean)
        return self.choose_secret(clean, candidates)

    def guess(self, prompt: str, secret: str, guess: str) -> dict:
        clean = guess.strip()
        if not clean:
            raise ValueError("推測する言葉を入力してください。")
        if clean == secret:
            similarity = 1.0
        else:
            secret_trace = self.trace(f"{prompt} ｜ {secret}", learn=False)
            guess_trace = self.trace(f"{prompt} ｜ {clean}", learn=False)
            node_overlap = _jaccard(secret_trace.activated_nodes, guess_trace.activated_nodes)
            edge_overlap = _jaccard(secret_trace.traversed_edges, guess_trace.traversed_edges)
            similarity = (0.35 * node_overlap) + (0.65 * edge_overlap)

        # The attempt itself becomes a light experience so play gradually changes the Core.
        self.trace(f"{prompt} ｜ {clean}", learn=True, noise=0.003)

        percent = int(round(max(0.0, min(1.0, similarity)) * 100))
        if clean == secret:
            label = "正解！"
        elif percent >= 70:
            label = "かなり近い"
        elif percent >= 45:
            label = "近い"
        elif percent >= 25:
            label = "少し近い"
        else:
            label = "遠い"
        return {"guess": clean, "similarity": similarity, "percent": percent, "label": label, "correct": clean == secret}

    def stats(self) -> dict:
        used_edges = int(np.count_nonzero(np.triu(self.brain.usage, k=1)))
        total_usage = int(np.triu(self.brain.usage, k=1).sum())
        experienced_nodes = int(np.count_nonzero(self.brain.node_usage))
        return {
            "used_edges": used_edges,
            "total_edge_usage": total_usage,
            "experienced_nodes": experienced_nodes,
            "max_edge_weight": round(float(self.brain.weights.max()), 6),
            "embedding_model": EMBEDDING_MODEL,
            "word_model": WORD_MODEL,
            "brain_file": str(BRAIN_FILE),
        }
