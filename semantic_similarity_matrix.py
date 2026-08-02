from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput, encode_and_experience, load_brain
from semantic_encoder_v2_contextual import observe_contextual


@dataclass(frozen=True)
class MatrixItem:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


def parse_items(text: str) -> list[MatrixItem]:
    items: list[MatrixItem] = []
    seen: set[tuple[str, str, str]] = set()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.replace("|", "｜").split("｜")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"{line_no}行目は 主体｜関係｜内容 の形式で入力してください。")
        key = tuple(parts)
        if key not in seen:
            seen.add(key)
            items.append(MatrixItem(*parts))
    if len(items) < 2:
        raise ValueError("比較する入力を2件以上入力してください。")
    if len(items) > 14:
        raise ValueError("一度に比較できるのは14件までです。")
    return items


def _weighted_similarity(left, right) -> float:
    a = {int(i): float(v) for i, v in enumerate(left) if float(v) > 0}
    b = {int(i): float(v) for i, v in enumerate(right) if float(v) > 0}
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def _jaccard(left, right) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _stage_result(experience, stage: str):
    return {
        "subject": experience.subject_result,
        "relation": experience.relation_result,
        "content": experience.content_result,
    }[stage]


def _pair_kind(left: MatrixItem, right: MatrixItem) -> str:
    same = []
    if left.subject == right.subject:
        same.append("主体")
    if left.relation == right.relation:
        same.append("関係")
    if left.content == right.content:
        same.append("内容")
    if len(same) == 3:
        return "同一入力"
    if not same:
        return "共通要素なし"
    return "同一" + "＋".join(same)


def _observe_old(items: list[MatrixItem]):
    brain = load_brain()
    return [
        encode_and_experience(
            brain,
            StructuredInput(item.subject, item.relation, item.content),
            learn=False,
        )
        for item in items
    ]


def _observe_new(items: list[MatrixItem]):
    return [observe_contextual(item.subject, item.relation, item.content) for item in items]


def build_matrix(text: str, *, stage: str = "content", metric: str = "activation") -> dict:
    if stage not in {"subject", "relation", "content"}:
        raise ValueError("不明な段階です。")
    if metric not in {"activation", "nodes", "edges"}:
        raise ValueError("不明な指標です。")

    items = parse_items(text)
    old_results = _observe_old(items)
    new_results = _observe_new(items)

    def score(left_exp, right_exp) -> float:
        left = _stage_result(left_exp, stage)
        right = _stage_result(right_exp, stage)
        if metric == "activation":
            return _weighted_similarity(left.final_activation, right.final_activation)
        if metric == "nodes":
            return _jaccard(left.activated_nodes, right.activated_nodes)
        return _jaccard(left.traversed_edges, right.traversed_edges)

    def matrix(results):
        return [[score(a, b) for b in results] for a in results]

    old_matrix = matrix(old_results)
    new_matrix = matrix(new_results)
    categories: dict[str, dict[str, list[float]]] = {}
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            kind = _pair_kind(items[i], items[j])
            old_score = old_matrix[i][j]
            new_score = new_matrix[i][j]
            bucket = categories.setdefault(kind, {"old": [], "new": []})
            bucket["old"].append(old_score)
            bucket["new"].append(new_score)
            pairs.append({
                "left": items[i].label,
                "right": items[j].label,
                "kind": kind,
                "old": old_score,
                "new": new_score,
                "change": new_score - old_score,
            })

    summaries = []
    for kind, values in categories.items():
        summaries.append({
            "kind": kind,
            "count": len(values["old"]),
            "old": sum(values["old"]) / len(values["old"]),
            "new": sum(values["new"]) / len(values["new"]),
        })
    summaries.sort(key=lambda row: (-row["new"], row["kind"]))
    pairs.sort(key=lambda row: (row["new"], row["left"], row["right"]))

    return {
        "items": [{"label": item.label} for item in items],
        "stage": stage,
        "metric": metric,
        "old_matrix": old_matrix,
        "new_matrix": new_matrix,
        "summaries": summaries,
        "pairs": pairs,
    }
