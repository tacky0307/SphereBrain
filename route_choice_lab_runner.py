from __future__ import annotations

import sqlite3
from typing import Any

import route_choice_lab as lab


def structural_route_signature(candidate: lab.Candidate) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return a direction/order independent identity for one internal route.

    Different internal routes are allowed to decode to the same Japanese word.
    Only candidates that are structurally the same route are treated as duplicates.
    """

    nodes = tuple(sorted({int(node) for node in candidate.nodes}))
    edges = tuple(
        sorted({lab.norm(int(a), int(b)) for a, b in candidate.edges})
    )
    return nodes, edges


def evaluate_without_structural_duplicates(
    text: str,
    count: int,
    decoy_count: int,
) -> dict[str, Any]:
    """Evaluate route candidates while suppressing duplicate route cards.

    The original decoder may legitimately render two different routes as the same
    word, such as two distinct routes both decoded as ``泳ぐ``. Those candidates
    remain visible and can both receive ○. This function removes only exact
    structural duplicates, including the same undirected route stored in reverse
    or in a different traversal order.
    """

    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")

    brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = lab.prefix_signature(sources)
    routes, counts = lab.load_routes()
    if not routes:
        raise RuntimeError(
            "memory.dbとpattern_candidates.dbの両方を調べましたが候補経路がありません。"
            "通常入力またはReflectionを一度実行してください。"
        )

    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix)
    for candidate in routes:
        candidate.score = lab.score_candidate(
            brain,
            sources,
            prefix,
            candidate,
            source_tree,
            feedback_lookup,
        )
    routes.sort(key=lambda candidate: (-candidate.score, candidate.key))

    requested_count = max(1, int(count))
    requested_decoys = max(0, min(int(decoy_count), requested_count - 1))
    requested_real = max(1, requested_count - requested_decoys)

    selected: list[lab.Candidate] = []
    seen_structures: set[
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
    ] = set()
    seen_keys: set[str] = set()

    def append_if_unique(candidate: lab.Candidate) -> bool:
        signature = structural_route_signature(candidate)
        if candidate.key in seen_keys or signature in seen_structures:
            return False
        seen_keys.add(candidate.key)
        seen_structures.add(signature)
        selected.append(candidate)
        return True

    for candidate in routes:
        if append_if_unique(candidate) and sum(not item.decoy for item in selected) >= requested_real:
            break

    # Generate a wider decoy pool and then keep only structurally unique routes.
    # Multiple seeds make it very unlikely that duplicate removal reduces the
    # number of displayed decoys.
    if requested_decoys:
        real_pool = routes[: min(120, len(routes))]
        for attempt in range(4):
            remaining = requested_decoys - sum(item.decoy for item in selected)
            if remaining <= 0:
                break
            pool_size = max(8, remaining * 5)
            generated = lab.decoys(
                real_pool,
                pool_size,
                f"{text}|structural-dedup|{attempt}",
            )
            for candidate in generated:
                if append_if_unique(candidate):
                    remaining -= 1
                    if remaining <= 0:
                        break

    # If the graph cannot produce enough unique decoys, preserve the requested
    # total candidate count by filling the remainder with additional real routes.
    if len(selected) < requested_count:
        for candidate in routes:
            append_if_unique(candidate)
            if len(selected) >= requested_count:
                break

    selected = selected[:requested_count]
    for candidate in selected:
        lab.decode_candidate(text, candidate, routes)
        candidate.score = lab.score_candidate(
            brain,
            sources,
            prefix,
            candidate,
            source_tree,
            feedback_lookup,
        )
    lab.normalize_candidates(selected)

    return {
        "prefix": prefix,
        "sources": sources,
        "candidates": selected,
        "source_counts": counts,
        "latest_reflection_run": lab.latest_reflection_run(),
    }


def feedback_cycle_count(text: str) -> int:
    """Return the persisted number of completed teaching cycles for this text."""

    try:
        lab.init_feedback()
        with sqlite3.connect(lab.FEEDBACK_DB, timeout=30) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM feedback_sessions WHERE prefix_text=?",
                (str(text or "").strip(),),
            ).fetchone()
        return int(row[0]) if row else 0
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return 0


def install_work_friendly_ui() -> None:
    """Add persistent progress and a one-click next-cycle action for ChatGPT Work."""

    page = lab.PAGE
    page = page.replace(
        "<p><button>経路候補を評価する</button></p>",
        '<p><button id="evaluate-button">経路候補を評価する</button></p>',
        1,
    )
    page = page.replace(
        "</form></div>\n{% if message %}",
        "</form></div>\n"
        '<div class="card" id="education-progress">'
        '<div class="eyebrow">EDUCATION PROGRESS</div>'
        '<h2>「{{text}}」の累積教育回数：'
        '<strong>{{feedback_cycle_count(text)}}</strong>回</h2>'
        '<p class="muted">教育を確定し、feedback_sessionsへ正常保存された時だけ1回増えます。'
        '一時停止や再読込の後も、この数字を基準に再開できます。</p>'
        "</div>\n{% if message %}",
        1,
    )
    page = page.replace(
        "<button>選んだ○・△・×で教育を確定</button>",
        '<button id="confirm-teaching-button">選んだ○・△・×で教育を確定</button>',
        1,
    )
    page = page.replace(
        "</main></body></html>",
        "{% if comparison %}"
        '<div class="card" id="next-cycle-card">'
        '<div class="eyebrow">NEXT CYCLE</div>'
        '<h2>同じ設定で次の教育サイクルへ進む</h2>'
        '<form method="post">'
        '<input type="hidden" name="action" value="evaluate">'
        '<input type="hidden" name="text" value="{{text}}">'
        '<input type="hidden" name="count" value="{{count}}">'
        '<input type="hidden" name="decoys" value="{{decoys}}">'
        '<button id="next-cycle-button">同じ設定で次の候補を評価</button>'
        "</form>"
        '<p class="muted">途中入力・候補数・偽経路数を変えずに次へ進みます。</p>'
        "</div>"
        "{% endif %}"
        "</main></body></html>",
        1,
    )
    lab.PAGE = page
    lab.app.jinja_env.globals["feedback_cycle_count"] = feedback_cycle_count


def main() -> None:
    # Flask's index handler resolves ``evaluate`` from route_choice_lab's module
    # globals at request time, so replacing it here keeps all v0.3 teaching and
    # Reflection behavior unchanged while fixing only candidate deduplication.
    lab.evaluate = evaluate_without_structural_duplicates
    install_work_friendly_ui()
    lab.main()


if __name__ == "__main__":
    main()
