from __future__ import annotations

from pathlib import Path
import argparse

from research_store import ResearchStore


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="同じ入力で生じた球体脳の経路を比較します。")
    parser.add_argument("text", help="比較する入力文字列。例: こんにちは")
    parser.add_argument("--source", default="input", help="input / audio / idle")
    parser.add_argument("--limit", type=int, default=10, help="比較する直近試行数")
    parser.add_argument("--database", default="data/research.db", help="研究DBのパス")
    args = parser.parse_args()

    store = ResearchStore(Path(args.database))
    report = store.repeated_input_comparison(args.text, source=args.source, limit=args.limit)

    print(f"入力: {report['input']}")
    print(f"記録された試行: {report['trial_count']}件")
    if report["trial_count"] < 2:
        print("比較には同じ入力が2回以上必要です。")
        return

    print("\n変化の履歴")
    for index, item in enumerate(report["comparisons"], start=2):
        print(
            f"{index - 1}回目 → {index}回目: "
            f"経路一致 {percent(item['edge_similarity'])}, "
            f"序盤一致 {percent(item['ordered_similarity'])}, "
            f"共通 {item['shared_edges']}, 新規 {item['new_edges']}, 消失 {item['lost_edges']}"
        )

    latest = report["latest"]
    print("\n直近の比較")
    print(f"経路の集合一致率: {percent(latest['edge_similarity'])}")
    print(f"最初から同じ順序で進んだ割合: {percent(latest['ordered_similarity'])}")
    print(f"新しく現れた経路: {latest['new_edges']}本")
    print(f"今回は使われなかった経路: {latest['lost_edges']}本")


if __name__ == "__main__":
    main()
