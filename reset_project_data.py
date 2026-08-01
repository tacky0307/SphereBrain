from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET_FILES = (
    "brain.json",
    "branch_observer.db",
    "memory.db",
    "pattern_candidates.db",
    "route_choice_feedback.db",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_targets(root: Path) -> list[Path]:
    """Find generated data in both supported layouts.

    Older work folders keep the files beside this script, while the current
    repository normally stores them under data/.
    """
    candidates: list[Path] = []
    for directory in (root, root / "data"):
        for name in TARGET_FILES:
            path = directory / name
            if path.is_file():
                candidates.append(path)
    return candidates


def backup_targets(root: Path, targets: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "backups" / f"data_reset_{stamp}"

    for source in targets:
        relative = source.relative_to(root)
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return backup_dir


def delete_targets(targets: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    deleted: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for path in targets:
        try:
            path.unlink()
            deleted.append(path)
        except OSError as exc:
            failed.append((path, str(exc)))

    return deleted, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SphereBrainの生成データをバックアップして初期化します。"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="確認入力を省略します。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    targets = find_targets(root)

    print("SphereBrain データ初期化")
    print(f"対象フォルダ: {root}")

    if not targets:
        print("初期化対象のデータファイルはありません。")
        return 0

    print("\n対象ファイル:")
    for path in targets:
        print(f"  - {path.relative_to(root)}")

    if not args.yes:
        print("\n実行前に backups フォルダへ退避します。")
        answer = input("初期化するには RESET と入力してください: ").strip()
        if answer != "RESET":
            print("キャンセルしました。")
            return 0

    try:
        backup_dir = backup_targets(root, targets)
    except OSError as exc:
        print(f"バックアップに失敗したため、初期化を中止しました: {exc}")
        return 1

    deleted, failed = delete_targets(targets)

    print(f"\nバックアップ先: {backup_dir.relative_to(root)}")
    for path in deleted:
        print(f"削除: {path.relative_to(root)}")

    if failed:
        print("\n削除できなかったファイルがあります。")
        print("SphereBrainや関連画面を閉じて、もう一度実行してください。")
        for path, error in failed:
            print(f"  - {path.relative_to(root)}: {error}")
        return 1

    print("\n初期化が完了しました。次回起動時に新しいデータが作成されます。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nキャンセルしました。")
        raise SystemExit(130)
