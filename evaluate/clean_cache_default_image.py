"""
Clean cached default_image table screenshots under outs/.

The script only targets runtime-rendered default_image caches in experiment
outputs. It does not touch dataset pre-rendered image directories.

Examples:
  # 先查看会删除哪些缓存，不实际删除
  python evaluate/clean_cache_default_image.py --results_root outs

  # 真正删除
  python evaluate/clean_cache_default_image.py --results_root outs --apply

  # 只清理某个模型或某个数据集目录
  python evaluate/clean_cache_default_image.py \
      --results_root outs/spreadsheetbench_verified_400/Qwen3.5-9B --apply
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REPO_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CacheTarget:
    path: Path
    file_count: int
    byte_count: int


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_DIR / p
    return p.resolve()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_image_files(path: Path) -> Iterable[Path]:
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def cache_stats(path: Path) -> CacheTarget:
    file_count = 0
    byte_count = 0
    for image_path in iter_image_files(path):
        file_count += 1
        byte_count += image_path.stat().st_size
    return CacheTarget(path=path, file_count=file_count, byte_count=byte_count)


def looks_like_default_image_experiment(path: Path) -> bool:
    # 只清理实验组路径名包含 default_image 的 table_images 缓存，避免误删 image/excel_1_image 预生成图片。
    return "default_image" in str(path)


def find_cache_targets(results_root: Path) -> List[CacheTarget]:
    outs_root = (REPO_DIR / "outs").resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"results_root does not exist: {results_root}")
    if not is_under(results_root, outs_root) and results_root != outs_root:
        raise ValueError(f"Refuse to clean outside outs/: {results_root}")

    targets: List[CacheTarget] = []
    for table_images_dir in results_root.rglob("table_images"):
        if not table_images_dir.is_dir():
            continue
        if not looks_like_default_image_experiment(table_images_dir):
            continue
        target = cache_stats(table_images_dir)
        if target.file_count > 0:
            targets.append(target)
    return sorted(targets, key=lambda item: str(item.path))


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TB"


def delete_target(target: CacheTarget) -> None:
    # 删除整个 table_images 目录；之后如果再次运行 default_image 实验，代码会重新生成缓存。
    shutil.rmtree(target.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean default_image table_images caches under outs/.")
    parser.add_argument(
        "--results_root",
        type=str,
        default="outs",
        help="Path under outs/ to scan. Can be outs, a dataset dir, a model dir, or one experiment dir.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete cache directories. Without this flag the script only prints a dry-run summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    targets = find_cache_targets(results_root)

    total_files = sum(item.file_count for item in targets)
    total_bytes = sum(item.byte_count for item in targets)
    mode = "APPLY" if args.apply else "DRY-RUN"

    print(f"[{mode}] results_root: {results_root}")
    print(f"Found {len(targets)} default_image table_images cache dir(s).")
    print(f"Total cached images: {total_files}")
    print(f"Total cache size: {format_size(total_bytes)}")

    for target in targets:
        rel = target.path.relative_to(REPO_DIR)
        print(f"- {rel} | {target.file_count} image(s), {format_size(target.byte_count)}")

    if not args.apply:
        print("\nDry run only. Add --apply to delete these cache directories.")
        return

    for target in targets:
        delete_target(target)
    print(f"\nDeleted {len(targets)} cache dir(s), {total_files} image(s), {format_size(total_bytes)}.")


if __name__ == "__main__":
    main()
