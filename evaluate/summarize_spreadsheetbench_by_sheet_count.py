"""
Summarize SpreadsheetBench evaluation results by worksheet/image count.

The default SpreadsheetBench summary reports all/sheet/cell metrics. This
script keeps those metrics and further splits them by the number of worksheets:
1 worksheet, 2 worksheets, and 3+ worksheets.

Example:
# 加overwrite选项可以覆盖已有的summary文件
# results_root可以是直接的实验组路径，也可以是上级路径

python evaluate/summarize_spreadsheetbench_by_sheet_count.py \
    --results_root outs/spreadsheetbench_verified_400/Qwen3.5-9B/pot_markdown_40ktoken
python evaluate/summarize_spreadsheetbench_by_sheet_count.py \
    --results_root outs/spreadsheetbench_verified_400/Qwen3.5-9B/pot_excel_1_image_40ktoken
python evaluate/summarize_spreadsheetbench_by_sheet_count.py \
    --results_root outs/spreadsheetbench_verified_400/Qwen3.5-9B/pot_markdown+excel_1_image_40ktoken
        
python evaluate/summarize_spreadsheetbench_by_sheet_count.py \
    --results_root outs/spreadsheetbench_verified_400/Qwen3.5-9B
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import openpyxl


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_GLOB = "*_eval.json"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sheet_count_bucket(count: Optional[int]) -> str:
    if count == 1:
        return "1_sheet"
    if count == 2:
        return "2_sheets"
    if count is not None and count >= 3:
        return "3plus_sheets"
    return "unknown"


def count_workbook_sheets(xlsx_path: Optional[str]) -> Optional[int]:
    if not xlsx_path:
        return None

    path = Path(xlsx_path)
    if not path.exists():
        return None

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            return len(wb.sheetnames)
        finally:
            wb.close()
    except Exception:
        return None


def infer_sheet_count(row: Dict[str, Any]) -> Optional[int]:
    metadata = row.get("table_metadata") or {}
    image_paths = metadata.get("image_paths") or []

    # 图像实验中每个工作表对应一张图，因此优先使用 image_paths 的数量。
    if image_paths:
        return len(image_paths)

    # 文本实验没有 image_paths，用原始 workbook 的 sheet 数作为分桶依据。
    return count_workbook_sheets(metadata.get("xlsx_path"))


def new_score_lists() -> Dict[str, List[float]]:
    return {
        "soft_all": [],
        "hard_all": [],
        "soft_sheet": [],
        "hard_sheet": [],
        "soft_cell": [],
        "hard_cell": [],
    }


def add_row(score_lists: Dict[str, List[float]], row: Dict[str, Any]) -> None:
    soft = safe_float(row.get("total_soft_restriction"))
    hard = safe_float(row.get("total_hard_restriction"))
    instruction_type = str(row.get("instruction_type") or "")
    is_sheet = "Sheet" in instruction_type

    score_lists["soft_all"].append(soft)
    score_lists["hard_all"].append(hard)
    if is_sheet:
        score_lists["soft_sheet"].append(soft)
        score_lists["hard_sheet"].append(hard)
    else:
        score_lists["soft_cell"].append(soft)
        score_lists["hard_cell"].append(hard)


def average_scores(score_lists: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in score_lists.items():
        out[key] = round(sum(values) / len(values), 4) if values else None
    out["num_samples"] = len(score_lists["soft_all"])
    out["num_sheet_samples"] = len(score_lists["soft_sheet"])
    out["num_cell_samples"] = len(score_lists["soft_cell"])
    return out


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "1_sheet": new_score_lists(),
        "2_sheets": new_score_lists(),
        "3plus_sheets": new_score_lists(),
        "unknown": new_score_lists(),
    }
    overall = new_score_lists()
    bucket_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"format_valid": 0, "execution_success": 0})

    for row in rows:
        bucket = sheet_count_bucket(infer_sheet_count(row))
        add_row(overall, row)
        add_row(buckets[bucket], row)

        if row.get("format_valid"):
            bucket_counts[bucket]["format_valid"] += 1
        if row.get("execution_success"):
            bucket_counts[bucket]["execution_success"] += 1

    summary = {
        "overall": average_scores(overall),
        "by_sheet_count": {},
    }

    for bucket, score_lists in buckets.items():
        scores = average_scores(score_lists)
        n = scores["num_samples"]
        counts = bucket_counts[bucket]
        scores["format_valid"] = round(counts["format_valid"] / n, 4) if n else None
        scores["execution_success"] = round(counts["execution_success"] / n, 4) if n else None
        summary["by_sheet_count"][bucket] = scores

    return summary


def find_eval_files(results_root: Path, eval_glob: str) -> List[Path]:
    if results_root.is_file():
        return [results_root]

    direct = sorted(results_root.glob(eval_glob))
    if direct:
        return [path for path in direct if path.is_file()]

    return sorted(path for path in results_root.rglob(eval_glob) if path.is_file())


def summarize_eval_file(eval_path: Path, output_name: str, overwrite: bool) -> Dict[str, Any]:
    output_path = eval_path.parent / output_name
    if output_path.exists() and not overwrite:
        print(f"WARNING: output already exists, skip: {output_path}", file=sys.stderr)
        return {
            "eval_path": str(eval_path),
            "output_path": str(output_path),
        }

    rows = load_json(eval_path)
    if not isinstance(rows, list):
        raise ValueError(f"{eval_path} is not a list-style eval JSON file.")

    summary = summarize_rows(rows)
    summary.update(
        {
            "eval_path": str(eval_path),
            "run_dir": str(eval_path.parent),
            "num_rows": len(rows),
        }
    )

    save_json(summary, output_path)

    return {
        "eval_path": str(eval_path),
        "output_path": str(output_path),
        "num_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SpreadsheetBench eval files by worksheet/image count."
    )
    parser.add_argument(
        "--results_root",
        required=True,
        help="A run directory containing spreadsheet_pot_eval.json, an eval JSON file, or a parent directory containing runs.",
    )
    parser.add_argument(
        "--output_name",
        default="spreadsheet_pot_accuracy_by_sheet_count.json",
        help="Output JSON filename written into each run directory.",
    )
    parser.add_argument(
        "--eval_glob",
        default=DEFAULT_EVAL_GLOB,
        help="Eval filename glob to summarize. Defaults to *_eval.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    if not root.is_absolute():
        root = REPO_DIR / root

    eval_files = find_eval_files(root, args.eval_glob)
    if not eval_files:
        raise SystemExit(f"No eval files found under {root}")

    outputs = []
    for eval_path in eval_files:
        outputs.append(summarize_eval_file(eval_path, args.output_name, args.overwrite))

    print(json.dumps({"num_eval_files": len(outputs), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
