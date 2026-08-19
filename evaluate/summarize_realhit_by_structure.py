"""
Summarize RealHiTBench results by task type and table structure.

The original RealHiTBench score file reports metrics for three question types.
This script additionally reports:
- All_FC_NR_SC: Fact Checking + Numerical Reasoning + Structure Comprehending
- Metrics grouped by CompStrucCata
- Metrics grouped by the paper-level Complex Structures

Example:

python evaluate/summarize_realhit_by_structure.py \
    --results_root outs/realhitbench/Qwen3.5-9B/cot_latex

python evaluate/summarize_realhit_by_structure.py \
    --results_root outs/realhitbench/Qwen3.5-9B

"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_DIR = Path(__file__).resolve().parents[1]

METRICS = ("F1", "EM", "ROUGE-L", "SacreBLEU")
TASK_TYPES = (
    "Fact Checking",
    "Numerical Reasoning",
    "Structure Comprehending",
    "All_FC_NR_SC",
)

COMP_STRUCTURE_ORDER = (
    "ColumnHeaderMerge",
    "MultiColumnClassified",
    "SingleRowClassified",
    "ContentCompound",
    "ExternalSupply",
    "StructureCompound",
    "BackgroundColor",
)

COMP_TO_COMPLEX = {
    "ColumnHeaderMerge": "Hierarchical Column Header",
    "MultiColumnClassified": "Hierarchical Row Header",
    "SingleRowClassified": "Hierarchical Row Header",
    "ContentCompound": "Nested Sub-Tables",
    "ExternalSupply": "Miscellaneous",
    "StructureCompound": "Multi-Table Join",
    "BackgroundColor": "Miscellaneous",
}

COMPLEX_STRUCTURE_ORDER = (
    "Hierarchical Column Header",
    "Hierarchical Row Header",
    "Nested Sub-Tables",
    "Multi-Table Join",
    "Miscellaneous",
)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def row_task_type(row: Dict[str, Any]) -> str:
    return str(row.get("QuestionType") or "Unknown")


def row_comp_structure(row: Dict[str, Any]) -> str:
    return str(row.get("CompStrucCata") or "Unknown")


def row_complex_structure(row: Dict[str, Any]) -> str:
    return COMP_TO_COMPLEX.get(row_comp_structure(row), "Unknown")


def new_metric_lists() -> Dict[str, List[float]]:
    return {metric: [] for metric in METRICS} | {"FormatValid": []}


def add_metric_row(metric_lists: Dict[str, List[float]], row: Dict[str, Any]) -> None:
    eval_scores = row.get("eval") or {}
    for metric in METRICS:
        value = eval_scores.get(metric)
        if value is not None:
            metric_lists[metric].append(safe_float(value))
        else:
            # 与 realhit_cot_score.json 保持一致：format invalid 时该项通常为 None，按 0 计入整体均值。
            metric_lists[metric].append(0.0)
    metric_lists["FormatValid"].append(100.0 if row.get("format_valid") else 0.0)


def average_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in (*METRICS, "FormatValid"):
        values = metric_lists[metric]
        out[metric] = sum(values) / len(values) if values else None
    out["num_samples"] = len(metric_lists["FormatValid"])
    return out


def make_task_bucket() -> Dict[str, Dict[str, List[float]]]:
    return {task_type: new_metric_lists() for task_type in TASK_TYPES}


def add_to_task_bucket(bucket: Dict[str, Dict[str, List[float]]], row: Dict[str, Any]) -> None:
    task_type = row_task_type(row)
    if task_type in bucket:
        add_metric_row(bucket[task_type], row)
    if task_type in TASK_TYPES[:-1]:
        add_metric_row(bucket["All_FC_NR_SC"], row)


def average_task_bucket(bucket: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    return {task_type: average_metric_lists(metric_lists) for task_type, metric_lists in bucket.items()}


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    overall_by_task = make_task_bucket()
    by_comp = {name: make_task_bucket() for name in COMP_STRUCTURE_ORDER}
    by_comp["Unknown"] = make_task_bucket()
    by_complex = {name: make_task_bucket() for name in COMPLEX_STRUCTURE_ORDER}
    by_complex["Unknown"] = make_task_bucket()

    for row in rows:
        comp = row_comp_structure(row)
        complex_name = row_complex_structure(row)

        if comp not in by_comp:
            comp = "Unknown"
        if complex_name not in by_complex:
            complex_name = "Unknown"

        add_to_task_bucket(overall_by_task, row)
        add_to_task_bucket(by_comp[comp], row)
        add_to_task_bucket(by_complex[complex_name], row)

    return {
        "overall_by_task": average_task_bucket(overall_by_task),
        "by_CompStrucCata": {
            name: average_task_bucket(bucket) for name, bucket in by_comp.items()
        },
        "by_ComplexStructures": {
            name: average_task_bucket(bucket) for name, bucket in by_complex.items()
        },
        "CompStrucCata_to_ComplexStructures": COMP_TO_COMPLEX,
    }


def choose_result_file(run_dir: Path) -> Path:
    # realhit_cot.jsonl 包含 CompStrucCata；eval.json 通常不包含，所以优先使用 jsonl。
    candidates = (
        run_dir / "realhit_cot.jsonl",
        run_dir / "realhit_cot.partial.jsonl",
        run_dir / "realhit_cot_eval.json",
        run_dir / "realhit_cot_eval.partial.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No RealHiT result file found in {run_dir}")


def load_result_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a list-style result file.")
    return rows


def discover_run_dirs(results_root: Path) -> List[Path]:
    if results_root.is_file():
        return [results_root.parent]

    try:
        choose_result_file(results_root)
        return [results_root]
    except FileNotFoundError:
        pass

    run_dirs = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            choose_result_file(path)
            run_dirs.append(path)
        except FileNotFoundError:
            continue
    return run_dirs


def summarize_run(run_dir: Path, output_name: str, overwrite: bool) -> Dict[str, Any]:
    result_path = choose_result_file(run_dir)
    output_path = run_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"WARNING: output already exists, skip: {output_path}", file=sys.stderr)
        return {
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "output_path": str(output_path),
        }

    rows = load_result_rows(result_path)
    summary = summarize_rows(rows)
    summary.update(
        {
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "num_rows": len(rows),
        }
    )

    save_json(summary, output_path)
    return {
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "output_path": str(output_path),
        "num_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RealHiTBench results by CompStrucCata and Complex Structures."
    )
    parser.add_argument(
        "--results_root",
        required=True,
        help="A single RealHiT run directory, a result file, or a parent directory containing multiple runs.",
    )
    parser.add_argument(
        "--output_name",
        default="realhit_cot_score_by_structure.json",
        help="Output JSON filename written into each run directory.",
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

    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        raise SystemExit(f"No RealHiT run directories found under {root}")

    outputs = [summarize_run(run_dir, args.output_name, args.overwrite) for run_dir in run_dirs]
    print(json.dumps({"num_runs": len(outputs), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
