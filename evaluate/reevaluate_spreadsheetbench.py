"""
Re-evaluate SpreadsheetBench output workbooks.

Default usage for the current verified_400 Qwen3.5-9B results:
    cd sheet_router
    python evaluate/reevaluate_spreadsheetbench.py

This script does not send model requests. It compares generated output Excel
files in each pot_*/spreadsheet directory against the dataset golden files.
By default it writes new *_recalc.json files so existing evaluation files are
left untouched.
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.eval.spreadsheet import compare_workbooks  # noqa: E402


SPLIT_CONFIGS = {
    "all_912": {
        "root": "dataset/spreadsheetbench/all_data_912_v0.1",
        "answer_suffix": "answer",
        "num_test_cases": 3,
    },
    "verified_400": {
        "root": "dataset/spreadsheetbench/spreadsheetbench_verified_400",
        "answer_suffix": "golden",
        "num_test_cases": 1,
    },
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def average_scores(score_lists):
    return {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in score_lists.items()
    }


def add_scores(score_lists, result):
    is_sheet = "Sheet" in str(result.get("instruction_type", ""))
    soft = result.get("total_soft_restriction", 0.0)
    hard = result.get("total_hard_restriction", 0.0)
    score_lists["soft_all"].append(soft)
    score_lists["hard_all"].append(hard)
    score_lists["soft_sheet" if is_sheet else "soft_cell"].append(soft)
    score_lists["hard_sheet" if is_sheet else "hard_cell"].append(hard)


def get_dataset(split_name):
    split_config = SPLIT_CONFIGS[split_name]
    dataset_root = REPO_DIR / split_config["root"]
    dataset_path = dataset_root / "dataset.json"
    data = load_json(dataset_path)
    return data, dataset_root, split_config


def discover_run_dirs(results_root, run_names):
    results_root = Path(results_root)
    if run_names:
        candidates = [results_root / name for name in run_names]
    else:
        candidates = sorted(path for path in results_root.glob("pot_*") if path.is_dir())

    run_dirs = []
    for path in candidates:
        spreadsheet_dir = path if path.name == "spreadsheet" else path / "spreadsheet"
        if spreadsheet_dir.is_dir():
            run_dirs.append(path.parent if path.name == "spreadsheet" else path)
        else:
            print(f"Skip {path}: spreadsheet directory not found.")
    return run_dirs


def find_gold_file(real_dir, idx, sample_id, answer_suffix):
    expected = real_dir / f"{idx}_{sample_id}_{answer_suffix}.xlsx"
    if expected.exists():
        return expected

    # Some verified_400 files have small filename inconsistencies. Fall back to
    # the only matching golden/answer file in that sample directory.
    candidates = sorted(real_dir.glob(f"{idx}_*_{answer_suffix}.xls*"))
    if len(candidates) == 1:
        return candidates[0]

    candidates = sorted(real_dir.glob(f"*_{answer_suffix}.xls*"))
    if len(candidates) == 1:
        return candidates[0]

    return expected


def load_previous_eval(run_dir):
    previous = {}
    for filename in ("spreadsheet_pot_eval.json", "spreadsheet_pot_eval.partial.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            rows = load_json(path)
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")
            continue
        if isinstance(rows, list):
            previous = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
            break
    return previous


def compare_one_sample(item, dataset_root, split_config, spreadsheet_dir, previous_eval):
    sample_id = str(item["id"])
    real_dir = dataset_root / item.get("spreadsheet_path", os.path.join("spreadsheet", sample_id))

    messages = []
    results = []
    for idx in range(1, split_config["num_test_cases"] + 1):
        gt_file = find_gold_file(real_dir, idx, sample_id, split_config["answer_suffix"])
        proc_file = spreadsheet_dir / f"{idx}_{sample_id}_output.xlsx"

        if not gt_file.exists():
            ok = False
            message = f"Ground truth file not found: {gt_file}"
        else:
            try:
                ok, message = compare_workbooks(
                    str(gt_file),
                    str(proc_file),
                    item.get("instruction_type", ""),
                    item.get("answer_position", ""),
                )
            except Exception:
                ok = False
                message = traceback.format_exc()

        results.append(int(bool(ok)))
        messages.append(message)

    prev = previous_eval.get(sample_id, {})
    entry = {
        "id": sample_id,
        "instruction": item.get("instruction"),
        "spreadsheet_path": item.get("spreadsheet_path"),
        "instruction_type": item.get("instruction_type"),
        "answer_position": item.get("answer_position"),
        "answer_sheet": item.get("answer_sheet"),
        "execution_success": prev.get("execution_success"),
        "format_valid": prev.get("format_valid"),
        "error": prev.get("error"),
        "test_case_results": results,
        "test_case_messages": messages,
        "total_soft_restriction": sum(results) / len(results) if results else 0.0,
        "total_hard_restriction": 1.0 if results and all(results) else 0.0,
        "table_metadata": prev.get("table_metadata"),
    }
    return entry


def output_paths(run_dir, tag, overwrite):
    if overwrite:
        return run_dir / "spreadsheet_pot_eval.json", run_dir / "spreadsheet_pot_accuracy.json"
    suffix = f"_{tag}" if tag else ""
    return run_dir / f"spreadsheet_pot_eval{suffix}.json", run_dir / f"spreadsheet_pot_accuracy{suffix}.json"


def evaluate_run(run_dir, data, dataset_root, split_config, args):
    spreadsheet_dir = run_dir / "spreadsheet"
    previous_eval = load_previous_eval(run_dir)
    eval_results = []
    score_lists = defaultdict(list)

    progress = tqdm(data, desc=f"Re-evaluating {run_dir.name}", leave=False)
    for item in progress:
        result = compare_one_sample(item, dataset_root, split_config, spreadsheet_dir, previous_eval)
        eval_results.append(result)
        add_scores(score_lists, result)

    scores = average_scores(score_lists)
    eval_path, accuracy_path = output_paths(run_dir, args.output_tag, args.overwrite)
    save_json(eval_results, eval_path)
    save_json(scores, accuracy_path)

    return {
        "run": run_dir.name,
        "spreadsheet_dir": str(spreadsheet_dir),
        "eval_path": str(eval_path),
        "accuracy_path": str(accuracy_path),
        "scores": scores,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate SpreadsheetBench generated Excel outputs.")
    parser.add_argument(
        "--results_root",
        default=str(REPO_DIR / "outs/spreadsheetbench_verified_400/Qwen3.5-9B"),
        help="Directory containing pot_* result folders.",
    )
    parser.add_argument(
        "--data_split",
        default="verified_400",
        choices=sorted(SPLIT_CONFIGS),
        help="SpreadsheetBench split used by the result directory.",
    )
    parser.add_argument(
        "--run_names",
        default=None,
        help="Comma-separated result folders to evaluate, e.g. pot_markdown,pot_excel_1_image. "
        "If omitted, all pot_* folders under results_root are evaluated.",
    )
    parser.add_argument(
        "--output_tag",
        default="recalc",
        help="Tag appended to new output files when --overwrite is not used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite spreadsheet_pot_eval.json and spreadsheet_pot_accuracy.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data, dataset_root, split_config = get_dataset(args.data_split)
    run_dirs = discover_run_dirs(args.results_root, split_csv(args.run_names))
    if not run_dirs:
        raise SystemExit(f"No result folders found under {args.results_root}")

    summaries = []
    for run_dir in run_dirs:
        summaries.append(evaluate_run(run_dir, data, dataset_root, split_config, args))

    summary_path = Path(args.results_root) / f"spreadsheet_pot_reeval_summary_{args.output_tag}.json"
    save_json(summaries, summary_path)

    print("\nRe-evaluation summary:")
    for summary in summaries:
        scores = summary["scores"]
        print(
            f"- {summary['run']}: "
            f"soft_all={scores.get('soft_all', 0):.4f}, "
            f"hard_all={scores.get('hard_all', 0):.4f}, "
            f"eval={summary['eval_path']}"
        )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
