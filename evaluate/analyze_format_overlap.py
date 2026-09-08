"""Analyze per-instance complementarity across SheetFlex input formats.

The conditional overlap follows FLEXTAF Figure 3 exactly:

    overlap[row, column] = |S_row intersect S_column| / |S_column|

where S_format is the set of instances correctly solved with that format. The
exclusive rate follows FLEXTAF Table 4:

    exclusive_rate[format] = |S_format minus union(other sets)| / |S_format|

The script additionally reports the distribution of the number of correct
formats, symmetric Jaccard overlap, and best-fixed/SheetFlex/oracle accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FORMATS = (
    "latex",
    "markdown",
    "json_cells",
    "json_rows",
    "image",
    "excel_1_image",
)

FORMAT_LABELS = {
    "latex": "LaTeX",
    "markdown": "Markdown",
    "json_cells": "JSON-Cell",
    "json_rows": "JSON-Row",
    "image": "Image",
    "excel_1_image": "Excel-Image",
}

FORMAT_PLOT_LABELS = {
    "latex": "LaTeX",
    "markdown": "Markdown",
    "json_cells": "JSON\nCell",
    "json_rows": "JSON\nRow",
    "image": "Image",
    "excel_1_image": "Excel\nImage",
}

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "realhit": {
        "display_name": "RealHiTBench",
        "default_root": REPO_DIR / "outs" / "realhitbench",
        "run_dir": "cot_{format}_100ktoken",
        "eval_file": "realhit_cot_eval.json",
        "aggregate_subdir": "realhit",
        "aggregate_file": "sheetflex_vote_eval.json",
        "task_field": "QuestionType",
        "correctness": "EM = 100",
    },
    "spreadsheet": {
        "display_name": "SpreadsheetBench",
        "default_root": REPO_DIR / "outs" / "spreadsheetbench_verified_400",
        "run_dir": "pot_{format}_40ktoken",
        "eval_file": "spreadsheet_pot_eval.json",
        "aggregate_subdir": "spreadsheet",
        "aggregate_file": "spreadsheet_pot_eval.json",
        "task_field": "instruction_type",
        "correctness": "total_hard_restriction = 1",
    },
}

MODEL_ORDER = (
    "Qwen3.5-9B",
    "Qwen3-VL-30B-A3B-Instruct",
    "gemma-3-12b-it",
    "gemma-4-12B-it",
    "gemma-4-26B-A4B-it",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_is_correct(dataset: str, row: Mapping[str, Any]) -> bool:
    if dataset == "realhit":
        value = safe_number((row.get("eval") or {}).get("EM"))
        return value is not None and math.isclose(value, 100.0, abs_tol=1e-9)
    if dataset == "spreadsheet":
        value = safe_number(row.get("total_hard_restriction"))
        return value is not None and math.isclose(value, 1.0, abs_tol=1e-9)
    raise ValueError(f"Unknown dataset: {dataset}")


def index_rows(rows: Any, path: Path) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Expected object at {path}[{position}]")
        sample_id = str(row.get("id"))
        if row.get("id") is None:
            raise ValueError(f"Missing id at {path}[{position}]")
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        indexed[sample_id] = row
    return indexed


def ordered_models(models: Iterable[str]) -> list[str]:
    model_set = set(models)
    known = [model for model in MODEL_ORDER if model in model_set]
    return known + sorted(model_set - set(known), key=str.casefold)


def discover_models(
    datasets: Sequence[str], roots: Mapping[str, Path], formats: Sequence[str]
) -> list[str]:
    available: set[str] | None = None
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        dataset_models = set()
        root = roots[dataset]
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset result root does not exist: {root}")
        for model_dir in root.iterdir():
            if not model_dir.is_dir():
                continue
            if all(
                (
                    model_dir
                    / config["run_dir"].format(format=format_name)
                    / config["eval_file"]
                ).is_file()
                for format_name in formats
            ):
                dataset_models.add(model_dir.name)
        available = dataset_models if available is None else available & dataset_models
    if not available:
        raise FileNotFoundError(
            "No model has complete results for every requested dataset and format."
        )
    return ordered_models(available)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def exact_mcnemar_p(rescued: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value for paired binary predictions."""
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(rescued, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{100.0 * value:.{digits}f}%"


def count_pct(count: int, denominator: int) -> str:
    return f"{count} ({pct(ratio(count, denominator))})"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def load_experiment(
    dataset: str,
    model: str,
    root: Path,
    aggregate_root: Path,
    formats: Sequence[str],
) -> tuple[
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, Mapping[str, Any]] | None,
    dict[str, str],
]:
    config = DATASET_CONFIGS[dataset]
    rows_by_format: dict[str, dict[str, Mapping[str, Any]]] = {}
    source_paths: dict[str, str] = {}

    for format_name in formats:
        path = (
            root
            / model
            / config["run_dir"].format(format=format_name)
            / config["eval_file"]
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset}/{model}/{format_name}: {path}")
        rows_by_format[format_name] = index_rows(load_json(path), path)
        source_paths[format_name] = str(path.resolve())

    base_format = formats[0]
    expected_ids = set(rows_by_format[base_format])
    for format_name in formats[1:]:
        actual_ids = set(rows_by_format[format_name])
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)[:5]
            extra = sorted(actual_ids - expected_ids)[:5]
            raise ValueError(
                f"ID mismatch for {dataset}/{model}/{format_name}: "
                f"missing={missing}, extra={extra}"
            )

    aggregate_path = (
        aggregate_root
        / model
        / config["aggregate_subdir"]
        / config["aggregate_file"]
    )
    aggregate_rows = None
    if aggregate_path.is_file():
        aggregate_rows = index_rows(load_json(aggregate_path), aggregate_path)
        aggregate_ids = set(aggregate_rows)
        if aggregate_ids != expected_ids:
            missing = sorted(expected_ids - aggregate_ids)[:5]
            extra = sorted(aggregate_ids - expected_ids)[:5]
            raise ValueError(
                f"Aggregate ID mismatch for {dataset}/{model}: "
                f"missing={missing}, extra={extra}"
            )
        source_paths["sheetflex"] = str(aggregate_path.resolve())

    return rows_by_format, aggregate_rows, source_paths


def summarize_binary_vectors(
    *,
    dataset: str,
    model: str,
    rows_by_format: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aggregate_rows: Mapping[str, Mapping[str, Any]] | None,
    formats: Sequence[str],
    source_paths: Mapping[str, str],
) -> dict[str, Any]:
    sample_ids = list(rows_by_format[formats[0]])
    n_samples = len(sample_ids)
    correct_sets = {
        format_name: {
            sample_id
            for sample_id in sample_ids
            if row_is_correct(dataset, rows_by_format[format_name][sample_id])
        }
        for format_name in formats
    }
    union_correct = set().union(*correct_sets.values())
    intersection_correct = set(sample_ids).intersection(*correct_sets.values())
    correct_count_by_id = {
        sample_id: sum(sample_id in correct_sets[name] for name in formats)
        for sample_id in sample_ids
    }
    count_distribution = Counter(correct_count_by_id.values())

    unique_sets = {
        format_name: correct_sets[format_name]
        - set().union(
            *(correct_sets[other] for other in formats if other != format_name)
        )
        for format_name in formats
    }

    format_rows = []
    for format_name in formats:
        correct_count = len(correct_sets[format_name])
        unique_count = len(unique_sets[format_name])
        format_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "format": format_name,
                "format_label": FORMAT_LABELS.get(format_name, format_name),
                "num_samples": n_samples,
                "correct_count": correct_count,
                "accuracy": ratio(correct_count, n_samples),
                "unique_count": unique_count,
                "unique_rate_among_format_correct": ratio(unique_count, correct_count),
                "unique_rate_among_all": ratio(unique_count, n_samples),
                "unique_share_of_oracle_union": ratio(unique_count, len(union_correct)),
            }
        )

    pairwise_rows = []
    conditional_matrix = []
    jaccard_matrix = []
    for row_format in formats:
        conditional_row = []
        jaccard_row = []
        for column_format in formats:
            intersection = correct_sets[row_format] & correct_sets[column_format]
            union = correct_sets[row_format] | correct_sets[column_format]
            conditional_overlap = ratio(
                len(intersection), len(correct_sets[column_format])
            )
            jaccard = ratio(len(intersection), len(union))
            conditional_row.append(conditional_overlap)
            jaccard_row.append(jaccard)
            pairwise_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "row_format": row_format,
                    "column_format": column_format,
                    "row_correct_count": len(correct_sets[row_format]),
                    "column_correct_count": len(correct_sets[column_format]),
                    "intersection_count": len(intersection),
                    "conditional_overlap": conditional_overlap,
                    "jaccard": jaccard,
                }
            )
        conditional_matrix.append(conditional_row)
        jaccard_matrix.append(jaccard_row)

    task_field = DATASET_CONFIGS[dataset]["task_field"]
    per_sample_rows = []
    combination_counts: Counter[tuple[str, ...]] = Counter()
    for sample_id in sample_ids:
        correct_formats = tuple(
            format_name
            for format_name in formats
            if sample_id in correct_sets[format_name]
        )
        combination_counts[correct_formats] += 1
        source_row = rows_by_format[formats[0]][sample_id]
        per_sample = {
            "dataset": dataset,
            "model": model,
            "sample_id": sample_id,
            "task_type": source_row.get(task_field) or "Unknown",
            "correct_format_count": len(correct_formats),
            "correct_formats": "+".join(correct_formats) if correct_formats else "none",
            "oracle_correct": bool(correct_formats),
            "sheetflex_correct": (
                row_is_correct(dataset, aggregate_rows[sample_id])
                if aggregate_rows is not None
                else None
            ),
        }
        per_sample.update(
            {
                f"correct_{format_name}": sample_id in correct_sets[format_name]
                for format_name in formats
            }
        )
        per_sample_rows.append(per_sample)

    combination_rows = []
    for combination, count in sorted(
        combination_counts.items(), key=lambda item: (-item[1], len(item[0]), item[0])
    ):
        combination_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "correct_format_count": len(combination),
                "correct_formats": "+".join(combination) if combination else "none",
                "num_samples": count,
                "share_all": ratio(count, n_samples),
                "share_oracle_solvable": (
                    ratio(count, len(union_correct)) if combination else 0.0
                ),
            }
        )

    distribution_rows = []
    for correct_format_count in range(len(formats) + 1):
        count = count_distribution.get(correct_format_count, 0)
        distribution_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "correct_format_count": correct_format_count,
                "num_samples": count,
                "share_all": ratio(count, n_samples),
                "share_oracle_solvable": (
                    ratio(count, len(union_correct))
                    if correct_format_count > 0
                    else 0.0
                ),
            }
        )

    best_format_row = max(
        format_rows,
        key=lambda row: (row["correct_count"], -formats.index(row["format"])),
    )
    best_format = str(best_format_row["format"])
    best_set = correct_sets[best_format]
    aggregate_set = (
        {
            sample_id
            for sample_id in sample_ids
            if row_is_correct(dataset, aggregate_rows[sample_id])
        }
        if aggregate_rows is not None
        else None
    )
    aggregate_count = len(aggregate_set) if aggregate_set is not None else None
    aggregation_gain_count = (
        aggregate_count - len(best_set) if aggregate_count is not None else None
    )
    oracle_headroom_count = len(union_correct) - len(best_set)
    captured_headroom = (
        ratio(aggregation_gain_count, oracle_headroom_count)
        if aggregation_gain_count is not None
        else None
    )
    rescued = len(aggregate_set - best_set) if aggregate_set is not None else None
    regressed = len(best_set - aggregate_set) if aggregate_set is not None else None
    mcnemar_p = (
        exact_mcnemar_p(rescued, regressed)
        if rescued is not None and regressed is not None
        else None
    )

    summary = {
        "dataset": dataset,
        "dataset_display_name": DATASET_CONFIGS[dataset]["display_name"],
        "model": model,
        "num_samples": n_samples,
        "num_formats": len(formats),
        "correctness": DATASET_CONFIGS[dataset]["correctness"],
        "none_correct_count": count_distribution.get(0, 0),
        "none_correct_rate": ratio(count_distribution.get(0, 0), n_samples),
        "exactly_one_count": count_distribution.get(1, 0),
        "exactly_one_rate": ratio(count_distribution.get(1, 0), n_samples),
        "partial_2_to_f_minus_1_count": sum(
            count_distribution.get(k, 0) for k in range(2, len(formats))
        ),
        "partial_2_to_f_minus_1_rate": ratio(
            sum(count_distribution.get(k, 0) for k in range(2, len(formats))),
            n_samples,
        ),
        "all_formats_count": len(intersection_correct),
        "all_formats_rate": ratio(len(intersection_correct), n_samples),
        "oracle_union_count": len(union_correct),
        "oracle_union_accuracy": ratio(len(union_correct), n_samples),
        "best_fixed_format": best_format,
        "best_fixed_correct_count": len(best_set),
        "best_fixed_accuracy": ratio(len(best_set), n_samples),
        "sheetflex_correct_count": aggregate_count,
        "sheetflex_accuracy": ratio(aggregate_count, n_samples) if aggregate_count is not None else None,
        "aggregation_gain_count": aggregation_gain_count,
        "aggregation_gain": (
            ratio(aggregation_gain_count, n_samples)
            if aggregation_gain_count is not None
            else None
        ),
        "oracle_headroom_count": oracle_headroom_count,
        "oracle_headroom": ratio(oracle_headroom_count, n_samples),
        "oracle_headroom_captured": captured_headroom,
        "aggregation_rescued_vs_best_count": rescued,
        "aggregation_regressed_vs_best_count": regressed,
        "mcnemar_exact_p_vs_best": mcnemar_p,
        "source_paths": dict(source_paths),
    }

    return {
        "summary": summary,
        "format_rows": format_rows,
        "distribution_rows": distribution_rows,
        "combination_rows": combination_rows,
        "pairwise_rows": pairwise_rows,
        "per_sample_rows": per_sample_rows,
        "conditional_matrix": conditional_matrix,
        "jaccard_matrix": jaccard_matrix,
    }


def task_breakdown(
    per_sample_rows: Sequence[Mapping[str, Any]], formats: Sequence[str]
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        buckets[(str(row["dataset"]), str(row["model"]), str(row["task_type"]))].append(row)

    results = []
    for (dataset, model, task_type), rows in sorted(buckets.items()):
        n = len(rows)
        format_counts = {
            format_name: sum(bool(row[f"correct_{format_name}"]) for row in rows)
            for format_name in formats
        }
        best_format = max(
            formats,
            key=lambda name: (format_counts[name], -formats.index(name)),
        )
        best_count = format_counts[best_format]
        oracle_count = sum(bool(row["oracle_correct"]) for row in rows)
        sheetflex_values = [row["sheetflex_correct"] for row in rows]
        sheetflex_count = (
            sum(bool(value) for value in sheetflex_values)
            if all(value is not None for value in sheetflex_values)
            else None
        )
        results.append(
            {
                "dataset": dataset,
                "model": model,
                "task_type": task_type,
                "num_samples": n,
                "none_correct_count": sum(row["correct_format_count"] == 0 for row in rows),
                "exactly_one_count": sum(row["correct_format_count"] == 1 for row in rows),
                "partial_2_to_f_minus_1_count": sum(
                    2 <= row["correct_format_count"] < len(formats) for row in rows
                ),
                "all_formats_count": sum(
                    row["correct_format_count"] == len(formats) for row in rows
                ),
                "best_fixed_format": best_format,
                "best_fixed_accuracy": ratio(best_count, n),
                "sheetflex_accuracy": (
                    ratio(sheetflex_count, n) if sheetflex_count is not None else None
                ),
                "oracle_union_accuracy": ratio(oracle_count, n),
            }
        )
    return results


def build_report(
    analyses: Sequence[Mapping[str, Any]],
    formats: Sequence[str],
    output_dir: Path,
    main_model: str,
) -> str:
    summaries = [analysis["summary"] for analysis in analyses]
    lines = [
        "# SheetFlex Format Complementarity Analysis",
        "",
        "This report mirrors FLEXTAF Figure 3 and Table 4 for the six SheetFlex candidate representations.",
        "",
        "## Definitions",
        "",
        "- RealHiTBench correctness: exact match (`EM = 100`).",
        "- SpreadsheetBench correctness: every test restriction passes (`total_hard_restriction = 1`).",
        "- Conditional overlap at row `r`, column `c`: `|S_r intersect S_c| / |S_c|`. This is asymmetric and matches FLEXTAF Figure 3.",
        "- Exclusive rate for format `f`: instances solved only by `f`, divided by all instances solved by `f`. This matches FLEXTAF Table 4.",
        "- Oracle union: an instance is correct if at least one format is correct. It is an upper bound on selecting one existing candidate with gold knowledge.",
        "",
        "## Coverage And Aggregation",
        "",
        "| Dataset | Model | N | None | Exactly 1 | 2 to 5 | All 6 | Best fixed | SheetFlex | Oracle | Gain vs best | Rescue / regress | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        n = summary["num_samples"]
        lines.append(
            "| {dataset} | {model} | {n} | {none} | {one} | {partial} | {all_formats} | {best_name}: {best_acc} | {sheetflex} | {oracle} | {gain} | {rescued} / {regressed} | {p_value} |".format(
                dataset=summary["dataset_display_name"],
                model=summary["model"],
                n=n,
                none=count_pct(summary["none_correct_count"], n),
                one=count_pct(summary["exactly_one_count"], n),
                partial=count_pct(summary["partial_2_to_f_minus_1_count"], n),
                all_formats=count_pct(summary["all_formats_count"], n),
                best_name=FORMAT_LABELS.get(summary["best_fixed_format"], summary["best_fixed_format"]),
                best_acc=pct(summary["best_fixed_accuracy"]),
                sheetflex=pct(summary["sheetflex_accuracy"]),
                oracle=pct(summary["oracle_union_accuracy"]),
                gain=pct(summary["aggregation_gain"]),
                rescued=summary["aggregation_rescued_vs_best_count"],
                regressed=summary["aggregation_regressed_vs_best_count"],
                p_value=(
                    "N/A"
                    if summary["mcnemar_exact_p_vs_best"] is None
                    else f"{summary['mcnemar_exact_p_vs_best']:.3g}"
                ),
            )
        )

    lines.extend(
        [
            "",
            "`McNemar p` is the two-sided exact paired test using SheetFlex rescues and regressions relative to the observed best fixed format. Treat it as exploratory because that baseline is selected on the same evaluation set; a confirmatory test should preselect the baseline on a development split.",
            "",
            "## FLEXTAF-Style Exclusive Rates",
            "",
            "Each cell is `exclusive count / all correct count for that format (rate)`.",
            "",
            "| Dataset | Model | " + " | ".join(FORMAT_LABELS.get(name, name) for name in formats) + " |",
            "|---|---|" + "---:|" * len(formats),
        ]
    )
    for analysis in analyses:
        by_format = {row["format"]: row for row in analysis["format_rows"]}
        cells = []
        for format_name in formats:
            row = by_format[format_name]
            cells.append(
                f"{row['unique_count']} / {row['correct_count']} ({pct(row['unique_rate_among_format_correct'])})"
            )
        lines.append(
            f"| {analysis['summary']['dataset_display_name']} | {analysis['summary']['model']} | "
            + " | ".join(cells)
            + " |"
        )

    main_analyses = [
        analysis for analysis in analyses if analysis["summary"]["model"] == main_model
    ]
    if main_analyses:
        lines.extend(
            [
                "",
                f"## Conditional Overlap Matrices: {main_model}",
                "",
                "Rows are the formats that also solve an instance; columns are the conditioning correct sets. Values are asymmetric `P(row correct | column correct)`.",
            ]
        )
        labels = [FORMAT_LABELS.get(name, name) for name in formats]
        for analysis in main_analyses:
            lines.extend(
                [
                    "",
                    f"### {analysis['summary']['dataset_display_name']}",
                    "",
                    "| Also solved by / conditioned on | " + " | ".join(labels) + " |",
                    "|---|" + "---:|" * len(formats),
                ]
            )
            for format_name, matrix_row in zip(formats, analysis["conditional_matrix"]):
                values = ["N/A" if value is None else f"{value:.2f}" for value in matrix_row]
                lines.append(
                    f"| {FORMAT_LABELS.get(format_name, format_name)} | "
                    + " | ".join(values)
                    + " |"
                )

    lines.extend(["", "## Key Findings", ""])
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_dataset[str(summary["dataset"])].append(summary)
    finding_number = 1
    for dataset, dataset_summaries in by_dataset.items():
        n = sum(summary["num_samples"] for summary in dataset_summaries)
        one = sum(summary["exactly_one_count"] for summary in dataset_summaries)
        partial = sum(
            summary["partial_2_to_f_minus_1_count"] for summary in dataset_summaries
        )
        all_formats = sum(summary["all_formats_count"] for summary in dataset_summaries)
        best_macro = sum(summary["best_fixed_accuracy"] for summary in dataset_summaries) / len(dataset_summaries)
        oracle_macro = sum(summary["oracle_union_accuracy"] for summary in dataset_summaries) / len(dataset_summaries)
        sheetflex_values = [
            summary["sheetflex_accuracy"]
            for summary in dataset_summaries
            if summary["sheetflex_accuracy"] is not None
        ]
        sheetflex_macro = (
            sum(sheetflex_values) / len(sheetflex_values) if sheetflex_values else None
        )
        display = DATASET_CONFIGS[dataset]["display_name"]
        lines.append(
            f"{finding_number}. **{display}:** across {len(dataset_summaries)} models ({n} model-instance pairs), "
            f"{one} ({pct(ratio(one, n))}) pairs are solved by exactly one format, "
            f"{partial} ({pct(ratio(partial, n))}) by two to five formats, and "
            f"{all_formats} ({pct(ratio(all_formats, n))}) by all six."
        )
        finding_number += 1
        lines.append(
            f"{finding_number}. **{display} aggregation:** macro accuracy is {pct(sheetflex_macro)} for SheetFlex, "
            f"versus {pct(best_macro)} for the per-model best fixed format and {pct(oracle_macro)} for the oracle union."
        )
        finding_number += 1

    total_unique_by_format = {
        format_name: sum(
            row["unique_count"]
            for analysis in analyses
            for row in analysis["format_rows"]
            if row["format"] == format_name
        )
        for format_name in formats
    }
    unique_ranking = sorted(
        total_unique_by_format.items(), key=lambda item: (-item[1], formats.index(item[0]))
    )
    lines.append(
        f"{finding_number}. **Unique contributions:** pooled across all dataset-model evaluations, "
        + ", ".join(
            f"{FORMAT_LABELS.get(name, name)} contributes {count} exclusive cases"
            for name, count in unique_ranking
        )
        + "."
    )
    finding_number += 1
    lines.append(
        f"{finding_number}. **Interpretation:** unique and partial coverage establishes representation complementarity and oracle headroom. "
        "The SheetFlex-vs-best-fixed comparison measures whether the current aggregation rule converts that headroom into realized accuracy; complementarity alone is not proof that every aggregation rule helps."
    )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.csv`: best fixed, SheetFlex, oracle, and coverage headline numbers.",
            "- `correct_format_count_distribution.csv`: exact `k = 0..6` counts.",
            "- `format_exclusive_rates.csv`: FLEXTAF Table 4 analogue.",
            "- `pairwise_overlap.csv`: conditional overlap and symmetric Jaccard values.",
            "- `format_combination_distribution.csv`: counts for every exact subset of correct formats.",
            "- `per_sample_correct_formats.csv`: auditable sample-level correctness matrix.",
            "- `coverage_by_task_type.csv`: task-type breakdown.",
            "- `analysis.json`: full matrices, tables, source paths, and sample-level records.",
            "- `figures/`: paper-ready PNG and PDF plots.",
            "",
            f"Output directory: `{output_dir.resolve()}`",
            "",
        ]
    )
    return "\n".join(lines)


def import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib and numpy; rerun with --skip-plots otherwise."
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return plt, np


def save_figure(fig: Any, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def plot_heatmap(
    analysis: Mapping[str, Any], formats: Sequence[str], output_base: Path
) -> None:
    plt, np = import_plotting()
    matrix = np.array(analysis["conditional_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    labels = [FORMAT_PLOT_LABELS.get(name, name) for name in formats]
    ax.set_xticks(range(len(formats)), labels=labels)
    ax.set_yticks(range(len(formats)), labels=labels)
    ax.set_xlabel("Conditioning format (column)")
    ax.set_ylabel("Also solved by format (row)")
    summary = analysis["summary"]
    ax.set_title(
        f"Conditional Correct-Set Overlap: {summary['dataset_display_name']} / {summary['model']}"
    )
    for row_index in range(len(formats)):
        for column_index in range(len(formats)):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else "#17242b",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("P(row correct | column correct)")
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)


def plot_main_heatmaps(
    analyses: Sequence[Mapping[str, Any]],
    formats: Sequence[str],
    model: str,
    output_base: Path,
) -> None:
    selected = [analysis for analysis in analyses if analysis["summary"]["model"] == model]
    if not selected:
        return
    plt, np = import_plotting()
    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(6.2 * len(selected), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    labels = [FORMAT_PLOT_LABELS.get(name, name) for name in formats]
    image = None
    for ax, analysis in zip(axes[0], selected):
        matrix = np.array(analysis["conditional_matrix"], dtype=float)
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_xticks(range(len(formats)), labels=labels)
        ax.set_yticks(range(len(formats)), labels=labels)
        ax.set_xlabel("Conditioning format (column)")
        ax.set_ylabel("Also solved by format (row)")
        ax.set_title(analysis["summary"]["dataset_display_name"])
        for row_index in range(len(formats)):
            for column_index in range(len(formats)):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.58 else "#17242b",
                    fontsize=8,
                )
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes[0]), fraction=0.025, pad=0.06)
        colorbar.set_label("P(row correct | column correct)")
    fig.suptitle(f"Conditional Correct-Set Overlap / {model}", fontsize=12)
    save_figure(fig, output_base)
    plt.close(fig)


def plot_correct_count_distribution(
    analyses: Sequence[Mapping[str, Any]], formats: Sequence[str], output_base: Path
) -> None:
    plt, _ = import_plotting()
    dataset_order = [
        dataset for dataset in DATASET_CONFIGS if any(a["summary"]["dataset"] == dataset for a in analyses)
    ]
    colors = ("#176b87", "#bc4b51", "#2f855a", "#7b5ea7", "#c47f17")
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(6.2 * len(dataset_order), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], dataset_order):
        dataset_analyses = [a for a in analyses if a["summary"]["dataset"] == dataset]
        for model_index, analysis in enumerate(dataset_analyses):
            rows = analysis["distribution_rows"]
            ax.plot(
                [row["correct_format_count"] for row in rows],
                [100.0 * row["share_all"] for row in rows],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=analysis["summary"]["model"],
                color=colors[model_index % len(colors)],
            )
        ax.set_xticks(range(len(formats) + 1))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of correct formats")
        ax.set_ylabel("Instances (%)")
        ax.set_title(DATASET_CONFIGS[dataset]["display_name"])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=min(3, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output_base)
    plt.close(fig)


def plot_exclusive_rates(
    analyses: Sequence[Mapping[str, Any]], formats: Sequence[str], output_base: Path
) -> None:
    plt, np = import_plotting()
    dataset_order = [
        dataset for dataset in DATASET_CONFIGS if any(a["summary"]["dataset"] == dataset for a in analyses)
    ]
    colors = ("#176b87", "#bc4b51", "#2f855a", "#7b5ea7", "#c47f17")
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(6.5 * len(dataset_order), 4.3), squeeze=False)
    x = np.arange(len(formats))
    for ax, dataset in zip(axes[0], dataset_order):
        dataset_analyses = [a for a in analyses if a["summary"]["dataset"] == dataset]
        width = 0.78 / max(1, len(dataset_analyses))
        for model_index, analysis in enumerate(dataset_analyses):
            by_format = {row["format"]: row for row in analysis["format_rows"]}
            values = [
                100.0 * (by_format[name]["unique_rate_among_format_correct"] or 0.0)
                for name in formats
            ]
            offset = (model_index - (len(dataset_analyses) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=analysis["summary"]["model"],
                color=colors[model_index % len(colors)],
            )
        ax.set_xticks(x, labels=[FORMAT_PLOT_LABELS.get(name, name) for name in formats])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Exclusive among format-correct (%)")
        ax.set_title(DATASET_CONFIGS[dataset]["display_name"])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=min(3, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output_base)
    plt.close(fig)


def plot_accuracy_headroom(
    analyses: Sequence[Mapping[str, Any]], output_base: Path
) -> None:
    plt, np = import_plotting()
    dataset_order = [
        dataset for dataset in DATASET_CONFIGS if any(a["summary"]["dataset"] == dataset for a in analyses)
    ]
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(6.5 * len(dataset_order), 4.3), squeeze=False)
    colors = ("#5b6573", "#197278", "#c44536")
    labels = ("Best fixed", "SheetFlex", "Oracle union")
    for ax, dataset in zip(axes[0], dataset_order):
        summaries = [a["summary"] for a in analyses if a["summary"]["dataset"] == dataset]
        x = np.arange(len(summaries))
        width = 0.24
        series = (
            [100.0 * row["best_fixed_accuracy"] for row in summaries],
            [100.0 * (row["sheetflex_accuracy"] or 0.0) for row in summaries],
            [100.0 * row["oracle_union_accuracy"] for row in summaries],
        )
        for index, (label, values, color) in enumerate(zip(labels, series, colors)):
            ax.bar(x + (index - 1) * width, values, width=width, label=label, color=color)
        ax.set_xticks(x, labels=[row["model"] for row in summaries], rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Strict accuracy (%)")
        ax.set_title(DATASET_CONFIGS[dataset]["display_name"])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)
    plt.close(fig)


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SheetFlex per-instance format complementarity."
    )
    parser.add_argument(
        "--datasets",
        default="realhit,spreadsheet",
        help="Comma-separated dataset keys: realhit,spreadsheet.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated candidate formats in display/order priority.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated models. Defaults to complete models discovered in all selected datasets.",
    )
    parser.add_argument(
        "--realhit-root",
        type=Path,
        default=DATASET_CONFIGS["realhit"]["default_root"],
    )
    parser.add_argument(
        "--spreadsheet-root",
        type=Path,
        default=DATASET_CONFIGS["spreadsheet"]["default_root"],
    )
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        default=REPO_DIR / "outs" / "sheetflex_vote",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "analysis" / "format_overlap",
    )
    parser.add_argument(
        "--main-model",
        default="Qwen3.5-9B",
        help="Model used for the two-panel main conditional-overlap figure.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write tables and report without matplotlib figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_comma_list(args.datasets)
    formats = parse_comma_list(args.formats)
    invalid_datasets = sorted(set(datasets) - set(DATASET_CONFIGS))
    if invalid_datasets:
        raise ValueError(f"Unsupported dataset keys: {invalid_datasets}")
    if len(formats) < 2 or len(set(formats)) != len(formats):
        raise ValueError("--formats must contain at least two distinct names")

    roots = {
        "realhit": args.realhit_root.resolve(),
        "spreadsheet": args.spreadsheet_root.resolve(),
    }
    models = (
        ordered_models(parse_comma_list(args.models))
        if args.models
        else discover_models(datasets, roots, formats)
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for dataset in datasets:
        for model in models:
            rows_by_format, aggregate_rows, source_paths = load_experiment(
                dataset,
                model,
                roots[dataset],
                args.aggregate_root.resolve(),
                formats,
            )
            analyses.append(
                summarize_binary_vectors(
                    dataset=dataset,
                    model=model,
                    rows_by_format=rows_by_format,
                    aggregate_rows=aggregate_rows,
                    formats=formats,
                    source_paths=source_paths,
                )
            )

    summaries = [analysis["summary"] for analysis in analyses]
    format_rows = [row for analysis in analyses for row in analysis["format_rows"]]
    distribution_rows = [
        row for analysis in analyses for row in analysis["distribution_rows"]
    ]
    combination_rows = [
        row for analysis in analyses for row in analysis["combination_rows"]
    ]
    pairwise_rows = [row for analysis in analyses for row in analysis["pairwise_rows"]]
    per_sample_rows = [
        row for analysis in analyses for row in analysis["per_sample_rows"]
    ]
    task_rows = task_breakdown(per_sample_rows, formats)

    summary_csv_rows = [
        {key: value for key, value in summary.items() if key != "source_paths"}
        for summary in summaries
    ]
    save_csv(summary_csv_rows, output_dir / "summary.csv")
    save_csv(format_rows, output_dir / "format_exclusive_rates.csv")
    save_csv(
        distribution_rows, output_dir / "correct_format_count_distribution.csv"
    )
    save_csv(
        combination_rows, output_dir / "format_combination_distribution.csv"
    )
    save_csv(pairwise_rows, output_dir / "pairwise_overlap.csv")
    save_csv(per_sample_rows, output_dir / "per_sample_correct_formats.csv")
    save_csv(task_rows, output_dir / "coverage_by_task_type.csv")
    save_json(
        {
            "formats": formats,
            "models": models,
            "datasets": datasets,
            "analyses": analyses,
        },
        output_dir / "analysis.json",
    )

    report = build_report(analyses, formats, output_dir, args.main_model)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    if not args.skip_plots:
        figures_dir = output_dir / "figures"
        for analysis in analyses:
            summary = analysis["summary"]
            plot_heatmap(
                analysis,
                formats,
                figures_dir
                / "conditional_overlap"
                / f"{summary['dataset']}__{slug(summary['model'])}",
            )
        plot_main_heatmaps(
            analyses,
            formats,
            args.main_model,
            figures_dir / f"conditional_overlap__{slug(args.main_model)}",
        )
        plot_correct_count_distribution(
            analyses, formats, figures_dir / "correct_format_count_distribution"
        )
        plot_exclusive_rates(
            analyses, formats, figures_dir / "format_exclusive_rates"
        )
        plot_accuracy_headroom(
            analyses, figures_dir / "accuracy_headroom"
        )

    print(
        f"Analyzed {len(datasets)} datasets, {len(models)} models, and "
        f"{len(formats)} formats. Report: {output_dir / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()
