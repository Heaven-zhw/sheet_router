"""Common, gold-free utilities for SheetFlex aggregation."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


FORMAT_ORDER = (
    "latex",
    "markdown",
    "json_cells",
    "json_rows",
    "image",
    "excel_1_image",
)
FORMAT_RANK = {name: index for index, name in enumerate(FORMAT_ORDER)}
SCORE_REL_TOL = 1e-12
SCORE_ABS_TOL = 1e-12


class SheetFlexError(ValueError):
    pass


@dataclass(frozen=True)
class TieDecision:
    selected: Mapping[str, Any]
    source: str
    all_logprobs_available: bool
    reason: str


def scores_tied(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=SCORE_REL_TOL,
        abs_tol=SCORE_ABS_TOL,
    )


def format_rank(format_name: str) -> int:
    try:
        return FORMAT_RANK[format_name]
    except KeyError as exc:
        raise SheetFlexError(f"Unknown SheetFlex format: {format_name!r}") from exc


def load_run_map(path) -> Dict[str, Path]:
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SheetFlexError(f"Failed to load run-map JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SheetFlexError("Run-map JSON must be an object mapping format to run_dir")

    missing = [name for name in FORMAT_ORDER if name not in payload]
    extra = sorted(set(payload) - set(FORMAT_ORDER))
    if missing or extra:
        raise SheetFlexError(
            f"Run-map formats mismatch; missing={missing or []}, extra={extra or []}"
        )

    run_map = {}
    for format_name in FORMAT_ORDER:
        value = payload[format_name]
        if not isinstance(value, str) or not value.strip():
            raise SheetFlexError(f"Invalid run_dir for {format_name}: {value!r}")
        run_dir = Path(value)
        if not run_dir.is_absolute():
            run_dir = (path.parent / run_dir).resolve()
        if not run_dir.is_dir():
            raise SheetFlexError(f"Run directory not found for {format_name}: {run_dir}")
        run_map[format_name] = run_dir
    return run_map


def load_result_rows(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise SheetFlexError(f"Full result file not found: {path}")
    try:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SheetFlexError(f"Failed to load result file {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise SheetFlexError(f"Result file must contain a list/JSONL rows: {path}")
    return rows


def index_rows_by_id(rows: Iterable[dict], source: str = "results") -> Dict[str, dict]:
    indexed = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SheetFlexError(f"Non-object row at {source}[{row_index}]")
        if row.get("id") is None:
            raise SheetFlexError(f"Missing id at {source}[{row_index}]")
        sample_id = str(row["id"])
        if sample_id in indexed:
            raise SheetFlexError(f"Duplicate id {sample_id!r} in {source}")
        indexed[sample_id] = row
    return indexed


def load_indexed_runs(
    run_map: Mapping[str, Path], result_filename: str
) -> Dict[str, Dict[str, dict]]:
    return {
        format_name: index_rows_by_id(
            load_result_rows(run_map[format_name] / result_filename),
            source=str(run_map[format_name] / result_filename),
        )
        for format_name in FORMAT_ORDER
    }


def logprob_summary(record: Mapping[str, Any] | None) -> Dict[str, Any]:
    record = record or {}
    value = record.get("sequence_logprob_sum")
    available = (
        record.get("logprob_available") is True
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
    if available:
        return {
            "logprob_available": True,
            "sequence_logprob_sum": float(value),
            "sequence_logprob_mean": record.get("sequence_logprob_mean"),
            "sequence_token_count": record.get("sequence_token_count"),
            "logprob_unavailable_reason": None,
        }
    return {
        "logprob_available": False,
        "sequence_logprob_sum": None,
        "sequence_logprob_mean": None,
        "sequence_token_count": record.get("sequence_token_count", 0),
        "logprob_unavailable_reason": record.get("logprob_unavailable_reason")
        or "Candidate result has no valid cumulative sequence logprob.",
    }


def break_tie(
    tied_items: Sequence[Mapping[str, Any]],
    *,
    format_getter: Callable[[Mapping[str, Any]], str] = lambda item: item["format"],
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    fallback_source: str = "format_order",
) -> TieDecision:
    if not tied_items:
        raise SheetFlexError("Cannot break an empty tie")

    if rank_getter is None:
        rank_getter = lambda item: format_rank(format_getter(item))

    all_available = all(
        item.get("logprob_available") is True
        and isinstance(item.get("sequence_logprob_sum"), (int, float))
        and not isinstance(item.get("sequence_logprob_sum"), bool)
        and math.isfinite(item["sequence_logprob_sum"])
        for item in tied_items
    )
    if all_available:
        best_logprob = max(float(item["sequence_logprob_sum"]) for item in tied_items)
        best = [
            item
            for item in tied_items
            if float(item["sequence_logprob_sum"]) == best_logprob
        ]
        if len(best) == 1:
            return TieDecision(best[0], "logprob", True, "highest_sequence_logprob_sum")
        selected = min(best, key=rank_getter)
        return TieDecision(
            selected,
            fallback_source,
            True,
            "equal_sequence_logprob_sum",
        )

    selected = min(tied_items, key=rank_getter)
    return TieDecision(
        selected,
        fallback_source,
        False,
        "missing_logprob_in_tied_set",
    )


def max_score_items(
    items: Sequence[Mapping[str, Any]], score_key: str = "aggregation_score"
) -> tuple[float, list[Mapping[str, Any]]]:
    if not items:
        raise SheetFlexError("Cannot select a score from an empty item list")
    best_score = max(float(item[score_key]) for item in items)
    tied = [item for item in items if scores_tied(item[score_key], best_score)]
    return best_score, tied


def count_distribution(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def prepare_output_dir(path) -> Path:
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise SheetFlexError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_jsonl(rows: Iterable[Mapping[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
