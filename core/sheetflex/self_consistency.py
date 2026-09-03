"""Self-Consistency adapters over the shared SheetFlex-vote algorithms."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import (
    count_distribution,
    index_rows_by_id,
    load_result_rows,
    numeric_summary,
    safe_rate,
)
from .realhit import aggregate_answer_vote, build_realhit_candidate_trace
from .spreadsheet import (
    aggregate_spreadsheet_candidates,
    candidate_output_path,
)


NUM_SELF_CONSISTENCY_SAMPLES = 6


class SelfConsistencyError(ValueError):
    pass


def load_self_consistency_manifest(path, expected_dataset: str) -> Dict[str, Any]:
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SelfConsistencyError(f"Failed to load manifest {path}: {exc}") from exc
    validate_self_consistency_manifest(manifest, expected_dataset)
    manifest = dict(manifest)
    manifest["manifest_path"] = str(path)
    manifest["runs"] = [dict(run) for run in manifest["runs"]]
    for run in manifest["runs"]:
        run_dir = Path(run["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (path.parent / run_dir).resolve()
        run["run_dir"] = str(run_dir)
    return manifest


def validate_self_consistency_manifest(
    manifest: Mapping[str, Any], expected_dataset: str
) -> None:
    if manifest.get("method") != "self_consistency":
        raise SelfConsistencyError("Manifest method must be 'self_consistency'")
    if manifest.get("stage") != "candidate_generation":
        raise SelfConsistencyError("Manifest stage must be 'candidate_generation'")
    if manifest.get("dataset") != expected_dataset:
        raise SelfConsistencyError(
            f"Manifest dataset mismatch: expected {expected_dataset!r}, "
            f"got {manifest.get('dataset')!r}"
        )
    if manifest.get("num_samples") != NUM_SELF_CONSISTENCY_SAMPLES:
        raise SelfConsistencyError(
            f"Self-Consistency requires num_samples={NUM_SELF_CONSISTENCY_SAMPLES}"
        )
    if manifest.get("base_seed") != 42:
        raise SelfConsistencyError("Self-Consistency requires base_seed=42")
    try:
        temperature = float(manifest.get("temperature"))
        top_p = float(manifest.get("top_p"))
    except (TypeError, ValueError) as exc:
        raise SelfConsistencyError(
            "Manifest temperature and top_p must be numeric"
        ) from exc
    if not math.isclose(temperature, 0.1):
        raise SelfConsistencyError("Self-Consistency requires temperature=0.1")
    if not math.isclose(top_p, 1.0):
        raise SelfConsistencyError("Self-Consistency requires top_p=1.0")
    if manifest.get("save_logprobs") is not True:
        raise SelfConsistencyError("Self-Consistency requires save_logprobs=true")
    if not isinstance(manifest.get("table_format"), str) or not manifest["table_format"]:
        raise SelfConsistencyError("Manifest table_format must be explicit")
    if not isinstance(manifest.get("model_name"), str) or not manifest["model_name"]:
        raise SelfConsistencyError("Manifest model_name is missing")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != NUM_SELF_CONSISTENCY_SAMPLES:
        raise SelfConsistencyError("Manifest must contain exactly six candidate runs")
    expected_indices = list(range(NUM_SELF_CONSISTENCY_SAMPLES))
    if [run.get("sample_index") for run in runs] != expected_indices:
        raise SelfConsistencyError("Manifest runs must be ordered by sample_index 0..5")
    expected_ids = [f"sample_{index}" for index in expected_indices]
    if [run.get("candidate_id") for run in runs] != expected_ids:
        raise SelfConsistencyError(
            "Manifest candidate_id values must be sample_0 ... sample_5"
        )
    expected_seeds = [manifest.get("base_seed") + index for index in expected_indices]
    if [run.get("seed") for run in runs] != expected_seeds:
        raise SelfConsistencyError(
            "Manifest seeds must equal base_seed + sample_index"
        )
    run_dirs = [run.get("run_dir") for run in runs]
    if any(not isinstance(run_dir, str) or not run_dir for run_dir in run_dirs):
        raise SelfConsistencyError("Every manifest run must provide run_dir")
    if len(set(run_dirs)) != NUM_SELF_CONSISTENCY_SAMPLES:
        raise SelfConsistencyError("Manifest run_dir values must be unique")

    shared_fields = (
        "model_name",
        "dataset",
        "table_format",
        "temperature",
        "top_p",
        "save_logprobs",
    )
    for run in runs:
        for field in shared_fields:
            if run.get(field) != manifest.get(field):
                raise SelfConsistencyError(
                    f"Run {run.get('candidate_id')} has inconsistent {field}"
                )


def manifest_identity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "method": manifest.get("method"),
        "stage": manifest.get("stage"),
        "dataset": manifest.get("dataset"),
        "model_name": manifest.get("model_name"),
        "table_format": manifest.get("table_format"),
        "num_samples": manifest.get("num_samples"),
        "temperature": manifest.get("temperature"),
        "top_p": manifest.get("top_p"),
        "base_seed": manifest.get("base_seed"),
        "runs": [
            {
                "candidate_id": run.get("candidate_id"),
                "sample_index": run.get("sample_index"),
                "seed": run.get("seed"),
                "run_dir": str(Path(run.get("run_dir", "")).resolve()),
            }
            for run in manifest.get("runs", [])
        ],
    }


def load_indexed_candidate_runs(
    manifest: Mapping[str, Any], result_filename: str
) -> Dict[str, Dict[str, dict]]:
    indexed_runs = {}
    expected_ids = None
    for run in manifest["runs"]:
        result_path = Path(run["run_dir"]) / result_filename
        if not result_path.is_file():
            raise SelfConsistencyError(
                f"Candidate result file missing for {run['candidate_id']}: {result_path}"
            )
        indexed = index_rows_by_id(
            load_result_rows(result_path), source=str(result_path)
        )
        for sample_id, row in indexed.items():
            for field in ("candidate_id", "sample_index", "seed"):
                if row.get(field) != run[field]:
                    raise SelfConsistencyError(
                        f"Candidate metadata mismatch in {result_path} for sample "
                        f"{sample_id}: expected {field}={run[field]!r}, "
                        f"got {row.get(field)!r}"
                    )
        row_ids = set(indexed)
        if expected_ids is None:
            expected_ids = row_ids
        elif row_ids != expected_ids:
            missing = sorted(expected_ids - row_ids)
            extra = sorted(row_ids - expected_ids)
            raise SelfConsistencyError(
                f"Candidate sample ID mismatch in {result_path}: "
                f"missing={missing}, extra={extra}"
            )
        indexed_runs[run["candidate_id"]] = indexed
    return indexed_runs


def _source_record(
    record: Mapping[str, Any] | None, structure_key: str | None = None
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    if structure_key is None:
        return record
    source = record.get(structure_key)
    return source if isinstance(source, Mapping) else {}


def generation_attempt_count(
    record: Mapping[str, Any] | None, structure_key: str | None = None
) -> int:
    attempts = _source_record(record, structure_key).get("attempts")
    return len(attempts) if isinstance(attempts, list) else 0


def _rank_map(manifest: Mapping[str, Any]) -> Dict[str, int]:
    return {
        run["candidate_id"]: run["sample_index"] for run in manifest["runs"]
    }


def _selected_candidate(vote: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next(
        (candidate for candidate in vote["candidates"] if candidate["selected"]),
        None,
    )


def _realhit_vote(
    manifest: Mapping[str, Any],
    records_by_candidate: Mapping[str, Mapping[str, Any] | None],
    structure_key: str | None = None,
    logprob_field: str = "sequence_logprob_sum",
) -> Dict[str, Any]:
    rank_map = _rank_map(manifest)
    candidates = []
    for run in manifest["runs"]:
        record = records_by_candidate.get(run["candidate_id"])
        candidates.append(
            build_realhit_candidate_trace(
                run["candidate_id"],
                record,
                candidate_id_key="candidate_id",
                structure_key=structure_key,
                run_dir=run["run_dir"],
                trace_metadata={
                    "table_format": manifest["table_format"],
                    "sample_index": run["sample_index"],
                    "seed": run["seed"],
                    "generation_attempt_count": generation_attempt_count(
                        record, structure_key
                    ),
                },
            )
        )
    vote = aggregate_answer_vote(
        candidates,
        candidate_id_key="candidate_id",
        selected_id_field="selected_candidate_id",
        group_ids_field="candidate_ids",
        rank_getter=lambda item: rank_map[item["candidate_id"]],
        fallback_source="sample_index",
        logprob_field=logprob_field,
    )
    for candidate in vote["candidates"]:
        candidate["candidate_valid"] = candidate["valid"]
    return vote


def aggregate_self_consistency_realhit_sample(
    item: Mapping[str, Any],
    records_by_candidate: Mapping[str, Mapping[str, Any] | None],
    manifest: Mapping[str, Any],
    logprob_field: str = "sequence_logprob_sum",
) -> Dict[str, Any]:
    sample_id = str(item["id"])
    question_type = item.get("QuestionType", "Unknown")
    common = {
        "id": sample_id,
        "QuestionType": question_type,
        "table_format": manifest["table_format"],
        "num_samples": manifest["num_samples"],
        "base_seed": manifest["base_seed"],
    }
    if question_type == "Structure Comprehending":
        reference_vote = _realhit_vote(
            manifest,
            records_by_candidate,
            "structure_reference_run",
            logprob_field,
        )
        swap_vote = _realhit_vote(
            manifest,
            records_by_candidate,
            "structure_swap_run",
            logprob_field,
        )
        reference_selected = _selected_candidate(reference_vote)
        swap_selected = _selected_candidate(swap_vote)
        valid_reference = {
            candidate["candidate_id"]
            for candidate in reference_vote["candidates"]
            if candidate["valid"]
        }
        valid_swap = {
            candidate["candidate_id"]
            for candidate in swap_vote["candidates"]
            if candidate["valid"]
        }
        return {
            **common,
            "format_valid": reference_vote["format_valid"]
            and swap_vote["format_valid"],
            "model_answer": swap_vote["selected_answer"],
            "selected_candidate_id": {
                "reference": reference_vote["selected_candidate_id"],
                "swap": swap_vote["selected_candidate_id"],
            },
            "selected_sample_index": {
                "reference": (
                    reference_selected.get("sample_index")
                    if reference_selected
                    else None
                ),
                "swap": swap_selected.get("sample_index") if swap_selected else None,
            },
            "selected_seed": {
                "reference": reference_selected.get("seed") if reference_selected else None,
                "swap": swap_selected.get("seed") if swap_selected else None,
            },
            "valid_candidate_count": len(valid_reference & valid_swap),
            "structure_reference_answer": reference_vote["selected_answer"],
            "structure_swap_answer": swap_vote["selected_answer"],
            "generation_attempts_total": sum(
                candidate["generation_attempt_count"]
                for candidate in reference_vote["candidates"]
                + swap_vote["candidates"]
            ),
            "unique_normalized_answer_count": {
                "reference": len(reference_vote["answer_groups"]),
                "swap": len(swap_vote["answer_groups"]),
            },
            "six_candidates_identical": (
                reference_vote["valid_candidate_count"] == manifest["num_samples"]
                and swap_vote["valid_candidate_count"] == manifest["num_samples"]
                and len(reference_vote["answer_groups"]) == 1
                and len(swap_vote["answer_groups"]) == 1
            ),
            "tie_occurred": {
                "reference": reference_vote["tie"],
                "swap": swap_vote["tie"],
            },
            "tie_break_source": {
                "reference": reference_vote["tie_break_source"],
                "swap": swap_vote["tie_break_source"],
            },
            "trace": {
                "aggregation": "equal_weight_answer_group_vote",
                "structure_reference_vote": reference_vote,
                "structure_swap_vote": swap_vote,
            },
        }

    vote = _realhit_vote(
        manifest, records_by_candidate, logprob_field=logprob_field
    )
    selected = _selected_candidate(vote)
    return {
        **common,
        "format_valid": vote["format_valid"],
        "model_answer": vote["selected_answer"],
        "selected_candidate_id": vote["selected_candidate_id"],
        "selected_sample_index": selected.get("sample_index") if selected else None,
        "selected_seed": selected.get("seed") if selected else None,
        "valid_candidate_count": vote["valid_candidate_count"],
        "generation_attempts_total": sum(
            candidate["generation_attempt_count"] for candidate in vote["candidates"]
        ),
        "unique_normalized_answer_count": len(vote["answer_groups"]),
        "six_candidates_identical": (
            vote["valid_candidate_count"] == manifest["num_samples"]
            and len(vote["answer_groups"]) == 1
        ),
        "tie_occurred": vote["tie"],
        "tie_break_source": vote["tie_break_source"],
        "trace": {
            "aggregation": "equal_weight_answer_group_vote",
            "answer_vote": vote,
        },
    }


def aggregate_self_consistency_spreadsheet_sample(
    item: Mapping[str, Any],
    records_by_candidate: Mapping[str, Mapping[str, Any] | None],
    manifest: Mapping[str, Any],
    input_path: Path,
    logprob_field: str = "sequence_logprob_sum",
) -> Dict[str, Any]:
    sample_id = str(item["id"])
    rank_map = _rank_map(manifest)
    candidate_specs = []
    for run in manifest["runs"]:
        candidate_id = run["candidate_id"]
        record = records_by_candidate.get(candidate_id)
        candidate_specs.append(
            {
                "candidate_id": candidate_id,
                "record": record,
                "run_dir": Path(run["run_dir"]),
                "output_path": candidate_output_path(
                    sample_id,
                    candidate_id,
                    record,
                    {candidate_id: Path(run["run_dir"])},
                ),
                "trace_metadata": {
                    "table_format": manifest["table_format"],
                    "sample_index": run["sample_index"],
                    "seed": run["seed"],
                    "generation_attempt_count": generation_attempt_count(record),
                },
            }
        )
    aggregate = aggregate_spreadsheet_candidates(
        item,
        candidate_specs,
        input_path,
        candidate_id_key="candidate_id",
        selected_id_field="selected_candidate_id",
        rank_getter=lambda candidate: rank_map[candidate["candidate_id"]],
        fallback_source="sample_index",
        logprob_field=logprob_field,
    )
    for candidate in aggregate["trace"]["candidates"]:
        candidate["candidate_valid"] = candidate["valid"]
    selected = _selected_candidate(aggregate["trace"])
    valid_hashes = {
        candidate["region_hash"]
        for candidate in aggregate["trace"]["candidates"]
        if candidate["valid"]
    }
    aggregate.update(
        {
            "table_format": manifest["table_format"],
            "num_samples": manifest["num_samples"],
            "base_seed": manifest["base_seed"],
            "selected_sample_index": (
                selected.get("sample_index") if selected else None
            ),
            "selected_seed": selected.get("seed") if selected else None,
            "generation_attempts_total": sum(
                candidate["generation_attempt_count"]
                for candidate in aggregate["trace"]["candidates"]
            ),
            "unique_region_hash_count": len(valid_hashes),
            "six_candidates_identical": (
                aggregate["valid_candidate_count"] == manifest["num_samples"]
                and len(valid_hashes) == 1
            ),
            "tie_occurred": aggregate["trace"]["tie"],
            "tie_break_source": aggregate["trace"]["tie_break_source"],
        }
    )
    return aggregate


def _realhit_vote_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if row["QuestionType"] == "Structure Comprehending":
        return [
            row["trace"]["structure_reference_vote"],
            row["trace"]["structure_swap_vote"],
        ]
    return [row["trace"]["answer_vote"]]


def self_consistency_realhit_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    vote_events = [event for row in rows for event in _realhit_vote_events(row)]
    answer_group_ties = [event for event in vote_events if event["tie"]]
    representative_ties = [
        event for event in vote_events if event["winning_group_size"] > 1
    ]
    tie_sources = [event["tie_break_source"] for event in answer_group_ties]
    tie_sources.extend(
        event["representative_selection_source"] for event in representative_ties
    )
    selected_indices = [
        candidate["sample_index"]
        for event in vote_events
        for candidate in event["candidates"]
        if candidate["selected"]
    ]
    unique_counts = [len(event["answer_groups"]) for event in vote_events]
    identical_events = [
        event
        for event in vote_events
        if event["valid_candidate_count"] == NUM_SELF_CONSISTENCY_SAMPLES
        and len(event["answer_groups"]) == 1
    ]
    invalid_reasons = [
        candidate["invalid_reason"]
        for event in vote_events
        for candidate in event["candidates"]
        if not candidate["valid"]
    ]
    attempts = [row["generation_attempts_total"] for row in rows]
    return {
        "num_samples": len(rows),
        "num_vote_events": len(vote_events),
        "valid_candidate_count_distribution": count_distribution(
            row["valid_candidate_count"] for row in rows
        ),
        "all_candidates_invalid_samples": sum(
            1 for row in rows if not row["format_valid"]
        ),
        "answer_group_tie_events": len(answer_group_ties),
        "tie_rate": safe_rate(len(answer_group_ties), len(vote_events)),
        "total_tie_break_events": len(tie_sources),
        "logprob_tie_breaks": tie_sources.count("logprob"),
        "logprob_tie_break_coverage": safe_rate(
            tie_sources.count("logprob"), len(tie_sources)
        ),
        "sample_index_fallbacks": tie_sources.count("sample_index"),
        "selected_sample_index_distribution": count_distribution(selected_indices),
        "unique_normalized_answer_count_distribution": count_distribution(
            unique_counts
        ),
        "six_candidates_identical_vote_events": len(identical_events),
        "six_candidates_identical_vote_event_rate": safe_rate(
            len(identical_events), len(vote_events)
        ),
        "six_candidates_identical_samples": sum(
            1 for row in rows if row["six_candidates_identical"]
        ),
        "six_candidates_identical_sample_rate": safe_rate(
            sum(1 for row in rows if row["six_candidates_identical"]), len(rows)
        ),
        "generation_attempts_total": sum(attempts),
        "generation_attempts_per_sample": numeric_summary(attempts),
        "generation_attempts_distribution": count_distribution(attempts),
        "invalid_candidate_reason_distribution": count_distribution(invalid_reasons),
    }


def self_consistency_spreadsheet_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ties = [row for row in rows if row["trace"]["tie"]]
    selected_indices = [
        row["selected_sample_index"]
        for row in rows
        if row["selected_sample_index"] is not None
    ]
    pairwise = [
        row["trace"]["average_pairwise_region_agreement"]
        for row in rows
        if row["trace"]["average_pairwise_region_agreement"] is not None
    ]
    medoid_scores = [
        row["trace"]["medoid_score"]
        for row in rows
        if row["trace"]["medoid_score"] is not None
    ]
    attempts = [row["generation_attempts_total"] for row in rows]
    invalid_reasons = [
        candidate["invalid_reason"]
        for row in rows
        for candidate in row["trace"]["candidates"]
        if not candidate["valid"]
    ]
    unique_hash_counts = [row["unique_region_hash_count"] for row in rows]
    identical_count = sum(1 for row in rows if row["six_candidates_identical"])
    tie_sources = [row["trace"]["tie_break_source"] for row in ties]
    return {
        "num_samples": len(rows),
        "valid_candidate_count_distribution": count_distribution(
            row["valid_candidate_count"] for row in rows
        ),
        "all_candidates_invalid_samples": sum(
            1 for row in rows if not row["format_valid"]
        ),
        "tie_samples": len(ties),
        "tie_rate": safe_rate(len(ties), len(rows)),
        "logprob_tie_breaks": tie_sources.count("logprob"),
        "logprob_tie_break_coverage": safe_rate(
            tie_sources.count("logprob"), len(ties)
        ),
        "sample_index_fallbacks": tie_sources.count("sample_index"),
        "selected_sample_index_distribution": count_distribution(selected_indices),
        "unique_region_hash_count_distribution": count_distribution(
            unique_hash_counts
        ),
        "average_pairwise_region_agreement": numeric_summary(pairwise),
        "medoid_score_summary": numeric_summary(medoid_scores),
        "six_candidates_identical_samples": identical_count,
        "six_candidates_identical_sample_rate": safe_rate(
            identical_count, len(rows)
        ),
        "generation_attempts_total": sum(attempts),
        "generation_attempts_per_sample": numeric_summary(attempts),
        "generation_attempts_distribution": count_distribution(attempts),
        "invalid_candidate_reason_distribution": count_distribution(invalid_reasons),
    }
