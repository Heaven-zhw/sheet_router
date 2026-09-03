"""Gold-free SheetFlex-vote aggregation and labeled evaluation for RealHiTBench."""

from collections import defaultdict
from typing import Any, Callable, Dict, Mapping, Sequence

from ..solver.metrics.qa_metrics import QAMetric, normalize_answer, process_decimal
from .common import (
    FORMAT_ORDER,
    TIE_BREAK_LOGPROB_FIELDS,
    break_tie,
    count_distribution,
    format_rank,
    has_valid_logprob,
    logprob_summary,
    max_score_items,
    safe_rate,
    validate_tie_break_logprob_field,
)


REALHIT_METRICS = ("F1", "EM", "ROUGE-L", "SacreBLEU")
SCORE_ORDER = (
    "Overall",
    "Fact Checking",
    "Numerical Reasoning",
    "Structure Comprehending",
)


def normalized_vote_answer(answer: Any) -> str:
    return normalize_answer(process_decimal(str(answer)))


def build_realhit_candidate_trace(
    candidate_id: str,
    record: Mapping[str, Any] | None,
    *,
    candidate_id_key: str = "format",
    structure_key: str | None = None,
    run_dir: str | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if record is None:
        source = None
        valid = False
        reason = "sample_id_missing_from_run_results"
    elif structure_key is not None:
        source = record.get(structure_key)
        valid = isinstance(source, Mapping)
        reason = None if valid else f"missing_{structure_key}"
    else:
        source = record
        valid = True
        reason = None

    source = source if isinstance(source, Mapping) else {}
    answer = source.get("model_answer")
    if valid and source.get("format_valid") is not True:
        valid = False
        reason = "format_valid_is_not_true"
    if valid and (answer is None or not str(answer).strip()):
        valid = False
        reason = "model_answer_is_empty"

    trace = {
        candidate_id_key: candidate_id,
        "source_run_dir": run_dir,
        "valid": valid,
        "invalid_reason": reason,
        "model_answer": answer,
        "normalized_answer": normalized_vote_answer(answer) if valid else None,
        "aggregation_score": None,
        "selected": False,
    }
    trace.update(trace_metadata or {})
    trace.update(logprob_summary(source))
    return trace


def aggregate_answer_vote(
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_id_key: str = "format",
    selected_id_field: str = "selected_format",
    group_ids_field: str = "formats",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    fallback_source: str = "format_order",
    logprob_field: str = "sequence_logprob_sum",
) -> Dict[str, Any]:
    validate_tie_break_logprob_field(logprob_field)
    candidates = [dict(candidate) for candidate in candidates]
    if rank_getter is None:
        rank_getter = lambda item: format_rank(item[candidate_id_key])
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if not valid_candidates:
        return {
            "format_valid": False,
            "valid_candidate_count": 0,
            "candidates": candidates,
            "answer_groups": [],
            "winning_normalized_answer": None,
            "winning_group_size": 0,
            selected_id_field: None,
            "selected_answer": "",
            "tie": False,
            "tied_group_count": 0,
            "tie_break_source": "not_applicable",
            "tie_break_reason": "all_candidates_invalid",
            "representative_selection_source": "not_applicable",
            "representative_selection_reason": "all_candidates_invalid",
            "tie_break_logprob_field": logprob_field,
        }

    grouped = {}
    for candidate in valid_candidates:
        grouped.setdefault(candidate["normalized_answer"], []).append(candidate)

    answer_groups = []
    for normalized_answer, members in grouped.items():
        group_logprobs = {
            field: (
                max(float(member[field]) for member in members)
                if all(has_valid_logprob(member, field) for member in members)
                else None
            )
            for field in TIE_BREAK_LOGPROB_FIELDS.values()
        }
        representative = min(members, key=rank_getter)
        group = {
            "normalized_answer": normalized_answer,
            group_ids_field: [member[candidate_id_key] for member in members],
            "size": len(members),
            "aggregation_score": float(len(members)),
            candidate_id_key: representative[candidate_id_key],
            "logprob_available": group_logprobs[logprob_field] is not None,
            **group_logprobs,
        }
        answer_groups.append(group)
        for member in members:
            member["aggregation_score"] = group["aggregation_score"]

    _, tied_groups = max_score_items(answer_groups)
    group_tie = len(tied_groups) > 1
    if group_tie:
        group_decision = break_tie(
            tied_groups,
            rank_getter=rank_getter,
            fallback_source=fallback_source,
            logprob_field=logprob_field,
        )
        winning_group = group_decision.selected
        tie_source = group_decision.source
        tie_reason = group_decision.reason
    else:
        winning_group = tied_groups[0]
        tie_source = "not_needed"
        tie_reason = "unique_largest_answer_group"

    winning_members = grouped[winning_group["normalized_answer"]]
    if len(winning_members) > 1:
        member_decision = break_tie(
            winning_members,
            rank_getter=rank_getter,
            fallback_source=fallback_source,
            logprob_field=logprob_field,
        )
        selected = member_decision.selected
        representative_source = member_decision.source
        representative_reason = member_decision.reason
    else:
        selected = winning_members[0]
        representative_source = "not_needed"
        representative_reason = "single_member_in_winning_group"
    selected["selected"] = True

    return {
        "format_valid": True,
        "valid_candidate_count": len(valid_candidates),
        "candidates": candidates,
        "answer_groups": answer_groups,
        "winning_normalized_answer": winning_group["normalized_answer"],
        "winning_group_size": winning_group["size"],
        selected_id_field: selected[candidate_id_key],
        "selected_answer": selected["model_answer"],
        "tie": group_tie,
        "tied_group_count": len(tied_groups),
        "tie_break_source": tie_source,
        "tie_break_reason": tie_reason,
        "representative_selection_source": representative_source,
        "representative_selection_reason": representative_reason,
        "tie_break_logprob_field": logprob_field,
    }


def aggregate_realhit_sample(
    sample_id: str,
    question_type: str,
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    run_dirs: Mapping[str, str] | None = None,
    format_order: Sequence[str] = FORMAT_ORDER,
    logprob_field: str = "sequence_logprob_sum",
) -> Dict[str, Any]:
    run_dirs = run_dirs or {}
    rank_map = {format_name: index for index, format_name in enumerate(format_order)}
    rank_getter = lambda item: rank_map[item["format"]]
    if question_type == "Structure Comprehending":
        reference_vote = aggregate_answer_vote(
            [
                build_realhit_candidate_trace(
                    format_name,
                    records_by_format.get(format_name),
                    structure_key="structure_reference_run",
                    run_dir=run_dirs.get(format_name),
                )
                for format_name in format_order
            ],
            rank_getter=rank_getter,
            logprob_field=logprob_field,
        )
        swap_vote = aggregate_answer_vote(
            [
                build_realhit_candidate_trace(
                    format_name,
                    records_by_format.get(format_name),
                    structure_key="structure_swap_run",
                    run_dir=run_dirs.get(format_name),
                )
                for format_name in format_order
            ],
            rank_getter=rank_getter,
            logprob_field=logprob_field,
        )
        valid_formats = {
            candidate["format"]
            for candidate in reference_vote["candidates"]
            if candidate["valid"]
        } & {
            candidate["format"]
            for candidate in swap_vote["candidates"]
            if candidate["valid"]
        }
        return {
            "id": str(sample_id),
            "QuestionType": question_type,
            "format_valid": reference_vote["format_valid"] and swap_vote["format_valid"],
            "model_answer": swap_vote["selected_answer"],
            "selected_format": {
                "reference": reference_vote["selected_format"],
                "swap": swap_vote["selected_format"],
            },
            "valid_candidate_count": len(valid_formats),
            "structure_reference_answer": reference_vote["selected_answer"],
            "structure_swap_answer": swap_vote["selected_answer"],
            "trace": {
                "aggregation": "equal_weight_answer_group_vote",
                "structure_reference_vote": reference_vote,
                "structure_swap_vote": swap_vote,
            },
        }

    vote = aggregate_answer_vote(
        [
            build_realhit_candidate_trace(
                format_name,
                records_by_format.get(format_name),
                run_dir=run_dirs.get(format_name),
            )
            for format_name in format_order
        ],
        rank_getter=rank_getter,
        logprob_field=logprob_field,
    )
    return {
        "id": str(sample_id),
        "QuestionType": question_type,
        "format_valid": vote["format_valid"],
        "model_answer": vote["selected_answer"],
        "selected_format": vote["selected_format"],
        "valid_candidate_count": vote["valid_candidate_count"],
        "trace": {
            "aggregation": "equal_weight_answer_group_vote",
            "answer_vote": vote,
        },
    }


def evaluate_realhit_vote(
    aggregated_rows: Sequence[Mapping[str, Any]],
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    metric: QAMetric | None = None,
    *,
    selected_id_field: str = "selected_format",
) -> tuple[list[dict], dict]:
    metric = metric or QAMetric()
    eval_rows = []
    score_lists = defaultdict(lambda: defaultdict(list))

    for aggregate in aggregated_rows:
        item = dataset_by_id[str(aggregate["id"])]
        question_type = item.get("QuestionType", "Unknown")
        scores = {name: None for name in REALHIT_METRICS}
        reference = item.get("ProcessedAnswer", "")
        prediction = aggregate.get("model_answer", "")
        if aggregate.get("format_valid"):
            if question_type == "Structure Comprehending":
                reference = aggregate.get("structure_reference_answer", "")
                prediction = aggregate.get("structure_swap_answer", "")
            if reference and prediction:
                scores.update(metric.compute([reference], [prediction]))

        eval_row = {
            "id": str(aggregate["id"]),
            "Question": item.get("Question"),
            "QuestionType": question_type,
            "SubQType": item.get("SubQType"),
            "FileName": item.get("FileName"),
            "format_valid": bool(aggregate.get("format_valid")),
            selected_id_field: aggregate.get(selected_id_field),
            "eval": {
                "Model_Answer": prediction,
                "Reference_Answer": reference,
                **scores,
            },
        }
        eval_rows.append(eval_row)
        for bucket in ("Overall", question_type):
            for metric_name in REALHIT_METRICS:
                score_lists[bucket][metric_name].append(scores[metric_name])
            score_lists[bucket]["FormatValid"].append(
                100.0 if aggregate.get("format_valid") else 0.0
            )

    score = {}
    for bucket in SCORE_ORDER:
        if bucket not in score_lists:
            continue
        score[bucket] = {}
        for metric_name, values in score_lists[bucket].items():
            numeric = [value for value in values if value is not None]
            score[bucket][metric_name] = sum(numeric) / len(values) if values else 0.0
    return eval_rows, score


def realhit_diagnostics(aggregated_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vote_events = []
    effective_counts = []
    all_invalid = 0
    for row in aggregated_rows:
        effective_counts.append(row["valid_candidate_count"])
        if not row["format_valid"]:
            all_invalid += 1
        trace = row["trace"]
        if row["QuestionType"] == "Structure Comprehending":
            vote_events.extend(
                [
                    trace["structure_reference_vote"],
                    trace["structure_swap_vote"],
                ]
            )
        else:
            vote_events.append(trace["answer_vote"])

    answer_group_ties = [event for event in vote_events if event["tie"]]
    representative_ties = [
        event for event in vote_events if event["winning_group_size"] > 1
    ]
    tie_break_sources = [event["tie_break_source"] for event in answer_group_ties]
    tie_break_sources.extend(
        event["representative_selection_source"] for event in representative_ties
    )
    logprob_breaks = [source for source in tie_break_sources if source == "logprob"]
    format_fallbacks = [
        source for source in tie_break_sources if source == "format_order"
    ]
    selected_formats = [
        event["selected_format"]
        for event in vote_events
        if event["selected_format"] is not None
    ]
    all_group_sizes = [
        group["size"] for event in vote_events for group in event["answer_groups"]
    ]
    winning_group_sizes = [
        event["winning_group_size"]
        for event in vote_events
        if event["format_valid"]
    ]
    return {
        "num_samples": len(aggregated_rows),
        "num_vote_events": len(vote_events),
        "valid_candidate_count_distribution": count_distribution(effective_counts),
        "all_candidates_invalid_samples": all_invalid,
        "tie_events": len(answer_group_ties),
        "tie_rate": safe_rate(len(answer_group_ties), len(vote_events)),
        "answer_group_tie_events": len(answer_group_ties),
        "representative_tie_events": len(representative_ties),
        "total_tie_break_events": len(tie_break_sources),
        "logprob_tie_breaks": len(logprob_breaks),
        "logprob_tie_break_coverage": safe_rate(
            len(logprob_breaks), len(tie_break_sources)
        ),
        "format_order_fallbacks": len(format_fallbacks),
        "selected_format_distribution": count_distribution(selected_formats),
        "all_answer_group_size_distribution": count_distribution(all_group_sizes),
        "winning_answer_group_size_distribution": count_distribution(
            winning_group_sizes
        ),
    }
