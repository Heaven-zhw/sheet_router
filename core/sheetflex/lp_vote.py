"""Gold-free sequence-mean weighted aggregation for SheetFlex-LPVote."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from .common import (
    FORMAT_ORDER,
    RECOMMEND_FORMAT_ORDER,
    SheetFlexError,
    count_distribution,
    has_valid_logprob,
    max_score_items,
    numeric_summary,
    safe_rate,
)
from .realhit import aggregate_answer_vote, build_realhit_candidate_trace
from .spreadsheet import (
    aggregate_spreadsheet_sample,
    build_similarity_matrix,
    select_spreadsheet_medoid,
)


LOGPROB_FIELD = "sequence_logprob_mean"
MISSING_LOGPROB_POLICIES = ("vote", "error")
NEAR_EQUAL_WEIGHT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class LPWeightResult:
    raw_weights: tuple[float, ...]
    weights: tuple[float, ...]
    fallback: bool
    fallback_reason: str | None
    valid_count: int
    complete_count: int
    missing_candidate_ids: tuple[str, ...]


def _candidate_label(candidate: Mapping[str, Any], index: int) -> str:
    return str(
        candidate.get("format", candidate.get("candidate_id", f"candidate_{index}"))
    )


def validate_lp_weight_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise SheetFlexError("lp_weight_strength must be a finite number >= 0")
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise SheetFlexError("lp_weight_strength must be a finite number >= 0")
    return strength


def compute_lp_weights(
    candidates: Sequence[Mapping[str, Any]],
    strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    *,
    forced_fallback_reason: str | None = None,
) -> LPWeightResult:
    """Compute stable softmax weights over valid candidates only."""
    strength = validate_lp_weight_strength(strength)
    if missing_logprob_policy not in MISSING_LOGPROB_POLICIES:
        raise SheetFlexError(
            "missing_logprob_policy must be one of "
            f"{MISSING_LOGPROB_POLICIES}, got {missing_logprob_policy!r}"
        )

    valid_indices = [
        index for index, candidate in enumerate(candidates) if candidate.get("valid")
    ]
    complete_indices = [
        index
        for index in valid_indices
        if has_valid_logprob(candidates[index], LOGPROB_FIELD)
    ]
    missing_indices = [index for index in valid_indices if index not in complete_indices]
    missing_ids = tuple(
        _candidate_label(candidates[index], index) for index in missing_indices
    )
    size = len(candidates)

    if not valid_indices:
        return LPWeightResult(
            raw_weights=(0.0,) * size,
            weights=(0.0,) * size,
            fallback=False,
            fallback_reason=None,
            valid_count=0,
            complete_count=0,
            missing_candidate_ids=missing_ids,
        )

    # A lone valid candidate needs no confidence comparison.
    if len(valid_indices) == 1 and forced_fallback_reason is None:
        raw = [0.0] * size
        weights = [0.0] * size
        raw[valid_indices[0]] = 1.0
        weights[valid_indices[0]] = 1.0
        return LPWeightResult(
            raw_weights=tuple(raw),
            weights=tuple(weights),
            fallback=False,
            fallback_reason=None,
            valid_count=1,
            complete_count=len(complete_indices),
            missing_candidate_ids=missing_ids,
        )

    missing_reason = None
    if missing_indices:
        missing_reason = (
            f"missing_or_invalid_{LOGPROB_FIELD}: " + ", ".join(missing_ids)
        )
        if missing_logprob_policy == "error":
            raise SheetFlexError(missing_reason)

    fallback_reason = forced_fallback_reason or missing_reason
    if fallback_reason is not None:
        equal_weight = 1.0 / len(valid_indices)
        raw = [1.0 if index in valid_indices else 0.0 for index in range(size)]
        weights = [
            equal_weight if index in valid_indices else 0.0 for index in range(size)
        ]
        return LPWeightResult(
            raw_weights=tuple(raw),
            weights=tuple(weights),
            fallback=True,
            fallback_reason=fallback_reason,
            valid_count=len(valid_indices),
            complete_count=len(complete_indices),
            missing_candidate_ids=missing_ids,
        )

    means = [float(candidates[index][LOGPROB_FIELD]) for index in valid_indices]
    max_logprob = max(means)
    raw = [0.0] * size
    if strength == 0.0:
        for index in valid_indices:
            raw[index] = 1.0
    else:
        for index in valid_indices:
            raw[index] = math.exp(
                strength
                * (float(candidates[index][LOGPROB_FIELD]) - max_logprob)
            )
    denominator = sum(raw)
    weights = [value / denominator for value in raw]
    return LPWeightResult(
        raw_weights=tuple(raw),
        weights=tuple(weights),
        fallback=False,
        fallback_reason=None,
        valid_count=len(valid_indices),
        complete_count=len(complete_indices),
        missing_candidate_ids=missing_ids,
    )


def _with_lp_weights(
    candidates: Sequence[Mapping[str, Any]], result: LPWeightResult
) -> list[dict]:
    weighted = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["lp_raw_weight"] = result.raw_weights[index]
        item["lp_weight"] = result.weights[index]
        weighted.append(item)
    return weighted


def _recommend_rank(item: Mapping[str, Any]) -> int:
    return RECOMMEND_FORMAT_ORDER.index(str(item["format"]))


def aggregate_answer_lp_vote(
    candidates: Sequence[Mapping[str, Any]],
    *,
    strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    candidate_id_key: str = "format",
    selected_id_field: str = "selected_format",
    group_ids_field: str = "formats",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    tie_break_order: str = "recommend",
    forced_fallback_reason: str | None = None,
) -> Dict[str, Any]:
    """Weight normalized RealHiT answer groups by sequence logprob mean."""
    rank_getter = rank_getter or _recommend_rank
    candidates = [dict(candidate) for candidate in candidates]
    fixed_baseline = aggregate_answer_vote(
        candidates,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        group_ids_field=group_ids_field,
        rank_getter=_recommend_rank,
        logprob_field=None,
    )
    mean_baseline = aggregate_answer_vote(
        candidates,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        group_ids_field=group_ids_field,
        rank_getter=_recommend_rank,
        logprob_field=LOGPROB_FIELD,
    )
    weight_result = compute_lp_weights(
        candidates,
        strength,
        missing_logprob_policy,
        forced_fallback_reason=forced_fallback_reason,
    )
    candidates = _with_lp_weights(candidates, weight_result)
    valid = [candidate for candidate in candidates if candidate["valid"]]

    if not valid:
        return {
            "format_valid": False,
            "valid_candidate_count": 0,
            "candidates": candidates,
            "answer_groups": [],
            "winning_normalized_answer": None,
            "winning_group_size": 0,
            "winning_weighted_score": None,
            selected_id_field: None,
            "selected_answer": "",
            "tie": False,
            "tied_group_count": 0,
            "tie_break_source": "not_applicable",
            "tie_break_reason": "all_candidates_invalid",
            "tie_break_order": tie_break_order,
            "representative_selection_source": "not_applicable",
            "representative_selection_reason": "all_candidates_invalid",
            "lpvote_fallback": False,
            "lpvote_fallback_reason": None,
            "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
            "mean_vote_selected_format": mean_baseline[selected_id_field],
            "fixed_recommend_tie": fixed_baseline["tie"],
        }

    # The required missing-mean fallback is specifically fixed-recommend.
    effective_rank = _recommend_rank if weight_result.fallback else rank_getter
    effective_order = "recommend" if weight_result.fallback else tie_break_order
    grouped: Dict[str, list[dict]] = {}
    for candidate in valid:
        grouped.setdefault(candidate["normalized_answer"], []).append(candidate)

    answer_groups = []
    for normalized_answer, members in grouped.items():
        score = sum(float(member["lp_weight"]) for member in members)
        representative = min(members, key=effective_rank)
        group = {
            "normalized_answer": normalized_answer,
            group_ids_field: [member[candidate_id_key] for member in members],
            "size": len(members),
            "aggregation_score": score,
            "weighted_score": score,
            candidate_id_key: representative[candidate_id_key],
        }
        answer_groups.append(group)
        for member in members:
            member["aggregation_score"] = score

    best_score, tied_groups = max_score_items(answer_groups)
    winning_group = min(tied_groups, key=effective_rank)
    group_tie = len(tied_groups) > 1
    winning_members = grouped[winning_group["normalized_answer"]]
    best_member_weight = max(float(member["lp_weight"]) for member in winning_members)
    tied_members = [
        member
        for member in winning_members
        if math.isclose(
            float(member["lp_weight"]), best_member_weight, rel_tol=1e-12, abs_tol=1e-12
        )
    ]
    selected = min(tied_members, key=effective_rank)
    selected["selected"] = True

    return {
        "format_valid": True,
        "valid_candidate_count": len(valid),
        "candidates": candidates,
        "answer_groups": answer_groups,
        "winning_normalized_answer": winning_group["normalized_answer"],
        "winning_group_size": winning_group["size"],
        "winning_weighted_score": best_score,
        selected_id_field: selected[candidate_id_key],
        "selected_answer": selected["model_answer"],
        "tie": group_tie,
        "tied_group_count": len(tied_groups),
        "tie_break_source": "format_order" if group_tie else "not_needed",
        "tie_break_reason": (
            "equal_weighted_answer_score" if group_tie else "unique_weighted_answer_score"
        ),
        "tie_break_order": effective_order,
        "representative_selection_source": (
            "format_order" if len(tied_members) > 1 else "lp_weight"
        ),
        "representative_selection_reason": (
            "equal_lp_weight" if len(tied_members) > 1 else "highest_lp_weight"
        ),
        "lpvote_fallback": weight_result.fallback,
        "lpvote_fallback_reason": weight_result.fallback_reason,
        "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
        "mean_vote_selected_format": mean_baseline[selected_id_field],
        "fixed_recommend_selected_answer": fixed_baseline["selected_answer"],
        "mean_vote_selected_answer": mean_baseline["selected_answer"],
        "fixed_recommend_tie": fixed_baseline["tie"],
    }


def _realhit_branch_candidates(
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    run_dirs: Mapping[str, str],
    structure_key: str | None = None,
) -> list[dict]:
    return [
        build_realhit_candidate_trace(
            format_name,
            records_by_format.get(format_name),
            structure_key=structure_key,
            run_dir=run_dirs.get(format_name),
        )
        for format_name in FORMAT_ORDER
    ]


def aggregate_realhit_lp_sample(
    sample_id: str,
    question_type: str,
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    *,
    run_dirs: Mapping[str, str] | None = None,
    strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    format_order: Sequence[str] = RECOMMEND_FORMAT_ORDER,
    tie_break_order: str = "recommend",
) -> Dict[str, Any]:
    run_dirs = run_dirs or {}
    rank_map = {format_name: index for index, format_name in enumerate(format_order)}
    rank_getter = lambda item: rank_map[item["format"]]

    if question_type == "Structure Comprehending":
        reference_candidates = _realhit_branch_candidates(
            records_by_format, run_dirs, "structure_reference_run"
        )
        swap_candidates = _realhit_branch_candidates(
            records_by_format, run_dirs, "structure_swap_run"
        )
        reference_probe = compute_lp_weights(
            reference_candidates, strength, missing_logprob_policy
        )
        swap_probe = compute_lp_weights(swap_candidates, strength, missing_logprob_policy)
        fallback_reasons = [
            f"{name}: {probe.fallback_reason}"
            for name, probe in (("reference", reference_probe), ("swap", swap_probe))
            if probe.fallback
        ]
        forced_reason = "; ".join(fallback_reasons) or None
        reference_vote = aggregate_answer_lp_vote(
            reference_candidates,
            strength=strength,
            missing_logprob_policy=missing_logprob_policy,
            rank_getter=rank_getter,
            tie_break_order=tie_break_order,
            forced_fallback_reason=forced_reason,
        )
        swap_vote = aggregate_answer_lp_vote(
            swap_candidates,
            strength=strength,
            missing_logprob_policy=missing_logprob_policy,
            rank_getter=rank_getter,
            tie_break_order=tie_break_order,
            forced_fallback_reason=forced_reason,
        )
        valid_reference = {
            candidate["format"]
            for candidate in reference_vote["candidates"]
            if candidate["valid"]
        }
        valid_swap = {
            candidate["format"]
            for candidate in swap_vote["candidates"]
            if candidate["valid"]
        }
        selected_format = {
            "reference": reference_vote["selected_format"],
            "swap": swap_vote["selected_format"],
        }
        weighted_score = {
            "reference": reference_vote["winning_weighted_score"],
            "swap": swap_vote["winning_weighted_score"],
        }
        tie = {"reference": reference_vote["tie"], "swap": swap_vote["tie"]}
        effective_order = (
            "recommend" if forced_reason is not None else tie_break_order
        )
        return {
            "id": str(sample_id),
            "QuestionType": question_type,
            "method": "SheetFlex-LPVote",
            "format_valid": reference_vote["format_valid"] and swap_vote["format_valid"],
            "model_answer": swap_vote["selected_answer"],
            "selected_format": selected_format,
            "valid_candidate_count": len(valid_reference & valid_swap),
            "structure_reference_answer": reference_vote["selected_answer"],
            "structure_swap_answer": swap_vote["selected_answer"],
            "trace": {
                "aggregation": "lp_weighted_answer_group_vote",
                "method": "SheetFlex-LPVote",
                "lp_weight_strength": float(strength),
                "missing_logprob_policy": missing_logprob_policy,
                "lpvote_fallback": forced_reason is not None,
                "lpvote_fallback_reason": forced_reason,
                "selected_format": selected_format,
                "weighted_score": weighted_score,
                "tie": tie,
                "tied_count": {
                    "reference": reference_vote["tied_group_count"],
                    "swap": swap_vote["tied_group_count"],
                },
                "tie_break_order": effective_order,
                "tie_break_reason": {
                    "reference": reference_vote["tie_break_reason"],
                    "swap": swap_vote["tie_break_reason"],
                },
                "structure_reference_vote": reference_vote,
                "structure_swap_vote": swap_vote,
            },
        }

    candidates = _realhit_branch_candidates(records_by_format, run_dirs)
    vote = aggregate_answer_lp_vote(
        candidates,
        strength=strength,
        missing_logprob_policy=missing_logprob_policy,
        rank_getter=rank_getter,
        tie_break_order=tie_break_order,
    )
    return {
        "id": str(sample_id),
        "QuestionType": question_type,
        "method": "SheetFlex-LPVote",
        "format_valid": vote["format_valid"],
        "model_answer": vote["selected_answer"],
        "selected_format": vote["selected_format"],
        "valid_candidate_count": vote["valid_candidate_count"],
        "trace": {
            "aggregation": "lp_weighted_answer_group_vote",
            "method": "SheetFlex-LPVote",
            "lp_weight_strength": float(strength),
            "missing_logprob_policy": missing_logprob_policy,
            "lpvote_fallback": vote["lpvote_fallback"],
            "lpvote_fallback_reason": vote["lpvote_fallback_reason"],
            "selected_format": vote["selected_format"],
            "weighted_score": vote["winning_weighted_score"],
            "tie": vote["tie"],
            "tied_count": vote["tied_group_count"],
            "tie_break_order": vote["tie_break_order"],
            "tie_break_reason": vote["tie_break_reason"],
            "answer_vote": vote,
        },
    }


def select_spreadsheet_lp_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_id_key: str = "format",
    selected_id_field: str = "selected_format",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    fallback_source: str = "format_order",
    logprob_field: str | None = LOGPROB_FIELD,
    strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    tie_break_order: str = "recommend",
) -> Dict[str, Any]:
    """Select an existing workbook using supporter-weighted region similarity."""
    del fallback_source, logprob_field
    rank_getter = rank_getter or _recommend_rank
    candidates = [dict(candidate) for candidate in candidates]
    fixed_baseline = select_spreadsheet_medoid(
        candidates,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        rank_getter=_recommend_rank,
        logprob_field=None,
    )
    mean_baseline = select_spreadsheet_medoid(
        candidates,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        rank_getter=_recommend_rank,
        logprob_field=LOGPROB_FIELD,
    )
    weight_result = compute_lp_weights(
        candidates, strength, missing_logprob_policy
    )
    candidates = _with_lp_weights(candidates, weight_result)
    valid = [candidate for candidate in candidates if candidate["valid"]]
    matrix, pairwise_values = build_similarity_matrix(
        candidates, candidate_id_key=candidate_id_key
    )
    if not valid:
        for candidate in candidates:
            candidate.pop("_cells", None)
        return {
            "format_valid": False,
            "valid_candidate_count": 0,
            "candidates": candidates,
            "similarity_matrix": matrix,
            "average_pairwise_region_agreement": None,
            "medoid_score": None,
            "weighted_score": None,
            selected_id_field: None,
            "selected_source_file": None,
            "tie": False,
            "tied_candidate_count": 0,
            "tie_break_source": "not_applicable",
            "tie_break_reason": "all_candidates_invalid",
            "tie_break_order": tie_break_order,
            "lpvote_fallback": False,
            "lpvote_fallback_reason": None,
            "equal_weight_tie": fixed_baseline["tie"],
            "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
            "mean_vote_selected_format": mean_baseline[selected_id_field],
            "fixed_recommend_tie": fixed_baseline["tie"],
        }

    effective_rank = _recommend_rank if weight_result.fallback else rank_getter
    effective_order = "recommend" if weight_result.fallback else tie_break_order
    for candidate in valid:
        candidate["aggregation_score"] = sum(
            float(other["lp_weight"])
            * float(matrix[candidate[candidate_id_key]][other[candidate_id_key]])
            for other in valid
        )
        candidate["weighted_score"] = candidate["aggregation_score"]
    best_score, tied = max_score_items(valid)
    selected = min(tied, key=effective_rank)
    selected["selected"] = True
    for candidate in candidates:
        candidate.pop("_cells", None)

    return {
        "format_valid": True,
        "valid_candidate_count": len(valid),
        "candidates": candidates,
        "similarity_matrix": matrix,
        "average_pairwise_region_agreement": (
            sum(pairwise_values) / len(pairwise_values) if pairwise_values else None
        ),
        "medoid_score": best_score,
        "weighted_score": best_score,
        selected_id_field: selected[candidate_id_key],
        "selected_source_file": selected["output_file"],
        "tie": len(tied) > 1,
        "tied_candidate_count": len(tied),
        "tie_break_source": "format_order" if len(tied) > 1 else "not_needed",
        "tie_break_reason": (
            "equal_weighted_similarity_score"
            if len(tied) > 1
            else "unique_weighted_similarity_score"
        ),
        "tie_break_order": effective_order,
        "lpvote_fallback": weight_result.fallback,
        "lpvote_fallback_reason": weight_result.fallback_reason,
        "equal_weight_tie": fixed_baseline["tie"],
        "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
        "mean_vote_selected_format": mean_baseline[selected_id_field],
        "fixed_recommend_tie": fixed_baseline["tie"],
    }


def aggregate_spreadsheet_lp_sample(
    item: Mapping[str, Any],
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    run_dirs: Mapping[str, Any],
    input_path: Any,
    *,
    strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    format_order: Sequence[str] = RECOMMEND_FORMAT_ORDER,
    tie_break_order: str = "recommend",
    exclude_unchanged_target_values: bool = False,
) -> Dict[str, Any]:
    aggregate = aggregate_spreadsheet_sample(
        item,
        records_by_format,
        run_dirs,
        input_path,
        format_order=format_order,
        logprob_field=LOGPROB_FIELD,
        selection_fn=select_spreadsheet_lp_candidate,
        selection_kwargs={
            "strength": strength,
            "missing_logprob_policy": missing_logprob_policy,
            "tie_break_order": tie_break_order,
        },
        aggregation_name="lp_weighted_region_medoid",
        exclude_unchanged_target_values=exclude_unchanged_target_values,
    )
    trace = aggregate["trace"]
    aggregate["method"] = "SheetFlex-LPVote"
    trace.update(
        {
            "method": "SheetFlex-LPVote",
            "lp_weight_strength": float(strength),
            "missing_logprob_policy": missing_logprob_policy,
            "selected_format": aggregate["selected_format"],
            "tied_count": trace["tied_candidate_count"],
        }
    )
    return aggregate


def _realhit_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = row["trace"]
    if row["QuestionType"] == "Structure Comprehending":
        return [trace["structure_reference_vote"], trace["structure_swap_vote"]]
    return [trace["answer_vote"]]


def _weight_snapshot(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict:
    event_snapshots = []
    all_weights = []
    near_equal = True
    for event in events:
        weights = sorted(
            (
                float(candidate["lp_weight"])
                for candidate in event["candidates"]
                if candidate["valid"]
            ),
            reverse=True,
        )
        all_weights.extend(weights)
        if weights:
            equal = 1.0 / len(weights)
            near_equal = near_equal and all(
                abs(weight - equal) <= NEAR_EQUAL_WEIGHT_TOLERANCE
                for weight in weights
            )
        event_snapshots.append(
            {
                "max_weight": weights[0] if weights else None,
                "top_two_weight_gap": (
                    weights[0] - weights[1] if len(weights) > 1 else None
                ),
            }
        )
    sorted_all = sorted(all_weights, reverse=True)
    return {
        "id": str(row["id"]),
        "max_weight": sorted_all[0] if sorted_all else None,
        "top_two_weight_gap": (
            sorted_all[0] - sorted_all[1] if len(sorted_all) > 1 else None
        ),
        "near_equal": bool(all_weights) and near_equal,
        "events": event_snapshots,
    }


def _shared_lp_diagnostics(rows: Sequence[Mapping[str, Any]], events_by_row) -> dict:
    all_events = [event for row in rows for event in events_by_row(row)]
    valid_candidates = [
        candidate
        for event in all_events
        for candidate in event["candidates"]
        if candidate["valid"]
    ]
    complete = [
        candidate
        for candidate in valid_candidates
        if has_valid_logprob(candidate, LOGPROB_FIELD)
    ]
    snapshots = [_weight_snapshot(row, events_by_row(row)) for row in rows]
    tied_events = [event for event in all_events if event["tie"]]
    fallback_rows = [row for row in rows if row["trace"]["lpvote_fallback"]]
    changed_fixed_events = [
        event
        for event in all_events
        if event.get("selected_format")
        != event.get("fixed_recommend_selected_format")
    ]
    changed_mean_events = [
        event
        for event in all_events
        if event.get("selected_format") != event.get("mean_vote_selected_format")
    ]
    return {
        "num_samples": len(rows),
        "num_aggregation_events": len(all_events),
        "valid_candidate_count_distribution": count_distribution(
            event["valid_candidate_count"] for event in all_events
        ),
        "mean_logprob_valid_candidates": len(complete),
        "mean_logprob_effective_candidates": len(valid_candidates),
        "mean_logprob_completeness_rate": safe_rate(
            len(complete), len(valid_candidates)
        ),
        "lpvote_fallback_samples": len(fallback_rows),
        "lpvote_fallback_rate": safe_rate(len(fallback_rows), len(rows)),
        "lpvote_final_tie_events": len(tied_events),
        "lpvote_final_tie_rate": safe_rate(len(tied_events), len(all_events)),
        "selection_changes_vs_fixed_recommend": sum(
            1
            for row in rows
            if any(
                event.get("selected_format")
                != event.get("fixed_recommend_selected_format")
                for event in events_by_row(row)
            )
        ),
        "selection_changes_vs_mean_vote": sum(
            1
            for row in rows
            if any(
                event.get("selected_format")
                != event.get("mean_vote_selected_format")
                for event in events_by_row(row)
            )
        ),
        "selection_change_events_vs_fixed_recommend": len(changed_fixed_events),
        "selection_change_events_vs_mean_vote": len(changed_mean_events),
        "near_equal_weight_tolerance": NEAR_EQUAL_WEIGHT_TOLERANCE,
        "near_equal_weight_samples": sum(
            1 for snapshot in snapshots if snapshot["near_equal"]
        ),
        "near_equal_weight_sample_rate": safe_rate(
            sum(1 for snapshot in snapshots if snapshot["near_equal"]), len(rows)
        ),
        "max_candidate_weight_summary": numeric_summary(
            snapshot["max_weight"] for snapshot in snapshots
        ),
        "top_two_weight_gap_summary": numeric_summary(
            snapshot["top_two_weight_gap"] for snapshot in snapshots
        ),
        "sample_weight_summaries": snapshots,
        "selected_format_distribution": count_distribution(
            event["selected_format"]
            for event in all_events
            if event.get("selected_format") is not None
        ),
    }


def realhit_lp_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnostics = _shared_lp_diagnostics(rows, _realhit_events)
    events = [event for row in rows for event in _realhit_events(row)]
    diagnostics.update(
        {
            "answer_group_count_distribution": count_distribution(
                len(event["answer_groups"]) for event in events
            ),
            "all_answer_group_size_distribution": count_distribution(
                group["size"] for event in events for group in event["answer_groups"]
            ),
            "ordinary_majority_answer_changed_by_lpvote": sum(
                1
                for row in rows
                if row["QuestionType"] != "Structure Comprehending"
                for event in _realhit_events(row)
                if not event.get("fixed_recommend_tie")
                and event.get("selected_answer")
                != event.get("fixed_recommend_selected_answer")
            ),
        }
    )
    return diagnostics


def spreadsheet_lp_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    diagnostics = _shared_lp_diagnostics(rows, lambda row: [row["trace"]])
    diagnostics.update(
        {
            "equal_weight_tie_samples": sum(
                1 for row in rows if row["trace"]["equal_weight_tie"]
            ),
            "two_valid_candidate_samples": sum(
                1 for row in rows if row["valid_candidate_count"] == 2
            ),
            "average_pairwise_region_agreement": numeric_summary(
                row["trace"]["average_pairwise_region_agreement"] for row in rows
            ),
        }
    )
    by_scope = {}
    for scope, marker in (("Cell-Level", "Cell"), ("Sheet-Level", "Sheet")):
        subset = [
            row for row in rows if marker in str(row.get("instruction_type", ""))
        ]
        by_scope[scope] = {
            "samples": len(subset),
            "equal_weight_ties": sum(
                1 for row in subset if row["trace"]["equal_weight_tie"]
            ),
            "lpvote_ties": sum(1 for row in subset if row["trace"]["tie"]),
            "selection_changes_vs_fixed_recommend": sum(
                1
                for row in subset
                if row["selected_format"]
                != row["trace"]["fixed_recommend_selected_format"]
            ),
            "selection_changes_vs_mean_vote": sum(
                1
                for row in subset
                if row["selected_format"]
                != row["trace"]["mean_vote_selected_format"]
            ),
        }
    diagnostics["by_instruction_scope"] = by_scope
    return diagnostics
