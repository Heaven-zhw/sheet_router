"""Gold-free class-aware confidence-gated aggregation for SheetFlex."""

import math
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Sequence

from .common import (
    RECOMMEND_FORMAT_ORDER,
    SCORE_ABS_TOL,
    SCORE_REL_TOL,
    SheetFlexError,
    count_distribution,
    has_valid_logprob,
    max_score_items,
    numeric_summary,
    safe_rate,
    scores_tied,
)
from .lp_vote import (
    LOGPROB_FIELD,
    MISSING_LOGPROB_POLICIES,
    aggregate_answer_lp_vote,
    compute_lp_weights,
    select_spreadsheet_lp_candidate,
    validate_lp_weight_strength,
)
from .realhit import aggregate_answer_vote, build_realhit_candidate_trace
from .spreadsheet import (
    aggregate_spreadsheet_sample,
    build_similarity_matrix,
    select_spreadsheet_medoid,
)


METHOD = "SheetFlex-CGVote"
SCORE_COMPARISON_SPACE = "max_shifted_exp_log_confidence_gated_score"


def validate_confidence_gate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise SheetFlexError(
            "confidence_gate_strength must be a finite number >= 0"
        )
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise SheetFlexError(
            "confidence_gate_strength must be a finite number >= 0"
        )
    return strength


def _recommend_rank(item: Mapping[str, Any]) -> int:
    return RECOMMEND_FORMAT_ORDER.index(str(item["format"]))


def _candidate_label(
    candidate: Mapping[str, Any], candidate_id_key: str, index: int
) -> str:
    return str(candidate.get(candidate_id_key, f"candidate_{index}"))


def check_cg_logprobs(
    candidates: Sequence[Mapping[str, Any]],
    missing_logprob_policy: str,
    *,
    sample_id: str,
    branch: str,
    candidate_id_key: str = "format",
) -> str | None:
    """Apply CGVote's strict precheck before LPWeight's single-item shortcut."""
    if missing_logprob_policy not in MISSING_LOGPROB_POLICIES:
        raise SheetFlexError(
            "missing_logprob_policy must be one of "
            f"{MISSING_LOGPROB_POLICIES}, got {missing_logprob_policy!r}"
        )
    missing = [
        _candidate_label(candidate, candidate_id_key, index)
        for index, candidate in enumerate(candidates)
        if candidate.get("valid") and not has_valid_logprob(candidate, LOGPROB_FIELD)
    ]
    if not missing:
        return None
    reason = (
        f"sample_id={sample_id} branch={branch} missing_or_invalid_"
        f"{LOGPROB_FIELD} formats={','.join(missing)}"
    )
    if missing_logprob_policy == "error":
        raise SheetFlexError(reason)
    return reason


def attach_lp_weights(
    candidates: Sequence[Mapping[str, Any]],
    *,
    strength: float,
    missing_logprob_policy: str,
    forced_fallback_reason: str | None = None,
) -> tuple[list[dict], Any]:
    result = compute_lp_weights(
        candidates,
        strength,
        missing_logprob_policy,
        forced_fallback_reason=forced_fallback_reason,
    )
    weighted = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["lp_raw_weight"] = result.raw_weights[index]
        item["lp_weight"] = result.weights[index]
        weighted.append(item)
    return weighted, result


def build_equivalence_classes(
    candidates: Sequence[Mapping[str, Any]],
    *,
    class_key: str,
    candidate_id_key: str = "format",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Copy candidates and group valid predictions without losing weight mass."""
    rank_getter = rank_getter or _recommend_rank
    copied = [dict(candidate) for candidate in candidates]
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for candidate in copied:
        candidate["class_id"] = None
        if not candidate.get("valid"):
            continue
        class_id = str(candidate[class_key])
        candidate["class_id"] = class_id
        grouped.setdefault(class_id, []).append(candidate)

    classes = []
    for class_id, members in grouped.items():
        representative = min(members, key=rank_getter)
        classes.append(
            {
                "class_id": class_id,
                "member_formats": [
                    str(member[candidate_id_key]) for member in members
                ],
                "member_count": len(members),
                "class_weight_mass": sum(
                    float(member["lp_weight"]) for member in members
                ),
                "class_confidence": max(
                    float(member["lp_weight"]) for member in members
                ),
                "representative_format": str(
                    representative[candidate_id_key]
                ),
                "rank": rank_getter(representative),
                "selected": False,
            }
        )
    return copied, classes


def score_equivalence_classes(
    classes: Sequence[Mapping[str, Any]],
    class_similarity_matrix: Mapping[str, Mapping[str, float]],
    confidence_gate_strength: float,
) -> list[dict]:
    """Compute W, Q, C and C * Q**beta for each output class."""
    beta = validate_confidence_gate_strength(confidence_gate_strength)
    scored = []
    for output_class in classes:
        item = dict(output_class)
        class_id = item["class_id"]
        consensus = sum(
            float(other["class_weight_mass"])
            * float(class_similarity_matrix[class_id][other["class_id"]])
            for other in classes
        )
        confidence = float(item["class_confidence"])
        if beta == 0.0:
            gated_score = consensus
            log_score = math.log(consensus) if consensus > 0 else None
        elif consensus > 0.0 and confidence > 0.0:
            raw_log_score = math.log(consensus) + beta * math.log(confidence)
            log_score = raw_log_score if math.isfinite(raw_log_score) else None
            gated_score = math.exp(raw_log_score) if log_score is not None else 0.0
        else:
            log_score = None
            gated_score = 0.0
        item.update(
            {
                "consensus_score": consensus,
                "confidence_gated_score": gated_score,
                "log_confidence_gated_score": log_score,
                "log_confidence_gated_score_state": (
                    "finite" if log_score is not None else "negative_infinity"
                ),
            }
        )
        scored.append(item)
    return scored


def select_confidence_gated_class(
    classes: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    beta: float,
) -> tuple[dict, list[dict], str]:
    """Select a class while preserving alpha=0 and beta=0 baseline ordering."""
    if not classes:
        raise SheetFlexError("Cannot select from an empty equivalence-class list")
    alpha = validate_lp_weight_strength(alpha)
    beta = validate_confidence_gate_strength(beta)
    copied = [dict(item) for item in classes]

    if beta == 0.0:
        _, tied = max_score_items(copied, "consensus_score")
        comparison_space = "consensus_score_beta_zero_compatibility"
    elif alpha == 0.0:
        _, tied = max_score_items(copied, "consensus_score")
        comparison_space = "consensus_score_alpha_zero_compatibility"
    else:
        finite_logs = [
            float(item["log_confidence_gated_score"])
            for item in copied
            if item["log_confidence_gated_score"] is not None
        ]
        if not finite_logs:
            tied = copied
        else:
            best_log = max(finite_logs)
            for item in copied:
                log_score = item["log_confidence_gated_score"]
                item["score_comparison_value"] = (
                    math.exp(float(log_score) - best_log)
                    if log_score is not None
                    else 0.0
                )
            best_value = max(item["score_comparison_value"] for item in copied)
            tied = [
                item
                for item in copied
                if scores_tied(item["score_comparison_value"], best_value)
            ]
        comparison_space = SCORE_COMPARISON_SPACE

    selected = min(tied, key=lambda item: item["rank"])
    selected["selected"] = True
    selected_id = selected["class_id"]
    for item in copied:
        item["selected"] = item["class_id"] == selected_id
    return selected, copied, comparison_space


def _tied_classes(
    classes: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    comparison_space: str,
) -> list[Mapping[str, Any]]:
    if comparison_space == SCORE_COMPARISON_SPACE:
        return [
            item
            for item in classes
            if scores_tied(
                item.get("score_comparison_value", 0.0),
                selected.get("score_comparison_value", 0.0),
            )
        ]
    return [
        item
        for item in classes
        if scores_tied(item["consensus_score"], selected["consensus_score"])
    ]


def _class_matrix_for_realhit(classes: Sequence[Mapping[str, Any]]) -> dict:
    return {
        left["class_id"]: {
            right["class_id"]: float(left["class_id"] == right["class_id"])
            for right in classes
        }
        for left in classes
    }


def _selected_class_id_from_answer_vote(vote: Mapping[str, Any]) -> str | None:
    return vote.get("winning_normalized_answer")


def aggregate_answer_cg_vote(
    candidates: Sequence[Mapping[str, Any]],
    *,
    strength: float = 1.0,
    confidence_gate_strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    candidate_id_key: str = "format",
    selected_id_field: str = "selected_format",
    group_ids_field: str = "formats",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    tie_break_order: str = "recommend",
    sample_id: str = "unknown",
    branch: str = "ordinary",
    forced_fallback_reason: str | None = None,
) -> Dict[str, Any]:
    """Apply class-aware CGVote to one RealHiT answer branch."""
    alpha = validate_lp_weight_strength(strength)
    beta = validate_confidence_gate_strength(confidence_gate_strength)
    rank_getter = rank_getter or _recommend_rank
    original = [dict(candidate) for candidate in candidates]
    local_issue = check_cg_logprobs(
        original,
        missing_logprob_policy,
        sample_id=str(sample_id),
        branch=branch,
        candidate_id_key=candidate_id_key,
    )
    fallback_reason = forced_fallback_reason or local_issue

    fixed_baseline = aggregate_answer_vote(
        original,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        group_ids_field=group_ids_field,
        rank_getter=_recommend_rank,
        logprob_field=None,
    )
    lp_baseline = aggregate_answer_lp_vote(
        original,
        strength=alpha,
        missing_logprob_policy=missing_logprob_policy,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        group_ids_field=group_ids_field,
        rank_getter=rank_getter,
        tie_break_order=tie_break_order,
        forced_fallback_reason=fallback_reason,
    )
    weighted, weight_result = attach_lp_weights(
        original,
        strength=alpha,
        missing_logprob_policy=missing_logprob_policy,
        forced_fallback_reason=fallback_reason,
    )
    effective_rank = _recommend_rank if weight_result.fallback else rank_getter
    effective_order = "recommend" if weight_result.fallback else tie_break_order
    weighted, classes = build_equivalence_classes(
        weighted,
        class_key="normalized_answer",
        candidate_id_key=candidate_id_key,
        rank_getter=effective_rank,
    )
    if not classes:
        return {
            "format_valid": False,
            "valid_candidate_count": 0,
            "candidates": weighted,
            "equivalence_classes": [],
            "answer_groups": [],
            "equivalence_class_count": 0,
            "selected_class_id": None,
            selected_id_field: None,
            "selected_answer": "",
            "tie": False,
            "tied_class_ids": [],
            "tied_class_count": 0,
            "tie_break_source": "not_applicable",
            "tie_break_reason": "all_candidates_invalid",
            "tie_break_order": effective_order,
            "requested_tie_break_order": tie_break_order,
            "representative_selection_source": "not_applicable",
            "representative_selection_reason": "all_candidates_invalid",
            "fallback": weight_result.fallback,
            "fallback_reason": weight_result.fallback_reason,
            "fixed_recommend_selected_class_id": _selected_class_id_from_answer_vote(
                fixed_baseline
            ),
            "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
            "lpvote_selected_class_id": lp_baseline.get("winning_normalized_answer"),
            "lpvote_selected_format": lp_baseline[selected_id_field],
            "score_comparison_space": "not_applicable",
            "score_rel_tol": SCORE_REL_TOL,
            "score_abs_tol": SCORE_ABS_TOL,
        }

    class_matrix = _class_matrix_for_realhit(classes)
    scored = score_equivalence_classes(classes, class_matrix, beta)
    winning_class, scored, comparison_space = select_confidence_gated_class(
        scored,
        alpha=0.0 if weight_result.fallback else alpha,
        beta=beta,
    )
    tied_classes = _tied_classes(scored, winning_class, comparison_space)
    winning_members = [
        item for item in weighted if item.get("class_id") == winning_class["class_id"]
    ]
    best_weight = max(float(item["lp_weight"]) for item in winning_members)
    tied_members = [
        item
        for item in winning_members
        if math.isclose(
            float(item["lp_weight"]),
            best_weight,
            rel_tol=SCORE_REL_TOL,
            abs_tol=SCORE_ABS_TOL,
        )
    ]
    selected = min(tied_members, key=effective_rank)
    for candidate in weighted:
        candidate["selected"] = candidate is selected

    tie = len(tied_classes) > 1
    fixed_class = _selected_class_id_from_answer_vote(fixed_baseline)
    lp_class = lp_baseline.get("winning_normalized_answer")
    return {
        "format_valid": True,
        "valid_candidate_count": sum(bool(item.get("valid")) for item in weighted),
        "candidates": weighted,
        "equivalence_classes": scored,
        "answer_groups": scored,
        "equivalence_class_count": len(scored),
        "class_similarity_matrix": class_matrix,
        "selected_class_id": winning_class["class_id"],
        "winning_normalized_answer": winning_class["class_id"],
        "winning_group_size": winning_class["member_count"],
        selected_id_field: selected[candidate_id_key],
        "selected_answer": selected["model_answer"],
        "tie": tie,
        "tied_class_ids": [item["class_id"] for item in tied_classes],
        "tied_class_count": len(tied_classes),
        "tie_break_source": "format_order" if tie else "not_needed",
        "tie_break_reason": (
            "equal_confidence_gated_class_score"
            if tie
            else "unique_confidence_gated_class_score"
        ),
        "tie_break_order": effective_order,
        "requested_tie_break_order": tie_break_order,
        "representative_selection_source": (
            "format_order" if len(tied_members) > 1 else "lp_weight"
        ),
        "representative_selection_reason": (
            "equal_lp_weight" if len(tied_members) > 1 else "highest_lp_weight"
        ),
        "fallback": weight_result.fallback,
        "fallback_reason": weight_result.fallback_reason,
        "fixed_recommend_selected_class_id": fixed_class,
        "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
        "lpvote_selected_class_id": lp_class,
        "lpvote_selected_format": lp_baseline[selected_id_field],
        "selected_class_changed_vs_fixed_recommend": winning_class["class_id"]
        != fixed_class,
        "selected_format_changed_vs_fixed_recommend": selected[candidate_id_key]
        != fixed_baseline[selected_id_field],
        "selected_class_changed_vs_lpvote": winning_class["class_id"] != lp_class,
        "selected_format_changed_vs_lpvote": selected[candidate_id_key]
        != lp_baseline[selected_id_field],
        "score_comparison_space": comparison_space,
        "score_rel_tol": SCORE_REL_TOL,
        "score_abs_tol": SCORE_ABS_TOL,
    }


def _realhit_candidates(
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    run_dirs: Mapping[str, str],
    format_order: Sequence[str],
    structure_key: str | None = None,
) -> list[dict]:
    return [
        build_realhit_candidate_trace(
            format_name,
            records_by_format.get(format_name),
            structure_key=structure_key,
            run_dir=run_dirs.get(format_name),
        )
        for format_name in format_order
    ]


def aggregate_realhit_cg_sample(
    sample_id: str,
    question_type: str,
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    *,
    run_dirs: Mapping[str, str] | None = None,
    strength: float = 1.0,
    confidence_gate_strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    format_order: Sequence[str] = RECOMMEND_FORMAT_ORDER,
    tie_break_order: str = "recommend",
) -> Dict[str, Any]:
    run_dirs = run_dirs or {}
    alpha = validate_lp_weight_strength(strength)
    beta = validate_confidence_gate_strength(confidence_gate_strength)
    rank_map = {name: index for index, name in enumerate(format_order)}
    rank_getter = lambda item: rank_map[item["format"]]

    if question_type == "Structure Comprehending":
        reference_candidates = _realhit_candidates(
            records_by_format,
            run_dirs,
            format_order,
            "structure_reference_run",
        )
        swap_candidates = _realhit_candidates(
            records_by_format,
            run_dirs,
            format_order,
            "structure_swap_run",
        )
        reference_issue = check_cg_logprobs(
            reference_candidates,
            missing_logprob_policy,
            sample_id=str(sample_id),
            branch="structure_reference_run",
        )
        swap_issue = check_cg_logprobs(
            swap_candidates,
            missing_logprob_policy,
            sample_id=str(sample_id),
            branch="structure_swap_run",
        )
        issues = [issue for issue in (reference_issue, swap_issue) if issue]
        forced_reason = "structure_sample_fallback: " + "; ".join(issues) if issues else None
        reference = aggregate_answer_cg_vote(
            reference_candidates,
            strength=alpha,
            confidence_gate_strength=beta,
            missing_logprob_policy=missing_logprob_policy,
            rank_getter=rank_getter,
            tie_break_order=tie_break_order,
            sample_id=str(sample_id),
            branch="structure_reference_run",
            forced_fallback_reason=forced_reason,
        )
        swap = aggregate_answer_cg_vote(
            swap_candidates,
            strength=alpha,
            confidence_gate_strength=beta,
            missing_logprob_policy=missing_logprob_policy,
            rank_getter=rank_getter,
            tie_break_order=tie_break_order,
            sample_id=str(sample_id),
            branch="structure_swap_run",
            forced_fallback_reason=forced_reason,
        )
        valid_reference = {
            item["format"] for item in reference["candidates"] if item["valid"]
        }
        valid_swap = {
            item["format"] for item in swap["candidates"] if item["valid"]
        }
        return {
            "id": str(sample_id),
            "QuestionType": question_type,
            "method": METHOD,
            "format_valid": reference["format_valid"] and swap["format_valid"],
            "model_answer": swap["selected_answer"],
            "selected_format": {
                "reference": reference["selected_format"],
                "swap": swap["selected_format"],
            },
            "selected_class_id": {
                "reference": reference["selected_class_id"],
                "swap": swap["selected_class_id"],
            },
            "valid_candidate_count": len(valid_reference & valid_swap),
            "branch_valid_candidate_count": {
                "reference": reference["valid_candidate_count"],
                "swap": swap["valid_candidate_count"],
            },
            "equivalence_class_count": {
                "reference": reference["equivalence_class_count"],
                "swap": swap["equivalence_class_count"],
            },
            "structure_reference_answer": reference["selected_answer"],
            "structure_swap_answer": swap["selected_answer"],
            "trace": {
                "aggregation": "confidence_gated_answer_class_vote",
                "method": METHOD,
                "lp_weight_strength": alpha,
                "confidence_gate_strength": beta,
                "missing_logprob_policy": missing_logprob_policy,
                "fallback": bool(forced_reason),
                "fallback_reason": forced_reason,
                "structure_reference_vote": reference,
                "structure_swap_vote": swap,
            },
        }

    candidates = _realhit_candidates(records_by_format, run_dirs, format_order)
    vote = aggregate_answer_cg_vote(
        candidates,
        strength=alpha,
        confidence_gate_strength=beta,
        missing_logprob_policy=missing_logprob_policy,
        rank_getter=rank_getter,
        tie_break_order=tie_break_order,
        sample_id=str(sample_id),
    )
    return {
        "id": str(sample_id),
        "QuestionType": question_type,
        "method": METHOD,
        "format_valid": vote["format_valid"],
        "model_answer": vote["selected_answer"],
        "selected_format": vote["selected_format"],
        "selected_class_id": vote["selected_class_id"],
        "valid_candidate_count": vote["valid_candidate_count"],
        "equivalence_class_count": vote["equivalence_class_count"],
        "trace": {
            "aggregation": "confidence_gated_answer_class_vote",
            "method": METHOD,
            "lp_weight_strength": alpha,
            "confidence_gate_strength": beta,
            "missing_logprob_policy": missing_logprob_policy,
            "fallback": vote["fallback"],
            "fallback_reason": vote["fallback_reason"],
            "answer_vote": vote,
        },
    }


def _selected_region_hash(selection: Mapping[str, Any]) -> str | None:
    selected_format = selection.get("selected_format")
    if selected_format is None:
        return None
    return next(
        (
            candidate.get("region_hash")
            for candidate in selection["candidates"]
            if candidate.get("format") == selected_format
        ),
        None,
    )


def select_spreadsheet_cg_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_id_key: str = "format",
    selected_id_field: str = "selected_format",
    rank_getter: Callable[[Mapping[str, Any]], Any] | None = None,
    fallback_source: str = "format_order",
    logprob_field: str | None = LOGPROB_FIELD,
    strength: float = 1.0,
    confidence_gate_strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    tie_break_order: str = "recommend",
    sample_id: str = "unknown",
) -> Dict[str, Any]:
    """Select a Spreadsheet prediction class without reading a golden workbook."""
    del fallback_source, logprob_field
    alpha = validate_lp_weight_strength(strength)
    beta = validate_confidence_gate_strength(confidence_gate_strength)
    rank_getter = rank_getter or _recommend_rank
    original = [dict(candidate) for candidate in candidates]
    issue = check_cg_logprobs(
        original,
        missing_logprob_policy,
        sample_id=str(sample_id),
        branch="spreadsheet",
        candidate_id_key=candidate_id_key,
    )
    fixed_baseline = select_spreadsheet_medoid(
        original,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        rank_getter=_recommend_rank,
        logprob_field=None,
    )
    lp_baseline = select_spreadsheet_lp_candidate(
        original,
        candidate_id_key=candidate_id_key,
        selected_id_field=selected_id_field,
        rank_getter=rank_getter,
        strength=alpha,
        missing_logprob_policy=missing_logprob_policy,
        tie_break_order=tie_break_order,
    )
    weighted, weight_result = attach_lp_weights(
        original,
        strength=alpha,
        missing_logprob_policy=missing_logprob_policy,
        forced_fallback_reason=issue,
    )
    effective_rank = _recommend_rank if weight_result.fallback else rank_getter
    effective_order = "recommend" if weight_result.fallback else tie_break_order
    weighted, classes = build_equivalence_classes(
        weighted,
        class_key="region_hash",
        candidate_id_key=candidate_id_key,
        rank_getter=effective_rank,
    )
    candidate_matrix, pairwise_values = build_similarity_matrix(
        weighted, candidate_id_key=candidate_id_key
    )
    fixed_class_id = _selected_region_hash(fixed_baseline)
    lp_class_id = _selected_region_hash(lp_baseline)
    if not classes:
        for candidate in weighted:
            candidate.pop("_cells", None)
        return {
            "format_valid": False,
            "valid_candidate_count": 0,
            "candidates": weighted,
            "equivalence_classes": [],
            "equivalence_class_count": 0,
            "similarity_matrix": candidate_matrix,
            "class_similarity_matrix": {},
            "average_pairwise_region_agreement": None,
            "medoid_score": None,
            "confidence_gated_score": None,
            "selected_class_id": None,
            selected_id_field: None,
            "selected_source_file": None,
            "tie": False,
            "tied_class_ids": [],
            "tied_class_count": 0,
            "tied_candidate_count": 0,
            "tie_break_source": "not_applicable",
            "tie_break_reason": "all_candidates_invalid",
            "tie_break_order": effective_order,
            "requested_tie_break_order": tie_break_order,
            "fallback": weight_result.fallback,
            "fallback_reason": weight_result.fallback_reason,
            "equivalent_class_representative_selection": False,
            "fixed_recommend_selected_class_id": fixed_class_id,
            "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
            "lpvote_selected_class_id": lp_class_id,
            "lpvote_selected_format": lp_baseline[selected_id_field],
            "selected_class_changed_vs_fixed_recommend": False,
            "selected_format_changed_vs_fixed_recommend": False,
            "selected_class_changed_vs_lpvote": False,
            "selected_format_changed_vs_lpvote": False,
            "equal_weight_tie": fixed_baseline["tie"],
            "equal_weight_tie_kind": "none",
            "equal_weight_tied_candidate_count": 0,
            "equal_weight_tied_class_count": 0,
            "score_comparison_space": "not_applicable",
            "score_rel_tol": SCORE_REL_TOL,
            "score_abs_tol": SCORE_ABS_TOL,
        }

    by_id = {candidate[candidate_id_key]: candidate for candidate in weighted}
    class_matrix = {}
    for left in classes:
        left_id = left["class_id"]
        left_member = left["member_formats"][0]
        class_matrix[left_id] = {}
        for member in left["member_formats"][1:]:
            for candidate_id, candidate in by_id.items():
                if not candidate["valid"]:
                    continue
                if not scores_tied(
                    candidate_matrix[left_member][candidate_id],
                    candidate_matrix[member][candidate_id],
                ):
                    raise SheetFlexError(
                        "Candidates with the same region_hash have different "
                        f"similarity vectors: class={left_id} members="
                        f"{left_member},{member}"
                    )
        for right in classes:
            right_member = right["member_formats"][0]
            class_matrix[left_id][right["class_id"]] = float(
                candidate_matrix[left_member][right_member]
            )
    scored = score_equivalence_classes(classes, class_matrix, beta)
    winning_class, scored, comparison_space = select_confidence_gated_class(
        scored,
        alpha=0.0 if weight_result.fallback else alpha,
        beta=beta,
    )
    tied_classes = _tied_classes(scored, winning_class, comparison_space)
    winning_members = [
        by_id[member] for member in winning_class["member_formats"]
    ]
    selected = min(winning_members, key=effective_rank)
    for candidate in weighted:
        candidate["selected"] = candidate is selected
        candidate.pop("_cells", None)

    fixed_tied = [
        item
        for item in fixed_baseline["candidates"]
        if item.get("valid")
        and scores_tied(item["aggregation_score"], fixed_baseline["medoid_score"])
    ]
    equal_tied_hashes = {item["region_hash"] for item in fixed_tied}
    equal_tie_kind = "none"
    if fixed_baseline["tie"]:
        if len(equal_tied_hashes) == 1:
            equal_tie_kind = "equivalent"
        elif fixed_baseline["valid_candidate_count"] == 2:
            equal_tie_kind = "forced"
        else:
            equal_tie_kind = "conflict"

    tie = len(tied_classes) > 1
    return {
        "format_valid": True,
        "valid_candidate_count": len([item for item in weighted if item["valid"]]),
        "candidates": weighted,
        "equivalence_classes": scored,
        "equivalence_class_count": len(scored),
        "similarity_matrix": candidate_matrix,
        "class_similarity_matrix": class_matrix,
        "average_pairwise_region_agreement": (
            sum(pairwise_values) / len(pairwise_values) if pairwise_values else None
        ),
        "medoid_score": winning_class["consensus_score"],
        "confidence_gated_score": winning_class["confidence_gated_score"],
        "selected_class_id": winning_class["class_id"],
        selected_id_field: selected[candidate_id_key],
        "selected_source_file": selected["output_file"],
        "tie": tie,
        "tied_class_ids": [item["class_id"] for item in tied_classes],
        "tied_class_count": len(tied_classes),
        "tied_candidate_count": sum(
            item["member_count"] for item in tied_classes
        ),
        "tie_break_source": "format_order" if tie else "not_needed",
        "tie_break_reason": (
            "equal_confidence_gated_class_score"
            if tie
            else "unique_confidence_gated_class_score"
        ),
        "tie_break_order": effective_order,
        "requested_tie_break_order": tie_break_order,
        "fallback": weight_result.fallback,
        "fallback_reason": weight_result.fallback_reason,
        "equivalent_class_representative_selection": len(winning_members) > 1,
        "fixed_recommend_selected_class_id": fixed_class_id,
        "fixed_recommend_selected_format": fixed_baseline[selected_id_field],
        "lpvote_selected_class_id": lp_class_id,
        "lpvote_selected_format": lp_baseline[selected_id_field],
        "selected_class_changed_vs_fixed_recommend": winning_class["class_id"]
        != fixed_class_id,
        "selected_format_changed_vs_fixed_recommend": selected[candidate_id_key]
        != fixed_baseline[selected_id_field],
        "selected_class_changed_vs_lpvote": winning_class["class_id"]
        != lp_class_id,
        "selected_format_changed_vs_lpvote": selected[candidate_id_key]
        != lp_baseline[selected_id_field],
        "equal_weight_tie": fixed_baseline["tie"],
        "equal_weight_tie_kind": equal_tie_kind,
        "equal_weight_tied_candidate_count": len(fixed_tied),
        "equal_weight_tied_class_count": len(equal_tied_hashes),
        "score_comparison_space": comparison_space,
        "score_rel_tol": SCORE_REL_TOL,
        "score_abs_tol": SCORE_ABS_TOL,
    }


def aggregate_spreadsheet_cg_sample(
    item: Mapping[str, Any],
    records_by_format: Mapping[str, Mapping[str, Any] | None],
    run_dirs: Mapping[str, Any],
    input_path: Any,
    *,
    strength: float = 1.0,
    confidence_gate_strength: float = 1.0,
    missing_logprob_policy: str = "vote",
    format_order: Sequence[str] = RECOMMEND_FORMAT_ORDER,
    tie_break_order: str = "recommend",
    exclude_unchanged_target_values: bool = False,
) -> Dict[str, Any]:
    alpha = validate_lp_weight_strength(strength)
    beta = validate_confidence_gate_strength(confidence_gate_strength)
    aggregate = aggregate_spreadsheet_sample(
        item,
        records_by_format,
        run_dirs,
        input_path,
        format_order=format_order,
        logprob_field=LOGPROB_FIELD,
        selection_fn=select_spreadsheet_cg_candidate,
        selection_kwargs={
            "strength": alpha,
            "confidence_gate_strength": beta,
            "missing_logprob_policy": missing_logprob_policy,
            "tie_break_order": tie_break_order,
            "sample_id": str(item["id"]),
        },
        aggregation_name="confidence_gated_region_class_vote",
        exclude_unchanged_target_values=exclude_unchanged_target_values,
    )
    aggregate["method"] = METHOD
    aggregate["selected_class_id"] = aggregate["trace"]["selected_class_id"]
    aggregate["equivalence_class_count"] = aggregate["trace"][
        "equivalence_class_count"
    ]
    aggregate["trace"].update(
        {
            "method": METHOD,
            "lp_weight_strength": alpha,
            "confidence_gate_strength": beta,
            "missing_logprob_policy": missing_logprob_policy,
            "selected_format": aggregate["selected_format"],
        }
    )
    return aggregate


def _realhit_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = row["trace"]
    if row["QuestionType"] == "Structure Comprehending":
        return [trace["structure_reference_vote"], trace["structure_swap_vote"]]
    return [trace["answer_vote"]]


def _shared_cg_diagnostics(rows, events_by_row) -> dict:
    events = [event for row in rows for event in events_by_row(row)]
    valid_candidates = [
        candidate
        for event in events
        for candidate in event["candidates"]
        if candidate["valid"]
    ]
    complete = [
        candidate
        for candidate in valid_candidates
        if has_valid_logprob(candidate, LOGPROB_FIELD)
    ]
    classes = [item for event in events for item in event["equivalence_classes"]]
    weights = [
        sorted(
            [
                float(candidate["lp_weight"])
                for candidate in event["candidates"]
                if candidate["valid"]
            ],
            reverse=True,
        )
        for event in events
    ]
    class_ties = [event for event in events if event["tie"]]
    invalid_reasons = [
        candidate["invalid_reason"]
        for event in events
        for candidate in event["candidates"]
        if not candidate["valid"]
    ]
    class_changed_lp = sum(
        bool(event.get("selected_class_changed_vs_lpvote")) for event in events
    )
    format_changed_lp = sum(
        bool(event.get("selected_format_changed_vs_lpvote")) for event in events
    )
    class_changed_fixed = sum(
        bool(event.get("selected_class_changed_vs_fixed_recommend"))
        for event in events
    )
    format_changed_fixed = sum(
        bool(event.get("selected_format_changed_vs_fixed_recommend"))
        for event in events
    )
    return {
        "num_samples": len(rows),
        "num_aggregation_events": len(events),
        "all_candidates_invalid_samples": sum(
            not bool(row["format_valid"]) for row in rows
        ),
        "valid_candidate_count_distribution": count_distribution(
            event["valid_candidate_count"] for event in events
        ),
        "equivalence_class_count_distribution": count_distribution(
            event["equivalence_class_count"] for event in events
        ),
        "equivalence_class_size_distribution": count_distribution(
            item["member_count"] for item in classes
        ),
        "mean_logprob_valid_candidates": len(complete),
        "mean_logprob_effective_candidates": len(valid_candidates),
        "mean_logprob_completeness_rate": safe_rate(
            len(complete), len(valid_candidates)
        ),
        "fallback_events": sum(1 for event in events if event["fallback"]),
        "fallback_event_rate": safe_rate(
            sum(1 for event in events if event["fallback"]), len(events)
        ),
        "fallback_samples": sum(
            any(event["fallback"] for event in events_by_row(row)) for row in rows
        ),
        "fallback_sample_rate": safe_rate(
            sum(
                any(event["fallback"] for event in events_by_row(row))
                for row in rows
            ),
            len(rows),
        ),
        "final_class_tie_events": len(class_ties),
        "final_class_tie_rate": safe_rate(len(class_ties), len(events)),
        "equivalent_class_representative_selections": sum(
            1
            for event in events
            if event.get("equivalent_class_representative_selection")
            or any(
                item["selected"] and item["member_count"] > 1
                for item in event["equivalence_classes"]
            )
        ),
        "selected_format_distribution": count_distribution(
            event["selected_format"]
            for event in events
            if event.get("selected_format") is not None
        ),
        "selected_class_changed_vs_lpvote": class_changed_lp,
        "selected_format_changed_vs_lpvote": format_changed_lp,
        "selected_class_change_events_vs_lpvote": class_changed_lp,
        "selected_format_change_events_vs_lpvote": format_changed_lp,
        "selected_class_change_samples_vs_lpvote": sum(
            any(
                event.get("selected_class_changed_vs_lpvote")
                for event in events_by_row(row)
            )
            for row in rows
        ),
        "selected_format_change_samples_vs_lpvote": sum(
            any(
                event.get("selected_format_changed_vs_lpvote")
                for event in events_by_row(row)
            )
            for row in rows
        ),
        "selected_class_changed_vs_fixed_recommend": class_changed_fixed,
        "selected_format_changed_vs_fixed_recommend": format_changed_fixed,
        "selected_class_change_events_vs_fixed_recommend": class_changed_fixed,
        "selected_format_change_events_vs_fixed_recommend": format_changed_fixed,
        "selected_class_change_samples_vs_fixed_recommend": sum(
            any(
                event.get("selected_class_changed_vs_fixed_recommend")
                for event in events_by_row(row)
            )
            for row in rows
        ),
        "selected_format_change_samples_vs_fixed_recommend": sum(
            any(
                event.get("selected_format_changed_vs_fixed_recommend")
                for event in events_by_row(row)
            )
            for row in rows
        ),
        "max_candidate_weight_summary": numeric_summary(
            values[0] if values else None for values in weights
        ),
        "top_two_candidate_weight_gap_summary": numeric_summary(
            values[0] - values[1] if len(values) > 1 else None
            for values in weights
        ),
        "class_weight_mass_summary": numeric_summary(
            item["class_weight_mass"] for item in classes
        ),
        "class_confidence_summary": numeric_summary(
            item["class_confidence"] for item in classes
        ),
        "consensus_score_summary": numeric_summary(
            item["consensus_score"] for item in classes
        ),
        "confidence_gated_score_summary": numeric_summary(
            item["confidence_gated_score"] for item in classes
        ),
        "invalid_candidate_reason_distribution": count_distribution(
            invalid_reasons
        ),
        "score_comparison": {
            "default_space": SCORE_COMPARISON_SPACE,
            "beta_zero_space": "consensus_score_beta_zero_compatibility",
            "alpha_zero_space": "consensus_score_alpha_zero_compatibility",
            "rel_tol": SCORE_REL_TOL,
            "abs_tol": SCORE_ABS_TOL,
        },
    }


def realhit_cg_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnostics = _shared_cg_diagnostics(rows, _realhit_events)
    diagnostics["answer_class_count_distribution"] = diagnostics[
        "equivalence_class_count_distribution"
    ]
    return diagnostics


def spreadsheet_cg_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    diagnostics = _shared_cg_diagnostics(rows, lambda row: [row["trace"]])
    diagnostics.update(
        {
            "equal_weight_equivalent_tie_samples": sum(
                row["trace"]["equal_weight_tie_kind"] == "equivalent"
                for row in rows
            ),
            "equal_weight_forced_tie_samples": sum(
                row["trace"]["equal_weight_tie_kind"] == "forced" for row in rows
            ),
            "equal_weight_conflict_tie_samples": sum(
                row["trace"]["equal_weight_tie_kind"] == "conflict" for row in rows
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
            "equivalent_ties": sum(
                row["trace"]["equal_weight_tie_kind"] == "equivalent"
                for row in subset
            ),
            "forced_ties": sum(
                row["trace"]["equal_weight_tie_kind"] == "forced"
                for row in subset
            ),
            "conflict_ties": sum(
                row["trace"]["equal_weight_tie_kind"] == "conflict"
                for row in subset
            ),
            "cg_final_class_ties": sum(row["trace"]["tie"] for row in subset),
        }
    diagnostics["by_instruction_scope"] = by_scope
    return diagnostics
