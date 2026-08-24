import argparse
import copy
import os
import shutil
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

import realhit_cot as realhit_entry
import spreadsheet_pot as spreadsheet_entry
from core.routing import RouteDecision, SpreadsheetProfiler, unique_keep_order
from core.solver.realhit_cot import RealHiTCoTSolver
from core.solver.spreadsheet_pot import SPREADSHEET_DATA_SPLITS, SpreadSheetPoTSolver
from core.utils import load_jsonl, save_jsonl


repo_dir = os.path.abspath(os.path.dirname(__file__))


QA_PRIORITY = (
    "latex",
    "json_cells",
    "markdown",
    "json_rows",
    "html",
    "dataframe",
    "csv",
    "image",
    "excel_1_image",
    "default_image",
)
MANIPULATION_PRIORITY = (
    "json_cells",
    "json_rows",
    "markdown",
    "latex",
    "html",
    "dataframe",
    "csv",
    "excel_1_image",
    "image",
    "default_image",
)
QA_BASE_SCORES = {
    "latex": 3.0,
    "json_cells": 2.6,
    "markdown": 2.4,
    "json_rows": 2.0,
    "html": 1.7,
    "dataframe": 1.3,
    "csv": 1.2,
    "image": 0.8,
    "excel_1_image": 0.6,
    "default_image": 0.3,
}
MANIPULATION_BASE_SCORES = {
    "json_cells": 4.0,
    "json_rows": 3.2,
    "markdown": 2.4,
    "latex": 2.3,
    "html": 2.0,
    "dataframe": 1.8,
    "csv": 1.6,
    "excel_1_image": 1.3,
    "image": 1.0,
    "default_image": 0.7,
}
IMAGE_FORMATS = {"image", "excel_1_image", "default_image"}
TEXT_FORMATS = {"official_latex", "latex", "json_cells", "markdown", "json_rows", "html", "dataframe", "csv"}


STYLE_TERMS = (
    "color",
    "colour",
    "background",
    "fill",
    "font",
    "bold",
    "border",
    "highlight",
    "shade",
    "conditional formatting",
    "format reference",
    "formatting reference",
    "format as",
    "same format",
    "keep format",
    "keep formatting",
    "preserve format",
    "preserve formatting",
    "cell format",
    "number format",
    "row height",
    "column width",
    "chart",
    "picture",
    "screenshot",
)
FORMAT_FALSE_POSITIVES = (
    "answer format",
    "output format",
    "final answer format",
    "format your answer",
    "format the answer",
)
RECORD_TERMS = (
    "match",
    "matching",
    "duplicate",
    "lookup",
    "filter",
    "sort",
    "group",
    "group by",
    "sum by",
    "for each row",
    "each row",
    "append row",
    "add row",
    "total row",
    "date",
    "ref",
)
NUMERIC_TERMS = (
    "max",
    "min",
    "sum",
    "total",
    "average",
    "mean",
    "median",
    "count",
    "rate",
    "percentage",
    "percent",
    "exceed",
    "below",
    "above",
    "between",
    "top",
    "bottom",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "greater",
    "less",
    "calculate",
)
STRUCTURE_TERMS = (
    "header",
    "row header",
    "column header",
    "multi-level",
    "multilevel",
    "hierarchical",
    "hierarchy",
    "nested",
    "sub-table",
    "subtable",
    "above",
    "below",
    "left",
    "right",
    "parent",
    "child",
    "section",
    "category",
    "join",
    "merge",
)
COORDINATE_PATTERNS = (
    r"\b[A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?\b",
    r"\brow\s+\d+\b",
    r"\bcolumn\s+[A-Z]{1,3}\b",
    r"\bsheet\s+name\b",
    r"\bworksheet\s+name\b",
)


def safe_name(value: Optional[str]) -> str:
    return (value or "default_model").replace("/", "_").replace("\\", "_")


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def matches_any(text: str, patterns: Sequence[str]) -> bool:
    import re

    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def first_available(order: Sequence[str], available: Set[str]) -> Optional[str]:
    for value in order:
        if value in available:
            return value
    return None


def sanitize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    return {key: round(value, 4) for key, value in sorted(scores.items())}


def build_available_formats(profile: Dict[str, Any]) -> Set[str]:
    return set(profile.get("available_text_formats") or []) | set(profile.get("available_image_formats") or [])


class SingleResponseHeuristicRouter:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def route(self, dataset: str, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        if dataset == "realhitbench":
            return self._route_realhit(item, profile)
        if dataset == "spreadsheetbench":
            return self._route_spreadsheet(item, profile)
        raise ValueError(f"Unsupported dataset: {dataset}")

    def _extract_common_features(
        self,
        dataset: str,
        query: str,
        item: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        qf = profile.get("question_features") or {}
        lowered = (query or "").lower()
        style_query = contains_any(query, STYLE_TERMS)
        if "format" in lowered and not contains_any(query, STYLE_TERMS):
            style_query = False
        if contains_any(query, FORMAT_FALSE_POSITIVES) and not any(
            term in lowered for term in ("color", "background", "fill", "font", "border", "bold", "format reference")
        ):
            style_query = False

        coordinate_query = bool(qf.get("mentions_exact_cell_or_range")) or matches_any(query, COORDINATE_PATTERNS)
        if dataset == "spreadsheetbench":
            coordinate_query = coordinate_query or bool(item.get("answer_position")) or bool(item.get("answer_sheet"))

        record_query = bool(qf.get("requires_lookup") or qf.get("mentions_sort_filter_pivot")) or contains_any(
            query, RECORD_TERMS
        )
        numeric_query = bool(qf.get("requires_aggregation") or qf.get("requires_comparison")) or contains_any(
            query, NUMERIC_TERMS
        )
        structure_query = bool(qf.get("mentions_header_or_structure")) or contains_any(query, STRUCTURE_TERMS)

        density = float(profile.get("nonempty_ratio") or 0.0)
        max_rows = int(profile.get("max_rows") or 0)
        max_cols = int(profile.get("max_cols") or 0)
        num_sheets = int(profile.get("num_sheets") or 0)
        num_merged = int(profile.get("num_merged_ranges") or 0)
        estimated_tokens = int(profile.get("estimated_text_tokens") or 0)
        token_budget = int(self.args.max_text_tokens or 0)
        is_large = bool(token_budget and estimated_tokens > token_budget * 0.7)
        has_style_signal = bool(
            profile.get("has_background_color")
            or profile.get("has_charts_or_images")
            or int(profile.get("num_bold_cells") or 0) > 0
            or int(profile.get("num_bordered_cells") or 0) > 0
            or int(profile.get("hidden_rows") or 0) > 0
            or int(profile.get("hidden_cols") or 0) > 0
        )
        instruction_mentions_format_reference = any(
            term in lowered
            for term in (
                "format reference",
                "formatting reference",
                "same format",
                "keep format",
                "keep formatting",
                "preserve format",
                "preserve formatting",
            )
        )

        return {
            "style_query": bool(style_query),
            "coordinate_query": bool(coordinate_query),
            "record_query": bool(record_query),
            "numeric_query": bool(numeric_query),
            "structure_query": bool(structure_query),
            "manipulation_type": item.get("instruction_type"),
            "n_sheets": num_sheets,
            "density": density,
            "max_rows": max_rows,
            "max_cols": max_cols,
            "non_empty_cells": int(profile.get("total_nonempty_cells") or 0),
            "merged_range_count": num_merged,
            "has_style_signal": has_style_signal,
            "is_sparse": density < 0.35 if density else False,
            "is_wide": max_cols >= 12,
            "is_long": max_rows >= 80,
            "is_large": is_large,
            "small_simple_table": max_rows <= 20 and max_cols <= 8 and num_merged == 0,
            "estimated_text_tokens": estimated_tokens,
            "max_text_tokens": token_budget,
            "instruction_mentions_format_reference": instruction_mentions_format_reference,
            "available_formats": sorted(build_available_formats(profile)),
        }

    def _apply_common_scores(self, dataset: str, scores: Dict[str, float], features: Dict[str, Any]) -> None:
        image_bonus = 0.0
        if features["style_query"]:
            image_bonus += 4.0
            for text_format in TEXT_FORMATS:
                if text_format in scores:
                    scores[text_format] -= 1.0
        if features["has_style_signal"] and not features["numeric_query"]:
            image_bonus += 1.5
        if image_bonus:
            if dataset == "realhitbench":
                weights = {"image": 1.0, "excel_1_image": 0.8, "default_image": 0.5}
            else:
                weights = {"excel_1_image": 1.0, "image": 0.8, "default_image": 0.5}
            for image_format, weight in weights.items():
                if image_format in scores:
                    scores[image_format] += image_bonus * weight

        if features["is_sparse"]:
            scores["json_cells"] = scores.get("json_cells", 0.0) + 1.5
            scores["json_rows"] = scores.get("json_rows", 0.0) - 0.5
        if features["is_large"]:
            scores["json_cells"] = scores.get("json_cells", 0.0) + 0.8
            for fmt, delta in {"markdown": -1.0, "html": -0.8, "dataframe": -0.8}.items():
                if fmt in scores:
                    scores[fmt] += delta

    def _select_by_score(
        self,
        scores: Dict[str, float],
        priority: Sequence[str],
        available: Set[str],
    ) -> Tuple[str, float]:
        candidate_order = [fmt for fmt in priority if fmt in available and fmt in scores]
        if not candidate_order:
            fallback = first_available(priority, available) or next(iter(available), priority[0])
            return fallback, scores.get(fallback, 0.0)

        rank = {fmt: idx for idx, fmt in enumerate(priority)}
        best = max(candidate_order, key=lambda fmt: (scores[fmt], -rank.get(fmt, 999)))
        return best, scores[best]

    def _strong_visual_route(
        self,
        dataset: str,
        available: Set[str],
        features: Dict[str, Any],
    ) -> Optional[str]:
        if not (
            features["style_query"]
            or features["instruction_mentions_format_reference"]
            or (
                features["has_style_signal"]
                and not features["numeric_query"]
                and not features["record_query"]
            )
        ):
            return None

        if dataset == "realhitbench":
            return first_available(("image", "excel_1_image", "default_image"), available)
        return first_available(("excel_1_image", "image", "default_image"), available)

    def _route_realhit(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        query = item.get("Question", "")
        available = build_available_formats(profile)
        available.discard("official_latex")
        features = self._extract_common_features("realhitbench", query, item, profile)
        features["available_formats"] = sorted(available)
        scores = copy.deepcopy(QA_BASE_SCORES)
        self._apply_common_scores("realhitbench", scores, features)

        if features["numeric_query"]:
            scores["latex"] += 1.0
            scores["json_cells"] += 0.6
        if features["structure_query"]:
            scores["json_cells"] += 1.2
            scores["latex"] += 0.8
            scores["markdown"] -= 0.3
        if features["coordinate_query"]:
            scores["json_cells"] += 1.5
        if features["record_query"]:
            scores["json_rows"] += 1.2
            scores["markdown"] += 0.5
        if features["small_simple_table"]:
            scores["markdown"] += 1.0

        rule = "score"
        reasons = ["QA/CoT uses latex as the default single representation."]
        visual_format = self._strong_visual_route("realhitbench", available, features)
        if visual_format:
            table_format = visual_format
            rule = "Rule A: visual/style priority"
            reasons.append("Detected visual/style evidence need; chose the best available RealHiT image format.")
        else:
            table_format, _ = self._select_by_score(scores, QA_PRIORITY, available)
            if table_format == "json_cells":
                reasons.append("Structure, coordinate, sparse, or large-table signals favor json_cells.")
            elif table_format == "json_rows":
                reasons.append("Record-like filtering or matching signals favor json_rows.")
            elif table_format == "markdown":
                reasons.append("Small/simple table signals favor markdown.")
            elif table_format != "latex":
                reasons.append(f"Selected {table_format} after availability and token-budget scoring.")

        if features["is_large"] and table_format in {"markdown", "html", "dataframe"}:
            compact = "json_cells" if features["is_sparse"] else first_available(("latex", "csv"), available)
            if compact and compact in available:
                table_format = compact
                rule = "Rule D: token protection"
                reasons.append("Large-table token risk switched to a more compact text representation.")

        score = scores.get(table_format, 0.0)
        return RouteDecision(
            solver_mode="cot_qa",
            table_format=table_format,
            fallback_formats=[],
            reason="; ".join(reasons),
            stages={
                "task_router": "qa",
                "representation_router": table_format,
                "reasoning_router": "cot_qa",
                "rule": rule,
                "features": features,
                "scores": sanitize_scores(scores),
                "selected_score": round(score, 4),
                "single_response": True,
            },
        )

    def _route_spreadsheet(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        query = item.get("instruction", "")
        features = self._extract_common_features("spreadsheetbench", query, item, profile)
        available = build_available_formats(profile)
        scores = copy.deepcopy(MANIPULATION_BASE_SCORES)
        self._apply_common_scores("spreadsheetbench", scores, features)

        if features["manipulation_type"] == "Cell-Level Manipulation":
            scores["json_cells"] += 1.5
        if features["manipulation_type"] == "Sheet-Level Manipulation":
            scores["json_cells"] += 0.8
            scores["json_rows"] += 0.8
        if features["coordinate_query"]:
            scores["json_cells"] += 1.5
        if features["record_query"] and features["density"] >= 0.35 and not features["style_query"]:
            scores["json_rows"] += 1.5
        if features["n_sheets"] >= 2:
            scores["json_cells"] += 1.0
        if features["instruction_mentions_format_reference"]:
            scores["excel_1_image"] += 1.5

        rule = "score"
        reasons = ["Manipulation/PoT uses json_cells as the default single representation."]
        visual_format = self._strong_visual_route("spreadsheetbench", available, features)
        if visual_format:
            table_format = visual_format
            rule = "Rule A: visual/style priority"
            reasons.append("Detected visual/style evidence need; chose the best available SpreadsheetBench image format.")
        else:
            table_format, _ = self._select_by_score(scores, MANIPULATION_PRIORITY, available)
            if table_format == "json_rows":
                reasons.append("Dense record-like row operations favor json_rows.")
            elif table_format == "json_cells":
                reasons.append("Coordinates, output ranges, multi-sheet, sparse, or cell-level signals favor json_cells.")
            elif table_format != "json_cells":
                reasons.append(f"Selected {table_format} after availability and token-budget scoring.")

        if features["is_large"] and table_format in {"markdown", "html", "dataframe"}:
            compact = "json_cells" if features["is_sparse"] else "json_rows"
            if compact in available:
                table_format = compact
                rule = "Rule D: token protection"
                reasons.append("Large-table token risk switched to a compact JSON representation.")

        score = scores.get(table_format, 0.0)
        return RouteDecision(
            solver_mode="pot_code",
            table_format=table_format,
            fallback_formats=[],
            reason="; ".join(reasons),
            stages={
                "task_router": "operation",
                "representation_router": table_format,
                "reasoning_router": "pot_code",
                "rule": rule,
                "features": features,
                "scores": sanitize_scores(scores),
                "selected_score": round(score, 4),
                "single_response": True,
            },
        )


def build_router_suffix(args: argparse.Namespace) -> str:
    parts = ["single_resp_router", "heuristic"]
    if args.fill_merged:
        parts.append("fillmerged")
    if not args.include_coordinates:
        parts.append("nocoord")
    if args.max_text_tokens:
        parts.append(f"{int(args.max_text_tokens / 1000)}ktoken")
    if args.top_p != 1.0 or args.temperature != 0:
        parts.append(f"tp{args.top_p}_temp{args.temperature}")
    if args.suffix:
        parts.append(args.suffix)
    return "_".join(parts)


def build_dataset_dir(args: argparse.Namespace) -> str:
    if args.dataset == "realhitbench":
        return "realhitbench"
    return f"spreadsheetbench_{args.data_split}".replace("/", "_").replace("\\", "_")


def build_output_dir(args: argparse.Namespace) -> str:
    return os.path.join(args.output_root, build_dataset_dir(args), safe_name(args.model_name), build_router_suffix(args))


def make_route_args(args: argparse.Namespace, table_format: str) -> argparse.Namespace:
    route_args = argparse.Namespace(**vars(args))
    route_args.table_format = table_format
    route_args.dry_run = False
    return route_args


def build_solver(dataset: str, args: argparse.Namespace, out_dir: str):
    if dataset == "realhitbench":
        return RealHiTCoTSolver(**vars(args))
    return SpreadSheetPoTSolver(**vars(args), output_dir=out_dir)


def run_solver(dataset: str, args: argparse.Namespace, item: Dict[str, Any], table_format: str, out_dir: str) -> Dict[str, Any]:
    route_args = make_route_args(args, table_format)
    solver = build_solver(dataset, route_args, out_dir)
    return solver(item)


def build_error_result(dataset: str, item: Dict[str, Any], error: str, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    result = copy.deepcopy(item)
    if dataset == "realhitbench":
        result.update(
            {
                "format_valid": False,
                "error": error,
                "model_answer": "",
                "eval": {
                    "Model_Answer": "",
                    "Reference_Answer": item.get("ProcessedAnswer", ""),
                    "F1": None,
                    "EM": None,
                    "ROUGE-L": None,
                    "SacreBLEU": None,
                },
            }
        )
    else:
        split_name = args.data_split if args is not None else "all_912"
        split_config = SPREADSHEET_DATA_SPLITS.get(split_name, SPREADSHEET_DATA_SPLITS["all_912"])
        n = int(split_config["num_test_cases"])
        result.update(
            {
                "format_valid": False,
                "execution_success": False,
                "error": error,
                "test_case_results": [0] * n,
                "test_case_messages": [error] * n,
                "total_soft_restriction": 0.0,
                "total_hard_restriction": 0.0,
            }
        )
    return result


def attach_router_metadata(
    result: Dict[str, Any],
    decision: RouteDecision,
    profile: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    result = copy.deepcopy(result)
    result["router_policy"] = "single_resp_heuristic"
    result["router_decision"] = asdict(decision)
    result["router_profile"] = profile
    result["router_fallback_used"] = False
    result["router_attempts"] = [
        {
            "attempt_index": 1,
            "is_primary": True,
            "table_format": decision.table_format,
            "summary": {
                "table_format": decision.table_format,
                "format_valid": bool(result.get("format_valid")),
                "error": result.get("error"),
                "table_metadata": result.get("table_metadata"),
            },
        }
    ]
    return result


def solve_one(
    dataset: str,
    args: argparse.Namespace,
    item: Dict[str, Any],
    out_dir: str,
    profiler: SpreadsheetProfiler,
    router: SingleResponseHeuristicRouter,
) -> Dict[str, Any]:
    profile = profiler.profile(dataset, item)
    decision = router.route(dataset, item, profile)
    try:
        result = run_solver(dataset, args, item, decision.table_format, out_dir)
    except Exception:
        result = build_error_result(dataset, item, traceback.format_exc(), args)
    return attach_router_metadata(result, decision, profile, args)


def build_realhit_eval_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry = realhit_entry.build_eval_entry(result)
    entry.update(
        {
            "router_decision": result.get("router_decision"),
            "router_fallback_used": result.get("router_fallback_used"),
            "table_metadata": result.get("table_metadata"),
        }
    )
    return entry


def build_spreadsheet_eval_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry = spreadsheet_entry.build_eval_entry(result)
    entry.update(
        {
            "router_decision": result.get("router_decision"),
            "router_fallback_used": result.get("router_fallback_used"),
        }
    )
    return entry


def build_eval_entry(dataset: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if dataset == "realhitbench":
        return build_realhit_eval_entry(result)
    return build_spreadsheet_eval_entry(result)


def add_scores(dataset: str, score_lists: Dict[str, Any], result: Dict[str, Any]) -> None:
    if dataset == "realhitbench":
        realhit_entry.add_scores(score_lists, result)
    else:
        spreadsheet_entry.add_scores(score_lists, result)


def average_scores(dataset: str, score_lists: Dict[str, Any]) -> Dict[str, Any]:
    if dataset == "realhitbench":
        return realhit_entry.average_scores(score_lists)
    return spreadsheet_entry.average_scores(score_lists)


def report_scores(dataset: str, score_lists: Dict[str, Any]) -> None:
    if dataset == "realhitbench":
        realhit_entry.report_scores(score_lists)
    else:
        spreadsheet_entry.report_scores(score_lists)


def output_filenames(dataset: str, partial: bool = False) -> Tuple[str, str, str]:
    suffix = ".partial" if partial else ""
    if dataset == "realhitbench":
        return (
            f"realhit_cot{suffix}.jsonl",
            f"realhit_cot_eval{suffix}.json",
            f"realhit_cot_score{suffix}.json",
        )
    return (
        f"spreadsheet_pot{suffix}.jsonl",
        f"spreadsheet_pot_eval{suffix}.json",
        f"spreadsheet_pot_accuracy{suffix}.json",
    )


def cleanup_intermediate_files(dataset: str, out_dir: str) -> None:
    for name in output_filenames(dataset, partial=True):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def save_outputs(
    dataset: str,
    out_dir: str,
    outs: List[Optional[Dict[str, Any]]],
    eval_results: List[Any],
    scores: Dict[str, Any],
    partial: bool = False,
) -> None:
    out_name, eval_name, score_name = output_filenames(dataset, partial=partial)
    save_jsonl(outs, os.path.join(out_dir, out_name))
    save_jsonl(eval_results, os.path.join(out_dir, eval_name))
    save_jsonl(scores, os.path.join(out_dir, score_name))

    if not partial:
        decisions = [
            {
                "id": result.get("id") if result else None,
                "router_decision": result.get("router_decision") if result else None,
                "router_fallback_used": result.get("router_fallback_used") if result else None,
            }
            for result in outs
        ]
        save_jsonl(decisions, os.path.join(out_dir, "router_decisions.jsonl"))


def dry_run_payload(dataset: str, prompt: Any, metadata: Dict[str, Any], profile: Dict[str, Any], decision: RouteDecision) -> Dict[str, Any]:
    if dataset == "realhitbench":
        payload = realhit_entry.dry_run_payload(prompt, metadata)
    else:
        payload = spreadsheet_entry.dry_run_payload(prompt, metadata)
    payload.update({"router_profile": profile, "router_decision": asdict(decision)})
    return payload


def get_dataset(args: argparse.Namespace, out_dir: str) -> List[Dict[str, Any]]:
    if args.dataset == "realhitbench":
        return realhit_entry.get_dataset(args, output_dir=out_dir)
    spreadsheet_out_dir = os.path.join(out_dir, "spreadsheet")
    if os.path.exists(spreadsheet_out_dir) and not args.resume and not args.dry_run:
        shutil.rmtree(spreadsheet_out_dir)
    os.makedirs(spreadsheet_out_dir, exist_ok=True)
    os.chmod(spreadsheet_out_dir, 0o777)
    return spreadsheet_entry.get_dataset(args, spreadsheet_out_dir)


def load_resume_state(dataset: str, args: argparse.Namespace, out_dir: str, data: List[Dict[str, Any]]):
    outs, eval_results = [None] * len(data), [None] * len(data)
    score_lists = defaultdict(lambda: defaultdict(list)) if dataset == "realhitbench" else defaultdict(list)
    partial_out_path = os.path.join(out_dir, output_filenames(dataset, partial=True)[0])
    partial_eval_path = os.path.join(out_dir, output_filenames(dataset, partial=True)[1])

    if not args.resume:
        return outs, eval_results, score_lists

    if not os.path.exists(partial_out_path):
        print(f"Resume enabled, but no partial result found in {out_dir}; starting from scratch.")
        return outs, eval_results, score_lists

    loaded_outs = load_jsonl(partial_out_path)
    loaded_eval_results = load_jsonl(partial_eval_path) if os.path.exists(partial_eval_path) else []
    resumed_count = 0
    for idx, result in enumerate(loaded_outs[: len(data)]):
        if result is None:
            continue
        if str(result.get("id")) != str(data[idx].get("id")):
            continue
        outs[idx] = result
        eval_results[idx] = loaded_eval_results[idx] if idx < len(loaded_eval_results) else build_eval_entry(dataset, result)
        add_scores(dataset, score_lists, result)
        resumed_count += 1

    print(f"Resume loaded {resumed_count} completed results from {partial_out_path}")
    return outs, eval_results, score_lists


def solution(args: argparse.Namespace) -> None:
    if args.base_url:
        os.environ["BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["API_KEY"] = args.api_key

    out_dir = build_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    data = get_dataset(args, out_dir)
    profiler = SpreadsheetProfiler(args)
    router = SingleResponseHeuristicRouter(args)

    if args.dry_run:
        if not data:
            print("No data selected.")
            return
        item = data[min(args.dry_run_index, len(data) - 1)]
        profile = profiler.profile(args.dataset, item)
        decision = router.route(args.dataset, item, profile)
        route_args = make_route_args(args, decision.table_format)
        solver = build_solver(args.dataset, route_args, out_dir)
        prompt, metadata = solver.build_prompt(item)
        path = os.path.join(out_dir, "dry_run_prompt.json")
        save_jsonl(dry_run_payload(args.dataset, prompt, metadata, profile, decision), path)
        print(f"Dry-run prompt saved to {path}")
        return

    outs, eval_results, score_lists = load_resume_state(args.dataset, args, out_dir, data)
    pending_indices = [idx for idx, result in enumerate(outs) if result is None]

    if not pending_indices:
        scores = average_scores(args.dataset, score_lists)
        save_outputs(args.dataset, out_dir, outs, eval_results, scores, partial=False)
        cleanup_intermediate_files(args.dataset, out_dir)
        print("No pending examples. Final files have been saved.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(solve_one, args.dataset, args, data[idx], out_dir, profiler, router): idx
            for idx in pending_indices
        }
        for solved_count, future in tqdm(
            enumerate(as_completed(futures), start=1),
            total=len(pending_indices),
            desc=f"Solving SingleRespRouter-{args.dataset}",
        ):
            idx = futures[future]
            try:
                result = future.result()
            except Exception:
                result = build_error_result(args.dataset, data[idx], traceback.format_exc(), args)
            outs[idx] = result
            eval_results[idx] = build_eval_entry(args.dataset, result)
            add_scores(args.dataset, score_lists, result)

            if args.save_every and solved_count % args.save_every == 0:
                scores = average_scores(args.dataset, score_lists)
                save_outputs(args.dataset, out_dir, outs, eval_results, scores, partial=True)

            if solved_count % args.report_every == 0 or solved_count == len(pending_indices):
                report_scores(args.dataset, score_lists)

    scores = average_scores(args.dataset, score_lists)
    save_outputs(args.dataset, out_dir, outs, eval_results, scores, partial=False)
    cleanup_intermediate_files(args.dataset, out_dir)
    print(f"Saved SingleRespRouter outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-response heuristic table representation router for RealHiTBench and SpreadsheetBench."
    )

    parser.add_argument("--dataset", type=str, required=True, choices=["realhitbench", "spreadsheetbench"])
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None, help="Optional OpenAI-compatible base URL, e.g. http://host:port/v1.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key. Also readable from API_KEY/OPENAI_API_KEY.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--code_exec_url", type=str, default="localhost:8081")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument("--include_coordinates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill_merged", action="store_true")
    parser.add_argument(
        "--max_text_tokens",
        type=int,
        default=100000,
        help="Prompt table-text token budget used both for profiling and selected solver prompts.",
    )
    parser.add_argument("--render_formulas_before_eval", action="store_true")

    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--question_types", type=str, default=None)
    parser.add_argument("--instruction_types", type=str, default=None)
    parser.add_argument("--data_split", type=str, default="all_912", choices=sorted(SPREADSHEET_DATA_SPLITS))
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-s", "--suffix", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(repo_dir, "outs"))

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    solution(parse_args())
