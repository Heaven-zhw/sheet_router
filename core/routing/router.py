import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .profile import query_features, unique_keep_order


@dataclass
class RouteDecision:
    solver_mode: str
    table_format: str
    fallback_formats: List[str]
    reason: str
    stages: Dict[str, Any] = field(default_factory=dict)


class HeuristicRouter:
    def __init__(self, args):
        self.args = args

    def route(self, dataset: str, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        if dataset == "realhitbench":
            return self._route_realhit(item, profile)
        if dataset == "spreadsheetbench":
            return self._route_spreadsheet(item, profile)
        raise ValueError(f"Unsupported dataset: {dataset}")

    def _best_text_format(self, profile: Dict[str, Any], preferred: str, default_text: str) -> str:
        available = set(profile.get("available_text_formats") or [])
        if preferred and preferred != "auto" and preferred in available:
            return preferred
        if default_text == "latex" and "latex" in available:
            return "latex"
        if default_text in available:
            return default_text
        return next(iter(available), default_text)

    def _best_image_format(self, profile: Dict[str, Any], order: Tuple[str, ...]) -> Optional[str]:
        available = set(profile.get("available_image_formats") or [])
        if self.args.image_preference == "none":
            return None
        if self.args.image_preference != "auto":
            return self.args.image_preference if self.args.image_preference in available else None
        for image_format in order:
            if image_format in available:
                return image_format
        return None

    def _text_image_format(self, text_format: str, image_format: Optional[str]) -> Optional[str]:
        if not image_format:
            return None
        if text_format not in {"latex", "markdown", "html"}:
            text_format = "markdown"
        return f"{text_format}+{image_format}"

    def _route_realhit(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        qf = profile.get("question_features") or {}
        qtype = item.get("QuestionType") or ""
        sub_qtype = item.get("SubQType") or ""
        comp = item.get("CompStrucCata") or ""
        text_format = self._best_text_format(profile, self.args.qa_text_format, "latex")
        image_format = self._best_image_format(profile, ("image", "excel_1_image", "default_image"))
        text_image = self._text_image_format(text_format, image_format)

        large_table = (profile.get("total_nonempty_cells") or 0) >= self.args.large_cell_threshold
        small_table = (profile.get("total_nonempty_cells") or 0) <= self.args.small_cell_threshold
        structure_signal = (
            qtype == "Structure Comprehending"
            or qf.get("mentions_header_or_structure")
            or profile.get("merged_cell_signal")
            or comp
            in {
                "ColumnHeaderMerge",
                "MultiColumnClassified",
                "SingleRowClassified",
                "ContentCompound",
                "StructureCompound",
            }
        )
        style_signal = qf.get("mentions_color_or_style") or comp == "BackgroundColor" or profile.get("has_background_color")
        reasoning_signal = qtype == "Numerical Reasoning" or qf.get("requires_aggregation") or "Reasoning" in sub_qtype
        retrieval_like = qtype == "Fact Checking" and not reasoning_signal and not structure_signal and not style_signal

        reason = []
        if text_image and (structure_signal or style_signal):
            primary = text_image
            reason.append("structure/style signal uses text+image evidence")
        elif text_image and reasoning_signal and (small_table or not large_table):
            primary = text_image
            reason.append("reasoning on non-large table uses text+image evidence")
        elif text_image and not retrieval_like:
            primary = text_image
            reason.append("default QA route uses complementary text+image evidence")
        else:
            primary = text_format
            reason.append("large or retrieval-like QA route keeps text-only evidence")

        fallbacks = []
        if primary != text_format:
            fallbacks.append(text_format)
        if text_image and primary != text_image:
            fallbacks.append(text_image)
        if text_format != "markdown" and image_format:
            fallbacks.append(self._text_image_format("markdown", image_format))
            fallbacks.append("markdown")

        return RouteDecision(
            solver_mode="cot_qa",
            table_format=primary,
            fallback_formats=unique_keep_order(fallbacks),
            reason="; ".join(reason),
            stages={
                "task_router": "qa",
                "representation_router": primary,
                "reasoning_router": "cot_qa",
                "signals": {
                    "large_table": large_table,
                    "small_table": small_table,
                    "structure_signal": bool(structure_signal),
                    "style_signal": bool(style_signal),
                    "reasoning_signal": bool(reasoning_signal),
                    "retrieval_like": bool(retrieval_like),
                    "image_format": image_format,
                    "text_format": text_format,
                },
            },
        )

    def _route_spreadsheet(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        qf = profile.get("question_features") or {}
        text_format = self._best_text_format(profile, self.args.operation_text_format, "markdown")
        if text_format != "markdown":
            text_format = "markdown"
        image_format = self._best_image_format(profile, ("excel_1_image", "image", "default_image"))
        text_image = self._text_image_format(text_format, image_format)

        style_signal = qf.get("mentions_color_or_style") or qf.get("requires_formatting")
        workbook_style = profile.get("has_background_color") or profile.get("num_bordered_cells", 0) > 0
        truncation_risk = profile.get("truncation_risk")
        multi_image_risk = profile.get("multi_image_risk")
        exact_cell_signal = qf.get("mentions_exact_cell_or_range") or bool(item.get("answer_position"))

        reason = []
        if text_image and style_signal and workbook_style:
            primary = text_image
            reason.append("operation requires visual/style evidence, keep markdown plus image")
        elif text_image and truncation_risk and not multi_image_risk:
            primary = text_image
            reason.append("text truncation risk with low image count, add image as complementary evidence")
        else:
            primary = text_format
            reason.append("operation route defaults to markdown for code generation stability")

        fallbacks = []
        if primary != text_format:
            fallbacks.append(text_format)
        if text_image and primary != text_image:
            fallbacks.append(text_image)
        if image_format != "default_image" and "default_image" in set(profile.get("available_image_formats") or []):
            default_combo = self._text_image_format(text_format, "default_image")
            if default_combo != primary:
                fallbacks.append(default_combo)

        return RouteDecision(
            solver_mode="pot_code",
            table_format=primary,
            fallback_formats=unique_keep_order(fallbacks),
            reason="; ".join(reason),
            stages={
                "task_router": "operation",
                "representation_router": primary,
                "reasoning_router": "pot_code",
                "signals": {
                    "style_signal": bool(style_signal),
                    "workbook_style": bool(workbook_style),
                    "truncation_risk": bool(truncation_risk),
                    "multi_image_risk": bool(multi_image_risk),
                    "exact_cell_signal": bool(exact_cell_signal),
                    "image_format": image_format,
                    "text_format": text_format,
                },
            },
        )


class BlackBoxHeuristicRouter(HeuristicRouter):
    def _route_realhit(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        qf = query_features(item.get("Question", ""), "qa")
        text_format = self._best_text_format(profile, self.args.qa_text_format, "markdown")
        if text_format == "latex" and self.args.qa_text_format == "auto":
            text_format = "markdown"

        image_format = self._best_image_format(profile, ("image", "excel_1_image", "default_image"))
        text_image = self._text_image_format(text_format, image_format)
        total_cells = profile.get("total_nonempty_cells") or 0
        large_table = total_cells >= self.args.large_cell_threshold
        small_table = total_cells <= self.args.small_cell_threshold
        low_density = (profile.get("nonempty_ratio") or 0) < 0.55

        structure_need_score = 0
        structure_need_score += 2 if qf.get("mentions_header_or_structure") else 0
        structure_need_score += 2 if profile.get("merged_cell_signal") else 0
        structure_need_score += 1 if (profile.get("num_merged_ranges") or 0) >= 3 else 0
        structure_need_score += 1 if (profile.get("num_sheets") or 0) >= 2 else 0
        structure_need_score += 1 if low_density else 0
        structure_need_score += 1 if (profile.get("max_cols") or 0) >= 10 and (profile.get("max_rows") or 0) >= 20 else 0

        visual_need_score = 0
        visual_need_score += 3 if qf.get("mentions_color_or_style") else 0
        visual_need_score += 2 if profile.get("has_background_color") else 0
        visual_need_score += 2 if profile.get("has_charts_or_images") else 0
        visual_need_score += 1 if (profile.get("num_bordered_cells") or 0) > 0 else 0

        reasoning_need_score = 0
        reasoning_need_score += 2 if qf.get("requires_aggregation") else 0
        reasoning_need_score += 1 if qf.get("requires_comparison") else 0
        reasoning_need_score += 1 if qf.get("requires_lookup") else 0
        reasoning_need_score += 1 if qf.get("mentions_sort_filter_pivot") else 0
        reasoning_need_score += 1 if qf.get("mentions_date_time") else 0
        reasoning_need_score += min(2, int(qf.get("num_operations_in_instruction") or 0) // 2)

        retrieval_like = (
            not qf.get("requires_aggregation")
            and not qf.get("requires_lookup")
            and not qf.get("requires_comparison")
            and structure_need_score <= 1
            and visual_need_score == 0
            and (qf.get("query_tokens_est") or 0) <= 50
        )

        reason = []
        if text_image and not (large_table and retrieval_like):
            primary = text_image
            reason.append("black-box QA defaults to markdown+image when visual evidence is available")
        else:
            primary = text_format
            reason.append("black-box QA keeps text-only for large retrieval-like cases or unavailable images")

        fallbacks = []
        if primary != text_format:
            fallbacks.append(text_format)
        if image_format and text_format != "latex" and "latex" in set(profile.get("available_text_formats") or []):
            fallbacks.append(self._text_image_format("latex", image_format))
            fallbacks.append("latex")
        if image_format != "excel_1_image" and "excel_1_image" in set(profile.get("available_image_formats") or []):
            excel_combo = self._text_image_format(text_format, "excel_1_image")
            if excel_combo != primary:
                fallbacks.append(excel_combo)
        if image_format != "default_image" and "default_image" in set(profile.get("available_image_formats") or []):
            default_combo = self._text_image_format(text_format, "default_image")
            if default_combo != primary:
                fallbacks.append(default_combo)

        return RouteDecision(
            solver_mode="cot_qa",
            table_format=primary,
            fallback_formats=unique_keep_order(fallbacks),
            reason="; ".join(reason),
            stages={
                "task_router": "qa",
                "representation_router": primary,
                "reasoning_router": "cot_qa",
                "blackbox_constraints": {
                    "uses_benchmark_labels": False,
                    "uses_answer_position": False,
                    "uses_answer_sheet": False,
                    "uses_llm_router": False,
                },
                "signals": {
                    "task_family": "qa",
                    "large_table": bool(large_table),
                    "small_table": bool(small_table),
                    "low_density": bool(low_density),
                    "structure_need_score": structure_need_score,
                    "visual_need_score": visual_need_score,
                    "reasoning_need_score": reasoning_need_score,
                    "retrieval_like": bool(retrieval_like),
                    "image_format": image_format,
                    "text_format": text_format,
                    "query_features": qf,
                },
            },
        )

    def _route_spreadsheet(self, item: Dict[str, Any], profile: Dict[str, Any]) -> RouteDecision:
        qf = query_features(item.get("instruction", ""), "operation")
        text_format = "markdown"
        image_format = self._best_image_format(profile, ("excel_1_image", "image", "default_image"))
        excel_image = "excel_1_image" if "excel_1_image" in set(profile.get("available_image_formats") or []) else None

        image_count = (profile.get("image_counts") or {}).get(image_format or "", 0)
        total_cells = profile.get("total_nonempty_cells") or 0
        large_table = total_cells >= self.args.large_cell_threshold
        low_density = (profile.get("nonempty_ratio") or 0) < 0.55
        multi_image_risk = (profile.get("num_sheets") or 0) >= 3

        visual_need_score = 0
        visual_need_score += 3 if qf.get("mentions_color_or_style") else 0
        visual_need_score += 2 if profile.get("has_background_color") else 0
        visual_need_score += 2 if profile.get("has_charts_or_images") else 0
        visual_need_score += 1 if (profile.get("num_bordered_cells") or 0) > 0 else 0

        layout_need_score = 0
        layout_need_score += 2 if qf.get("mentions_header_or_structure") else 0
        layout_need_score += 2 if qf.get("mentions_cross_sheet") or qf.get("mentions_sheet_name") else 0
        layout_need_score += 1 if qf.get("requires_lookup") else 0
        layout_need_score += 1 if qf.get("mentions_sort_filter_pivot") else 0
        layout_need_score += 1 if qf.get("mentions_insert_delete") else 0
        layout_need_score += 1 if profile.get("merged_cell_signal") else 0
        layout_need_score += 1 if (profile.get("num_sheets") or 0) >= 2 else 0
        layout_need_score += 1 if low_density else 0

        operation_complexity_score = 0
        operation_complexity_score += 2 if qf.get("requires_lookup") else 0
        operation_complexity_score += 2 if qf.get("requires_aggregation") else 0
        operation_complexity_score += 1 if qf.get("mentions_sort_filter_pivot") else 0
        operation_complexity_score += 1 if qf.get("mentions_insert_delete") else 0
        operation_complexity_score += min(3, int(qf.get("num_operations_in_instruction") or 0) // 2)

        image_risk_score = 0
        image_risk_score += 2 if multi_image_risk else 0
        image_risk_score += 1 if image_count and image_count >= 3 else 0
        image_risk_score += 1 if large_table else 0
        image_risk_score += 1 if image_format == "default_image" else 0

        truncation_risk = bool(profile.get("truncation_risk"))
        reason = []
        if excel_image and (
            visual_need_score >= 4
            or (layout_need_score >= 5 and operation_complexity_score >= 3 and image_risk_score <= 3)
            or (truncation_risk and image_risk_score <= 2)
        ):
            primary = "markdown+excel_1_image"
            reason.append("black-box operation adds excel-style image for strong visual/layout need")
        elif image_format and image_format != "default_image" and visual_need_score >= 5 and image_risk_score <= 2:
            primary = self._text_image_format(text_format, image_format)
            reason.append("black-box operation adds non-default image for strong visual need")
        elif image_format == "default_image" and visual_need_score >= 6 and image_risk_score <= 1:
            primary = "markdown+default_image"
            reason.append("black-box operation uses default image only for very strong low-risk visual need")
        else:
            primary = text_format
            reason.append("black-box operation defaults to markdown for code-generation stability")

        fallbacks = []
        if primary != text_format:
            fallbacks.append(text_format)
        if primary == text_format:
            if excel_image:
                fallbacks.append("markdown+excel_1_image")
            elif image_format and image_format != "default_image":
                fallbacks.append(self._text_image_format(text_format, image_format))
            elif image_format == "default_image":
                fallbacks.append("markdown+default_image")
        if primary != "markdown+default_image" and "default_image" in set(profile.get("available_image_formats") or []):
            fallbacks.append("markdown+default_image")

        return RouteDecision(
            solver_mode="pot_code",
            table_format=primary,
            fallback_formats=unique_keep_order(fallbacks),
            reason="; ".join(reason),
            stages={
                "task_router": "operation",
                "representation_router": primary,
                "reasoning_router": "pot_code",
                "blackbox_constraints": {
                    "uses_benchmark_labels": False,
                    "uses_answer_position": False,
                    "uses_answer_sheet": False,
                    "uses_llm_router": False,
                },
                "signals": {
                    "task_family": "operation",
                    "large_table": bool(large_table),
                    "low_density": bool(low_density),
                    "multi_image_risk": bool(multi_image_risk),
                    "truncation_risk": truncation_risk,
                    "visual_need_score": visual_need_score,
                    "layout_need_score": layout_need_score,
                    "operation_complexity_score": operation_complexity_score,
                    "image_risk_score": image_risk_score,
                    "image_format": image_format,
                    "text_format": text_format,
                    "query_features": qf,
                },
            },
        )


def build_router(args) -> HeuristicRouter:
    if args.router_policy == "heuristic":
        return HeuristicRouter(args)
    if args.router_policy == "blackbox_heuristic":
        return BlackBoxHeuristicRouter(args)
    raise ValueError(f"Unsupported router policy: {args.router_policy}")


def sanitize_router_profile(dataset: str, args, item: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    if args.router_policy != "blackbox_heuristic":
        return profile

    cleaned = copy.deepcopy(profile)
    if dataset == "realhitbench":
        for key in ("question_type", "sub_question_type", "complex_structure"):
            cleaned.pop(key, None)
        cleaned["question_features"] = query_features(item.get("Question", ""), "qa")
    elif dataset == "spreadsheetbench":
        for key in ("instruction_type", "answer_position", "answer_sheet"):
            cleaned.pop(key, None)
        cleaned["question_features"] = query_features(item.get("instruction", ""), "operation")
    cleaned["blackbox_profile"] = True
    return cleaned
