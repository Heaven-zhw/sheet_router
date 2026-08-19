import argparse
import copy
import os
import shutil
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import realhit_cot as realhit_entry
import spreadsheet_pot as spreadsheet_entry
from core.routing import (
    RouteDecision,
    SpreadsheetProfiler,
    build_router,
    sanitize_router_profile,
    unique_keep_order,
)
from core.solver.realhit_cot import RealHiTCoTSolver
from core.solver.spreadsheet_pot import SPREADSHEET_DATA_SPLITS, SpreadSheetPoTSolver
from core.utils import load_jsonl, save_jsonl


repo_dir = os.path.abspath(os.path.dirname(__file__))


def safe_name(value: Optional[str]) -> str:
    return (value or "default_model").replace("/", "_").replace("\\", "_")


class ObservableVerifier:
    def verify(self, dataset: str, result: Dict[str, Any]) -> Dict[str, Any]:
        reasons = []
        if not result.get("format_valid"):
            reasons.append("format_invalid")

        if dataset == "realhitbench":
            if not result.get("model_answer"):
                reasons.append("empty_model_answer")
        else:
            if not result.get("execution_success"):
                reasons.append("execution_failed")
            if not result.get("solution"):
                reasons.append("empty_solution")

        return {
            "passed": not reasons,
            "reasons": reasons,
            "uses_gold_signal": False,
        }


def build_router_suffix(args: argparse.Namespace) -> str:
    parts = ["router", args.router_policy]
    if args.fallback:
        parts.append("fallback")
    if args.fill_merged:
        parts.append("fillmerged")
    if not args.include_coordinates:
        parts.append("nocoord")
    if args.max_text_tokens:
        token_suffix = f"{int(args.max_text_tokens / 1000)}ktoken"
        parts.append(token_suffix)
    else:
        token_suffix = None
    if args.top_p != 1.0 or args.temperature != 0:
        parts.append(f"tp{args.top_p}_temp{args.temperature}")
    if args.suffix and args.suffix != token_suffix:
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


def result_summary(dataset: str, result: Dict[str, Any], table_format: str, verifier: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "table_format": table_format,
        "format_valid": bool(result.get("format_valid")),
        "error": result.get("error"),
        "verifier": verifier,
        "table_metadata": result.get("table_metadata"),
    }
    if dataset == "realhitbench":
        eval_scores = result.get("eval") or {}
        summary.update(
            {
                "model_answer": result.get("model_answer"),
                "F1": eval_scores.get("F1"),
                "EM": eval_scores.get("EM"),
            }
        )
    else:
        summary.update(
            {
                "execution_success": bool(result.get("execution_success")),
                "total_soft_restriction": result.get("total_soft_restriction"),
                "total_hard_restriction": result.get("total_hard_restriction"),
                "test_case_results": result.get("test_case_results"),
            }
        )
    return summary


def attach_router_metadata(
    result: Dict[str, Any],
    decision: RouteDecision,
    profile: Dict[str, Any],
    route_attempts: List[Dict[str, Any]],
    failed_results: List[Dict[str, Any]],
    fallback_used: bool,
    final_verifier: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    result = copy.deepcopy(result)
    result["router_policy"] = args.router_policy
    result["router_decision"] = asdict(decision)
    result["router_profile"] = profile
    result["router_attempts"] = route_attempts
    result["router_verifier"] = final_verifier
    result["router_fallback_used"] = fallback_used
    if args.save_all_route_attempts and failed_results:
        result["router_failed_attempt_results"] = failed_results
    return result


def solve_one(
    dataset: str,
    args: argparse.Namespace,
    item: Dict[str, Any],
    out_dir: str,
    profiler: SpreadsheetProfiler,
    router: Any,
    verifier: ObservableVerifier,
) -> Dict[str, Any]:
    profile = profiler.profile(dataset, item)
    decision = router.route(dataset, item, profile)
    formats = unique_keep_order([decision.table_format] + (decision.fallback_formats if args.fallback else []))
    if args.max_fallbacks >= 0:
        formats = formats[: 1 + args.max_fallbacks]

    route_attempts = []
    failed_results = []
    final_result: Optional[Dict[str, Any]] = None
    final_verifier: Dict[str, Any] = {"passed": False, "reasons": ["not_run"], "uses_gold_signal": False}
    fallback_used = False

    for attempt_index, table_format in enumerate(formats, start=1):
        try:
            result = run_solver(dataset, args, item, table_format, out_dir)
        except Exception:
            result = build_error_result(dataset, item, traceback.format_exc(), args)

        observed = verifier.verify(dataset, result)
        route_attempts.append(
            {
                "attempt_index": attempt_index,
                "is_primary": attempt_index == 1,
                "table_format": table_format,
                "summary": result_summary(dataset, result, table_format, observed),
            }
        )

        final_result = result
        final_verifier = observed
        if observed["passed"]:
            fallback_used = attempt_index > 1
            break

        if attempt_index < len(formats):
            if args.save_all_route_attempts:
                failed_results.append(result)
            continue

    if final_result is None:
        final_result = build_error_result(dataset, item, "No route was available.", args)

    profile_for_output = sanitize_router_profile(dataset, args, item, profile)
    return attach_router_metadata(
        final_result,
        decision,
        profile_for_output,
        route_attempts,
        failed_results,
        fallback_used,
        final_verifier,
        args,
    )


def build_realhit_eval_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry = realhit_entry.build_eval_entry(result)
    entry.update(
        {
            "router_decision": result.get("router_decision"),
            "router_verifier": result.get("router_verifier"),
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
            "router_verifier": result.get("router_verifier"),
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
    names = list(output_filenames(dataset, partial=True))
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def save_outputs(dataset: str, out_dir: str, outs: List[Optional[Dict[str, Any]]], eval_results: List[Any], scores: Dict[str, Any], partial: bool = False) -> None:
    out_name, eval_name, score_name = output_filenames(dataset, partial=partial)
    save_jsonl(outs, os.path.join(out_dir, out_name))
    save_jsonl(eval_results, os.path.join(out_dir, eval_name))
    save_jsonl(scores, os.path.join(out_dir, score_name))

    if not partial:
        decisions = [
            {
                "id": result.get("id") if result else None,
                "router_decision": result.get("router_decision") if result else None,
                "router_verifier": result.get("router_verifier") if result else None,
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
    canonical_out, canonical_eval, _ = output_filenames(dataset, partial=True)
    partial_out_path = os.path.join(out_dir, canonical_out)
    partial_eval_path = os.path.join(out_dir, canonical_eval)

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
    router = build_router(args)
    verifier = ObservableVerifier()

    if args.dry_run:
        if not data:
            print("No data selected.")
            return
        item = data[min(args.dry_run_index, len(data) - 1)]
        profile = profiler.profile(args.dataset, item)
        decision = router.route(args.dataset, item, profile)
        profile_for_output = sanitize_router_profile(args.dataset, args, item, profile)
        route_args = make_route_args(args, decision.table_format)
        solver = build_solver(args.dataset, route_args, out_dir)
        prompt, metadata = solver.build_prompt(item)
        path = os.path.join(out_dir, "dry_run_prompt.json")
        save_jsonl(dry_run_payload(args.dataset, prompt, metadata, profile_for_output, decision), path)
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
            executor.submit(solve_one, args.dataset, args, data[idx], out_dir, profiler, router, verifier): idx
            for idx in pending_indices
        }
        for solved_count, future in tqdm(
            enumerate(as_completed(futures), start=1),
            total=len(pending_indices),
            desc=f"Solving SheetRouter-{args.dataset}",
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
    print(f"Saved SheetRouter outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic routing entry for RealHiTBench and SpreadsheetBench.")

    parser.add_argument("--dataset", type=str, required=True, choices=["realhitbench", "spreadsheetbench"])
    parser.add_argument("--router_policy", type=str, default="heuristic", choices=["heuristic", "blackbox_heuristic"])

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
        help="Prompt table-text token budget. Default caps text evidence at 100K tokens.",
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

    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_fallbacks", type=int, default=1, help="Maximum observable fallback routes to try after the primary route.")
    parser.add_argument("--save_all_route_attempts", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--qa_text_format", type=str, default="auto", choices=["auto", "latex", "markdown", "html"])
    parser.add_argument("--operation_text_format", type=str, default="markdown", choices=["markdown", "html", "latex"])
    parser.add_argument("--image_preference", type=str, default="auto", choices=["auto", "image", "excel_1_image", "default_image", "none"])
    parser.add_argument("--large_cell_threshold", type=int, default=1500)
    parser.add_argument("--small_cell_threshold", type=int, default=400)

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    solution(parse_args())
