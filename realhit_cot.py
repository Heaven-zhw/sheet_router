import argparse
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from core.solver.realhit_cot import SUPPORTED_TABLE_FORMATS, RealHiTCoTSolver
from core.utils import load_jsonl, save_jsonl


repo_dir = os.path.abspath(os.path.dirname(__file__))
OVERALL_SCORE_KEY = "Overall"
REALHIT_METRICS = ["F1", "EM", "ROUGE-L", "SacreBLEU"]
REALHIT_SCORE_ORDER = [
    OVERALL_SCORE_KEY,
    "Fact Checking",
    "Numerical Reasoning",
    "Structure Comprehending",
]


def get_dataset(args, output_dir=None):
    dataset_path = os.path.join(repo_dir, "dataset/realhitbench/realhit.json")
    data = load_jsonl(dataset_path)

    real_dir = os.path.join(repo_dir, "dataset/realhitbench/tables")
    latex_dir = os.path.join(repo_dir, "dataset/realhitbench/latex")
    image_dir = os.path.join(repo_dir, "dataset/realhitbench/image")
    excel_1_image_dir = os.path.join(repo_dir, "dataset/realhitbench/excel_1_images")

    queries = []
    selected_ids = None
    if args.ids:
        selected_ids = {int(item.strip()) for item in args.ids.split(",") if item.strip()}

    selected_types = None
    if args.question_types:
        selected_types = {item.strip() for item in args.question_types.split(",") if item.strip()}

    for item in data["queries"]:
        if selected_ids is not None and int(item["id"]) not in selected_ids:
            continue
        if selected_types is not None and item.get("QuestionType") not in selected_types:
            continue

        item = dict(item)
        item.update(
            {
                "real_dir": real_dir,
                "latex_dir": latex_dir,
                "image_dir": image_dir,
                "excel_1_image_dir": excel_1_image_dir,
                "image_cache_dir": (
                    os.path.join(output_dir, "table_images")
                    if output_dir
                    else None
                ),
                "mount_dir": {real_dir: "/mnt/data/input"},
            }
        )
        queries.append(item)

    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    return queries


def average_scores(score_lists):
    scores = {}
    ordered_question_types = [
        question_type for question_type in REALHIT_SCORE_ORDER if question_type in score_lists
    ]
    ordered_question_types.extend(
        question_type
        for question_type in sorted(score_lists)
        if question_type not in REALHIT_SCORE_ORDER
    )
    for question_type in ordered_question_types:
        metric_values = score_lists[question_type]
        scores[question_type] = {}
        for metric_name, values in metric_values.items():
            numeric = [value for value in values if value is not None]
            scores[question_type][metric_name] = sum(numeric) / len(values) if values else 0.0
    return scores


def report_scores(score_lists):
    scores = average_scores(score_lists)
    for question_type in sorted(scores):
        pieces = []
        for metric_name in sorted(scores[question_type]):
            pieces.append(f"{metric_name}: {scores[question_type][metric_name]:.4f}")
        tqdm.write(f"{question_type} " + ", ".join(pieces))


def build_eval_entry(result):
    return {
        key: result.get(key, None)
        for key in [
            "id",
            "Question",
            "QuestionType",
            "SubQType",
            "FileName",
            "format_valid",
            "error",
            "eval",
        ]
    }


def add_scores(score_lists, result):
    question_type = result.get("QuestionType", "Unknown")
    eval_scores = result.get("eval", {}) or {}
    format_valid = 100.0 if result.get("format_valid") else 0.0

    for bucket in (OVERALL_SCORE_KEY, question_type):
        for metric_name in REALHIT_METRICS:
            score_lists[bucket][metric_name].append(eval_scores.get(metric_name))
        score_lists[bucket]["FormatValid"].append(format_valid)


def build_suffix(args):
    parts = ["cot", args.table_format]
    if args.fill_merged:
        parts.append("fillmerged")
    if not args.include_coordinates:
        parts.append("nocoord")
    if args.top_p != 1.0 or args.temperature != 0:
        parts.append(f"tp{args.top_p}_temp{args.temperature}")
    if args.suffix:
        parts.append(args.suffix)
    return "_".join(parts)


def build_model_dir(args):
    model_name = args.model_name or "default_model"
    return model_name.replace("/", "_").replace("\\", "_")


def build_dataset_dir(args):
    dataset_name = "realhitbench"
    if args.data_split:
        dataset_name = f"{dataset_name}_{args.data_split}"
    return dataset_name.replace("/", "_").replace("\\", "_")


def dry_run_payload(prompt, metadata):
    if isinstance(prompt, list):
        prompt_summary = []
        for item in prompt:
            if item.get("type") == "image_url":
                prompt_summary.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "<base64 image omitted from dry-run output>",
                        },
                    }
                )
            else:
                prompt_summary.append(item)
        return {"prompt": prompt_summary, "metadata": metadata}
    return {"prompt": prompt, "metadata": metadata}


def first_existing_path(out_dir, names):
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(out_dir, names[0])


def solution(args):
    out_dir = os.path.join(repo_dir, args.output_root, build_dataset_dir(args), build_model_dir(args), build_suffix(args))
    os.makedirs(out_dir, exist_ok=True)
    data = get_dataset(args, output_dir=out_dir)

    solver = RealHiTCoTSolver(**vars(args))

    score_lists = defaultdict(lambda: defaultdict(list))
    outs, eval_results = [None] * len(data), [None] * len(data)
    partial_out_path = first_existing_path(out_dir, ["realhit_cot.partial.jsonl", "realhit_cot.partial.json"])
    partial_eval_path = first_existing_path(out_dir, ["realhit_cot_eval.partial.json", "realhit_cot_eval.partial.jsonl"])

    if args.dry_run:
        if not data:
            print("No data selected.")
            return
        prompt, metadata = solver.build_prompt(data[0])
        # save_jsonl({"prompt": prompt, "metadata": metadata}, os.path.join(out_dir, "dry_run_prompt.json"))
        save_jsonl(dry_run_payload(prompt, metadata), os.path.join(out_dir, "dry_run_prompt.json"))
        print(f"Dry-run prompt saved to {os.path.join(out_dir, 'dry_run_prompt.json')}")
        return

    if args.resume and os.path.exists(partial_out_path):
        loaded_outs = load_jsonl(partial_out_path)
        loaded_eval_results = load_jsonl(partial_eval_path) if os.path.exists(partial_eval_path) else []
        resumed_count = 0

        for idx, result in enumerate(loaded_outs[: len(data)]):
            if result is None:
                continue
            if str(result.get("id")) != str(data[idx].get("id")):
                continue

            outs[idx] = result
            if idx < len(loaded_eval_results) and loaded_eval_results[idx] is not None:
                eval_results[idx] = loaded_eval_results[idx]
            else:
                eval_results[idx] = build_eval_entry(result)
            add_scores(score_lists, result)
            resumed_count += 1

        print(f"Resume loaded {resumed_count} completed results from {partial_out_path}")
    elif args.resume:
        print(f"Resume enabled, but no partial result found in {out_dir}; starting from scratch.")

    pending_indices = [idx for idx, result in enumerate(outs) if result is None]
    if not pending_indices:
        scores = average_scores(score_lists)
        save_jsonl(outs, os.path.join(out_dir, "realhit_cot.jsonl"))
        save_jsonl(scores, os.path.join(out_dir, "realhit_cot_score.json"))
        save_jsonl(eval_results, os.path.join(out_dir, "realhit_cot_eval.json"))
        print("No pending examples. Final files have been saved.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solver, data[idx]): idx for idx in pending_indices}
        for solved_count, future in tqdm(
            enumerate(as_completed(futures), start=1),
            total=len(pending_indices),
            desc="Solving RealHiT-CoT",
        ):
            idx = futures[future]
            result = future.result()
            outs[idx] = result
            eval_results[idx] = build_eval_entry(result)
            add_scores(score_lists, result)

            if args.save_every and (solved_count % args.save_every == 0):
                save_jsonl(outs, os.path.join(out_dir, "realhit_cot.partial.jsonl"))
                save_jsonl(eval_results, os.path.join(out_dir, "realhit_cot_eval.partial.json"))

            if solved_count % args.report_every == 0 or solved_count == len(pending_indices):
                report_scores(score_lists)

    scores = average_scores(score_lists)

    save_jsonl(outs, os.path.join(out_dir, "realhit_cot.jsonl"))
    save_jsonl(scores, os.path.join(out_dir, "realhit_cot_score.json"))
    save_jsonl(eval_results, os.path.join(out_dir, "realhit_cot_eval.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="Direct CoT baseline for RealHiTBench table question answering.")

    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument(
        "--table_format",
        type=str,
        default="official_latex",
        choices=SUPPORTED_TABLE_FORMATS,
        help="Spreadsheet input representation passed to the LLM.",
    )
    parser.add_argument(
        "--include_coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For non-LaTeX formats, include row/column coordinates in the serialized table.",
    )
    parser.add_argument(
        "--fill_merged",
        action="store_true",
        help="For non-LaTeX formats, repeat top-left merged-cell values across the merged range.",
    )
    parser.add_argument(
        "--max_text_tokens",
        type=int,
        default=0,
        help="Optional prompt table-text token limit. 0 means no truncation.",
    )

    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data_split", type=str, default=None, help="Optional RealHiTBench split name for output paths.")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated RealHiT ids to run.")
    parser.add_argument(
        "--question_types",
        type=str,
        default=None,
        help="Comma-separated QuestionType filters, e.g. 'Fact Checking,Numerical Reasoning'.",
    )
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from partial result files in the output directory.")
    parser.add_argument("-s", "--suffix", type=str, help="可用于超参数调试，加上后缀区分实验组", default=None)
    parser.add_argument("--output_root", type=str, default="outs", help="Root directory for result outputs.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore")
    solution(args)
