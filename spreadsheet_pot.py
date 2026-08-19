import argparse
import os
import shutil
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from core.solver.spreadsheet_pot import (
    SPREADSHEET_DATA_SPLITS,
    SUPPORTED_TABLE_FORMATS,
    SpreadSheetPoTSolver,
)
from core.utils import load_jsonl, save_jsonl


repo_dir = os.path.abspath(os.path.dirname(__file__))


def resolve_image_dir(image_root, item_id):
    if not os.path.isdir(image_root):
        return None

    # 首先按照image_root/id为路径读取
    item_dir = os.path.join(image_root, str(item_id))
    if os.path.isdir(item_dir):
        return item_dir

    # 兼容扁平目录：所有样例图片直接放在 image_root 下，例如 1_13-1_init___1.png。
    return image_root


def get_dataset(args, output_dir):
    split_config = SPREADSHEET_DATA_SPLITS[args.data_split]
    dataset_root = os.path.join(repo_dir, split_config["root"])
    data = load_jsonl(os.path.join(dataset_root, "dataset.json"))

    selected_ids = None
    if args.ids:
        selected_ids = {item.strip() for item in args.ids.split(",") if item.strip()}

    selected_instruction_types = None
    if args.instruction_types:
        selected_instruction_types = {item.strip() for item in args.instruction_types.split(",") if item.strip()}

    image_root = os.path.join(dataset_root, "image")
    excel_1_image_root = os.path.join(dataset_root, "excel_1_images")
    latex_root = os.path.join(dataset_root, "latex")

    outs = []
    for item in data:
        if selected_ids is not None and str(item.get("id")) not in selected_ids:
            continue
        if selected_instruction_types is not None and item.get("instruction_type") not in selected_instruction_types:
            continue

        item = dict(item)
        real_dir = os.path.join(
            dataset_root,
            item.get("spreadsheet_path", os.path.join("spreadsheet", str(item["id"]))),
        )
        input_file = f"1_{item['id']}_{split_config['input_suffix']}.xlsx"
        output_file = f"1_{item['id']}_output.xlsx"
        item.update(
            {
                "input_file": input_file,
                "output_file": output_file,
                "input_path": f"/mnt/data/input/{input_file}",
                "output_path": f"/mnt/data/output/{output_file}",
                "real_dir": real_dir,
                "dataset_root": dataset_root,
                "latex_dir": os.path.join(latex_root, str(item["id"])) if os.path.isdir(latex_root) else None,
                "image_dir": resolve_image_dir(image_root, item["id"]),
                "excel_1_image_dir": resolve_image_dir(excel_1_image_root, item["id"]),
                "image_cache_dir": os.path.join(output_dir, "table_images"),
                "mount_dir": {real_dir: "/mnt/data/input", output_dir: "/mnt/data/output"},
            }
        )
        outs.append(item)

    if args.limit and args.limit > 0:
        outs = outs[: args.limit]
    return outs


def average_scores(score_lists):
    return {key: round(sum(values) / len(values), 4) if values else 0.0 for key, values in score_lists.items()}


def report_scores(score_lists):
    scores = average_scores(score_lists)
    tqdm.write(
        "Current "
        + ", ".join(
            f"{key}: {scores[key]:.4f}"
            for key in ["soft_all", "hard_all", "soft_cell", "hard_cell", "soft_sheet", "hard_sheet"]
            if key in scores
        )
    )


def add_scores(score_lists, result):
    is_sheet = "Sheet" in str(result.get("instruction_type", ""))
    soft = result.get("total_soft_restriction", 0.0)
    hard = result.get("total_hard_restriction", 0.0)
    score_lists["soft_all"].append(soft)
    score_lists["hard_all"].append(hard)
    score_lists["soft_sheet" if is_sheet else "soft_cell"].append(soft)
    score_lists["hard_sheet" if is_sheet else "hard_cell"].append(hard)


def build_eval_entry(result):
    return {
        key: result.get(key, None)
        for key in [
            "id",
            "instruction",
            "spreadsheet_path",
            "instruction_type",
            "answer_position",
            "answer_sheet",
            "execution_success",
            "format_valid",
            "error",
            "test_case_results",
            "test_case_messages",
            "total_soft_restriction",
            "total_hard_restriction",
            "table_metadata",
        ]
    }


def build_suffix(args):
    parts = ["pot", args.table_format]
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
    dataset_name = "spreadsheetbench"
    if args.data_split:
        dataset_name = f"{dataset_name}_{args.data_split}"
    return dataset_name.replace("/", "_").replace("\\", "_")


def dry_run_payload(prompt, metadata):
    if isinstance(prompt, list):
        prompt_summary = []
        for item in prompt:
            if item.get("type") == "image_url":
                prompt_summary.append({"type": "image_url", "image_url": {"url": "<base64 image omitted>"}})
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
    spreadsheet_out_dir = os.path.join(out_dir, "spreadsheet")
    if os.path.exists(spreadsheet_out_dir) and not args.resume:
        shutil.rmtree(spreadsheet_out_dir)
    os.makedirs(spreadsheet_out_dir, exist_ok=True)
    os.chmod(spreadsheet_out_dir, 0o777)

    data = get_dataset(args, spreadsheet_out_dir)
    solver = SpreadSheetPoTSolver(**vars(args), output_dir=out_dir)

    if args.dry_run:
        if not data:
            print("No data selected.")
            return
        prompt, metadata = solver.build_prompt(data[0])
        save_jsonl(dry_run_payload(prompt, metadata), os.path.join(out_dir, "dry_run_prompt.json"))
        print(f"Dry-run prompt saved to {os.path.join(out_dir, 'dry_run_prompt.json')}")
        return

    outs, eval_results = [None] * len(data), [None] * len(data)
    score_lists = defaultdict(list)

    partial_out_path = first_existing_path(out_dir, ["spreadsheet_pot.partial.jsonl", "spreadsheet_pot.partial.json"])
    partial_eval_path = first_existing_path(
        out_dir,
        ["spreadsheet_pot_eval.partial.json", "spreadsheet_pot_eval.partial.jsonl"],
    )

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
            eval_results[idx] = loaded_eval_results[idx] if idx < len(loaded_eval_results) else build_eval_entry(result)
            add_scores(score_lists, result)
            resumed_count += 1
        print(f"Resume loaded {resumed_count} completed results from {partial_out_path}")
    elif args.resume:
        print(f"Resume enabled, but no partial result found in {out_dir}; starting from scratch.")

    pending_indices = [idx for idx, result in enumerate(outs) if result is None]
    if pending_indices:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(solver, data[idx]): idx for idx in pending_indices}
            for solved_count, future in tqdm(
                enumerate(as_completed(futures), start=1),
                total=len(pending_indices),
                desc="Solving Spreadsheet-PoT",
            ):
                idx = futures[future]
                result = future.result()
                outs[idx] = result
                eval_results[idx] = build_eval_entry(result)
                add_scores(score_lists, result)

                if args.save_every and (solved_count % args.save_every == 0):
                    save_jsonl(outs, os.path.join(out_dir, "spreadsheet_pot.partial.jsonl"))
                    save_jsonl(eval_results, os.path.join(out_dir, "spreadsheet_pot_eval.partial.json"))

                if solved_count % args.report_every == 0 or solved_count == len(pending_indices):
                    report_scores(score_lists)

    scores = average_scores(score_lists)
    save_jsonl(outs, os.path.join(out_dir, "spreadsheet_pot.jsonl"))
    save_jsonl(eval_results, os.path.join(out_dir, "spreadsheet_pot_eval.json"))
    save_jsonl(scores, os.path.join(out_dir, "spreadsheet_pot_accuracy.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="Direct PoT baseline for SpreadsheetBench spreadsheet manipulation.")

    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--code_exec_url", type=str, default="localhost:8081")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument(
        "--table_format",
        type=str,
        default="markdown",
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
    parser.add_argument(
        "--render_formulas_before_eval",
        action="store_true",
        help="Open and save each generated workbook with LibreOffice before comparing, so cached formula values are refreshed.",
    )

    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--data_split",
        type=str,
        default="all_912",
        choices=sorted(SPREADSHEET_DATA_SPLITS),
        help="SpreadsheetBench split name for output paths.",
    )
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated SpreadsheetBench ids to run.")
    parser.add_argument(
        "--instruction_types",
        type=str,
        default=None,
        help="Comma-separated instruction_type filters.",
    )
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from partial result files in the output directory.")
    parser.add_argument("-s", "--suffix", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outs", help="Root directory for result outputs.")
    
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    solution(parse_args())
