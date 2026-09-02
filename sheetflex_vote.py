"""Run the equal-weight SheetFlex-vote baseline from existing candidate runs."""

import argparse
import json
from pathlib import Path

from core.sheetflex.common import (
    DEFAULT_TIE_BREAK_ORDER,
    FORMAT_ORDER,
    FORMAT_ORDERS,
    get_format_order,
    index_rows_by_id,
    load_indexed_runs,
    load_result_rows,
    load_run_map,
    prepare_output_dir,
    save_json,
    save_jsonl,
)
from core.sheetflex.realhit import (
    aggregate_realhit_sample,
    evaluate_realhit_vote,
    realhit_diagnostics,
)
from core.sheetflex.spreadsheet import (
    aggregate_spreadsheet_sample,
    copy_selected_workbooks,
    evaluate_spreadsheet_vote,
    spreadsheet_diagnostics,
    spreadsheet_input_path,
)


REPO_DIR = Path(__file__).resolve().parent


def _parse_ids(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _select_rows(rows, ids=None, limit=0):
    selected_ids = _parse_ids(ids)
    selected = [
        row
        for row in rows
        if selected_ids is None or str(row.get("id")) in selected_ids
    ]
    if selected_ids is not None:
        found = {str(row["id"]) for row in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Requested dataset id(s) not found: {missing}")
    return selected[:limit] if limit and limit > 0 else selected


def _resolved_run_map_trace(run_map):
    return {format_name: str(run_map[format_name]) for format_name in FORMAT_ORDER}


def run_realhit(args):
    output_dir = prepare_output_dir(args.output_dir)
    format_order = get_format_order(args.tie_break_order)
    run_map = load_run_map(args.run_map)
    indexed_runs = load_indexed_runs(run_map, "realhit_cot.jsonl")

    dataset_payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_rows = dataset_payload.get("queries")
    if not isinstance(dataset_rows, list):
        raise ValueError("RealHiT dataset JSON must contain a queries list")
    selected_rows = _select_rows(dataset_rows, args.ids, args.limit)
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(args.dataset))
    run_dirs = _resolved_run_map_trace(run_map)

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            format_name: indexed_runs[format_name].get(sample_id)
            for format_name in FORMAT_ORDER
        }
        aggregate = aggregate_realhit_sample(
            sample_id,
            item.get("QuestionType", "Unknown"),
            records,
            run_dirs=run_dirs,
            format_order=format_order,
        )
        aggregate.update(
            {
                "Question": item.get("Question"),
                "SubQType": item.get("SubQType"),
                "FileName": item.get("FileName"),
                "CompStrucCata": item.get("CompStrucCata"),
            }
        )
        aggregates.append(aggregate)

    eval_rows, score = evaluate_realhit_vote(aggregates, dataset_by_id)
    diagnostics = realhit_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "realhitbench",
            "method": "SheetFlex-vote",
            "tie_break_order": args.tie_break_order,
            "run_map": run_dirs,
        }
    )
    save_jsonl(aggregates, output_dir / "sheetflex_vote.jsonl")
    save_json(eval_rows, output_dir / "sheetflex_vote_eval.json")
    save_json(score, output_dir / "sheetflex_vote_score.json")
    save_json(diagnostics, output_dir / "sheetflex_vote_diagnostics.json")
    print(
        f"RealHiT SheetFlex-vote: samples={len(aggregates)}, "
        f"valid={sum(1 for row in aggregates if row['format_valid'])}, "
        f"output={output_dir}"
    )


def run_spreadsheet(args):
    output_dir = prepare_output_dir(args.output_dir)
    format_order = get_format_order(args.tie_break_order)
    run_map = load_run_map(args.run_map)
    indexed_runs = load_indexed_runs(run_map, "spreadsheet_pot.jsonl")

    dataset_root = Path(args.dataset_root).resolve()
    dataset_path = dataset_root / "dataset.json"
    dataset_rows = load_result_rows(dataset_path)
    selected_rows = _select_rows(dataset_rows, args.ids, args.limit)
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(dataset_path))

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            format_name: indexed_runs[format_name].get(sample_id)
            for format_name in FORMAT_ORDER
        }
        aggregates.append(
            aggregate_spreadsheet_sample(
                item,
                records,
                run_map,
                spreadsheet_input_path(dataset_root, item),
                format_order=format_order,
            )
        )

    copied = copy_selected_workbooks(aggregates, output_dir)
    eval_rows, accuracy = evaluate_spreadsheet_vote(
        aggregates, dataset_by_id, dataset_root, output_dir
    )
    diagnostics = spreadsheet_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "spreadsheetbench_verified_400",
            "method": "SheetFlex-vote",
            "tie_break_order": args.tie_break_order,
            "run_map": _resolved_run_map_trace(run_map),
            "copied_output_workbooks": copied,
        }
    )
    save_jsonl(aggregates, output_dir / "sheetflex_vote.jsonl")
    save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
    save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
    save_json(diagnostics, output_dir / "sheetflex_vote_diagnostics.json")
    print(
        f"SpreadsheetBench SheetFlex-vote: samples={len(aggregates)}, "
        f"copied={copied}, soft_all={accuracy.get('soft_all', 0):.4f}, "
        f"output={output_dir}"
    )


def _common_parser(subparser):
    subparser.add_argument("--run_map", required=True, help="JSON mapping six formats to run directories.")
    subparser.add_argument("--output_dir", required=True, help="New, empty SheetFlex-vote output directory.")
    subparser.add_argument("--ids", default=None, help="Optional comma-separated sample IDs.")
    subparser.add_argument("--limit", type=int, default=0, help="Optional maximum samples after ID filtering.")
    subparser.add_argument(
        "--tie_break_order",
        choices=tuple(FORMAT_ORDERS),
        default=DEFAULT_TIE_BREAK_ORDER,
        help="Format fallback order used when cumulative logprobs cannot decide a tie.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Equal-weight SheetFlex-vote over six existing candidate runs."
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    realhit = subparsers.add_parser("realhit", help="Aggregate RealHiTBench candidates.")
    _common_parser(realhit)
    realhit.add_argument(
        "--dataset",
        default=str(REPO_DIR / "dataset/realhitbench/realhit.json"),
    )
    realhit.set_defaults(func=run_realhit)

    spreadsheet = subparsers.add_parser(
        "spreadsheet", help="Aggregate SpreadsheetBench verified_400 candidates."
    )
    _common_parser(spreadsheet)
    spreadsheet.add_argument(
        "--dataset_root",
        default=str(
            REPO_DIR
            / "dataset/spreadsheetbench/spreadsheetbench_verified_400"
        ),
    )
    spreadsheet.set_defaults(func=run_spreadsheet)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
