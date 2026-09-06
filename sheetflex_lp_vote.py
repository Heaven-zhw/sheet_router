"""Run SheetFlex-LPVote over existing sequence-logprob candidate outputs."""

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
from core.sheetflex.lp_vote import (
    MISSING_LOGPROB_POLICIES,
    aggregate_realhit_lp_sample,
    aggregate_spreadsheet_lp_sample,
    realhit_lp_diagnostics,
    spreadsheet_lp_diagnostics,
    validate_lp_weight_strength,
)
from core.sheetflex.realhit import evaluate_realhit_vote
from core.sheetflex.spreadsheet import (
    copy_selected_workbooks,
    evaluate_spreadsheet_vote,
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


def _run_map_trace(run_map):
    return {format_name: str(run_map[format_name]) for format_name in FORMAT_ORDER}


def _nonnegative_strength(value):
    try:
        return validate_lp_weight_strength(float(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def run_realhit(args):
    output_dir = prepare_output_dir(args.output_dir)
    run_map = load_run_map(args.run_map)
    indexed_runs = load_indexed_runs(run_map, "realhit_cot.jsonl")
    format_order = get_format_order(args.tie_break_order)

    dataset_payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_rows = dataset_payload.get("queries")
    if not isinstance(dataset_rows, list):
        raise ValueError("RealHiT dataset JSON must contain a queries list")
    selected_rows = _select_rows(dataset_rows, args.ids, args.limit)
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(args.dataset))
    run_dirs = _run_map_trace(run_map)

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            format_name: indexed_runs[format_name].get(sample_id)
            for format_name in FORMAT_ORDER
        }
        aggregate = aggregate_realhit_lp_sample(
            sample_id,
            item.get("QuestionType", "Unknown"),
            records,
            run_dirs=run_dirs,
            strength=args.lp_weight_strength,
            missing_logprob_policy=args.missing_logprob_policy,
            format_order=format_order,
            tie_break_order=args.tie_break_order,
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
    diagnostics = realhit_lp_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "realhitbench",
            "method": "SheetFlex-LPVote",
            "lp_weight_strength": args.lp_weight_strength,
            "missing_logprob_policy": args.missing_logprob_policy,
            "tie_break_order": args.tie_break_order,
            "logprob_field": "sequence_logprob_mean",
            "run_map": run_dirs,
        }
    )
    save_jsonl(aggregates, output_dir / "sheetflex_lp_vote.jsonl")
    save_json(eval_rows, output_dir / "sheetflex_lp_vote_eval.json")
    save_json(score, output_dir / "sheetflex_lp_vote_score.json")
    save_json(diagnostics, output_dir / "sheetflex_lp_vote_diagnostics.json")
    print(
        f"RealHiT SheetFlex-LPVote: samples={len(aggregates)}, "
        f"valid={sum(1 for row in aggregates if row['format_valid'])}, "
        f"fallbacks={diagnostics['lpvote_fallback_samples']}, output={output_dir}"
    )


def run_spreadsheet(args):
    output_dir = prepare_output_dir(args.output_dir)
    run_map = load_run_map(args.run_map)
    indexed_runs = load_indexed_runs(run_map, "spreadsheet_pot.jsonl")
    format_order = get_format_order(args.tie_break_order)

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
            aggregate_spreadsheet_lp_sample(
                item,
                records,
                run_map,
                spreadsheet_input_path(dataset_root, item),
                strength=args.lp_weight_strength,
                missing_logprob_policy=args.missing_logprob_policy,
                format_order=format_order,
                tie_break_order=args.tie_break_order,
            )
        )

    copied = copy_selected_workbooks(aggregates, output_dir)
    eval_rows, accuracy = evaluate_spreadsheet_vote(
        aggregates, dataset_by_id, dataset_root, output_dir
    )
    diagnostics = spreadsheet_lp_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "spreadsheetbench_verified_400",
            "method": "SheetFlex-LPVote",
            "lp_weight_strength": args.lp_weight_strength,
            "missing_logprob_policy": args.missing_logprob_policy,
            "tie_break_order": args.tie_break_order,
            "logprob_field": "sequence_logprob_mean",
            "run_map": _run_map_trace(run_map),
            "copied_output_workbooks": copied,
        }
    )
    save_jsonl(aggregates, output_dir / "sheetflex_lp_vote.jsonl")
    save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
    save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
    save_json(diagnostics, output_dir / "sheetflex_lp_vote_diagnostics.json")
    print(
        f"SpreadsheetBench SheetFlex-LPVote: samples={len(aggregates)}, "
        f"copied={copied}, fallbacks={diagnostics['lpvote_fallback_samples']}, "
        f"soft_all={accuracy.get('soft_all', 0):.4f}, output={output_dir}"
    )


def _common_parser(parser):
    parser.add_argument("--run_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ids", default=None, help="Comma-separated sample IDs.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--lp_weight_strength", type=_nonnegative_strength, default=1.0
    )
    parser.add_argument(
        "--missing_logprob_policy",
        choices=MISSING_LOGPROB_POLICIES,
        default="vote",
    )
    parser.add_argument(
        "--tie_break_order",
        choices=tuple(FORMAT_ORDERS),
        default=DEFAULT_TIE_BREAK_ORDER,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Offline SheetFlex-LPVote using existing sequence_logprob_mean values."
        )
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    realhit = subparsers.add_parser("realhit")
    _common_parser(realhit)
    realhit.add_argument(
        "--dataset", default=str(REPO_DIR / "dataset/realhitbench/realhit.json")
    )
    realhit.set_defaults(func=run_realhit)

    spreadsheet = subparsers.add_parser("spreadsheet")
    _common_parser(spreadsheet)
    spreadsheet.add_argument(
        "--dataset_root",
        default=str(
            REPO_DIR
            / "dataset/spreadsheetbench/spreadsheetbench_verified_400"
        ),
    )
    spreadsheet.set_defaults(func=run_spreadsheet)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
