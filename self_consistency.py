"""Offline Self-Consistency aggregation, evaluation, and diagnostics."""

import argparse
import json
from pathlib import Path

from core.sheetflex.common import (
    DEFAULT_TIE_BREAK_LOGPROB,
    SheetFlexError,
    TIE_BREAK_LOGPROB_FIELDS,
    get_tie_break_logprob_field,
    index_rows_by_id,
    load_result_rows,
    save_json,
    save_jsonl,
)
from core.sheetflex.realhit import evaluate_realhit_vote
from core.sheetflex.self_consistency import (
    SelfConsistencyError,
    aggregate_self_consistency_realhit_sample,
    aggregate_self_consistency_spreadsheet_sample,
    load_indexed_candidate_runs,
    load_self_consistency_manifest,
    manifest_identity,
    self_consistency_realhit_diagnostics,
    self_consistency_spreadsheet_diagnostics,
)
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


def select_dataset_rows(dataset_rows, candidate_ids, ids=None, limit=0):
    dataset_by_id = index_rows_by_id(dataset_rows, source="dataset")
    unknown_candidate_ids = sorted(set(candidate_ids) - set(dataset_by_id))
    if unknown_candidate_ids:
        raise SelfConsistencyError(
            f"Candidate result IDs are missing from dataset: {unknown_candidate_ids}"
        )
    requested_ids = _parse_ids(ids)
    if requested_ids is None:
        requested_ids = set(candidate_ids)
    missing = sorted(requested_ids - set(candidate_ids))
    if missing:
        raise SelfConsistencyError(
            f"Requested sample IDs are missing from candidate runs: {missing}"
        )
    selected = [
        row for row in dataset_rows if str(row.get("id")) in requested_ids
    ]
    return selected[:limit] if limit and limit > 0 else selected


def prepare_output_dir(
    output_dir,
    manifest,
    resume=False,
    tie_break_logprob=DEFAULT_TIE_BREAK_LOGPROB,
):
    output_dir = Path(output_dir).resolve()
    manifest_copy = output_dir / "manifest.json"
    output_manifest = dict(manifest)
    output_manifest["aggregation_config"] = {
        "tie_break_logprob": tie_break_logprob,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise SheetFlexError(
                f"Refusing to overwrite non-empty output directory: {output_dir}"
            )
        if not manifest_copy.is_file():
            raise SelfConsistencyError(
                f"Cannot resume without output manifest: {manifest_copy}"
            )
        existing = json.loads(manifest_copy.read_text(encoding="utf-8"))
        if manifest_identity(existing) != manifest_identity(manifest):
            raise SelfConsistencyError(
                "Cannot resume: output manifest does not match requested experiment"
            )
        existing_statistic = existing.get("aggregation_config", {}).get(
            "tie_break_logprob", "sum"
        )
        if existing_statistic != tie_break_logprob:
            raise SelfConsistencyError(
                "Cannot resume: output tie-break logprob statistic does not "
                "match requested strategy"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_manifest, manifest_copy)
    return output_dir


def _indexed_run_ids(indexed_runs):
    first = next(iter(indexed_runs.values()), {})
    return set(first)


def run_realhit(args):
    manifest = load_self_consistency_manifest(args.manifest, "realhitbench")
    logprob_field = get_tie_break_logprob_field(args.tie_break_logprob)
    indexed_runs = load_indexed_candidate_runs(manifest, "realhit_cot.jsonl")
    output_dir = prepare_output_dir(
        args.output_dir, manifest, args.resume, args.tie_break_logprob
    )

    dataset_payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_rows = dataset_payload.get("queries")
    if not isinstance(dataset_rows, list):
        raise SelfConsistencyError("RealHiT dataset must contain a queries list")
    selected_rows = select_dataset_rows(
        dataset_rows, _indexed_run_ids(indexed_runs), args.ids, args.limit
    )
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(args.dataset))

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            run["candidate_id"]: indexed_runs[run["candidate_id"]][sample_id]
            for run in manifest["runs"]
        }
        aggregate = aggregate_self_consistency_realhit_sample(
            item, records, manifest, logprob_field=logprob_field
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

    eval_rows, score = evaluate_realhit_vote(
        aggregates,
        dataset_by_id,
        selected_id_field="selected_candidate_id",
    )
    aggregate_by_id = {row["id"]: row for row in aggregates}
    for row in eval_rows:
        aggregate = aggregate_by_id[row["id"]]
        row.update(
            {
                "table_format": manifest["table_format"],
                "selected_sample_index": aggregate["selected_sample_index"],
                "selected_seed": aggregate["selected_seed"],
            }
        )
    diagnostics = self_consistency_realhit_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "realhitbench",
            "method": "Self-Consistency",
            "table_format": manifest["table_format"],
            "num_candidates": manifest["num_samples"],
            "base_seed": manifest["base_seed"],
            "manifest": manifest["manifest_path"],
            "tie_break_logprob": args.tie_break_logprob,
        }
    )
    save_jsonl(aggregates, output_dir / "self_consistency.jsonl")
    save_json(eval_rows, output_dir / "self_consistency_eval.json")
    save_json(score, output_dir / "self_consistency_score.json")
    save_json(
        diagnostics, output_dir / "self_consistency_diagnostics.json"
    )
    print(
        f"RealHiT Self-Consistency: samples={len(aggregates)}, "
        f"valid={sum(1 for row in aggregates if row['format_valid'])}, "
        f"output={output_dir}"
    )


def run_spreadsheet(args):
    manifest = load_self_consistency_manifest(
        args.manifest, "spreadsheetbench_verified_400"
    )
    logprob_field = get_tie_break_logprob_field(args.tie_break_logprob)
    indexed_runs = load_indexed_candidate_runs(
        manifest, "spreadsheet_pot.jsonl"
    )
    output_dir = prepare_output_dir(
        args.output_dir, manifest, args.resume, args.tie_break_logprob
    )

    dataset_root = Path(args.dataset_root).resolve()
    dataset_path = dataset_root / "dataset.json"
    dataset_rows = load_result_rows(dataset_path)
    selected_rows = select_dataset_rows(
        dataset_rows, _indexed_run_ids(indexed_runs), args.ids, args.limit
    )
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(dataset_path))

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            run["candidate_id"]: indexed_runs[run["candidate_id"]][sample_id]
            for run in manifest["runs"]
        }
        aggregates.append(
            aggregate_self_consistency_spreadsheet_sample(
                item,
                records,
                manifest,
                spreadsheet_input_path(dataset_root, item),
                logprob_field=logprob_field,
            )
        )

    spreadsheet_dir = output_dir / "spreadsheet"
    spreadsheet_dir.mkdir(parents=True, exist_ok=True)
    for row in aggregates:
        if not row["format_valid"]:
            stale = spreadsheet_dir / f"1_{row['id']}_output.xlsx"
            if stale.exists():
                stale.unlink()
    copied = copy_selected_workbooks(aggregates, output_dir)

    eval_rows, accuracy = evaluate_spreadsheet_vote(
        aggregates,
        dataset_by_id,
        dataset_root,
        output_dir,
        selected_id_field="selected_candidate_id",
    )
    aggregate_by_id = {row["id"]: row for row in aggregates}
    for row in eval_rows:
        aggregate = aggregate_by_id[row["id"]]
        row.update(
            {
                "table_format": manifest["table_format"],
                "selected_sample_index": aggregate["selected_sample_index"],
                "selected_seed": aggregate["selected_seed"],
            }
        )
    diagnostics = self_consistency_spreadsheet_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "spreadsheetbench_verified_400",
            "method": "Self-Consistency",
            "table_format": manifest["table_format"],
            "num_candidates": manifest["num_samples"],
            "base_seed": manifest["base_seed"],
            "manifest": manifest["manifest_path"],
            "tie_break_logprob": args.tie_break_logprob,
            "copied_output_workbooks": copied,
        }
    )
    save_jsonl(aggregates, output_dir / "self_consistency.jsonl")
    save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
    save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
    save_json(
        diagnostics, output_dir / "self_consistency_diagnostics.json"
    )
    print(
        f"SpreadsheetBench Self-Consistency: samples={len(aggregates)}, "
        f"copied={copied}, soft_all={accuracy.get('soft_all', 0):.4f}, "
        f"output={output_dir}"
    )


def _common_parser(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tie_break_logprob",
        choices=tuple(TIE_BREAK_LOGPROB_FIELDS),
        default=DEFAULT_TIE_BREAK_LOGPROB,
        help="Logprob statistic used after a vote/medoid tie (default: mean).",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Self-Consistency aggregation and evaluation."
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    realhit = subparsers.add_parser("realhit")
    _common_parser(realhit)
    realhit.add_argument(
        "--dataset",
        default=str(REPO_DIR / "dataset/realhitbench/realhit.json"),
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
