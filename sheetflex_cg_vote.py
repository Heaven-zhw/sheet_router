"""Run offline SheetFlex-CGVote over six existing format candidates."""

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from core.sheetflex.cg_vote import (
    METHOD,
    aggregate_realhit_cg_sample,
    aggregate_spreadsheet_cg_sample,
    realhit_cg_diagnostics,
    spreadsheet_cg_diagnostics,
    validate_confidence_gate_strength,
)
from core.sheetflex.common import (
    DEFAULT_TIE_BREAK_ORDER,
    FORMAT_ORDER,
    FORMAT_ORDERS,
    SCORE_ABS_TOL,
    SCORE_REL_TOL,
    SheetFlexError,
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
            raise SheetFlexError(f"Requested dataset id(s) not found: {missing}")
    return selected[:limit] if limit and limit > 0 else selected


def _nonnegative(value, validator):
    try:
        return validator(float(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _alpha(value):
    return _nonnegative(value, validate_lp_weight_strength)


def _beta(value):
    return _nonnegative(value, validate_confidence_gate_strength)


def _run_map_trace(run_map):
    return {format_name: str(run_map[format_name]) for format_name in FORMAT_ORDER}


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _git_dirty():
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_DIR,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        return None


def _validate_output_isolation(output_dir, run_map):
    output = Path(output_dir).resolve()
    for format_name, run_dir in run_map.items():
        candidate = Path(run_dir).resolve()
        if (
            output == candidate
            or output.is_relative_to(candidate)
            or candidate.is_relative_to(output)
        ):
            raise SheetFlexError(
                f"Output directory cannot be a candidate directory or its child: "
                f"format={format_name} output={output} candidate={candidate}"
            )


def _candidate_generation_config(run_map):
    observed = {}
    missing = []
    models = set()
    for format_name, run_dir in run_map.items():
        metadata_path = Path(run_dir) / "run_metadata.json"
        if not metadata_path.is_file():
            missing.append(format_name)
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SheetFlexError(
                f"Failed to read candidate metadata {metadata_path}: {exc}"
            ) from exc
        temperature = metadata.get("temperature")
        top_p = metadata.get("top_p")
        if temperature != 0 or top_p != 1:
            raise SheetFlexError(
                f"Candidate generation settings mismatch for {format_name}: "
                f"temperature={temperature!r}, top_p={top_p!r}"
            )
        recorded_format = metadata.get("table_format")
        if recorded_format is not None and recorded_format != format_name:
            raise SheetFlexError(
                f"Candidate format mismatch for {format_name}: "
                f"metadata has {recorded_format!r}"
            )
        model_name = metadata.get("model_name")
        if model_name:
            models.add(str(model_name))
        observed[format_name] = {
            "temperature": temperature,
            "top_p": top_p,
            "model_name": model_name,
            "table_format": recorded_format,
            "metadata_path": str(metadata_path.resolve()),
        }
    if len(models) > 1:
        raise SheetFlexError(f"Run-map mixes models: {sorted(models)}")
    status = (
        "verified"
        if not missing
        else "unverified_missing_run_metadata"
        if len(missing) == len(run_map)
        else "partially_verified_missing_run_metadata"
    )
    return {
        "status": status,
        "required": {"temperature": 0, "top_p": 1},
        "observed": observed,
        "missing_formats": missing,
        "model_name": next(iter(models), None),
    }


def _validate_candidate_result_identity(indexed_runs):
    models = set()
    metadata_rows = 0
    for format_name, rows in indexed_runs.items():
        for sample_id, row in rows.items():
            metadata = row.get("table_metadata")
            if not isinstance(metadata, Mapping):
                continue
            metadata_rows += 1
            recorded_format = metadata.get("table_format")
            if recorded_format is not None and recorded_format != format_name:
                raise SheetFlexError(
                    f"Candidate result format mismatch: sample_id={sample_id} "
                    f"run_format={format_name} record_format={recorded_format}"
                )
            model_name = metadata.get("token_model")
            if model_name:
                models.add(str(model_name))
    if len(models) > 1:
        raise SheetFlexError(f"Candidate results mix models: {sorted(models)}")
    return {
        "status": "verified_from_result_table_metadata" if metadata_rows else "unverified",
        "model_name": next(iter(models), None),
        "metadata_rows": metadata_rows,
    }


def _save_result_identity(manifest, output_dir, indexed_runs):
    manifest["candidate_result_identity"] = _validate_candidate_result_identity(
        indexed_runs
    )
    save_json(manifest, output_dir / "manifest.json")


def _prepare_run(args, dataset_path):
    run_map = load_run_map(args.run_map)
    _validate_output_isolation(args.output_dir, run_map)
    output_dir = prepare_output_dir(args.output_dir)
    format_order = get_format_order(args.tie_break_order)
    manifest = {
        "schema_version": 1,
        "method": METHOD,
        "lp_weight_strength": args.lp_weight_strength,
        "confidence_gate_strength": args.confidence_gate_strength,
        "missing_logprob_policy": args.missing_logprob_policy,
        "tie_break_order": args.tie_break_order,
        "format_order": list(format_order),
        "score_comparison": {
            "space": "max_shifted_exp_log_confidence_gated_score",
            "rel_tol": SCORE_REL_TOL,
            "abs_tol": SCORE_ABS_TOL,
        },
        "run_map": _run_map_trace(run_map),
        "candidate_generation_config": _candidate_generation_config(run_map),
        "dataset_path": str(Path(dataset_path).resolve()),
        "sample_filter": {"ids": args.ids, "limit": args.limit},
        "code_revision": _git_revision(),
        "code_dirty": _git_dirty(),
        "baseline_config": {
            "lpvote": {
                "lp_weight_strength": args.lp_weight_strength,
                "tie_break_order": args.tie_break_order,
            },
            "fixed_vote": {"tie_break_order": "recommend"},
        },
    }
    save_json(manifest, output_dir / "manifest.json")
    return run_map, output_dir, format_order, manifest


def run_realhit(args):
    run_map, output_dir, format_order, manifest = _prepare_run(
        args, args.dataset
    )
    indexed_runs = load_indexed_runs(run_map, "realhit_cot.jsonl")
    _save_result_identity(manifest, output_dir, indexed_runs)
    payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_rows = payload.get("queries")
    if not isinstance(dataset_rows, list):
        raise SheetFlexError("RealHiT dataset JSON must contain a queries list")
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
        aggregate = aggregate_realhit_cg_sample(
            sample_id,
            item.get("QuestionType", "Unknown"),
            records,
            run_dirs=run_dirs,
            strength=args.lp_weight_strength,
            confidence_gate_strength=args.confidence_gate_strength,
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

    trace_path = output_dir / "sheetflex_cg_vote.jsonl"
    save_jsonl(aggregates, trace_path)
    try:
        eval_rows, score = evaluate_realhit_vote(aggregates, dataset_by_id)
    except Exception as exc:
        raise RuntimeError(
            f"CGVote selections were saved to {trace_path}, but evaluation failed: {exc}"
        ) from exc
    diagnostics = realhit_cg_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "realhitbench",
            **{key: manifest[key] for key in (
                "method",
                "lp_weight_strength",
                "confidence_gate_strength",
                "missing_logprob_policy",
                "tie_break_order",
                "run_map",
            )},
        }
    )
    save_json(eval_rows, output_dir / "sheetflex_cg_vote_eval.json")
    save_json(score, output_dir / "sheetflex_cg_vote_score.json")
    save_json(diagnostics, output_dir / "sheetflex_cg_vote_diagnostics.json")
    print(
        f"RealHiT SheetFlex-CGVote: samples={len(aggregates)}, "
        f"valid={sum(bool(row['format_valid']) for row in aggregates)}, "
        f"output={output_dir}"
    )


def run_spreadsheet(args):
    dataset_root = Path(args.dataset_root).resolve()
    dataset_path = dataset_root / "dataset.json"
    run_map, output_dir, format_order, manifest = _prepare_run(
        args, dataset_path
    )
    indexed_runs = load_indexed_runs(run_map, "spreadsheet_pot.jsonl")
    _save_result_identity(manifest, output_dir, indexed_runs)
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
            aggregate_spreadsheet_cg_sample(
                item,
                records,
                run_map,
                spreadsheet_input_path(dataset_root, item),
                strength=args.lp_weight_strength,
                confidence_gate_strength=args.confidence_gate_strength,
                missing_logprob_policy=args.missing_logprob_policy,
                format_order=format_order,
                tie_break_order=args.tie_break_order,
            )
        )

    trace_path = output_dir / "sheetflex_cg_vote.jsonl"
    save_jsonl(aggregates, trace_path)
    copied = copy_selected_workbooks(aggregates, output_dir)
    try:
        eval_rows, accuracy = evaluate_spreadsheet_vote(
            aggregates, dataset_by_id, dataset_root, output_dir
        )
    except Exception as exc:
        raise RuntimeError(
            f"CGVote selections were saved to {trace_path}, but evaluation failed: {exc}"
        ) from exc
    diagnostics = spreadsheet_cg_diagnostics(aggregates)
    diagnostics.update(
        {
            "benchmark": "spreadsheetbench_verified_400",
            **{key: manifest[key] for key in (
                "method",
                "lp_weight_strength",
                "confidence_gate_strength",
                "missing_logprob_policy",
                "tie_break_order",
                "run_map",
            )},
            "copied_output_workbooks": copied,
        }
    )
    save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
    save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
    save_json(diagnostics, output_dir / "sheetflex_cg_vote_diagnostics.json")
    print(
        f"SpreadsheetBench SheetFlex-CGVote: samples={len(aggregates)}, "
        f"copied={copied}, hard_all={accuracy.get('hard_all', 0):.4f}, "
        f"output={output_dir}"
    )


def _common_parser(parser):
    parser.add_argument("--run_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ids", default=None, help="Comma-separated sample IDs.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lp_weight_strength", type=_alpha, default=1.0)
    parser.add_argument("--confidence_gate_strength", type=_beta, default=1.0)
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
        description="Offline class-aware SheetFlex Confidence-Gated Vote."
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
