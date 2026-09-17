"""Offline format-ablation aggregation helpers.

The default format set remains the existing four-format ablation.  The
``--format_set five`` option adds ``image`` for RealHiTBench and
``excel_1_image`` for SpreadsheetBench, while reusing the same aggregators.
For the five-format comparison, SC covers all six formats independently.
"""

import argparse
import json
from pathlib import Path

from core.sheetflex.cg_vote import (
    aggregate_realhit_cg_sample,
    aggregate_spreadsheet_cg_sample,
    realhit_cg_diagnostics,
    spreadsheet_cg_diagnostics,
)
from core.sheetflex.common import (
    RECOMMEND_FORMAT_ORDER,
    SheetFlexError,
    get_format_order,
    index_rows_by_id,
    load_result_rows,
    save_json,
    save_jsonl,
)
from core.sheetflex.lp_vote import (
    aggregate_realhit_lp_sample,
    aggregate_spreadsheet_lp_sample,
)
from core.sheetflex.realhit import aggregate_realhit_sample, evaluate_realhit_vote
from core.sheetflex.self_consistency import (
    aggregate_self_consistency_realhit_sample,
    aggregate_self_consistency_spreadsheet_sample,
    self_consistency_realhit_diagnostics,
    self_consistency_spreadsheet_diagnostics,
)
from core.sheetflex.spreadsheet import (
    aggregate_spreadsheet_sample,
    copy_selected_workbooks,
    evaluate_spreadsheet_vote,
    spreadsheet_input_path,
)


REPO_DIR = Path(__file__).resolve().parent
ABLATION_FORMATS = ("latex", "markdown", "json_rows", "json_cells")
FIVE_FORMATS_BY_BENCHMARK = {
    "realhit": ABLATION_FORMATS + ("image",),
    "spreadsheet": ABLATION_FORMATS + ("excel_1_image",),
}
ABLATION_ORDER = tuple(
    format_name
    for format_name in RECOMMEND_FORMAT_ORDER
    if format_name in ABLATION_FORMATS
)
DEFAULT_SC_SEEDS = (42, 43, 44, 45)
SC_FORMATS = ABLATION_FORMATS + ("image", "excel_1_image")
METHODS = ("vote_mean", "vote_fixed", "lpvote", "cgvote")


def _format_set_names(format_set, benchmark):
    if format_set == "four":
        return ABLATION_FORMATS
    return FIVE_FORMATS_BY_BENCHMARK[benchmark]


def _sc_format_names(format_set):
    return ABLATION_FORMATS if format_set == "four" else SC_FORMATS


def _parse_ids(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_seed_list(value):
    try:
        seeds = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise SheetFlexError(f"Invalid self-consistency seed list: {value!r}") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise SheetFlexError("Self-consistency seeds must be a non-empty unique list")
    return seeds


def _select_rows(rows, ids=None, limit=0):
    selected_ids = _parse_ids(ids)
    if selected_ids is not None:
        found = {str(row.get("id")) for row in rows}
        missing = sorted(selected_ids - found)
        if missing:
            raise SheetFlexError(f"Requested dataset IDs not found: {missing}")
    selected = [
        row
        for row in rows
        if selected_ids is None or str(row.get("id")) in selected_ids
    ]
    return selected[:limit] if limit and limit > 0 else selected


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise SheetFlexError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SheetFlexError(f"Failed to parse JSON {path}: {exc}") from exc


def _resolve_dir(value, base):
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_run_map(path, formats):
    path = Path(path).resolve()
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise SheetFlexError(f"Run map must be an object: {path}")
    missing = [name for name in formats if name not in payload]
    if missing:
        raise SheetFlexError(f"Run map is missing ablation formats: {missing}")
    result = {}
    for name in formats:
        result[name] = _resolve_dir(payload[name], path.parent)
        if not result[name].is_dir():
            raise SheetFlexError(f"Run directory not found for {name}: {result[name]}")
    return result


def _load_indexed(path):
    return index_rows_by_id(load_result_rows(path), source=str(path))


def _load_single_format_runs(candidate_root, benchmark, model, format_name, seeds):
    manifest_path = (
        Path(candidate_root)
        / "self_consistency_manifests"
        / benchmark
        / model
        / format_name
        / "manifest.json"
    )
    manifest = _load_json(manifest_path)
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise SheetFlexError(f"Manifest runs must be a list: {manifest_path}")
    selected = []
    for seed in seeds:
        matches = [run for run in runs if run.get("seed") == seed]
        if len(matches) != 1:
            raise SheetFlexError(
                f"Expected exactly one seed={seed} in {manifest_path}, found {len(matches)}"
            )
        run = dict(matches[0])
        run_dir = _resolve_dir(run.get("run_dir", ""), manifest_path.parent)
        if not run_dir.is_dir():
            raise SheetFlexError(f"Run directory not found: {run_dir}")
        selected.append(
            {
                "candidate_id": f"sample_{len(selected)}",
                "sample_index": len(selected),
                "seed": seed,
                "run_dir": str(run_dir),
                "format": format_name,
                "model_name": model,
                "dataset": manifest.get("dataset"),
                "source_manifest": str(manifest_path),
            }
        )
    expected_dataset = (
        "realhitbench" if benchmark == "realhit" else "spreadsheetbench_verified_400"
    )
    if manifest.get("dataset") != expected_dataset:
        raise SheetFlexError(
            f"Manifest dataset mismatch in {manifest_path}: {manifest.get('dataset')!r}"
        )
    return manifest, selected


def _load_sc_candidates(candidate_root, benchmark, model, format_name, filename, seeds, *, load_records=True):
    manifest, runs = _load_single_format_runs(
        candidate_root, benchmark, model, format_name, seeds
    )
    indexed = {}
    for run in runs:
        result_path = Path(run["run_dir"]) / filename
        if not result_path.is_file():
            raise SheetFlexError(
                f"Candidate result file missing for {run['candidate_id']}: {result_path}"
            )
        if load_records:
            indexed[run["candidate_id"]] = _load_indexed(result_path)
    reduced = {
        "method": "self_consistency",
        "stage": "candidate_generation",
        "dataset": manifest.get("dataset"),
        "model_name": model,
        "table_format": format_name,
        "num_samples": len(seeds),
        "base_seed": 42,
        "temperature": manifest.get("temperature"),
        "top_p": manifest.get("top_p"),
        "save_logprobs": manifest.get("save_logprobs"),
        "runs": runs,
        "source_manifest": str(
            Path(candidate_root)
            / "self_consistency_manifests"
            / benchmark
            / model
            / format_name
            / "manifest.json"
        ),
    }
    return reduced, indexed


def _load_dataset(benchmark, dataset_root):
    if benchmark == "realhit":
        path = REPO_DIR / "dataset/realhitbench/realhit.json"
        payload = _load_json(path)
        rows = payload.get("queries")
        if not isinstance(rows, list):
            raise SheetFlexError("RealHiT dataset must contain a queries list")
        return path, rows
    path = Path(dataset_root).resolve() / "dataset.json"
    return path, load_result_rows(path)


def _base_output_manifest(args, benchmark, model, formats, source_info):
    return {
        "schema_version": 1,
        "method": args.method,
        "benchmark": benchmark,
        "model_name": model,
        "formats": list(formats),
        "format_order": list(args.format_order),
        "seeds": list(args.self_consistency_seeds) if args.method == "self_consistency" else None,
        "num_candidates": len(args.self_consistency_seeds) if args.method == "self_consistency" else len(formats),
        "num_samples": len(args.self_consistency_seeds) if args.method == "self_consistency" else len(formats),
        "lp_weight_strength": args.lp_weight_strength,
        "confidence_gate_strength": args.confidence_gate_strength,
        "missing_logprob_policy": args.missing_logprob_policy,
        "tie_break_logprob": args.tie_break_logprob,
        "source": source_info,
        "dataset_path": str(_load_dataset(benchmark, args.dataset_root)[0].resolve()),
        "sample_filter": {"ids": sorted(_parse_ids(args.ids) or []), "limit": args.limit},
    }


def _output_config_matches(previous, expected):
    fields = ["method", "benchmark", "model_name", "formats", "num_candidates"]
    method = expected["method"]
    if method == "self_consistency":
        fields += ["seeds", "tie_break_logprob"]
        identity = lambda m: [(r["seed"], str(Path(r["run_dir"]).resolve())) for r in m.get("source_runs", [])]
        if identity(previous) != identity(expected):
            return False
    else:
        fields += ["format_order"]
        if previous.get("source", {}).get("run_dirs") != expected["source"]["run_dirs"]:
            return False
        if method == "vote_mean":
            fields += ["tie_break_logprob"]
        if method in ("lpvote", "cgvote"):
            fields += ["lp_weight_strength", "missing_logprob_policy"]
        if method == "cgvote":
            fields += ["confidence_gate_strength"]
    if previous.get("exclude_unchanged_target_values", False):
        return False
    if previous.get("dataset_path", expected["dataset_path"]) != expected["dataset_path"]:
        return False
    return all(previous.get(field) == expected.get(field) for field in fields)


def _completed_output(output_dir, method, benchmark, selected_rows):
    eval_name = f"{method}_eval.json" if benchmark == "realhit" else "spreadsheet_pot_eval.json"
    score_name = f"{method}_score.json" if benchmark == "realhit" else "spreadsheet_pot_accuracy.json"
    filenames = [f"{method}.jsonl", eval_name, score_name, f"{method}_diagnostics.json"]
    if any(not (output_dir / name).is_file() for name in filenames):
        return False
    try:
        expected_ids = {str(row["id"]) for row in selected_rows}
        traces = _load_indexed(output_dir / filenames[0])
        eval_rows = _load_indexed(output_dir / eval_name)
        if set(traces) != expected_ids or set(eval_rows) != expected_ids:
            return False
        score = _load_json(output_dir / score_name)
        diagnostics = _load_json(output_dir / filenames[3])
        if not isinstance(score, dict) or not score or diagnostics.get("num_samples") != len(expected_ids):
            return False
        if benchmark == "spreadsheet":
            for sample_id, row in traces.items():
                if row.get("execution_success") is True and not (output_dir / "spreadsheet" / f"1_{sample_id}_output.xlsx").is_file():
                    return False
        return True
    except (SheetFlexError, ValueError, TypeError, KeyError):
        return False


def _prepare_output(args, output_dir, manifest, selected_rows):
    """Return False for dry-run or a complete matching task; never skip by directory alone."""
    nonempty = output_dir.exists() and any(output_dir.iterdir())
    complete = False
    if nonempty:
        if not (args.resume or args.skip_completed or args.dry_run):
            raise SheetFlexError(f"Refusing to overwrite non-empty output directory: {output_dir}")
        previous = _load_json(output_dir / "manifest.json")
        if not _output_config_matches(previous, manifest):
            raise SheetFlexError(f"Output configuration mismatch; use a different output root: {output_dir}")
        if "sample_filter" in previous and previous["sample_filter"] != manifest["sample_filter"]:
            raise SheetFlexError(f"Output sample-filter mismatch; use a different output root: {output_dir}")
        # Old manifests did not save ids/limit: never replace a different completed scope.
        existing_eval = output_dir / (f"{args.method}_eval.json" if args.benchmark == "realhit" else "spreadsheet_pot_eval.json")
        if "sample_filter" not in previous and existing_eval.exists():
            if set(_load_indexed(existing_eval)) != {str(row["id"]) for row in selected_rows}:
                raise SheetFlexError(f"Output sample-ID mismatch; use a different output root: {output_dir}")
        complete = _completed_output(output_dir, args.method, args.benchmark, selected_rows)
    action = "skip" if args.skip_completed and complete else "resume" if nonempty else "run"
    if args.dry_run:
        print(json.dumps({"action": action, "method": args.method, "benchmark": args.benchmark,
                          "model": args.model, "formats": manifest["formats"], "seeds": manifest["seeds"],
                          "num_samples": len(selected_rows), "output_dir": str(output_dir)}, ensure_ascii=False))
        return False
    if action == "skip":
        print(f"[skip completed] {output_dir}")
        return False
    if nonempty and not args.resume:
        raise SheetFlexError(f"Incomplete/existing output; pass --resume to recompute: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, output_dir / "manifest.json")
    return True


def run_self_consistency(args, benchmark, model, format_name):
    filename = "realhit_cot.jsonl" if benchmark == "realhit" else "spreadsheet_pot.jsonl"
    manifest, _ = _load_sc_candidates(
        args.candidate_root, benchmark, model, format_name, filename,
        args.self_consistency_seeds,
        load_records=False,
    )
    dataset_path, dataset_rows = _load_dataset(benchmark, args.dataset_root)
    selected_rows = _select_rows(dataset_rows, args.ids, args.limit)
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(dataset_path))
    benchmark_output = (
        "realhitbench" if benchmark == "realhit" else "spreadsheetbench_verified_400"
    )
    output_dir = (
        Path(args.output_root)
        / "self_consistency"
        / benchmark_output
        / model
        / format_name
    ).resolve()
    output_manifest = _base_output_manifest(
        args,
        benchmark,
        model,
        (format_name,),
        {"candidate_root": str(Path(args.candidate_root).resolve()), "source_manifest": manifest.get("source_manifest")},
    )
    output_manifest.update({"source_manifest": manifest.get("source_manifest"), "source_runs": manifest["runs"]})
    if not _prepare_output(args, output_dir, output_manifest, selected_rows):
        return
    indexed = {run["candidate_id"]: _load_indexed(Path(run["run_dir"]) / filename) for run in manifest["runs"]}
    aggregates = []
    reduced_manifest = dict(manifest)
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            run["candidate_id"]: indexed[run["candidate_id"]].get(sample_id)
            for run in reduced_manifest["runs"]
        }
        if benchmark == "realhit":
            aggregate = aggregate_self_consistency_realhit_sample(
                item,
                records,
                reduced_manifest,
                logprob_field=(
                    "sequence_logprob_mean"
                    if args.tie_break_logprob == "mean"
                    else "sequence_logprob_sum"
                ),
            )
            aggregate.update(
                {
                    "Question": item.get("Question"),
                    "SubQType": item.get("SubQType"),
                    "FileName": item.get("FileName"),
                    "CompStrucCata": item.get("CompStrucCata"),
                }
            )
        else:
            aggregate = aggregate_self_consistency_spreadsheet_sample(
                item,
                records,
                reduced_manifest,
                spreadsheet_input_path(Path(args.dataset_root), item),
                logprob_field=(
                    "sequence_logprob_mean"
                    if args.tie_break_logprob == "mean"
                    else "sequence_logprob_sum"
                ),
            )
        aggregates.append(aggregate)

    save_jsonl(aggregates, output_dir / "self_consistency.jsonl")
    if benchmark == "realhit":
        eval_rows, score = evaluate_realhit_vote(
            aggregates, dataset_by_id, selected_id_field="selected_candidate_id"
        )
        diagnostics = self_consistency_realhit_diagnostics(
            aggregates, num_candidates=len(args.self_consistency_seeds)
        )
        save_json(eval_rows, output_dir / "self_consistency_eval.json")
        save_json(score, output_dir / "self_consistency_score.json")
        diagnostics.update({"benchmark": benchmark, "method": "Self-Consistency", "format": format_name})
    else:
        spreadsheet_dir = output_dir / "spreadsheet"
        spreadsheet_dir.mkdir(exist_ok=True)
        copied = copy_selected_workbooks(aggregates, output_dir)
        eval_rows, accuracy = evaluate_spreadsheet_vote(
            aggregates, dataset_by_id, Path(args.dataset_root), output_dir,
            selected_id_field="selected_candidate_id",
        )
        diagnostics = self_consistency_spreadsheet_diagnostics(
            aggregates, num_candidates=len(args.self_consistency_seeds)
        )
        save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
        save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
        diagnostics.update({"benchmark": benchmark, "method": "Self-Consistency", "format": format_name, "copied_output_workbooks": copied})
    diagnostics.update({"seed_subset": list(args.self_consistency_seeds), "output_manifest": str(output_dir / "manifest.json")})
    save_json(diagnostics, output_dir / "self_consistency_diagnostics.json")
    print(f"[format-ablation] self_consistency {benchmark} {model} {format_name}: samples={len(aggregates)} output={output_dir}")


def _load_cross_format_candidates(source_root, benchmark, model, formats):
    source_root = Path(source_root).resolve()
    paths = {}
    for format_name in formats:
        if benchmark == "realhit":
            paths[format_name] = source_root / "realhitbench" / model / f"cot_{format_name}_logprobs_100ktoken"
            filename = "realhit_cot.jsonl"
        else:
            paths[format_name] = source_root / "spreadsheetbench_verified_400" / model / f"pot_{format_name}_logprobs_40ktoken"
            filename = "spreadsheet_pot.jsonl"
        if not paths[format_name].is_dir():
            raise SheetFlexError(f"Cross-format run directory not found for {format_name}: {paths[format_name]}")
    return paths, filename


def _format_order_for(args, benchmark):
    allowed = set(_format_set_names(args.format_set, benchmark))
    return tuple(name for name in get_format_order(args.tie_break_order) if name in allowed)


def run_sheetflex(args, benchmark, model):
    formats = _format_order_for(args, benchmark)
    run_dirs, filename = _load_cross_format_candidates(
        args.source_root, benchmark, model, formats
    )
    for run_dir in run_dirs.values():
        if not (run_dir / filename).is_file():
            raise SheetFlexError(f"Candidate result file missing: {run_dir / filename}")
    dataset_path, dataset_rows = _load_dataset(benchmark, args.dataset_root)
    selected_rows = _select_rows(dataset_rows, args.ids, args.limit)
    dataset_by_id = index_rows_by_id(dataset_rows, source=str(dataset_path))
    output_dir = (
        Path(args.output_root) / args.method / benchmark / model
    ).resolve()
    output_manifest = _base_output_manifest(
        args,
        benchmark,
        model,
        formats,
        {"source_root": str(Path(args.source_root).resolve()), "run_dirs": {name: str(path) for name, path in run_dirs.items()}},
    )
    if not _prepare_output(args, output_dir, output_manifest, selected_rows):
        return
    indexed = {name: _load_indexed(run_dirs[name] / filename) for name in formats}

    aggregates = []
    for item in selected_rows:
        sample_id = str(item["id"])
        records = {
            format_name: indexed[format_name].get(sample_id)
            for format_name in formats
        }
        if args.method == "vote_mean":
            tie_logprob_field = {
                "mean": "sequence_logprob_mean",
                "sum": "sequence_logprob_sum",
            }[args.tie_break_logprob]
            if benchmark == "realhit":
                aggregate = aggregate_realhit_sample(
                    sample_id,
                    item.get("QuestionType", "Unknown"),
                    records,
                    run_dirs={name: str(path) for name, path in run_dirs.items()},
                    format_order=formats,
                    logprob_field=tie_logprob_field,
                )
            else:
                aggregate = aggregate_spreadsheet_sample(
                    item,
                    records,
                    run_dirs,
                    spreadsheet_input_path(Path(args.dataset_root), item),
                    format_order=formats,
                    logprob_field=tie_logprob_field,
                )
        elif args.method == "vote_fixed":
            if benchmark == "realhit":
                aggregate = aggregate_realhit_sample(
                    sample_id,
                    item.get("QuestionType", "Unknown"),
                    records,
                    run_dirs={name: str(path) for name, path in run_dirs.items()},
                    format_order=formats,
                    logprob_field=None,
                )
            else:
                aggregate = aggregate_spreadsheet_sample(
                    item,
                    records,
                    run_dirs,
                    spreadsheet_input_path(Path(args.dataset_root), item),
                    format_order=formats,
                    logprob_field=None,
                )
        elif args.method == "lpvote":
            if benchmark == "realhit":
                aggregate = aggregate_realhit_lp_sample(
                    sample_id,
                    item.get("QuestionType", "Unknown"),
                    records,
                    run_dirs={name: str(path) for name, path in run_dirs.items()},
                    strength=args.lp_weight_strength,
                    missing_logprob_policy=args.missing_logprob_policy,
                    format_order=formats,
                    tie_break_order=args.tie_break_order,
                )
            else:
                aggregate = aggregate_spreadsheet_lp_sample(
                    item,
                    records,
                    run_dirs,
                    spreadsheet_input_path(Path(args.dataset_root), item),
                    strength=args.lp_weight_strength,
                    missing_logprob_policy=args.missing_logprob_policy,
                    format_order=formats,
                    tie_break_order=args.tie_break_order,
                )
        elif args.method == "cgvote":
            if benchmark == "realhit":
                aggregate = aggregate_realhit_cg_sample(
                    sample_id,
                    item.get("QuestionType", "Unknown"),
                    records,
                    run_dirs={name: str(path) for name, path in run_dirs.items()},
                    strength=args.lp_weight_strength,
                    confidence_gate_strength=args.confidence_gate_strength,
                    missing_logprob_policy=args.missing_logprob_policy,
                    format_order=formats,
                    tie_break_order=args.tie_break_order,
                )
            else:
                aggregate = aggregate_spreadsheet_cg_sample(
                    item,
                    records,
                    run_dirs,
                    spreadsheet_input_path(Path(args.dataset_root), item),
                    strength=args.lp_weight_strength,
                    confidence_gate_strength=args.confidence_gate_strength,
                    missing_logprob_policy=args.missing_logprob_policy,
                    format_order=formats,
                    tie_break_order=args.tie_break_order,
                )
        else:
            raise SheetFlexError(f"Unknown ablation method: {args.method}")
        if benchmark == "realhit":
            aggregate.update(
                {
                    "Question": item.get("Question"),
                    "SubQType": item.get("SubQType"),
                    "FileName": item.get("FileName"),
                    "CompStrucCata": item.get("CompStrucCata"),
                }
            )
        aggregates.append(aggregate)

    save_jsonl(aggregates, output_dir / f"{args.method}.jsonl")
    if benchmark == "realhit":
        eval_rows, score = evaluate_realhit_vote(aggregates, dataset_by_id)
        diagnostics = (
            realhit_cg_diagnostics(aggregates)
            if args.method == "cgvote"
            else {"num_samples": len(aggregates)}
        )
        save_json(eval_rows, output_dir / f"{args.method}_eval.json")
        save_json(score, output_dir / f"{args.method}_score.json")
    else:
        copied = copy_selected_workbooks(aggregates, output_dir)
        eval_rows, accuracy = evaluate_spreadsheet_vote(
            aggregates, dataset_by_id, Path(args.dataset_root), output_dir
        )
        diagnostics = (
            spreadsheet_cg_diagnostics(aggregates)
            if args.method == "cgvote"
            else {"num_samples": len(aggregates)}
        )
        save_json(eval_rows, output_dir / "spreadsheet_pot_eval.json")
        save_json(accuracy, output_dir / "spreadsheet_pot_accuracy.json")
        diagnostics["copied_output_workbooks"] = copied
    diagnostics.update(
        {
            "benchmark": benchmark,
            "method": args.method,
            "model_name": model,
            "formats": list(formats),
            "format_order": list(formats),
            "lp_weight_strength": args.lp_weight_strength,
            "confidence_gate_strength": args.confidence_gate_strength,
            "missing_logprob_policy": args.missing_logprob_policy,
            "tie_break_order": args.tie_break_order,
        }
    )
    save_json(diagnostics, output_dir / f"{args.method}_diagnostics.json")
    print(f"[format-ablation] {args.method} {benchmark} {model}: samples={len(aggregates)} output={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline format ablation aggregation")
    parser.add_argument("method", choices=("self_consistency", *METHODS))
    parser.add_argument("benchmark", choices=("realhit", "spreadsheet"))
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--format",
        choices=tuple(dict.fromkeys(ABLATION_FORMATS + ("image", "excel_1_image"))),
        default=None,
        help="Format for one-format Self-Consistency; ignored by cross-format methods.",
    )
    parser.add_argument(
        "--format_set",
        choices=("four", "five"),
        default="four",
        help="Use the existing four-format set or the five-format benchmark-specific set.",
    )
    parser.add_argument("--candidate_root", default=str(REPO_DIR / "sc_outs/candidates"))
    parser.add_argument("--source_root", default=str(REPO_DIR / "lp_outs"))
    parser.add_argument("--output_root", default=str(REPO_DIR / "format_ablation_outs"))
    parser.add_argument("--dataset_root", default=str(REPO_DIR / "dataset/spreadsheetbench/spreadsheetbench_verified_400"))
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_completed", action="store_true", help="Skip complete results with the same settings, sources and IDs")
    parser.add_argument("--dry_run", action="store_true", help="Check inputs and output status without aggregation or writes")
    parser.add_argument(
        "--self_consistency_seeds",
        default=",".join(str(seed) for seed in DEFAULT_SC_SEEDS),
        help="Comma-separated seeds used by Self-Consistency aggregation.",
    )
    parser.add_argument("--lp_weight_strength", type=float, default=1.0)
    parser.add_argument("--confidence_gate_strength", type=float, default=1.0)
    parser.add_argument("--missing_logprob_policy", choices=("vote", "error"), default="error")
    parser.add_argument(
        "--tie_break_logprob", choices=("mean", "sum"), default="mean"
    )
    parser.add_argument("--tie_break_order", choices=("recommend", "legacy"), default="recommend")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def main():
    args = parse_args()
    args.self_consistency_seeds = _parse_seed_list(args.self_consistency_seeds)
    args.format_order = _format_order_for(args, args.benchmark)
    if args.method == "self_consistency":
        if args.format is None:
            raise SystemExit("--format is required for self_consistency")
        if args.format not in _sc_format_names(args.format_set):
            raise SystemExit(
                f"Format {args.format!r} is not in {args.format_set} set for {args.benchmark}"
            )
        if args.format_set == "five" and (
            len(args.self_consistency_seeds) != 5
            or not set(args.self_consistency_seeds) <= set(range(42, 48))
            or tuple(sorted(args.self_consistency_seeds)) != args.self_consistency_seeds
        ):
            raise SystemExit("Five-seed SC requires five distinct seeds from 42..47 in ascending order; pass --self_consistency_seeds")
        args.format_order = (args.format,)
        run_self_consistency(args, args.benchmark, args.model, args.format)
    else:
        run_sheetflex(args, args.benchmark, args.model)


if __name__ == "__main__":
    main()
