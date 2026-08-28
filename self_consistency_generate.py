"""Generate independent seeded candidates for a later Self-Consistency stage."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_NUM_SAMPLES = 6
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
DEFAULT_BASE_SEED = 42


class CandidateGenerationError(ValueError):
    pass


def trajectory_seeds(base_seed: int, num_samples: int) -> list[int]:
    if num_samples < 1:
        raise CandidateGenerationError("num_samples must be at least 1")
    return [base_seed + sample_index for sample_index in range(num_samples)]


def _model_dir(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _number_text(value: float) -> str:
    return str(float(value))


def candidate_run_suffix(args, seed: int) -> str:
    prefix = "cot" if args.dataset == "realhit" else "pot"
    parts = [prefix, args.table_format, "logprobs", f"seed{seed}"]
    if args.fill_merged:
        parts.append("fillmerged")
    if not args.include_coordinates:
        parts.append("nocoord")
    if args.top_p != 1.0 or args.temperature != 0:
        parts.append(f"tp{args.top_p}_temp{args.temperature}")
    if args.suffix:
        parts.append(args.suffix)
    return "_".join(parts)


def candidate_run_dir(args, seed: int) -> Path:
    dataset_dir = (
        "realhitbench"
        if args.dataset == "realhit"
        else "spreadsheetbench_verified_400"
    )
    return (
        Path(args.output_root).resolve()
        / dataset_dir
        / _model_dir(args.model_name)
        / candidate_run_suffix(args, seed)
    )


def default_manifest_path(args) -> Path:
    return (
        Path(args.output_root).resolve()
        / "self_consistency_manifests"
        / args.dataset
        / _model_dir(args.model_name)
        / args.table_format
        / "manifest.json"
    )


def build_solver_command(args, sample_index: int, seed: int) -> list[str]:
    entrypoint = "realhit_cot.py" if args.dataset == "realhit" else "spreadsheet_pot.py"
    python_executable = (
        str(Path(args.python).expanduser().resolve())
        if Path(args.python).is_absolute() or "/" in args.python
        else args.python
    )
    command = [
        python_executable,
        str(REPO_DIR / entrypoint),
        "--url",
        args.url,
        "--model_name",
        args.model_name,
        "--table_format",
        args.table_format,
        "--temperature",
        _number_text(args.temperature),
        "--top_p",
        _number_text(args.top_p),
        "--seed",
        str(seed),
        "--base_seed",
        str(args.base_seed),
        "--sample_index",
        str(sample_index),
        "--candidate_id",
        f"sample_{sample_index}",
        "--save_logprobs",
        "--max_retries",
        str(args.max_retries),
        "--workers",
        str(args.workers),
        "--max_text_tokens",
        str(args.max_text_tokens),
        "--report_every",
        str(args.report_every),
        "--save_every",
        str(args.save_every),
        "--output_root",
        str(Path(args.output_root).resolve()),
    ]
    if args.ids:
        command.extend(["--ids", args.ids])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.fill_merged:
        command.append("--fill_merged")
    if not args.include_coordinates:
        command.append("--no-include_coordinates")
    if args.save_prompts:
        command.append("--save_prompts")
    if args.resume:
        command.append("--resume")
    if args.suffix:
        command.extend(["--suffix", args.suffix])

    if args.dataset == "realhit":
        if args.question_types:
            command.extend(["--question_types", args.question_types])
    else:
        command.extend(
            ["--data_split", "verified_400", "--code_exec_url", args.code_exec_url]
        )
        if args.instruction_types:
            command.extend(["--instruction_types", args.instruction_types])
        if args.render_formulas_before_eval:
            command.append("--render_formulas_before_eval")
    return command


def build_manifest(args) -> Dict[str, Any]:
    seeds = trajectory_seeds(args.base_seed, args.num_samples)
    dataset_name = (
        "realhitbench"
        if args.dataset == "realhit"
        else "spreadsheetbench_verified_400"
    )
    runs = []
    for sample_index, seed in enumerate(seeds):
        runs.append(
            {
                "candidate_id": f"sample_{sample_index}",
                "sample_index": sample_index,
                "seed": seed,
                "run_dir": str(candidate_run_dir(args, seed)),
                "model_name": args.model_name,
                "dataset": dataset_name,
                "table_format": args.table_format,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "save_logprobs": True,
                "status": "pending",
            }
        )
    manifest = {
        "schema_version": 1,
        "method": "self_consistency",
        "stage": "candidate_generation",
        "execution_mode": "independent_solver_runs",
        "dataset": dataset_name,
        "model_name": args.model_name,
        "url": args.url,
        "table_format": args.table_format,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.base_seed,
        "save_logprobs": True,
        "sample_filter": {
            "ids": args.ids,
            "limit": args.limit,
            "question_types": args.question_types if args.dataset == "realhit" else None,
            "instruction_types": (
                args.instruction_types if args.dataset == "spreadsheet" else None
            ),
        },
        "solver_settings": {
            "max_retries": args.max_retries,
            "max_text_tokens": args.max_text_tokens,
            "include_coordinates": args.include_coordinates,
            "fill_merged": args.fill_merged,
            "render_formulas_before_eval": (
                args.render_formulas_before_eval
                if args.dataset == "spreadsheet"
                else False
            ),
            "suffix": args.suffix,
        },
        "runs": runs,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> None:
    runs = manifest.get("runs")
    num_samples = manifest.get("num_samples")
    if not isinstance(runs, list) or len(runs) != num_samples:
        raise CandidateGenerationError(
            f"Manifest must contain exactly num_samples={num_samples} ordered runs"
        )
    expected_indices = list(range(num_samples))
    if [run.get("sample_index") for run in runs] != expected_indices:
        raise CandidateGenerationError("Manifest runs must be ordered by sample_index")
    expected_seeds = trajectory_seeds(manifest["base_seed"], num_samples)
    if [run.get("seed") for run in runs] != expected_seeds:
        raise CandidateGenerationError("Manifest run seeds do not match base_seed + sample_index")
    run_dirs = [run.get("run_dir") for run in runs]
    if len(set(run_dirs)) != len(run_dirs):
        raise CandidateGenerationError("Manifest run directories must be unique")

    shared_fields = (
        "model_name",
        "dataset",
        "table_format",
        "temperature",
        "top_p",
        "save_logprobs",
    )
    for run in runs:
        for field in shared_fields:
            if run.get(field) != manifest.get(field):
                raise CandidateGenerationError(
                    f"Manifest run {run.get('candidate_id')} has inconsistent {field}"
                )


def _manifest_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "method",
        "stage",
        "dataset",
        "model_name",
        "url",
        "table_format",
        "num_samples",
        "temperature",
        "top_p",
        "base_seed",
        "save_logprobs",
        "sample_filter",
        "solver_settings",
    )
    return {key: manifest.get(key) for key in keys} | {
        "runs": [
            {
                "candidate_id": run.get("candidate_id"),
                "sample_index": run.get("sample_index"),
                "seed": run.get("seed"),
                "run_dir": run.get("run_dir"),
            }
            for run in manifest.get("runs", [])
        ]
    }


def load_compatible_manifest(path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return manifest
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateGenerationError(f"Failed to load existing manifest {path}: {exc}") from exc
    validate_manifest(existing)
    if _manifest_identity(existing) != _manifest_identity(manifest):
        raise CandidateGenerationError(
            f"Existing manifest is for a different experiment: {path}"
        )
    for new_run, old_run in zip(manifest["runs"], existing["runs"]):
        new_run["status"] = old_run.get("status", "pending")
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def requested_sample_indices(args) -> list[int]:
    if args.sample_index is None:
        return list(range(args.num_samples))
    if args.sample_index < 0 or args.sample_index >= args.num_samples:
        raise CandidateGenerationError(
            f"sample_index must be in [0, {args.num_samples - 1}]"
        )
    return [args.sample_index]


def completed_result_path(args, run: Dict[str, Any]) -> Path:
    filename = "realhit_cot.jsonl" if args.dataset == "realhit" else "spreadsheet_pot.jsonl"
    return Path(run["run_dir"]) / filename


def run_generation(args) -> Path:
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else default_manifest_path(args)
    )
    manifest = load_compatible_manifest(manifest_path, build_manifest(args))
    save_manifest(manifest, manifest_path)

    indices = requested_sample_indices(args)
    print(f"Manifest: {manifest_path}")
    for sample_index in indices:
        run = manifest["runs"][sample_index]
        if (
            args.resume
            and run.get("status") == "completed"
            and completed_result_path(args, run).is_file()
        ):
            print(
                f"[{run['candidate_id']}] seed={run['seed']} already completed; skipping"
            )
            continue
        command = build_solver_command(args, sample_index, run["seed"])
        print(f"[{run['candidate_id']}] seed={run['seed']} run_dir={run['run_dir']}")
        print(shlex.join(command))
        if args.dry_run:
            run["status"] = "dry_run"
            save_manifest(manifest, manifest_path)
            continue
        try:
            subprocess.run(command, cwd=REPO_DIR, check=True)
        except subprocess.CalledProcessError:
            run["status"] = "failed"
            save_manifest(manifest, manifest_path)
            raise
        run["status"] = "completed"
        save_manifest(manifest, manifest_path)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate independent seeded Self-Consistency candidate runs."
    )
    parser.add_argument("--dataset", required=True, choices=("realhit", "spreadsheet"))
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--table_format", required=True)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--output_root", default="sc_outs/candidates")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_text_tokens", type=int, default=0)
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question_types", default=None)
    parser.add_argument("--instruction_types", default=None)
    parser.add_argument("--code_exec_url", default="localhost:8081")
    parser.add_argument("--render_formulas_before_eval", action="store_true")
    parser.add_argument("--include_coordinates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill_merged", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--suffix", default=None)
    return parser.parse_args(argv)


def main():
    run_generation(parse_args())


if __name__ == "__main__":
    main()
