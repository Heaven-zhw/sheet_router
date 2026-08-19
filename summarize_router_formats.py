import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FORMAT_KEYS = ("table_format", "format", "selected_format", "postprocessed_format")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(row).__name__}")
            rows.append(row)
    return rows


def resolve_decision_file(path: Path) -> Path:
    if path.is_file():
        return path

    candidate = path / "router_decisions.jsonl"
    if candidate.exists():
        return candidate

    matches = sorted(path.rglob("router_decisions.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No router_decisions.jsonl found under {path}")
    raise ValueError(
        f"Multiple router_decisions.jsonl files found under {path}; pass a more specific directory:\n"
        + "\n".join(str(item) for item in matches)
    )


def extract_format(row: Dict[str, Any]) -> Optional[str]:
    decision = row.get("router_decision")
    if isinstance(decision, dict):
        for key in FORMAT_KEYS:
            value = decision.get(key)
            if value:
                return str(value)
        stages = decision.get("stages")
        if isinstance(stages, dict):
            for key in ("postprocessed_format", "representation_router"):
                value = stages.get(key)
                if value:
                    return str(value)

    for key in FORMAT_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    return None


def format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{count / total * 100:.2f}%"


def summarize(path: Path) -> Dict[str, Any]:
    decision_file = resolve_decision_file(path)
    rows = load_jsonl(decision_file)
    counts = Counter()
    missing = 0
    for row in rows:
        table_format = extract_format(row)
        if table_format:
            counts[table_format] += 1
        else:
            missing += 1
    return {
        "input": str(path),
        "decision_file": str(decision_file),
        "total": len(rows),
        "counts": counts,
        "missing": missing,
    }


def print_markdown_table(results: List[Dict[str, Any]]) -> None:
    print("| Input | Decision file | Format | Count | Percent |")
    print("|---|---|---:|---:|---:|")
    for result in results:
        total = result["total"]
        counts: Counter = result["counts"]
        if counts:
            for table_format, count in counts.most_common():
                print(
                    f"| {result['input']} | {result['decision_file']} | "
                    f"`{table_format}` | {count} / {total} | {format_percent(count, total)} |"
                )
        else:
            print(f"| {result['input']} | {result['decision_file']} | `<none>` | 0 / {total} | 0.00% |")

        if result["missing"]:
            print(
                f"| {result['input']} | {result['decision_file']} | `<missing>` | "
                f"{result['missing']} / {total} | {format_percent(result['missing'], total)} |"
            )


def print_text_summary(results: List[Dict[str, Any]]) -> None:
    for index, result in enumerate(results):
        if index:
            print()
        total = result["total"]
        print(f"Input: {result['input']}")
        print(f"Decision file: {result['decision_file']}")
        print(f"Total decisions: {total}")
        for table_format, count in result["counts"].most_common():
            print(f"  {table_format}: {count} / {total} ({format_percent(count, total)})")
        if result["missing"]:
            print(f"  <missing>: {result['missing']} / {total} ({format_percent(result['missing'], total)})")


def print_csv(results: List[Dict[str, Any]]) -> None:
    print("input,decision_file,format,count,total,percent")
    for result in results:
        total = result["total"]
        counts: Counter = result["counts"]
        items = counts.most_common() or [("<none>", 0)]
        for table_format, count in items:
            print(
                f"{json.dumps(result['input'], ensure_ascii=False)},"
                f"{json.dumps(result['decision_file'], ensure_ascii=False)},"
                f"{json.dumps(table_format, ensure_ascii=False)},"
                f"{count},{total},{count / total * 100 if total else 0:.4f}"
            )
        if result["missing"]:
            print(
                f"{json.dumps(result['input'], ensure_ascii=False)},"
                f"{json.dumps(result['decision_file'], ensure_ascii=False)},"
                f"\"<missing>\",{result['missing']},{total},{result['missing'] / total * 100 if total else 0:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize selected table-format counts from router_decisions.jsonl files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more result directories, or direct router_decisions.jsonl files.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [summarize(Path(path).expanduser().resolve()) for path in args.paths]
    if args.format == "markdown":
        print_markdown_table(results)
    elif args.format == "csv":
        print_csv(results)
    else:
        print_text_summary(results)


if __name__ == "__main__":
    main()
