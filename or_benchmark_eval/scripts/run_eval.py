#!/usr/bin/env python3
"""Summarize OR benchmark evaluation outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


TARGETS = (
    "nl4opt_solver",
    "optibench_solver",
    "bench4opt_feasible_solver",
    "bench4opt_orgeval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OR benchmark results")
    parser.add_argument("--skip_run", action="store_true", help="Kept for compatibility with scripts/run_eval.sh")
    parser.add_argument("--result_root", default="results")
    parser.add_argument("--summary_root", default=None)
    parser.add_argument("--summary_tag", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", default=None)
    parser.add_argument("--bench4opt_max_samples", default=None)
    parser.add_argument("--nl4opt_dataset", default="./data/NL4OPT")
    parser.add_argument("--optibench_dataset", default="./data/optibench")
    parser.add_argument("--bench4opt_dataset", default="data/bench4opt")
    parser.add_argument("--bench4opt_feasible_dataset", default="data/bench4opt_feasible")
    parser.add_argument("--skip_missing_results", action="store_true")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    return parser.parse_args()


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in model)


def range_suffix(args: argparse.Namespace) -> str:
    if args.start != 0 or args.end is not None:
        return f"_{args.start}_{args.end or 'none'}"
    return ""


def target_display_name(target_key: str) -> str:
    return {
        "nl4opt_solver": "nl4opt",
        "optibench_solver": "optibench",
        "bench4opt_feasible_solver": "bench4opt-feasible",
        "bench4opt_orgeval": "bench4opt",
    }[target_key]


def result_path(args: argparse.Namespace, target_key: str, model: str) -> Path:
    root = Path(args.result_root)
    safe_model = safe_model_name(model)
    suffix = range_suffix(args)
    if target_key == "nl4opt_solver":
        return root / "nl4opt" / f"{safe_model}{suffix}_solver.json"
    if target_key == "optibench_solver":
        return root / "optibench" / f"{safe_model}{suffix}_solver.json"
    if target_key == "bench4opt_feasible_solver":
        return root / "bench4opt" / f"{safe_model}{suffix}_solver.json"
    if target_key == "bench4opt_orgeval":
        max_suffix = f"_max{args.bench4opt_max_samples}" if args.bench4opt_max_samples else ""
        return root / "bench4opt" / f"{safe_model}{suffix}{max_suffix}_orgeval.json"
    raise ValueError(f"Unsupported target: {target_key}")


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON result file: {path}")
    return payload


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def solver_code_pass(item: dict[str, Any]) -> bool:
    return item.get("predicted_optimal_value") is not None


def bench4opt_code_pass(item: dict[str, Any]) -> bool:
    reward = item.get("reward") or {}
    try:
        return float(reward.get("code_reward", 0.0)) >= 1.0
    except (TypeError, ValueError):
        return False


def bench4opt_success(item: dict[str, Any]) -> bool:
    reward = item.get("reward") or {}
    try:
        return float(reward.get("wl_reward", 0.0)) >= 1.0
    except (TypeError, ValueError):
        return False


def classify_build_error(payload: Any) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    lower = text.lower()
    if "api_error" in lower:
        return "api_error"
    if "timeout" in lower:
        return "timeout"
    if "syntaxerror" in lower or "positional argument follows keyword argument" in lower:
        return "syntax_error"
    if "model_not_found" in lower:
        return "model_not_found"
    if "nameerror" in lower or "is not defined" in lower:
        return "name_error"
    if "keyerror" in lower:
        return "key_error"
    if "indexerror" in lower or "out of bounds" in lower:
        return "index_error"
    if (
        "typeerror" in lower
        or "unsupported operand type" in lower
        or "unhashable type" in lower
        or "not subscriptable" in lower
        or "can't multiply sequence" in lower
    ):
        return "type_error"
    if "valueerror" in lower or "could not convert" in lower or "invalid literal" in lower:
        return "value_error"
    if "addconstr" in lower or "addvars" in lower or "duplicate keys in model.addvars" in lower:
        return "gurobi_api_error"
    if "lp_build_error" in lower or "lp file was not created" in lower:
        return "lp_build_failure"
    return "other_build_error"


def classify_equivalence_error(payload: Any) -> str:
    if isinstance(payload, dict):
        if payload.get("var_num_check") is False:
            return "var_count_mismatch"
        if payload.get("cons_num_check") is False:
            return "constraint_count_mismatch"
        if payload.get("wl_check") is False:
            return "wl_graph_mismatch"
        false_keys = [key for key, value in payload.items() if value is False]
        if false_keys:
            return "structure_mismatch:" + ",".join(sorted(false_keys))
    text = str(payload or "").lower()
    if "no reference lp" in text:
        return "missing_reference_lp"
    if "normalization" in text:
        return "normalization_error"
    return "other_equivalence_error"


def classify_orgeval_failure(item: dict[str, Any]) -> str:
    if item.get("api_error", False):
        return "api_error"
    if bench4opt_success(item):
        return "correct"
    verification = item.get("verification") or {}
    if not bench4opt_code_pass(item):
        return classify_build_error(verification.get("code_verification", ""))
    return classify_equivalence_error(verification.get("wl_verification", ""))


def summarize_target(target_key: str, model: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    api_errors = sum(1 for item in items if item.get("api_error", False))
    if target_key == "bench4opt_orgeval":
        success_count = sum(1 for item in items if bench4opt_success(item))
        code_count = sum(1 for item in items if bench4opt_code_pass(item))
    else:
        success_count = sum(1 for item in items if item.get("success"))
        code_count = sum(1 for item in items if solver_code_pass(item))
    return {
        "dataset": target_display_name(target_key),
        "target": target_key,
        "model": model,
        "accuracy": ratio(success_count, total),
        "accuracy_count": success_count,
        "code_pass_rate": ratio(code_count, total),
        "code_pass_count": code_count,
        "api_error_count": api_errors,
        "total": total,
    }


def build_orgeval_error_summary(rows: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    target_key = "bench4opt_orgeval"
    for model, items in rows.get(target_key, {}).items():
        total = len(items)
        failed = [item for item in items if not bench4opt_success(item)]
        failed_total = len(failed)
        counts: dict[str, int] = {}
        for item in failed:
            key = classify_orgeval_failure(item)
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            error_rows.append(
                {
                    "dataset": "bench4opt",
                    "model": model,
                    "error_type": key,
                    "count": count,
                    "share_of_total": ratio(count, total),
                    "share_of_failures": ratio(count, failed_total),
                    "total": total,
                    "failed_total": failed_total,
                }
            )
    return error_rows


def markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Model | Score | Code pass | API errors | Count |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {model} | {score} ({ok}/{total}) | {code} ({code_ok}/{total}) | {api} | {total} |".format(
                dataset=row["dataset"],
                model=row["model"],
                score=fmt_pct(row["accuracy"]),
                ok=row["accuracy_count"],
                total=row["total"],
                code=fmt_pct(row["code_pass_rate"]),
                code_ok=row["code_pass_count"],
                api=row["api_error_count"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    result_paths: dict[str, dict[str, str]] = {}

    for target in args.targets:
        if target not in TARGETS:
            raise SystemExit(f"Unsupported target: {target}")
        result_map[target] = {}
        result_paths[target] = {}
        for model in args.models:
            path = result_path(args, target, model)
            if not path.exists():
                if args.skip_missing_results:
                    continue
                raise FileNotFoundError(path)
            items = load_results(path)
            result_map[target][model] = items
            result_paths[target][model] = str(path.resolve())

    summary_rows: list[dict[str, Any]] = []
    for target in args.targets:
        for model in args.models:
            items = result_map.get(target, {}).get(model)
            if items is not None:
                summary_rows.append(summarize_target(target, model, items))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "result_paths": result_paths,
        "summary": summary_rows,
        "orgeval_error_summary": build_orgeval_error_summary(result_map),
    }

    summary_root = Path(args.summary_root) if args.summary_root else Path(args.result_root) / "summary"
    summary_dir = summary_root / (args.summary_tag or "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = summary_dir / "summary.json"
    md_path = summary_dir / "summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Evaluation Summary\n\n" + markdown_table(summary_rows) + "\n",
        encoding="utf-8",
    )

    print(markdown_table(summary_rows))
    print(f"\nSummary JSON: {json_path}")
    print(f"Summary MD: {md_path}")


if __name__ == "__main__":
    main()
