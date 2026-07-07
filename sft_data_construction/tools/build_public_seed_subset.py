#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_data_distill.io_utils import sha256_json, write_json, write_jsonl
from or_data_distill.ir_schema import (
    ANSWER_STYLES,
    DATA_INTERFACES,
    DIFFICULTIES,
    DOMAINS,
    PROBLEM_MODES,
    STRUCTURES,
)
from or_data_distill.quality_gate import validate_sft_record


SOURCE_TRACE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_api_or_key",
        re.compile(r"sk-[A-Za-z0-9]|(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}", re.I),
    ),
    ("local_path", re.compile(r"/home/[^\s\"']+|/root/|/mnt/|/tmp/", re.I)),
    ("unfinished", re.compile(r"\bplaceholder\b|\bTODO\b|\bTBD\b|\bN/A\b|\bLocked\b|\bSc0\b|\)\s*Skip\b", re.I)),
)

DOMAIN_MAP = {
    "logistics_transport": "logistics",
    "production_manufacturing": "production",
    "scheduling_workforce": "production",
    "finance_marketing": "finance",
    "network_flow_graph": "logistics",
    "energy_utilities": "energy",
    "inventory_supply": "production",
    "assignment_selection": "generic",
    "health_pharma": "healthcare",
    "healthcare": "healthcare",
    "education_public": "education",
    "environment_sustainability": "environment",
    "other_domain": "generic",
}

STRUCTURE_MAP = {
    "assignment_matching": "assignment",
    "scheduling_timing": "scheduling",
    "network_flow": "routing",
    "transportation_distribution": "transportation",
    "blending_mixing": "blending",
    "inventory_balance": "capacity_planning",
    "capacity_planning": "capacity_planning",
    "facility_location": "facility_location",
    "production_planning": "production_planning",
    "portfolio_budget": "portfolio",
    "selection_knapsack": "generic_lp_mip",
    "covering_set": "generic_lp_mip",
    "packing_cutting": "generic_lp_mip",
}

INTERFACE_MAP = {
    "inline_text": "inline_text",
    "markdown_table": "markdown_table",
    "data_json": "attached_files",
    "csv_instance": "attached_files",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def nested_get(row: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_pool_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            ir_id = str(row.get("ir_id") or "")
            if not ir_id:
                continue
            buckets = nested_get(row, ["metadata", "buckets"], {}) or {}
            mode = nested_get(row, ["problem", "mode"], None)
            metadata_by_id[ir_id] = {
                "mode": mode,
                "domain": buckets.get("domain"),
                "structure": buckets.get("structure"),
                "difficulty": buckets.get("difficulty"),
                "data_interface": buckets.get("data_interface"),
                "file_schema_type": buckets.get("file_schema_type"),
            }
    return metadata_by_id


def normalize_mode(value: Any) -> str:
    value = str(value or "").strip().upper()
    return value if value in PROBLEM_MODES else "DP"


def map_domain(value: Any) -> str:
    value = str(value or "").strip()
    value = DOMAIN_MAP.get(value, value)
    return value if value in DOMAINS else "generic"


def map_structure(value: Any) -> str:
    value = str(value or "").strip()
    value = STRUCTURE_MAP.get(value, value)
    return value if value in STRUCTURES else "generic_lp_mip"


def map_difficulty(value: Any) -> str:
    value = str(value or "").strip()
    return value if value in DIFFICULTIES else "medium"


def infer_interface(mode: str, raw_interface: Any, problem: str) -> str:
    raw = str(raw_interface or "").strip()
    if raw in INTERFACE_MAP:
        return INTERFACE_MAP[raw]
    if mode == "DPS":
        return "attached_files"
    if mode == "DT" or re.search(r"\|[ \t:-]+\|", problem):
        return "markdown_table"
    return "inline_text"


def infer_answer_style(answer: str) -> str:
    stripped = answer.lstrip()
    if stripped.startswith("import ") or stripped.startswith("from "):
        return "code_only"
    if "<python>" in answer or "gurobipy" in answer or "model.optimize()" in answer:
        return "math_model_with_code"
    return "math_model" if "math_model" in ANSWER_STYLES else "math_model_with_code"


def has_trace(text: str, extra_patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    patterns = list(SOURCE_TRACE_PATTERNS) + extra_patterns
    return [name for name, pattern in patterns if pattern.search(text)]


def render_data_files(data_files: Any, *, max_chars: int) -> str:
    if not isinstance(data_files, list) or not data_files:
        return ""
    blocks = ["\n\nThe following structured data files are provided:"]
    for item in data_files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("name") or "data.txt")
        kind = str(item.get("kind") or "").lower()
        content = item.get("content", "")
        if isinstance(content, (dict, list)):
            language = "json" if kind == "json" or path.endswith(".json") else ""
            rendered = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            language = "csv" if kind == "csv" or path.endswith(".csv") else ""
            rendered = str(content)
        if len(rendered) > max_chars:
            return ""
        blocks.append(f"\n### {path}\n```{language}\n{rendered}\n```")
    return "\n".join(blocks)


def token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(tok) >= 3}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def make_candidate(
    row: dict[str, Any],
    pool_meta: dict[str, dict[str, Any]],
    index: int,
    max_data_chars: int,
    extra_forbidden_patterns: list[tuple[str, re.Pattern[str]]],
) -> tuple[dict[str, Any] | None, list[str]]:
    problem = str(row.get("problem_text") or row.get("problem") or "").strip()
    answer = str(row.get("answer_text") or row.get("answer") or "").strip()
    source_ir_id = str(row.get("source_ir_id") or "")
    source_meta = pool_meta.get(source_ir_id, {})
    mode = normalize_mode(row.get("problem_mode") or source_meta.get("mode"))
    attachment_text = render_data_files(row.get("data_files"), max_chars=max_data_chars)
    if attachment_text:
        problem = problem + attachment_text
    raw_interface = source_meta.get("data_interface") or source_meta.get("file_schema_type")
    metadata = {
        "mode": mode,
        "domain": map_domain(source_meta.get("domain")),
        "structure": map_structure(source_meta.get("structure")),
        "difficulty": map_difficulty(source_meta.get("difficulty")),
        "data_interface": infer_interface(mode, raw_interface, problem),
        "answer_style": infer_answer_style(answer),
        "seed_stage": "public_synthetic_seed",
    }
    issues: list[str] = []
    if not problem or not answer:
        issues.append("empty_problem_or_answer")
    if len(problem) < 40:
        issues.append("problem_too_short")
    if len(answer) < 80:
        issues.append("answer_too_short")
    if len(problem) + len(answer) > 30000:
        issues.append("sample_too_long")
    for label, text in (("problem", problem), ("answer", answer)):
        for issue in has_trace(text, extra_forbidden_patterns):
            issues.append(f"{label}:{issue}")
    sft_probe = {
        "messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": answer}],
        "metadata": {key: value for key, value in metadata.items() if key != "seed_stage"},
    }
    issues.extend(validate_sft_record(sft_probe))
    if issues:
        return None, issues
    payload = {"problem": problem, "answer": answer, "metadata": metadata}
    return {
        "id": f"public_seed_{sha256_json(payload)[:16]}",
        **payload,
    }, []


def mode_targets(total: int, available: collections.Counter[str]) -> dict[str, int]:
    weights = {"DP": 0.35, "DT": 0.25, "DPS": 0.40}
    targets = {mode: min(available.get(mode, 0), int(round(total * weight))) for mode, weight in weights.items()}
    while sum(targets.values()) < total:
        options = [mode for mode in ("DP", "DT", "DPS") if targets.get(mode, 0) < available.get(mode, 0)]
        if not options:
            break
        mode = max(options, key=lambda item: available[item] - targets.get(item, 0))
        targets[mode] = targets.get(mode, 0) + 1
    while sum(targets.values()) > total:
        mode = max(targets, key=targets.get)
        targets[mode] -= 1
    return targets


def diverse_select(candidates: list[dict[str, Any]], count: int, seed: int, max_similarity: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_mode: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_mode[row["metadata"]["mode"]].append(row)
    targets = mode_targets(count, collections.Counter({mode: len(rows) for mode, rows in by_mode.items()}))
    selected: list[dict[str, Any]] = []
    selected_tokens: list[set[str]] = []
    used_ids: set[str] = set()

    def try_add(row: dict[str, Any], *, enforce_similarity: bool) -> bool:
        if row["id"] in used_ids:
            return False
        tokens = token_set(row["problem"])
        if enforce_similarity and any(jaccard(tokens, old) > max_similarity for old in selected_tokens):
            return False
        selected.append(row)
        selected_tokens.append(tokens)
        used_ids.add(row["id"])
        return True

    for mode in ("DP", "DT", "DPS"):
        rows = by_mode.get(mode, [])
        rng.shuffle(rows)
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            meta = row["metadata"]
            grouped[(meta["domain"], meta["structure"], meta["difficulty"], meta["answer_style"])].append(row)
        buckets = list(grouped.values())
        rng.shuffle(buckets)
        mode_added = 0
        while mode_added < targets.get(mode, 0) and buckets:
            progressed = False
            for bucket in list(buckets):
                if not bucket:
                    buckets.remove(bucket)
                    continue
                row = bucket.pop()
                if try_add(row, enforce_similarity=True):
                    mode_added += 1
                    progressed = True
                    if mode_added >= targets.get(mode, 0):
                        break
            if not progressed:
                break

    if len(selected) < count:
        rest = [row for row in candidates if row["id"] not in used_ids]
        rng.shuffle(rest)
        for row in rest:
            if try_add(row, enforce_similarity=True) and len(selected) >= count:
                break
    if len(selected) < count:
        rest = [row for row in candidates if row["id"] not in used_ids]
        rng.shuffle(rest)
        for row in rest:
            if try_add(row, enforce_similarity=False) and len(selected) >= count:
                break
    selected.sort(key=lambda row: row["id"])
    return selected[:count]


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = collections.Counter(str(row.get("metadata", {}).get(key) or "") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean, diverse public seed subset from accepted generated data.")
    parser.add_argument("--answers", nargs="+", type=Path, required=True, help="Accepted problem-answer JSONL files.")
    parser.add_argument("--synthetic-pools", nargs="*", type=Path, default=[], help="Optional synthetic IR pool JSONL files for generic metadata.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-similarity", type=float, default=0.82)
    parser.add_argument("--max-data-file-chars", type=int, default=6000)
    parser.add_argument(
        "--forbidden-regex",
        action="append",
        default=[],
        help="Additional source-specific regex to reject from problem/answer text. May be repeated.",
    )
    args = parser.parse_args()

    pool_meta = load_pool_metadata(args.synthetic_pools)
    extra_forbidden_patterns = [
        (f"extra_forbidden_{idx}", re.compile(pattern, re.I)) for idx, pattern in enumerate(args.forbidden_regex)
    ]
    candidates: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()
    reject_counts: collections.Counter[str] = collections.Counter()
    rejected_candidate_rows = 0
    input_rows = 0
    for path in args.answers:
        for index, row in enumerate(read_jsonl(path)):
            input_rows += 1
            candidate, issues = make_candidate(
                row,
                pool_meta,
                index,
                args.max_data_file_chars,
                extra_forbidden_patterns,
            )
            if not candidate:
                rejected_candidate_rows += 1
                reject_counts.update(issues)
                continue
            fingerprint = sha256_json({"problem": candidate["problem"], "answer": candidate["answer"]})
            if fingerprint in seen_payloads:
                rejected_candidate_rows += 1
                reject_counts["duplicate_problem_answer"] += 1
                continue
            seen_payloads.add(fingerprint)
            candidates.append(candidate)

    selected = diverse_select(candidates, args.count, args.seed, args.max_similarity)
    write_jsonl(args.output, selected)
    manifest = {
        "schema": "or_data_distill_public_seed_subset_v0.1",
        "requested_count": args.count,
        "selected_count": len(selected),
        "input_rows": input_rows,
        "candidate_rows": len(candidates),
        "rejected_candidate_rows": rejected_candidate_rows,
        "rejected_issue_count": sum(reject_counts.values()),
        "reject_counts": dict(reject_counts.most_common()),
        "selection": {
            "seed": args.seed,
            "max_similarity": args.max_similarity,
            "mode_distribution": distribution(selected, "mode"),
            "domain_distribution": distribution(selected, "domain"),
            "structure_distribution": distribution(selected, "structure"),
            "difficulty_distribution": distribution(selected, "difficulty"),
            "data_interface_distribution": distribution(selected, "data_interface"),
            "answer_style_distribution": distribution(selected, "answer_style"),
        },
        "output": {
            "seed_file": str(args.output.name),
            "manifest_file": str(args.manifest.name),
        },
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
