from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cpt_cleaner.utils import json_compat

from or_cpt_engine.utils.io import iter_jsonl, write_jsonl, write_text


STAGE_SPECS: tuple[dict[str, str], ...] = (
    {
        "stage": "05_backtranslation",
        "path": "05_backtranslation/backtranslation_candidates.jsonl",
        "text_key": "problem_statement",
    },
    {
        "stage": "06_nl_quality_filter",
        "path": "06_nl_quality_filter/nl_validated_candidates.jsonl",
        "text_key": "problem_statement",
    },
    {
        "stage": "07_forward_modeling",
        "path": "07_forward_modeling/forward_modeling_outputs.jsonl",
        "text_key": "problem_statement",
    },
    {
        "stage": "08_forward_eval",
        "path": "08_forward_eval/accepted_pairs.jsonl",
        "text_key": "problem_statement",
    },
    {
        "stage": "09_cpt_rendering",
        "path": "09_cpt_rendering/cpt_documents.jsonl",
        "text_key": "text",
    },
    {
        "stage": "10_train_export",
        "path": "10_train_val_export/train.jsonl",
        "text_key": "text",
    },
)

_STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "also",
    "available",
    "because",
    "before",
    "between",
    "cannot",
    "could",
    "during",
    "each",
    "every",
    "from",
    "given",
    "goal",
    "have",
    "must",
    "need",
    "needs",
    "number",
    "objective",
    "problem",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "total",
    "using",
    "which",
    "with",
}


def analyze_run_diversity(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    near_duplicate_threshold: float = 0.72,
    max_rows_per_similarity_group: int = 80,
    max_reported_pairs: int = 100,
) -> dict[str, Any]:
    run = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else run / "reports" / "diversity"
    output.mkdir(parents=True, exist_ok=True)

    stage_reports: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    opening_rows: list[dict[str, Any]] = []
    near_duplicate_rows: list[dict[str, Any]] = []

    for spec in STAGE_SPECS:
        path = run / spec["path"]
        rows = list(iter_jsonl(path) or [])
        if not rows:
            continue
        report = _analyze_stage(
            spec["stage"],
            rows,
            text_key=spec["text_key"],
            near_duplicate_threshold=near_duplicate_threshold,
            max_rows_per_similarity_group=max_rows_per_similarity_group,
            max_reported_pairs=max_reported_pairs,
        )
        stage_reports.append(report)
        distribution_rows.extend(_distribution_rows(report))
        opening_rows.extend(_opening_rows(report))
        near_duplicate_rows.extend(report["near_duplicate_examples"])

    summary = {
        "run_dir": str(run),
        "stage_count": len(stage_reports),
        "near_duplicate_threshold": near_duplicate_threshold,
        "max_rows_per_similarity_group": max_rows_per_similarity_group,
        "stages": stage_reports,
    }
    (output / "diversity_summary.json").write_bytes(json_compat.dumps(summary))
    _write_csv(output / "diversity_distribution.csv", distribution_rows)
    _write_csv(output / "opening_template_dashboard.csv", opening_rows)
    write_jsonl(output / "near_duplicate_examples.jsonl", near_duplicate_rows[:max_reported_pairs])
    write_text(output / "diversity_dashboard.md", render_diversity_dashboard(summary))
    return {"stages": len(stage_reports), "output_dir": str(output)}


def render_diversity_dashboard(summary: dict[str, Any]) -> str:
    lines = [
        "# OR-CPT Diversity Dashboard",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Stages analyzed: `{summary['stage_count']}`",
        f"- Near-duplicate threshold: `{summary['near_duplicate_threshold']}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Rows | Unique Text | Exact Dup Rate | Opening Template Top Share | Candidate Pair Similarity | Sampled Near-Dup Rate | Generator HHI | Scenario HHI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in summary.get("stages") or []:
        lines.append(
            f"| {stage['stage']} | {stage['row_count']} | {stage['unique_text_count']} | "
            f"{_pct(stage['exact_duplicate_rate'])} | {_pct(stage['top_opening_template_share'])} | "
            f"{_fmt(stage.get('paired_candidate_similarity_avg'))} | {_pct(stage['sampled_near_duplicate_rate'])} | "
            f"{_fmt(stage['generator_hhi'])} | {_fmt(stage['scenario_hhi'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Exact Dup Rate` 只统计完全相同文本，通常会偏乐观。",
            "- `Opening Template Top Share` 越高，说明开头句模板越集中。",
            "- `Candidate Pair Similarity` 统计同一 seed 的不同 backtranslation candidate 是否只是换皮。",
            "- `Sampled Near-Dup Rate` 是按 generator/scenario 分组抽样的词集合 Jaccard 近重复率。",
            "- `Generator HHI` 和 `Scenario HHI` 越高，分布越集中；越接近 0 越均衡。",
            "",
        ]
    )
    for stage in summary.get("stages") or []:
        lines.extend(_render_stage_details(stage))
    return "\n".join(lines) + "\n"


def _analyze_stage(
    stage: str,
    rows: list[dict[str, Any]],
    *,
    text_key: str,
    near_duplicate_threshold: float,
    max_rows_per_similarity_group: int,
    max_reported_pairs: int,
) -> dict[str, Any]:
    texts = [_row_text(row, text_key) for row in rows]
    exact_counts = Counter(_normalize_text(text) for text in texts if text.strip())
    opening_counts = Counter(_opening_template(text) for text in texts if text.strip())
    generator_counts = Counter(_generator_id(row) for row in rows)
    scenario_counts = Counter(_scenario_id(row) for row in rows)
    base_scenario_counts = Counter(_base_scenario_id(row) for row in rows)
    scenario_variant_counts = Counter(_scenario_variant_id(row) for row in rows)
    industry_lens_counts = Counter(_industry_lens_id(row) for row in rows)
    narrative_angle_counts = Counter(_narrative_angle_id(row) for row in rows)
    task_family_counts = Counter(_task_family(row) for row in rows)
    trigger_counts = Counter(_business_trigger(row) for row in rows)
    doc_type_counts = Counter(_doc_type(row) for row in rows)
    near_duplicate = _sample_near_duplicates(
        stage,
        rows,
        text_key=text_key,
        threshold=near_duplicate_threshold,
        max_rows_per_group=max_rows_per_similarity_group,
        max_examples=max_reported_pairs,
    )
    pair_similarity = _paired_candidate_similarity(rows, text_key=text_key)
    top_opening_count = opening_counts.most_common(1)[0][1] if opening_counts else 0
    return {
        "stage": stage,
        "row_count": len(rows),
        "unique_text_count": len(exact_counts),
        "exact_duplicate_rows": sum(count - 1 for count in exact_counts.values() if count > 1),
        "exact_duplicate_rate": _rate(sum(count - 1 for count in exact_counts.values() if count > 1), len(rows)),
        "top_opening_template_share": _rate(top_opening_count, len(rows)),
        "paired_candidate_similarity_avg": pair_similarity.get("avg"),
        "paired_candidate_similarity_median": pair_similarity.get("median"),
        "paired_candidate_similarity_p90": pair_similarity.get("p90"),
        "sampled_near_duplicate_pairs": near_duplicate["near_duplicate_pairs"],
        "sampled_compared_pairs": near_duplicate["compared_pairs"],
        "sampled_near_duplicate_rate": _rate(near_duplicate["near_duplicate_pairs"], near_duplicate["compared_pairs"]),
        "near_duplicate_examples": near_duplicate["examples"],
        "generator_hhi": _hhi(generator_counts),
        "scenario_hhi": _hhi(scenario_counts),
        "top_generators": generator_counts.most_common(20),
        "top_scenarios": scenario_counts.most_common(20),
        "top_base_scenarios": base_scenario_counts.most_common(20),
        "top_scenario_variants": scenario_variant_counts.most_common(20),
        "top_industry_lenses": industry_lens_counts.most_common(20),
        "top_narrative_angles": narrative_angle_counts.most_common(20),
        "top_task_families": task_family_counts.most_common(20),
        "top_business_triggers": trigger_counts.most_common(20),
        "top_doc_types": doc_type_counts.most_common(20),
        "top_opening_templates": opening_counts.most_common(30),
    }


def _sample_near_duplicates(
    stage: str,
    rows: list[dict[str, Any]],
    *,
    text_key: str,
    threshold: float,
    max_rows_per_group: int,
    max_examples: int,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, dict[str, Any], set[str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        tokens = _token_set(_row_text(row, text_key))
        if not tokens:
            continue
        key = "|".join([_generator_id(row), _scenario_id(row), _doc_type(row)])
        if len(groups[key]) < max_rows_per_group:
            groups[key].append((index, row, tokens))

    compared_pairs = 0
    near_duplicate_pairs = 0
    examples: list[dict[str, Any]] = []
    for group_key, group_rows in groups.items():
        for left_index in range(len(group_rows)):
            for right_index in range(left_index + 1, len(group_rows)):
                left_pos, left_row, left_tokens = group_rows[left_index]
                right_pos, right_row, right_tokens = group_rows[right_index]
                compared_pairs += 1
                score = _jaccard(left_tokens, right_tokens)
                if score < threshold:
                    continue
                near_duplicate_pairs += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "stage": stage,
                            "group_key": group_key,
                            "similarity": round(score, 6),
                            "left_index": left_pos,
                            "right_index": right_pos,
                            "left_id": _row_id(left_row),
                            "right_id": _row_id(right_row),
                            "left_opening": _first_sentence(_row_text(left_row, text_key))[:300],
                            "right_opening": _first_sentence(_row_text(right_row, text_key))[:300],
                        }
                    )
    examples.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return {
        "compared_pairs": compared_pairs,
        "near_duplicate_pairs": near_duplicate_pairs,
        "examples": examples,
    }


def _paired_candidate_similarity(rows: list[dict[str, Any]], *, text_key: str) -> dict[str, float | None]:
    grouped: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        if row.get("candidate_index") is None or not row.get("instance_id"):
            continue
        try:
            candidate_index = int(row["candidate_index"])
        except (TypeError, ValueError):
            continue
        grouped[str(row["instance_id"])][candidate_index] = _row_text(row, text_key)
    similarities: list[float] = []
    for candidates in grouped.values():
        if 1 not in candidates or 2 not in candidates:
            continue
        left = _token_set(candidates[1])
        right = _token_set(candidates[2])
        if left or right:
            similarities.append(_jaccard(left, right))
    if not similarities:
        return {"avg": None, "median": None, "p90": None}
    ordered = sorted(similarities)
    return {
        "avg": sum(similarities) / len(similarities),
        "median": ordered[len(ordered) // 2],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
    }


def _distribution_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, values in (
        ("generator", report["top_generators"]),
        ("scenario", report["top_scenarios"]),
        ("base_scenario", report["top_base_scenarios"]),
        ("scenario_variant", report["top_scenario_variants"]),
        ("industry_lens", report["top_industry_lenses"]),
        ("narrative_angle", report["top_narrative_angles"]),
        ("task_family", report["top_task_families"]),
        ("business_trigger", report["top_business_triggers"]),
        ("doc_type", report["top_doc_types"]),
    ):
        total = report["row_count"]
        for value, count in values:
            rows.append(
                {
                    "stage": report["stage"],
                    "field": field,
                    "value": value,
                    "count": count,
                    "share": _rate(count, total),
                }
            )
    return rows


def _opening_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template, count in report["top_opening_templates"]:
        rows.append(
            {
                "stage": report["stage"],
                "opening_template": template,
                "count": count,
                "share": _rate(count, report["row_count"]),
            }
        )
    return rows


def _render_stage_details(stage: dict[str, Any]) -> list[str]:
    lines = [
        f"## {stage['stage']}",
        "",
        "### Top Generators",
        "",
        "| Generator | Count | Share |",
        "|---|---:|---:|",
    ]
    for value, count in stage["top_generators"][:10]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, stage['row_count']))} |")
    lines.extend(["", "### Top Scenarios", "", "| Scenario | Count | Share |", "|---|---:|---:|"])
    for value, count in stage["top_scenarios"][:10]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, stage['row_count']))} |")
    lines.extend(["", "### Top Industry Lenses", "", "| Industry Lens | Count | Share |", "|---|---:|---:|"])
    for value, count in stage["top_industry_lenses"][:10]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, stage['row_count']))} |")
    lines.extend(["", "### Top Narrative Angles", "", "| Narrative Angle | Count | Share |", "|---|---:|---:|"])
    for value, count in stage["top_narrative_angles"][:10]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, stage['row_count']))} |")
    lines.extend(["", "### Top Opening Templates", "", "| Opening Template | Count | Share |", "|---|---:|---:|"])
    for value, count in stage["top_opening_templates"][:10]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, stage['row_count']))} |")
    lines.append("")
    return lines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row_text(row: dict[str, Any], text_key: str) -> str:
    value = _nested_get(row, text_key)
    if value:
        return str(value)
    for fallback in (
        "problem_statement",
        "text",
        "candidate.problem_statement",
        "backtranslation.problem_statement",
        "metadata.problem_statement",
    ):
        value = _nested_get(row, fallback)
        if value:
            return str(value)
    return ""


def _row_id(row: dict[str, Any]) -> str:
    for key in ("bt_id", "fm_id", "pair_id", "doc_id", "id", "instance_id"):
        if row.get(key):
            return str(row[key])
    return "unknown"


def _generator_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("generator_id") or metadata.get("generator_id") or "unknown")


def _scenario_id(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.scenario_id")
        or _nested_get(row, "metadata.scenario_id")
        or _nested_get(row, "metadata.source_metadata.scenario_id")
        or _nested_get(row, "source_metadata.scenario_id")
        or "unknown"
    )


def _base_scenario_id(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.base_scenario_id")
        or _nested_get(row, "metadata.base_scenario_id")
        or _nested_get(row, "metadata.source_metadata.base_scenario_id")
        or _nested_get(row, "source_metadata.base_scenario_id")
        or "unknown"
    )


def _scenario_variant_id(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.scenario_variant_id")
        or _nested_get(row, "metadata.scenario_variant_id")
        or _nested_get(row, "metadata.source_metadata.scenario_variant_id")
        or _nested_get(row, "source_metadata.scenario_variant_id")
        or "unknown"
    )


def _industry_lens_id(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.industry_lens_id")
        or _nested_get(row, "metadata.industry_lens_id")
        or _nested_get(row, "metadata.source_metadata.industry_lens_id")
        or _nested_get(row, "source_metadata.industry_lens_id")
        or "unknown"
    )


def _narrative_angle_id(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.narrative_angle_id")
        or _nested_get(row, "metadata.narrative_angle_id")
        or _nested_get(row, "metadata.source_metadata.narrative_angle_id")
        or _nested_get(row, "source_metadata.narrative_angle_id")
        or "unknown"
    )


def _task_family(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.task_family")
        or _nested_get(row, "metadata.task_family")
        or _nested_get(row, "metadata.source_metadata.task_family")
        or _nested_get(row, "source_metadata.task_family")
        or "unknown"
    )


def _business_trigger(row: dict[str, Any]) -> str:
    return str(
        _nested_get(row, "llm_metadata.business_trigger")
        or _nested_get(row, "metadata.business_trigger")
        or _nested_get(row, "metadata.source_metadata.business_trigger")
        or _nested_get(row, "source_metadata.business_trigger")
        or "unknown"
    )


def _doc_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("doc_type") or metadata.get("doc_type") or "unknown")


def _nested_get(row: dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)
    return parts[0]


def _opening_template(text: str) -> str:
    first = _first_sentence(text).lower()
    first = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", first)
    first = re.sub(r"\b[a-z]+[_-]\d+\b", "<id>", first)
    first = re.sub(r"\b(facility|site|clinic|customer|parcel|item|container|machine|job|period|product|warehouse|store)\s+\d+\b", r"\1 <num>", first)
    first = re.sub(r"\s+", " ", first).strip()
    return first[:240] or "empty"


def _token_set(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z_'-]{2,}", text.lower()):
        normalized = token.strip("_'-")
        if len(normalized) < 4 or normalized in _STOPWORDS:
            continue
        tokens.add(normalized)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _hhi(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return round(sum((count / total) ** 2 for count in counter.values()), 6)


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.3f}"
