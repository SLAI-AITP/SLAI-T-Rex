from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from or_cpt_engine.generation.generator_profiles import load_generator_profiles
from or_cpt_engine.utils.io import read_jsonl, write_jsonl, write_text


STAGE_FILES = {
    "instance_generated": "03_instance_generation/instances_generated.jsonl",
    "instance_rejected": "03_instance_generation/instances_generation_rejected.jsonl",
    "solver_validated": "04_solver_validation/solver_validated_instances.jsonl",
    "solver_rejected": "04_solver_validation/solver_rejected_instances.jsonl",
    "instance_quality_validated": "04b_instance_quality/quality_validated_instances.jsonl",
    "instance_quality_review": "04b_instance_quality/quality_review_instances.jsonl",
    "instance_quality_rejected": "04b_instance_quality/quality_rejected_instances.jsonl",
    "backtranslation_candidates": "05_backtranslation/backtranslation_candidates.jsonl",
    "backtranslation_rejected": "05_backtranslation/backtranslation_rejected.jsonl",
    "nl_validated": "06_nl_quality_filter/nl_validated_candidates.jsonl",
    "nl_rejected": "06_nl_quality_filter/nl_rejected.jsonl",
    "forward_outputs": "07_forward_modeling/forward_modeling_outputs.jsonl",
    "forward_rejected": "07_forward_modeling/forward_modeling_rejected.jsonl",
    "forward_accepted": "08_forward_eval/accepted_pairs.jsonl",
    "forward_eval_rejected": "08_forward_eval/rejected_pairs.jsonl",
    "rendered": "09_cpt_rendering/cpt_documents.jsonl",
    "render_rejected": "09_cpt_rendering/cpt_rendering_rejected.jsonl",
}


def analyze_generator_quality(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    profiles_path: str | Path | None = None,
) -> dict[str, int]:
    run = Path(run_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profiles = load_generator_profiles(profiles_path)
    profile_rows = profiles.get("profiles") if isinstance(profiles.get("profiles"), dict) else {}

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    concept_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for stage_name, relative_path in STAGE_FILES.items():
        for row in read_jsonl(run / relative_path):
            generator_id = str(row.get("generator_id") or "unknown")
            counters[generator_id][stage_name] += 1
            for concept in row.get("concept_tags") or (row.get("metadata") or {}).get("concept_tags") or []:
                concept_counts[generator_id][str(concept)] += 1
            for reason in _row_rejection_reasons(stage_name, row):
                rejection_reasons[generator_id][reason] += 1

    generator_ids = set(counters) | set(profile_rows)
    report_rows: list[dict[str, Any]] = []
    for generator_id in sorted(generator_ids):
        counts = counters[generator_id]
        profile = profile_rows.get(generator_id) if isinstance(profile_rows, dict) else {}
        row = {
            "generator_id": generator_id,
            "priority": (profile or {}).get("priority"),
            "recommended_weight": (profile or {}).get("recommended_weight"),
            "sampling_weight": (profile or {}).get("sampling_weight"),
            "task_family": (profile or {}).get("task_family"),
            "sub_family": (profile or {}).get("sub_family"),
            "instance_generated": counts["instance_generated"],
            "instance_rejected": counts["instance_rejected"],
            "solver_validated": counts["solver_validated"],
            "solver_rejected": counts["solver_rejected"],
            "instance_quality_validated": counts["instance_quality_validated"],
            "instance_quality_review": counts["instance_quality_review"],
            "instance_quality_rejected": counts["instance_quality_rejected"],
            "backtranslation_candidates": counts["backtranslation_candidates"],
            "backtranslation_rejected": counts["backtranslation_rejected"],
            "nl_validated": counts["nl_validated"],
            "nl_rejected": counts["nl_rejected"],
            "forward_outputs": counts["forward_outputs"],
            "forward_rejected": counts["forward_rejected"],
            "forward_accepted": counts["forward_accepted"],
            "forward_eval_rejected": counts["forward_eval_rejected"],
            "rendered": counts["rendered"],
            "render_rejected": counts["render_rejected"],
            "solver_pass_rate": _rate(counts["solver_validated"], counts["solver_validated"] + counts["solver_rejected"]),
            "quality_pass_rate": _rate(
                counts["instance_quality_validated"],
                counts["instance_quality_validated"] + counts["instance_quality_review"] + counts["instance_quality_rejected"],
            ),
            "quality_review_rate": _rate(
                counts["instance_quality_review"],
                counts["instance_quality_validated"] + counts["instance_quality_review"] + counts["instance_quality_rejected"],
            ),
            "quality_reject_rate": _rate(
                counts["instance_quality_rejected"],
                counts["instance_quality_validated"] + counts["instance_quality_review"] + counts["instance_quality_rejected"],
            ),
            "generated_to_quality_pass_rate": _rate(counts["instance_quality_validated"], counts["instance_generated"]),
            "nl_pass_rate": _rate(counts["nl_validated"], counts["nl_validated"] + counts["nl_rejected"]),
            "forward_objective_match_rate": _rate(counts["forward_accepted"], counts["forward_accepted"] + counts["forward_eval_rejected"]),
            "render_pass_rate": _rate(counts["rendered"], counts["rendered"] + counts["render_rejected"]),
            "top_rejection_reasons": rejection_reasons[generator_id].most_common(8),
            "top_concepts": concept_counts[generator_id].most_common(8),
        }
        row["stage_scope"] = _stage_scope(counts)
        row["seed_quality_score"] = _seed_quality_score(row)
        row["downstream_score"] = _downstream_score(row) if row["stage_scope"] == "full_pipeline" else None
        row["generator_score"] = _generator_score(row)
        row["recommended_action"] = _recommended_action(row)
        row["quality_risk_flags"] = _quality_risk_flags(row)
        row["production_tier"], row["tier_reason"] = _production_tier(row)
        report_rows.append(row)

    write_jsonl(output / "generator_quality_report.jsonl", report_rows)
    _write_dashboard_csv(output / "generator_quality_dashboard.csv", report_rows)
    write_text(output / "generator_quality_summary.md", render_quality_summary(report_rows, run))
    return {"generators": len(report_rows)}


def render_quality_summary(rows: list[dict[str, Any]], run_dir: Path) -> str:
    lines = [
        "# Generator Quality Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Generators analyzed: {len(rows)}",
        "",
        "| generator_id | tier | scope | family | sub_family | score | seed_score | action | solver | seed_quality | nl | forward | render |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (float(item.get("generator_score") or 0), item["generator_id"])):
        lines.append(
            f"| {row['generator_id']} | {row.get('production_tier') or ''} | {row.get('stage_scope') or ''} | "
            f"{row.get('task_family') or ''} | {row.get('sub_family') or ''} | "
            f"{float(row.get('generator_score') or 0):.3f} | "
            f"{float(row.get('seed_quality_score') or 0):.3f} | "
            f"{row.get('recommended_action')} | {_fmt_rate(row.get('solver_pass_rate'))} | "
            f"{_fmt_rate(row.get('quality_pass_rate'))} | "
            f"{_fmt_rate(row.get('nl_pass_rate'))} | {_fmt_rate(row.get('forward_objective_match_rate'))} | "
            f"{_fmt_rate(row.get('render_pass_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def _write_dashboard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "generator_id",
        "priority",
        "task_family",
        "sub_family",
        "generator_score",
        "seed_quality_score",
        "downstream_score",
        "stage_scope",
        "production_tier",
        "tier_reason",
        "recommended_action",
        "quality_risk_flags",
        "solver_pass_rate",
        "quality_pass_rate",
        "quality_review_rate",
        "quality_reject_rate",
        "generated_to_quality_pass_rate",
        "nl_pass_rate",
        "forward_objective_match_rate",
        "render_pass_rate",
        "rendered",
        "forward_accepted",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _generator_score(row: dict[str, Any]) -> float:
    if row.get("stage_scope") != "full_pipeline":
        return float(row.get("seed_quality_score") or 0.0)
    seed = float(row.get("seed_quality_score") or 0.0)
    downstream = float(row.get("downstream_score") or 0.0)
    return round(0.45 * seed + 0.55 * downstream, 6)


def _seed_quality_score(row: dict[str, Any]) -> float:
    solver = _score_rate(row.get("solver_pass_rate"))
    quality = _optional_score_rate(row.get("quality_pass_rate"), fallback=solver)
    generated_to_pass = _optional_score_rate(row.get("generated_to_quality_pass_rate"), fallback=quality)
    concept = min(1.0, len(row.get("top_concepts") or []) / 4.0)
    return round(0.35 * solver + 0.40 * quality + 0.15 * generated_to_pass + 0.10 * concept, 6)


def _downstream_score(row: dict[str, Any]) -> float:
    forward = _score_rate(row.get("forward_objective_match_rate"))
    nl = _score_rate(row.get("nl_pass_rate"))
    render = _score_rate(row.get("render_pass_rate"))
    return round(0.42 * forward + 0.33 * nl + 0.25 * render, 6)


def _recommended_action(row: dict[str, Any]) -> str:
    score = float(row.get("generator_score") or 0)
    if _total_evidence_count(row) == 0:
        return "needs_data"
    quality_pass_rate = row.get("quality_pass_rate")
    solver_pass_rate = row.get("solver_pass_rate")
    if row.get("stage_scope") != "full_pipeline":
        if solver_pass_rate is not None and float(solver_pass_rate) < 0.80:
            return "optimize_generator_source"
        if quality_pass_rate is not None and float(quality_pass_rate) < 0.80:
            return "optimize_generator_source_or_quality_controller"
        if score >= 0.90:
            return "increase_weight"
        if score >= 0.75:
            return "keep"
        return "limit_until_reviewed"
    if quality_pass_rate is not None and float(quality_pass_rate) < 0.35:
        return "optimize_generator_source_or_quality_controller"
    if row.get("solver_validated", 0) > 0 and (row.get("forward_accepted", 0) + row.get("forward_eval_rejected", 0)) == 0:
        return "needs_downstream_eval"
    if score >= 0.80:
        return "increase_weight"
    if score >= 0.60:
        return "keep"
    if (row.get("solver_pass_rate") or 0) < 0.5:
        return "optimize_generator_source"
    if (row.get("forward_objective_match_rate") or 0) < 0.5:
        return "optimize_prompt_or_semantics"
    return "limit_until_reviewed"


def _production_tier(row: dict[str, Any]) -> tuple[str, str]:
    score = float(row.get("generator_score") or 0.0)
    seed_score = float(row.get("seed_quality_score") or 0.0)
    solver = row.get("solver_pass_rate")
    quality = row.get("quality_pass_rate")
    nl = row.get("nl_pass_rate")
    forward = row.get("forward_objective_match_rate")
    scope = row.get("stage_scope")

    if _total_evidence_count(row) == 0:
        return "D_hold", "No usable evidence yet."
    if solver is not None and float(solver) < 0.50:
        return "D_hold", "Solver pass rate is below 50%."
    if quality is not None and float(quality) < 0.35:
        return "D_hold", "Seed quality pass rate is below 35%."

    if scope == "full_pipeline":
        if forward is not None and float(forward) < 0.25:
            return "D_hold", "Forward objective match rate is below 25%."
        if score >= 0.80 and (forward is None or float(forward) >= 0.60) and (nl is None or float(nl) >= 0.70):
            return "A_core", "High seed quality and downstream acceptance."
        if score >= 0.55:
            return "B_improve", "Useful generator with improvement headroom."
        if score >= 0.35:
            return "C_limited", "Keep limited volume for diversity while improving."
        return "D_hold", "Full-pipeline score is too low for production."

    if seed_score >= 0.90:
        return "A_core", "Strong seed-only audit result."
    if seed_score >= 0.75:
        return "B_improve", "Seed generation is usable but still needs downstream confirmation."
    if seed_score >= 0.45:
        return "C_limited", "Seed generation is marginal; use limited volume."
    return "D_hold", "Seed-only score is too low for production."


def _total_evidence_count(row: dict[str, Any]) -> int:
    keys = (
        "instance_generated",
        "instance_rejected",
        "solver_validated",
        "solver_rejected",
        "instance_quality_validated",
        "instance_quality_review",
        "instance_quality_rejected",
        "backtranslation_candidates",
        "backtranslation_rejected",
        "nl_validated",
        "nl_rejected",
        "forward_outputs",
        "forward_rejected",
        "forward_accepted",
        "forward_eval_rejected",
        "rendered",
        "render_rejected",
    )
    return sum(int(row.get(key) or 0) for key in keys)


def _quality_risk_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if row.get("solver_pass_rate") is not None and float(row["solver_pass_rate"]) < 0.80:
        flags.append("LOW_SOLVER_PASS_RATE")
    if row.get("quality_pass_rate") is not None and float(row["quality_pass_rate"]) < 0.80:
        flags.append("LOW_SEED_QUALITY_PASS_RATE")
    if row.get("quality_review_rate") is not None and float(row["quality_review_rate"]) > 0.20:
        flags.append("HIGH_SEED_REVIEW_RATE")
    if row.get("nl_pass_rate") is not None and float(row["nl_pass_rate"]) < 0.60:
        flags.append("LOW_NL_PASS_RATE")
    if row.get("forward_objective_match_rate") is not None and float(row["forward_objective_match_rate"]) < 0.50:
        flags.append("LOW_FORWARD_OBJECTIVE_MATCH_RATE")
    top_reasons = {str(reason): count for reason, count in row.get("top_rejection_reasons") or []}
    if top_reasons.get("OBJECTIVE_MISMATCH", 0) > 0:
        flags.append("HAS_OBJECTIVE_MISMATCH")
    if top_reasons.get("FORWARD_CODE_EXECUTION_FAILED", 0) > 0:
        flags.append("HAS_FORWARD_CODE_EXECUTION_FAILED")
    if top_reasons.get("STATIC_CODE_CHECK_FAILED", 0) > 0 or top_reasons.get("FORWARD_CODE_STATIC_CHECK_FAILED", 0) > 0:
        flags.append("HAS_STATIC_CODE_FAILURES")
    return flags


def _stage_scope(counts: Counter[str]) -> str:
    downstream_total = sum(
        counts[key]
        for key in (
            "backtranslation_candidates",
            "backtranslation_rejected",
            "nl_validated",
            "nl_rejected",
            "forward_outputs",
            "forward_rejected",
            "forward_accepted",
            "forward_eval_rejected",
            "rendered",
            "render_rejected",
        )
    )
    if downstream_total > 0:
        return "full_pipeline"
    if counts["instance_generated"] or counts["solver_validated"] or counts["instance_quality_validated"]:
        return "seed_only"
    return "no_data"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _score_rate(value: Any) -> float:
    return 0.0 if value is None else max(0.0, min(1.0, float(value)))


def _optional_score_rate(value: Any, *, fallback: float) -> float:
    return fallback if value is None else _score_rate(value)


def _fmt_rate(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _reason_key(value: Any) -> str:
    text = str(value or "UNKNOWN")
    for separator in (";", ":"):
        if separator in text:
            return text.split(separator)[0]
    return text


def _row_rejection_reasons(stage_name: str, row: dict[str, Any]) -> list[str]:
    if stage_name == "instance_quality_rejected":
        quality = row.get("instance_quality") or {}
        reasons = quality.get("rejection_reasons") or quality.get("flags") or []
        return [_reason_key(reason) for reason in reasons] or ["INSTANCE_QUALITY_REJECTED"]
    if stage_name == "instance_quality_review":
        quality = row.get("instance_quality") or {}
        reasons = quality.get("review_reasons") or quality.get("flags") or []
        return [f"REVIEW:{_reason_key(reason)}" for reason in reasons] or ["INSTANCE_QUALITY_REVIEW"]
    if "rejected" in stage_name:
        return [_reason_key(row.get("rejection_reason"))]
    return []
