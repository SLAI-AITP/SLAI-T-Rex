from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cpt_cleaner.utils import json_compat

from or_cpt_engine.generation.generator_profiles import load_generator_profiles
from or_cpt_engine.utils.io import read_jsonl, write_text


def plan_production(
    *,
    target_accepted: int,
    profiles_path: str | Path,
    output_dir: str | Path,
    quality_report_path: str | Path | None = None,
    min_expected_acceptance_rate: float = 0.05,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile_payload = load_generator_profiles(profiles_path)
    profiles = profile_payload.get("profiles") or {}
    quality_by_id = _load_quality(quality_report_path)

    enabled_profiles = {
        generator_id: profile
        for generator_id, profile in profiles.items()
        if isinstance(profile, dict) and profile.get("enabled", True)
    }
    weighted: list[dict[str, Any]] = []
    for generator_id, profile in sorted(enabled_profiles.items()):
        quality = quality_by_id.get(generator_id, {})
        sampling_weight = profile.get("sampling_weight") if isinstance(profile.get("sampling_weight"), dict) else {}
        base_weight = float(sampling_weight.get("cpt_weight") or profile.get("recommended_weight") or 1.0)
        production_tier = _production_tier(profile, quality)
        quality_multiplier = _quality_multiplier(quality)
        tier_multiplier = _tier_multiplier(production_tier)
        expected_rate = max(
            min_expected_acceptance_rate,
            float(
                quality.get("forward_objective_match_rate")
                or quality.get("generated_to_quality_pass_rate")
                or quality.get("quality_pass_rate")
                or quality.get("generator_score")
                or profile.get("expected_acceptance_rate")
                or 0.30
            ),
        )
        weighted.append(
            {
                "generator_id": generator_id,
                "weight": base_weight * quality_multiplier * tier_multiplier,
                "base_weight": base_weight,
                "quality_multiplier": quality_multiplier,
                "tier_multiplier": tier_multiplier,
                "production_tier": production_tier,
                "tier_reason": quality.get("tier_reason") or _default_tier_reason(production_tier),
                "quality_risk_flags": quality.get("quality_risk_flags") or [],
                "generator_score": quality.get("generator_score"),
                "seed_quality_score": quality.get("seed_quality_score"),
                "downstream_score": quality.get("downstream_score"),
                "expected_acceptance_rate": expected_rate,
                "priority": profile.get("priority"),
                "task_family": profile.get("task_family"),
                "sub_family": profile.get("sub_family"),
                "min_quality_score": sampling_weight.get("min_quality_score"),
                "max_daily_samples": sampling_weight.get("max_daily_samples"),
                "recommended_action": quality.get("recommended_action"),
            }
        )

    active_weighted = [row for row in weighted if row["weight"] > 0]
    total_weight = sum(row["weight"] for row in active_weighted)
    if total_weight <= 0:
        raise ValueError("no enabled generator profile has positive production weight")

    plan_rows: list[dict[str, Any]] = []
    accepted_allocated = 0
    for row in weighted:
        if row["weight"] <= 0:
            plan_rows.append({**row, "target_accepted": 0, "planned_attempts": 0, "target_accepted_cap": 0})
            continue
        target_for_generator = math.floor(target_accepted * row["weight"] / total_weight)
        accepted_allocated += target_for_generator
        plan_rows.append(
            {
                **row,
                "target_accepted": target_for_generator,
                "target_accepted_cap": _target_accepted_cap(row, target_accepted, len(active_weighted)),
                "planned_attempts": math.ceil(target_for_generator / row["expected_acceptance_rate"]) if target_for_generator else 0,
            }
        )

    remainder = max(0, target_accepted - accepted_allocated)
    for row in sorted([item for item in plan_rows if item["weight"] > 0], key=lambda item: (-item["weight"], item["generator_id"]))[:remainder]:
        row["target_accepted"] += 1
        row["planned_attempts"] = math.ceil(row["target_accepted"] / row["expected_acceptance_rate"])

    _apply_target_caps(plan_rows, target_accepted)

    payload = {
        "target_accepted": target_accepted,
        "profile_version": profile_payload.get("version"),
        "quality_report_path": str(quality_report_path) if quality_report_path else None,
        "tier_counts": dict(sorted(_count_tiers(plan_rows).items())),
        "total_planned_attempts": sum(row["planned_attempts"] for row in plan_rows),
        "generators": plan_rows,
    }
    (output / "production_plan.json").write_bytes(json_compat.dumps(payload))
    write_text(output / "production_plan.md", render_production_plan(payload))
    return {
        "target_accepted": target_accepted,
        "generators": len(plan_rows),
        "planned_attempts": payload["total_planned_attempts"],
        "output": str(output),
    }


def render_production_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# OR-CPT Production Plan",
        "",
        f"- Target accepted documents: {payload['target_accepted']}",
        f"- Total planned attempts: {payload['total_planned_attempts']}",
        f"- Profile version: {payload.get('profile_version')}",
        f"- Tier counts: {payload.get('tier_counts')}",
        "",
        "| generator_id | tier | priority | family | sub_family | target accepted | planned attempts | expected pass | weight | action | risk flags |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(payload["generators"], key=lambda item: (-item["target_accepted"], item["generator_id"])):
        lines.append(
            f"| {row['generator_id']} | {row.get('production_tier') or ''} | {row.get('priority') or ''} | "
            f"{row.get('task_family') or ''} | {row.get('sub_family') or ''} | "
            f"{row['target_accepted']} | {row['planned_attempts']} | "
            f"{100 * float(row['expected_acceptance_rate']):.1f}% | {float(row.get('weight') or 0):.3f} | "
            f"{row.get('recommended_action') or ''} | {', '.join(row.get('quality_risk_flags') or [])} |"
        )
    return "\n".join(lines) + "\n"


def _load_quality(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {str(row.get("generator_id")): row for row in rows if row.get("generator_id")}


def _quality_multiplier(quality: dict[str, Any]) -> float:
    if not quality:
        return 1.0
    action = quality.get("recommended_action")
    if action == "increase_weight":
        return 1.25
    if action in {"optimize_generator_source", "optimize_prompt_or_semantics"}:
        return 0.50
    if action == "limit_until_reviewed":
        return 0.35
    score = quality.get("generator_score")
    if score is None:
        return 1.0
    return max(0.25, min(1.35, 0.5 + float(score)))


def _production_tier(profile: dict[str, Any], quality: dict[str, Any]) -> str:
    value = quality.get("production_tier") or profile.get("production_tier")
    if value in {"A_core", "B_improve", "C_limited", "D_hold"}:
        return str(value)
    if not quality:
        return "B_improve"
    action = quality.get("recommended_action")
    if action == "increase_weight":
        return "A_core"
    if action in {"keep", "needs_downstream_eval"}:
        return "B_improve"
    if action in {"limit_until_reviewed", "optimize_prompt_or_semantics"}:
        return "C_limited"
    if action in {"optimize_generator_source", "optimize_generator_source_or_quality_controller", "needs_data"}:
        return "D_hold"
    return "B_improve"


def _tier_multiplier(tier: str) -> float:
    return {
        "A_core": 1.20,
        "B_improve": 0.75,
        "C_limited": 0.30,
        "D_hold": 0.0,
    }.get(tier, 0.75)


def _target_accepted_cap(row: dict[str, Any], target_accepted: int, active_count: int) -> int | None:
    if active_count < 5:
        return None
    share = {
        "A_core": 0.20,
        "B_improve": 0.12,
        "C_limited": 0.05,
        "D_hold": 0.0,
    }.get(str(row.get("production_tier")), 0.12)
    cap = math.ceil(target_accepted * share)
    max_daily = row.get("max_daily_samples")
    if max_daily is not None:
        try:
            cap = min(cap, math.floor(float(max_daily) * float(row.get("expected_acceptance_rate") or 0)))
        except (TypeError, ValueError):
            pass
    return max(0, cap)


def _apply_target_caps(plan_rows: list[dict[str, Any]], target_accepted: int) -> None:
    capped_rows = [row for row in plan_rows if row.get("target_accepted_cap") is not None]
    if not capped_rows:
        return
    for row in capped_rows:
        cap = int(row.get("target_accepted_cap") or 0)
        if row["target_accepted"] > cap:
            row["target_accepted"] = cap
    remaining = target_accepted - sum(int(row.get("target_accepted") or 0) for row in plan_rows)
    while remaining > 0:
        candidates = [
            row
            for row in capped_rows
            if row.get("weight", 0) > 0 and int(row.get("target_accepted") or 0) < int(row.get("target_accepted_cap") or 0)
        ]
        if not candidates:
            break
        for row in sorted(candidates, key=lambda item: (-item["weight"], item["generator_id"])):
            if remaining <= 0:
                break
            row["target_accepted"] += 1
            remaining -= 1
    for row in plan_rows:
        row["planned_attempts"] = (
            math.ceil(row["target_accepted"] / row["expected_acceptance_rate"])
            if row.get("target_accepted")
            else 0
        )


def _count_tiers(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("production_tier") or "UNKNOWN")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _default_tier_reason(tier: str) -> str:
    return {
        "A_core": "High-confidence generator; suitable for larger production share.",
        "B_improve": "Usable generator with controlled production share.",
        "C_limited": "Limited production for diversity while issues are improved.",
        "D_hold": "Held out from formal production until quality improves.",
    }.get(tier, "Tier inferred from profile and quality report.")
