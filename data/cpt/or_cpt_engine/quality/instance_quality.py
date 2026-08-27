from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from or_cpt_engine.generators.numeric import count_long_decimals
from or_cpt_engine.generation.generator_profiles import infer_task_family, load_generator_profiles
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "objective_abs_tolerance": 1e-4,
    "feasibility_tolerance": 1e-6,
    "integer_feasibility_tolerance": 1e-5,
    "min_binding_constraints": 1,
    "reject_all_vars_at_upper_bound": True,
    "reject_all_vars_at_lower_bound": True,
    "reject_all_non_fixed_vars_zero": True,
    "reject_source_duplicate": True,
    "reject_parameter_duplicate": True,
    "reject_solution_duplicate": True,
    "seed_decimal_max_places": 2,
    "max_seed_long_decimal_count": 0,
    "review_min_nonzero_variables": 2,
}


def validate_instance_quality(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    thresholds: dict[str, Any] | None = None,
    profiles_path: str | Path | None = None,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "quality_validated_instances.jsonl"
    review_path = output / "quality_review_instances.jsonl"
    rejected_path = output / "quality_rejected_instances.jsonl"
    for path in (accepted_path, review_path, rejected_path):
        path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    resolved_thresholds = dict(DEFAULT_THRESHOLDS)
    resolved_thresholds.update(thresholds or {})
    profile_payload = load_generator_profiles(profiles_path)
    profile_rows = profile_payload.get("profiles") if isinstance(profile_payload.get("profiles"), dict) else {}
    seen_source: set[str] = set()
    seen_parameter: set[str] = set()
    seen_solution: set[str] = set()
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    by_generator: dict[str, Counter[str]] = defaultdict(Counter)

    for row in iter_jsonl(input_path) or []:
        profile = _profile_for_row(row, profile_rows)
        quality = evaluate_instance_quality(
            row,
            thresholds=resolved_thresholds,
            profile=profile,
            seen_source=seen_source,
            seen_parameter=seen_parameter,
            seen_solution=seen_solution,
        )
        enriched = dict(row)
        enriched["instance_quality"] = quality
        status = quality["status"]
        counts[status] += 1
        by_generator[str(row.get("generator_id") or "unknown")][status] += 1
        for reason in (quality.get("rejection_reasons") or []) + (quality.get("review_reasons") or []):
            reason_counts[str(reason)] += 1

        if status == "PASS":
            append_jsonl(accepted_path, enriched)
        elif status == "REVIEW":
            append_jsonl(review_path, enriched)
        else:
            append_jsonl(rejected_path, enriched)

    _write_dashboard_csv(output / "instance_quality_dashboard.csv", by_generator)
    write_text(
        output / "instance_quality_report.md",
        render_instance_quality_report(counts, reason_counts, by_generator, resolved_thresholds),
    )
    return {
        "accepted": counts["PASS"],
        "review": counts["REVIEW"],
        "rejected": counts["REJECT"],
        "total": sum(counts.values()),
    }


def evaluate_instance_quality(
    row: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    profile: dict[str, Any] | None = None,
    seen_source: set[str] | None = None,
    seen_parameter: set[str] | None = None,
    seen_solution: set[str] | None = None,
) -> dict[str, Any]:
    solver = row.get("solver_validation") or {}
    solution = solver.get("solution") or {}
    summary = solution.get("solution_summary") or {}
    feasibility = solution.get("feasibility") or {}
    hard_reasons: list[str] = []
    review_reasons: list[str] = []
    flags: list[str] = []

    status = solver.get("status") or (row.get("reference_answer") or {}).get("status")
    objective = solver.get("objective_value")
    if objective is None:
        objective = (row.get("reference_answer") or {}).get("objective_value")
    if status not in {"OPTIMAL", "SOURCE_REFERENCE"}:
        hard_reasons.append("SOLVER_NOT_OPTIMAL")
    if objective is None:
        hard_reasons.append("OBJECTIVE_MISSING")

    decimal_max_places = int(thresholds.get("seed_decimal_max_places", 2))
    max_seed_long_decimals = int(thresholds.get("max_seed_long_decimal_count", 0))
    seed_long_decimal_count = _seed_long_decimal_count(row, decimal_max_places)
    if seed_long_decimal_count > max_seed_long_decimals:
        hard_reasons.append("EXCESSIVE_SEED_DECIMAL_PRECISION")

    constraint_violation = _as_float(feasibility.get("constraint_violation"))
    bound_violation = _as_float(feasibility.get("bound_violation"))
    integer_violation = _as_float(feasibility.get("integer_violation"))
    feasibility_tolerance = float(thresholds["feasibility_tolerance"])
    integer_feasibility_tolerance = float(thresholds.get("integer_feasibility_tolerance", 1e-5))
    if (
        (constraint_violation is not None and constraint_violation > feasibility_tolerance)
        or (bound_violation is not None and bound_violation > feasibility_tolerance)
        or (integer_violation is not None and integer_violation > integer_feasibility_tolerance)
    ):
        hard_reasons.append("FEASIBILITY_VIOLATION")

    recomputed = _as_float(solution.get("objective_recomputed"))
    objective_value = _as_float(solution.get("objective_value") if solution.get("objective_value") is not None else objective)
    if recomputed is not None and objective_value is not None:
        if abs(recomputed - objective_value) > float(thresholds["objective_abs_tolerance"]):
            hard_reasons.append("OBJECTIVE_RECOMPUTE_MISMATCH")

    if thresholds.get("reject_all_vars_at_upper_bound", True) and summary.get("all_vars_at_ub") is True:
        hard_reasons.append("ALL_NON_FIXED_VARIABLES_AT_UPPER_BOUND")
    if thresholds.get("reject_all_vars_at_lower_bound", True) and summary.get("all_vars_at_lb") is True:
        hard_reasons.append("ALL_NON_FIXED_VARIABLES_AT_LOWER_BOUND")
    if thresholds.get("reject_all_non_fixed_vars_zero", True) and summary.get("all_non_fixed_vars_zero") is True:
        hard_reasons.append("ALL_NON_FIXED_VARIABLES_ZERO")

    binding_count = _as_int(summary.get("binding_constraint_count"))
    constraint_count = _as_int(summary.get("constraint_count"))
    if constraint_count and binding_count is not None and binding_count < int(thresholds["min_binding_constraints"]):
        hard_reasons.append("NO_BINDING_CORE_CONSTRAINT")

    nonzero_count = _as_int(solution.get("nonzero_variable_count"))
    if nonzero_count is not None and nonzero_count < int(thresholds["review_min_nonzero_variables"]):
        review_reasons.append("LOW_NONZERO_VARIABLE_COUNT")

    controller_reasons, controller_reviews, controller_metrics = _generator_specific_findings(
        row,
        solution,
        profile=profile,
        thresholds=thresholds,
    )
    hard_reasons.extend(controller_reasons)
    review_reasons.extend(controller_reviews)

    source_signature = _source_signature(row)
    parameter_signature = _parameter_signature(row)
    solution_signature = _solution_signature(row, solution)
    if seen_source is not None and thresholds.get("reject_source_duplicate", True):
        if source_signature in seen_source:
            hard_reasons.append("DUPLICATE_SOURCE_SIGNATURE")
        seen_source.add(source_signature)
    if seen_parameter is not None and thresholds.get("reject_parameter_duplicate", True):
        if parameter_signature in seen_parameter:
            hard_reasons.append("DUPLICATE_PARAMETER_SIGNATURE")
        seen_parameter.add(parameter_signature)
    if seen_solution is not None and thresholds.get("reject_solution_duplicate", True):
        if solution_signature in seen_solution:
            hard_reasons.append("DUPLICATE_SOLUTION_SIGNATURE")
        seen_solution.add(solution_signature)

    if hard_reasons:
        quality_status = "REJECT"
    elif review_reasons:
        quality_status = "REVIEW"
    else:
        quality_status = "PASS"

    flags.extend(hard_reasons)
    flags.extend(review_reasons)
    score = max(0.0, 1.0 - 0.25 * len(set(hard_reasons)) - 0.08 * len(set(review_reasons)))
    return {
        "status": quality_status,
        "score": round(score, 6),
        "flags": sorted(set(flags)),
        "rejection_reasons": sorted(set(hard_reasons)),
        "review_reasons": sorted(set(review_reasons)),
        "solver_status": status,
        "objective_recomputed_pass": "OBJECTIVE_RECOMPUTE_MISMATCH" not in hard_reasons,
        "binding_constraint_count": binding_count,
        "constraint_count": constraint_count,
        "all_vars_at_upper_bound": bool(summary.get("all_vars_at_ub")),
        "all_vars_at_lower_bound": bool(summary.get("all_vars_at_lb")),
        "all_non_fixed_vars_zero": bool(summary.get("all_non_fixed_vars_zero")),
        "nonzero_variable_count": nonzero_count,
        "seed_long_decimal_count": seed_long_decimal_count,
        "seed_decimal_max_places": decimal_max_places,
        "source_signature": source_signature,
        "parameter_signature": parameter_signature,
        "solution_signature": solution_signature,
        "generator_quality_metrics": controller_metrics,
        "recommended_data_class": _recommended_data_class(quality_status),
    }


def render_instance_quality_report(
    counts: Counter[str],
    reason_counts: Counter[str],
    by_generator: dict[str, Counter[str]],
    thresholds: dict[str, Any],
) -> str:
    total = sum(counts.values())
    lines = [
        "# Instance Quality Report",
        "",
        f"- Total input: {total}",
        f"- Quality pass: {counts['PASS']}",
        f"- Review: {counts['REVIEW']}",
        f"- Rejected: {counts['REJECT']}",
        "",
        "## Thresholds",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(thresholds.items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Top Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in reason_counts.most_common(20):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Per Generator", "", "| Generator | Pass | Review | Reject |", "| --- | ---: | ---: | ---: |"])
    for generator_id, counter in sorted(by_generator.items()):
        lines.append(f"| {generator_id} | {counter['PASS']} | {counter['REVIEW']} | {counter['REJECT']} |")
    return "\n".join(lines) + "\n"


def _write_dashboard_csv(path: Path, by_generator: dict[str, Counter[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generator_id", "pass", "review", "reject", "total", "pass_rate"])
        writer.writeheader()
        for generator_id, counter in sorted(by_generator.items()):
            total = counter["PASS"] + counter["REVIEW"] + counter["REJECT"]
            writer.writerow(
                {
                    "generator_id": generator_id,
                    "pass": counter["PASS"],
                    "review": counter["REVIEW"],
                    "reject": counter["REJECT"],
                    "total": total,
                    "pass_rate": (counter["PASS"] / total) if total else 0.0,
                }
            )


def _seed_long_decimal_count(row: dict[str, Any], max_decimal_places: int) -> int:
    chunks: list[str] = []
    source_compact_data = row.get("source_compact_data") or {}
    if source_compact_data:
        chunks.append(json.dumps(source_compact_data, ensure_ascii=False, sort_keys=True))
        return sum(count_long_decimals(chunk, max_decimal_places=max_decimal_places) for chunk in chunks)

    generation_params = row.get("generation_params") or {}
    if generation_params:
        compact_params = {
            str(key): value
            for key, value in generation_params.items()
            if str(key).startswith("compact_")
        }
        chunks.append(json.dumps(compact_params or generation_params, ensure_ascii=False, sort_keys=True))
    return sum(count_long_decimals(chunk, max_decimal_places=max_decimal_places) for chunk in chunks)


def _generator_specific_findings(
    row: dict[str, Any],
    solution: dict[str, Any],
    *,
    profile: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    generator_id = str(row.get("generator_id") or "")
    family = _task_family_for_row(row, profile)
    controller = _quality_controller_for(generator_id, family, profile)
    variables = solution.get("variable_diagnostics") or []
    constraints = solution.get("constraint_diagnostics") or []
    business_variables = _match_named_items(variables, controller.get("business_variable_patterns") or [])
    slack_variables = _match_named_items(variables, controller.get("slack_variable_patterns") or [])
    activation_variables = _match_named_items(variables, controller.get("activation_variable_patterns") or [])
    core_constraints = _match_named_items(constraints, controller.get("core_constraint_patterns") or [])
    hard_rules = set(controller.get("hard_reject_rules") or [])
    review_rules = set(controller.get("review_rules") or [])
    family_thresholds = dict(controller.get("family_specific_thresholds") or {})

    hard_reasons: list[str] = []
    review_reasons: list[str] = []
    business_sum = _sum_abs_values(business_variables)
    slack_sum = _sum_abs_values(slack_variables)
    activation_sum = _sum_abs_values(activation_variables)
    business_nonzero_count = _nonzero_count(business_variables)
    binary_business_variables = [
        variable for variable in business_variables
        if str(variable.get("vtype") or "").upper() in {"B", "I"} or _is_binary_like(variable)
    ]
    binary_active_count = _nonzero_count(binary_business_variables)
    core_binding_count = sum(1 for constraint in core_constraints if constraint.get("is_binding") is True)
    core_constraint_count = len(core_constraints)
    tolerance = float(family_thresholds.get("tolerance", thresholds.get("feasibility_tolerance", 1e-6)))

    if "reject_business_all_upper_bound" in hard_rules and business_variables and _all_nonfixed_at_bound(business_variables, "at_ub"):
        hard_reasons.append("BUSINESS_VARIABLES_ALL_AT_UPPER_BOUND")
    if "reject_business_all_lower_bound" in hard_rules and business_variables and _all_nonfixed_at_bound(business_variables, "at_lb"):
        hard_reasons.append("BUSINESS_VARIABLES_ALL_AT_LOWER_BOUND")
    if "reject_business_all_zero" in hard_rules and business_variables and business_sum <= tolerance:
        hard_reasons.append("BUSINESS_VARIABLES_ALL_ZERO")
    if "reject_no_business_nonzero" in hard_rules and business_variables and business_nonzero_count <= 0:
        hard_reasons.append("NO_BUSINESS_VARIABLE_NONZERO")
    if "reject_no_core_binding_constraint" in hard_rules and core_constraints:
        min_core_binding = int(family_thresholds.get("min_core_binding_constraints", 1))
        if core_binding_count < min_core_binding:
            hard_reasons.append("NO_CORE_CONSTRAINT_BINDING")
    if "reject_all_activation_zero" in hard_rules and activation_variables and activation_sum <= tolerance:
        hard_reasons.append("ACTIVATION_VARIABLES_ALL_ZERO")
    if "reject_all_activation_one" in hard_rules and activation_variables and _all_nonfixed_at_bound(activation_variables, "at_ub"):
        hard_reasons.append("ACTIVATION_VARIABLES_ALL_ONE")
    if "reject_slack_dominates_business" in hard_rules and slack_variables:
        total_service = business_sum + slack_sum
        slack_ratio = (slack_sum / total_service) if total_service > tolerance else 0.0
        max_slack_ratio = float(family_thresholds.get("max_slack_solution_ratio", 0.40))
        if slack_sum > tolerance and (business_sum <= tolerance or slack_ratio > max_slack_ratio):
            hard_reasons.append("SLACK_OR_RESERVE_DOMINATES_SOLUTION")
    if "reject_single_flow_variable_dominates" in hard_rules and len(business_variables) > 1 and business_sum > tolerance:
        max_share = max(abs(_as_float(variable.get("value")) or 0.0) for variable in business_variables) / business_sum
        if max_share >= float(family_thresholds.get("max_single_business_variable_share", 0.95)):
            hard_reasons.append("SINGLE_BUSINESS_VARIABLE_DOMINATES")
    if "reject_empty_selection" in hard_rules and binary_business_variables and binary_active_count <= 0:
        hard_reasons.append("EMPTY_SELECTION")
    if "reject_full_selection" in hard_rules and binary_business_variables and binary_active_count >= len(binary_business_variables):
        hard_reasons.append("FULL_SELECTION")
    if "reject_single_selected_item" in hard_rules and len(binary_business_variables) > 1 and 0 < binary_active_count <= 1:
        hard_reasons.append("SINGLE_SELECTED_ITEM")

    if "review_low_core_binding_ratio" in review_rules and core_constraints:
        min_core_binding_ratio = float(family_thresholds.get("min_core_binding_ratio", 0.35))
        core_binding_ratio = core_binding_count / core_constraint_count if core_constraint_count else 0.0
        if core_binding_ratio < min_core_binding_ratio:
            review_reasons.append("LOW_CORE_BINDING_RATIO")
    if "review_low_business_nonzero_count" in review_rules and business_variables:
        min_business_nonzero = int(family_thresholds.get("min_business_nonzero_count", 2))
        if business_nonzero_count < min_business_nonzero:
            review_reasons.append("LOW_BUSINESS_NONZERO_COUNT")

    metrics = {
        "task_family": family,
        "business_variable_count": len(business_variables),
        "business_nonzero_count": business_nonzero_count,
        "slack_variable_count": len(slack_variables),
        "slack_solution_abs_sum": round(slack_sum, 8),
        "activation_variable_count": len(activation_variables),
        "activation_solution_abs_sum": round(activation_sum, 8),
        "core_constraint_count": core_constraint_count,
        "core_binding_constraint_count": core_binding_count,
    }
    return sorted(set(hard_reasons)), sorted(set(review_reasons)), metrics


def _quality_controller_for(generator_id: str, family: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    base = _default_quality_controller(generator_id, family)
    profile_controller = (profile or {}).get("quality_controller")
    if isinstance(profile_controller, dict):
        return _merge_controller(base, profile_controller)
    return base


def _default_quality_controller(generator_id: str, family: str) -> dict[str, Any]:
    gid = generator_id.lower()
    common_review = ["review_low_business_nonzero_count"]
    if "staticlineplanning" in gid:
        return {
            "business_variable_patterns": [r"^x\["],
            "slack_variable_patterns": [r"^s\[", r"reserve", r"slack", r"unmet", r"short"],
            "activation_variable_patterns": [r"^x\["],
            "core_constraint_patterns": [r"demand", r"capacity", r"fleet", r"line"],
            "hard_reject_rules": [
                "reject_all_activation_zero",
                "reject_slack_dominates_business",
                "reject_no_business_nonzero",
            ],
            "review_rules": common_review,
            "family_specific_thresholds": {"max_slack_solution_ratio": 0.25, "min_business_nonzero_count": 1},
        }
    if family == "production_planning":
        return {
            "business_variable_patterns": [r"prod", r"production", r"make", r"output", r"^x\["],
            "slack_variable_patterns": [r"slack", r"short", r"unmet", r"reserve"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"capacity", r"resource", r"time", r"labor", r"machine", r"material"],
            "hard_reject_rules": [
                "reject_business_all_upper_bound",
                "reject_business_all_lower_bound",
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.08,
                "min_business_nonzero_count": 2,
            },
        }
    if family in {"facility_location", "network_design"}:
        return {
            "business_variable_patterns": [r"assign", r"serve", r"ship", r"flow", r"cover", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unmet", r"short", r"reserve"],
            "activation_variable_patterns": [r"open", r"activate", r"build", r"facility", r"site", r"selected", r"location", r"^y\["],
            "core_constraint_patterns": [r"capacity", r"demand", r"balance", r"budget", r"limit", r"cover"],
            "hard_reject_rules": [
                "reject_all_activation_zero",
                "reject_all_activation_one",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.05,
            },
        }
    if family in {"transportation", "network_flow", "supply_chain"}:
        return {
            "business_variable_patterns": [r"flow", r"ship", r"shipment", r"arc", r"lane", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unmet", r"short", r"backlog"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"supply", r"demand", r"balance", r"conservation", r"capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_single_flow_variable_dominates",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "max_single_business_variable_share": 0.95,
                "max_slack_solution_ratio": 0.15,
                "min_core_binding_ratio": 0.05,
            },
        }
    if family in {"packing", "covering"}:
        return {
            "business_variable_patterns": [r"item", r"select", r"set", r"cover", r"^x\["],
            "slack_variable_patterns": [r"slack", r"uncovered"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"capacity", r"cover", r"demand", r"weight", r"assignment"],
            "hard_reject_rules": [
                "reject_empty_selection",
                "reject_full_selection",
                "reject_single_selected_item",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "min_business_nonzero_count": 2,
                "min_core_binding_ratio": 0.20,
            },
        }
    if family == "lot_sizing":
        return {
            "business_variable_patterns": [
                r"amount",
                r"prod",
                r"production",
                r"order",
                r"quantity",
                r"stock",
                r"inventory",
                r"backlog",
                r"^x\[",
            ],
            "slack_variable_patterns": [r"unmet", r"short", r"slack"],
            "activation_variable_patterns": [r"setup", r"startup", r"production", r"^y\["],
            "core_constraint_patterns": [r"inventory", r"balance", r"demand", r"capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {"max_slack_solution_ratio": 0.25},
        }
    if family == "assignment":
        return {
            "business_variable_patterns": [r"assign", r"allocation", r"team", r"selected", r"^x\["],
            "slack_variable_patterns": [r"shortage", r"slack", r"unmet", r"gap"],
            "activation_variable_patterns": [r"assign", r"allocation", r"^x\["],
            "core_constraint_patterns": [
                r"assignment",
                r"staffing",
                r"capacity",
                r"skill",
                r"balance",
                r"exclusive",
                r"availability",
                r"demand",
                r"allocation",
            ],
            "hard_reject_rules": [
                "reject_no_business_nonzero",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.35,
                "min_business_nonzero_count": 2,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "scheduling":
        return {
            "business_variable_patterns": [r"^s_", r"start", r"completion", r"c_max", r"^x_", r"assign", r"landing"],
            "slack_variable_patterns": [r"slack", r"tard", r"lateness", r"short"],
            "activation_variable_patterns": [r"^x_", r"assign", r"sequence", r"order"],
            "core_constraint_patterns": [
                r"prec",
                r"machine",
                r"makespan",
                r"capacity",
                r"sequence",
                r"landing",
                r"runway",
                r"time",
                r"order",
                r"separation",
                r"earliest",
                r"latest",
                r"deviation",
            ],
            "hard_reject_rules": [
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.30,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "routing":
        return {
            "business_variable_patterns": [r"route", r"arc", r"travel", r"numplanes", r"vehicle", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unserved", r"unmet", r"late"],
            "activation_variable_patterns": [r"route", r"arc", r"numplanes", r"vehicle", r"^x\["],
            "core_constraint_patterns": [r"flow", r"balance", r"demand", r"visit", r"degree", r"capacity", r"route", r"time"],
            "hard_reject_rules": [
                "reject_no_business_nonzero",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "diet":
        return {
            "business_variable_patterns": [r"buy", r"servings", r"foods", r"food", r"^x\["],
            "slack_variable_patterns": [r"slack", r"excess", r"deficit"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"nutrient", r"requirement", r"volume", r"min", r"max"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": [*common_review],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "energy":
        return {
            "business_variable_patterns": [r"numgenerators", r"poweroutput", r"generation", r"output", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unmet", r"short", r"reserve_gap"],
            "activation_variable_patterns": [r"numgenerators", r"startup", r"^x\["],
            "core_constraint_patterns": [
                r"available",
                r"demand",
                r"mingeneration",
                r"maxgeneration",
                r"reserve",
                r"startup",
                r"capacity",
            ],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
                "reject_slack_dominates_business",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_ratio": 0.03,
                "min_business_nonzero_count": 2,
            },
        }
    if family == "workforce_deployment":
        return {
            "business_variable_patterns": [r"soldiers", r"deploy", r"assignment", r"^x\["],
            "slack_variable_patterns": [r"shortage", r"slack", r"unmet", r"gap"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"totalsoldiers", r"skillrequirement", r"skillavailability", r"staffing", r"capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": [*common_review],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "cutting_stock":
        return {
            "business_variable_patterns": [r"cut", r"pattern"],
            "slack_variable_patterns": [r"slack", r"waste"],
            "activation_variable_patterns": [r"cut", r"pattern"],
            "core_constraint_patterns": [r"fill", r"order", r"demand", r"width"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": [*common_review],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "blending":
        return {
            "business_variable_patterns": [r"alloy", r"blend", r"ingredient", r"^x\["],
            "slack_variable_patterns": [r"slack", r"deviation"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"element", r"total", r"blend", r"composition"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": [*common_review],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "contract_allocation":
        return {
            "business_variable_patterns": [r"generation", r"delivery", r"contract", r"allocation", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unmet", r"short"],
            "activation_variable_patterns": [r"incidence", r"selected", r"^y\["],
            "core_constraint_patterns": [r"capacity", r"contract", r"fulfillment", r"contributor", r"delivery", r"activation"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", *common_review],
            "family_specific_thresholds": {"min_core_binding_ratio": 0.03, "min_business_nonzero_count": 2},
        }
    if family == "marketing":
        return {
            "business_variable_patterns": [r"supply", r"market", r"allocation", r"^x\["],
            "slack_variable_patterns": [r"slack", r"unmet"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": [r"demand", r"market", r"capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": [*common_review],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    return {
        "business_variable_patterns": [],
        "slack_variable_patterns": [r"slack", r"unmet", r"short", r"reserve"],
        "activation_variable_patterns": [],
        "core_constraint_patterns": [],
        "hard_reject_rules": [],
        "review_rules": [],
        "family_specific_thresholds": {},
    }


def _merge_controller(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, list):
            merged[key] = _unique_list([*(merged.get(key) or []), *value])
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _unique_list(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _profile_for_row(row: dict[str, Any], profile_rows: Any) -> dict[str, Any] | None:
    if not isinstance(profile_rows, dict):
        return None
    profile = profile_rows.get(str(row.get("generator_id") or ""))
    return profile if isinstance(profile, dict) else None


def _task_family_for_row(row: dict[str, Any], profile: dict[str, Any] | None) -> str:
    return str((profile or {}).get("task_family") or row.get("task_family") or infer_task_family(str(row.get("generator_id") or "")))


def _match_named_items(items: list[dict[str, Any]], patterns: list[str]) -> list[dict[str, Any]]:
    if not patterns:
        return []
    matched: list[dict[str, Any]] = []
    for item in items:
        name = str(item.get("name") or "")
        if any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns):
            matched.append(item)
    return matched


def _sum_abs_values(variables: list[dict[str, Any]]) -> float:
    return sum(abs(_as_float(variable.get("value")) or 0.0) for variable in variables)


def _nonzero_count(variables: list[dict[str, Any]]) -> int:
    return sum(1 for variable in variables if abs(_as_float(variable.get("value")) or 0.0) > 1e-6)


def _all_nonfixed_at_bound(variables: list[dict[str, Any]], bound_key: str) -> bool:
    nonfixed = [variable for variable in variables if variable.get("is_fixed") is not True]
    return bool(nonfixed) and all(variable.get(bound_key) is True for variable in nonfixed)


def _is_binary_like(variable: dict[str, Any]) -> bool:
    lb = _as_float(variable.get("lb"))
    ub = _as_float(variable.get("ub"))
    return lb is not None and ub is not None and abs(lb) <= 1e-9 and abs(ub - 1.0) <= 1e-9


def _recommended_data_class(status: str) -> str:
    if status == "PASS":
        return "A_final_model_seed"
    if status == "REVIEW":
        return "B_review_seed"
    return "C_debug_or_drop_seed"


def _source_signature(row: dict[str, Any]) -> str:
    return str(row.get("instance_id") or "")


def _parameter_signature(row: dict[str, Any]) -> str:
    payload = {
        "generator_id": row.get("generator_id"),
        "model_type": row.get("model_type"),
        "optimization_sense": row.get("optimization_sense"),
        "lp_text": _normalize_text(row.get("lp_text")),
        "math_formula": _normalize_text(row.get("math_formula")),
        "canonical_math_signature": row.get("canonical_math_signature") or {},
    }
    return _hash_payload(payload)


def _solution_signature(row: dict[str, Any], solution: dict[str, Any]) -> str:
    variable_values = solution.get("nonzero_variable_values") or {}
    binding_names = [
        str(item.get("name") or "")
        for item in solution.get("constraint_diagnostics") or []
        if item.get("is_binding") is True
    ]
    rounded_nonzero_values = {
        str(name): _round_float(value)
        for name, value in variable_values.items()
    }
    payload = {
        "generator_id": row.get("generator_id"),
        "status": (row.get("solver_validation") or {}).get("status"),
        "objective_value": _round_float((row.get("solver_validation") or {}).get("objective_value")),
        "variable_count": solution.get("variable_count"),
        "nonzero_values": sorted(rounded_nonzero_values.items()),
        "binding_names": sorted(binding_names),
    }
    return _hash_payload(payload)


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _round_float(value: Any) -> float | None:
    numeric = _as_float(value)
    return None if numeric is None else round(numeric, 8)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
