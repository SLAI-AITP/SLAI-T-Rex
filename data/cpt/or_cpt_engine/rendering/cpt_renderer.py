from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from or_cpt_engine.generators.numeric import count_long_decimals
from or_cpt_engine.quality.semantic_coherence import semantic_coherence_issues
from or_cpt_engine.schemas.common import CPTDocument, EngineConfig
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text


FORBIDDEN_RENDER_PHRASES = (
    "user:",
    "assistant:",
    "chatml",
    "<|im_start|>",
    "<|im_end|>",
    "let me think",
    "downstream cpt training data",
    "downstream training",
    "cpt corpus",
    "sample is exported",
    "rejected upstream",
    "previous model",
    "previous formulation",
    "previous code",
    "failed because",
    "validation showed",
    "reference objective",
    "expected objective",
    "objective mismatch",
)
REPAIR_AUDIT_MARKERS = (
    "previous model",
    "previous formulation",
    "previous code",
    "previous attempt",
    "failed because",
    "validation showed",
    "validation failure",
    "reference objective",
    "expected objective",
    "objective mismatch",
    "the repair",
    "repair adds",
    "repair changes",
    "repaired model",
)
_MAX_RENDERED_SOLUTION_ROWS = 80


def render_cpt_documents(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    doc_path = output / "cpt_documents.jsonl"
    rejected_path = output / "cpt_rendering_rejected.jsonl"
    doc_count = 0
    rejected_count = 0
    type_counts: dict[str, int] = {}
    rejection_reasons: Counter[str] = Counter()
    views_by_instance: dict[str, int] = {}
    max_views = max(1, int(config.rendering.max_views_per_seed_instance or 1))
    rows = iter_jsonl(input_path)
    if rows is None:
        rows = []

    for index, row in enumerate(filter(None, rows)):
        if limit is not None and index >= limit:
            break
        accepted, record = render_cpt_document_row(
            row,
            config,
            index=index,
            views_by_instance=views_by_instance,
            max_views=max_views,
        )
        if not accepted:
            append_jsonl(rejected_path, record)
            rejected_count += 1
            if isinstance(record, dict):
                for reason in _split_rejection_reason(record.get("rejection_reason")):
                    rejection_reasons[reason] += 1
            continue
        append_jsonl(doc_path, record)
        doc_count += 1
        type_counts[record.doc_type] = type_counts.get(record.doc_type, 0) + 1

    write_text(output / "cpt_rendering_report.md", render_cpt_report(doc_count, rejected_count, type_counts, dict(rejection_reasons)))
    return {"documents": doc_count, "rejected": rejected_count}


def render_cpt_document_row(
    row: dict[str, Any],
    config: EngineConfig,
    *,
    index: int,
    views_by_instance: dict[str, int],
    max_views: int | None = None,
) -> tuple[bool, CPTDocument | dict[str, Any]]:
    resolved_max_views = max_views if max_views is not None else max(1, int(config.rendering.max_views_per_seed_instance or 1))
    instance_id = str(row.get("instance_id") or "")
    if views_by_instance.get(instance_id, 0) >= resolved_max_views:
        return False, {
            **row,
            "render_status": "REJECTED",
            "rejection_stage": "cpt_rendering",
            "rejection_reason": "MAX_VIEWS_PER_SEED_INSTANCE",
        }
    doc_type = _choose_doc_type(index, config.rendering)
    style_context = _choose_style_context(index, config.rendering)
    text = _render_by_doc_type(row, doc_type, style_context)
    text, numeric_formatting = _normalize_rendered_numeric_precision(text, config.rendering.numeric_rendering)
    quality = _check_rendered_text(
        text,
        doc_type,
        numeric_policy=config.rendering.numeric_rendering,
        generator_id=str(row.get("generator_id") or ""),
    )
    source_metadata = _source_metadata(row)
    generated = row.get("generated") or {}
    solver = generated.get("solver_result") or {}
    correctness = row.get("correctness") or {}
    doc = CPTDocument(
        doc_id=f"or_cpt_{_short_hash(row['pair_id'], doc_type)}",
        pair_id=row["pair_id"],
        instance_id=instance_id,
        generator_id=row.get("generator_id", ""),
        doc_type=doc_type,
        language=config.rendering.language,
        model_type=(row.get("reference") or {}).get("model_type"),
        difficulty_level=row.get("difficulty_level") or (row.get("reference") or {}).get("difficulty_level"),
        text=text,
        token_count=_approx_token_count(text),
        quality=quality,
        metadata={
            "bt_id": row.get("bt_id"),
            "fm_id": row.get("fm_id"),
            "source": "or_cpt_engine",
            "generator_profile_version": row.get("generator_profile_version"),
            "sub_family": row.get("sub_family"),
            "canonical_math_signature": row.get("canonical_math_signature") or {},
            "concept_tags": list(row.get("concept_tags") or []),
            "variant_id": row.get("variant_id"),
            "source_metadata": source_metadata,
            "scenario_id": source_metadata.get("scenario_id"),
            "base_scenario_id": source_metadata.get("base_scenario_id"),
            "scenario_variant_id": source_metadata.get("scenario_variant_id"),
            "scenario_source": source_metadata.get("scenario_source"),
            "task_family": source_metadata.get("task_family"),
            "business_trigger": source_metadata.get("business_trigger"),
            "industry_lens_id": source_metadata.get("industry_lens_id"),
            "narrative_angle_id": source_metadata.get("narrative_angle_id"),
            "organization_profile_id": source_metadata.get("organization_profile_id"),
            "planning_horizon_id": source_metadata.get("planning_horizon_id"),
            "entity_naming_style_id": source_metadata.get("entity_naming_style_id"),
            "style_context": style_context,
            "verbosity_level": style_context.get("verbosity_level"),
            "writing_style": style_context.get("writing_style"),
            "audience": style_context.get("audience"),
            "section_order_variant": style_context.get("section_order_variant"),
            "seed_instance_id": instance_id,
            "solver_objective_value": solver.get("objective_value"),
            "solver_status": solver.get("solver_status") or solver.get("status"),
            "forward_eval_status": row.get("acceptance_status") or "ACCEPTED",
            "objective_abs_error": correctness.get("abs_error"),
            "objective_rel_error": correctness.get("rel_error"),
            "objective_is_correct": correctness.get("is_correct"),
            "code_preparation": generated.get("code_preparation") or {},
            "family_contract_version": row.get("family_contract_version"),
            "family_contract_id": row.get("family_contract_id"),
            "family_contract": row.get("family_contract") or {},
            "numeric_formatting": numeric_formatting,
        },
    )
    if quality["status"] != "PASS":
        return False, {
            **doc.model_dump(mode="json"),
            "render_status": "REJECTED",
            "rejection_stage": "cpt_rendering",
            "rejection_reason": "; ".join(quality["issues"]),
        }
    views_by_instance[instance_id] = views_by_instance.get(instance_id, 0) + 1
    return True, doc


def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("source_metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def render_cpt_report(
    doc_count: int,
    rejected_count: int,
    type_counts: dict[str, int],
    rejection_reasons: dict[str, int] | None = None,
) -> str:
    lines = [
        "# CPT Rendering Report",
        "",
        f"- Rendered documents: {doc_count}",
        f"- Rejected documents: {rejected_count}",
        "",
        "| doc_type | count |",
        "|---|---:|",
    ]
    for doc_type, count in sorted(type_counts.items()):
        lines.append(f"| {doc_type} | {count} |")
    lines.extend(["", "## Rejection Reasons", "", "| reason | count |", "|---|---:|"])
    reasons = rejection_reasons or {}
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[:30]:
            lines.append(f"| `{_escape_table_cell(str(reason))}` | {count} |")
    else:
        lines.append("| n/a | 0 |")
    return "\n".join(lines) + "\n"


def _choose_doc_type(index: int, rendering_config: Any) -> str:
    policy = _normalized_doc_type_policy(rendering_config)
    cycle = _interleaved_doc_type_cycle(policy, cycle_size=100)
    return cycle[index % len(cycle)]


def _split_rejection_reason(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        reasons: list[str] = []
        for item in value:
            reasons.extend(_split_rejection_reason(item))
        return reasons or ["UNKNOWN"]
    text = str(value).strip()
    if not text:
        return ["UNKNOWN"]
    return [part.strip() for part in re.split(r";\s*", text) if part.strip()] or ["UNKNOWN"]


def _choose_style_context(index: int, rendering_config: Any) -> dict[str, str]:
    policy = getattr(rendering_config, "style_policy", None) or {}
    verbosity_levels = _style_values(policy, "verbosity_levels", ["standard"])
    writing_styles = _style_values(policy, "writing_styles", ["operations analyst memo"])
    audiences = _style_values(policy, "audiences", ["operations research practitioner"])
    section_variants = _style_values(
        policy,
        "section_order_variants",
        ["formulation_first", "implementation_first", "validation_first"],
    )
    return {
        "verbosity_level": verbosity_levels[index % len(verbosity_levels)],
        "writing_style": writing_styles[(index // max(1, len(verbosity_levels))) % len(writing_styles)],
        "audience": audiences[(index // max(1, len(verbosity_levels) * len(writing_styles))) % len(audiences)],
        "section_order_variant": section_variants[
            (index // max(1, len(verbosity_levels) * len(writing_styles) * len(audiences))) % len(section_variants)
        ],
    }


def _style_values(policy: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    raw_values = policy.get(key) if isinstance(policy, dict) else None
    values = [str(value).strip() for value in raw_values or [] if str(value).strip()]
    return values or fallback


def _interleaved_doc_type_cycle(policy: list[tuple[str, float]], *, cycle_size: int) -> list[str]:
    """Build a small weighted cycle that works for pilot runs as well as large runs."""
    if not policy:
        return ["final_model_report"]
    resolved_cycle_size = max(len(policy), int(cycle_size))
    target_counts = _largest_remainder_counts(policy, resolved_cycle_size)
    used_counts = {doc_type: 0 for doc_type, _ in policy}
    cycle: list[str] = []
    for position in range(1, resolved_cycle_size + 1):
        remaining_slots = resolved_cycle_size - len(cycle)
        forced = [
            doc_type
            for doc_type, target in target_counts.items()
            if target - used_counts[doc_type] >= remaining_slots
        ]
        if forced:
            selected = forced[0]
        else:
            selected = max(
                (doc_type for doc_type, target in target_counts.items() if used_counts[doc_type] < target),
                key=lambda doc_type: (
                    target_counts[doc_type] * position / resolved_cycle_size - used_counts[doc_type],
                    target_counts[doc_type],
                ),
            )
        cycle.append(selected)
        used_counts[selected] += 1
    return cycle or [policy[0][0]]


def _largest_remainder_counts(policy: list[tuple[str, float]], total: int) -> dict[str, int]:
    raw_counts = [(doc_type, ratio * total) for doc_type, ratio in policy]
    counts = {doc_type: int(raw) for doc_type, raw in raw_counts}
    remainder = total - sum(counts.values())
    for doc_type, _raw in sorted(raw_counts, key=lambda item: (-(item[1] - int(item[1])), item[0]))[:remainder]:
        counts[doc_type] += 1
    for doc_type, _ratio in policy:
        if counts[doc_type] == 0:
            donor = max(counts, key=lambda key: counts[key])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[doc_type] = 1
    return counts


def _normalized_doc_type_policy(rendering_config: Any) -> list[tuple[str, float]]:
    raw_policy = getattr(rendering_config, "doc_type_policy", None) or {}
    policy: list[tuple[str, float]] = []
    for doc_type, spec in raw_policy.items():
        if isinstance(spec, dict):
            ratio = spec.get("ratio")
        else:
            ratio = spec
        try:
            ratio_value = float(ratio)
        except (TypeError, ValueError):
            continue
        if ratio_value > 0:
            policy.append((str(doc_type), ratio_value))
    if not policy:
        policy = [
            ("final_model_report", float(getattr(rendering_config, "final_model_report_ratio", 0.70))),
            ("modeling_rationale", float(getattr(rendering_config, "modeling_rationale_ratio", 0.30))),
        ]
    total = sum(ratio for _, ratio in policy)
    if total <= 0:
        return [("final_model_report", 1.0)]
    return [(doc_type, ratio / total) for doc_type, ratio in policy]


def _render_by_doc_type(row: dict[str, Any], doc_type: str, style_context: dict[str, str]) -> str:
    if doc_type == "modeling_rationale":
        return _render_modeling_rationale(row, style_context)
    if doc_type == "solver_validation_note":
        return _render_solver_validation_note(row, style_context)
    return _render_final_model_report(row, style_context)


def _render_final_model_report(row: dict[str, Any], style_context: dict[str, str]) -> str:
    answer = _renderable_modeling_answer(row)
    generated = row.get("generated") or {}
    solver = generated.get("solver_result") or {}
    correctness = row.get("correctness") or {}
    sections = {
        "problem": _section("## Problem", str(row.get("problem_statement") or "").strip()),
        "modeling": _section(
            "## Modeling Explanation",
            str(
                answer.get("modeling_explanation")
                or "The model maps the business entities to decision variables, optimizes the stated objective, and enforces the resource and feasibility requirements from the problem statement."
            ).strip(),
        ),
        "formulation": _section(
            "## Model Formulation",
            str(answer.get("math_model") or "The mathematical formulation is represented by the executable Gurobi model below.").strip(),
        ),
        "implementation": _section("## Implementation", "```python\n" + str(answer.get("gurobipy_code") or "").strip() + "\n```"),
        "solution": _render_optimal_solution(solver, max_rows=_solution_row_limit(style_context)),
        "feasibility": _render_feasibility_check(solver, verbosity_level=style_context.get("verbosity_level", "standard")),
        "objective": _render_objective_recalculation(solver),
        "validation": _section("## Validation", _render_validation_summary(solver, correctness, style_context)),
    }
    return _join_rendered_sections(
        "# Optimization Modeling Report",
        sections,
        _section_order(
            style_context,
            default=["problem", "modeling", "formulation", "implementation", "solution", "feasibility", "objective", "validation"],
            implementation_first=["problem", "modeling", "implementation", "formulation", "solution", "feasibility", "objective", "validation"],
            validation_first=["problem", "validation", "modeling", "formulation", "implementation", "solution", "feasibility", "objective"],
        ),
    )


def _render_modeling_rationale(row: dict[str, Any], style_context: dict[str, str]) -> str:
    answer = _renderable_modeling_answer(row)
    generated = row.get("generated") or {}
    solver = generated.get("solver_result") or {}
    sections = {
        "scenario": _section("## Scenario Interpretation", str(row.get("problem_statement") or "").strip()),
        "business_mapping": _section("## Business-to-Model Mapping", _render_business_to_model_mapping(answer)),
        "variable_design": _section(
            "## Decision Variable Design",
            str(
                answer.get("modeling_explanation")
                or "The model introduces decision variables for the controllable quantities and links them with resource and feasibility constraints."
            ).strip(),
        ),
        "logic": _section(
            "## Objective and Constraint Logic",
            str(answer.get("math_model") or "The model optimizes the stated business objective while enforcing all provided constraints.").strip(),
        ),
        "code_mapping": _section("## Code-to-Model Mapping", _render_code_to_model_mapping(answer)),
        "executable": _section("## Executable Model", "```python\n" + str(answer.get("gurobipy_code") or "").strip() + "\n```"),
        "solution": _render_optimal_solution(solver, max_rows=_solution_row_limit(style_context)),
        "feasibility": _render_feasibility_check(solver, verbosity_level=style_context.get("verbosity_level", "standard")),
        "outcome": _section(
            "## Verified Outcome",
            f"The independent execution returned status `{solver.get('solver_status') or solver.get('status')}` and objective `{_format_number(solver.get('objective_value'))}`.",
        ),
    }
    return _join_rendered_sections(
        "# Modeling Rationale",
        sections,
        _section_order(
            style_context,
            default=["scenario", "business_mapping", "variable_design", "logic", "code_mapping", "executable", "solution", "feasibility", "outcome"],
            implementation_first=["scenario", "executable", "code_mapping", "business_mapping", "variable_design", "logic", "solution", "feasibility", "outcome"],
            validation_first=["scenario", "outcome", "solution", "feasibility", "business_mapping", "variable_design", "logic", "code_mapping", "executable"],
        ),
    )


def _render_solver_validation_note(row: dict[str, Any], style_context: dict[str, str]) -> str:
    answer = _renderable_modeling_answer(row)
    generated = row.get("generated") or {}
    solver = generated.get("solver_result") or {}
    correctness = row.get("correctness") or {}
    signature = row.get("canonical_math_signature") or {}
    return "\n\n".join(
        [
            "# Solver Validation Note",
            "",
            "## Problem Snapshot",
            str(row.get("problem_statement") or "").strip(),
            "## Model Artifact",
            (
                "The validated forward model is represented by this executable Gurobi artifact. "
                "It connects the natural-language problem, the mathematical structure, and the verified implementation."
            ),
            "```python\n" + str(answer.get("gurobipy_code") or "").strip() + "\n```",
            "## Canonical Math Signature",
            _render_signature_summary(signature),
            "## Solver Result",
            (
                f"The independent solver execution returned status `{solver.get('solver_status') or solver.get('status')}` "
                f"with objective value `{_format_number(solver.get('objective_value'))}`."
            ),
            _render_optimal_solution(solver, max_rows=_solution_row_limit(style_context)),
            _render_feasibility_check(solver, verbosity_level=style_context.get("verbosity_level", "standard")),
            "## Objective Check",
            (
                f"The forward model objective value was compared with the trusted target value. "
                f"The comparison result is `{correctness.get('is_correct')}`, with absolute error "
                f"`{_format_number(correctness.get('abs_error'))}` and relative error `{_format_number(correctness.get('rel_error'))}`."
            ),
            "## Feasibility Signals",
            (
                "The recorded feasibility metrics summarize whether the computed solution satisfies the model constraints and variable bounds. "
                "A non-optimal solver status, inconsistent objective comparison, or material feasibility violation would indicate that the formulation or implementation needs review."
            ),
            "## Verification Interpretation",
            "The solver evidence above indicates that the formulation, implementation, objective value, and feasibility checks are mutually consistent for this optimization instance.",
        ]
    ).strip() + "\n"


def _section(title: str, body: str) -> str:
    return f"{title}\n\n{body.strip()}".strip()


def _join_rendered_sections(title: str, sections: dict[str, str], order: list[str]) -> str:
    return "\n\n".join([title, "", *(sections[key] for key in order if key in sections and sections[key].strip())]).strip() + "\n"


def _section_order(
    style_context: dict[str, str],
    *,
    default: list[str],
    implementation_first: list[str],
    validation_first: list[str],
) -> list[str]:
    variant = style_context.get("section_order_variant")
    if variant == "implementation_first":
        return implementation_first
    if variant == "validation_first":
        return validation_first
    return default


def _solution_row_limit(style_context: dict[str, str]) -> int:
    verbosity = style_context.get("verbosity_level", "standard")
    if verbosity == "concise":
        return 25
    if verbosity == "detailed":
        return 120
    return _MAX_RENDERED_SOLUTION_ROWS


def _render_validation_summary(solver: dict[str, Any], correctness: dict[str, Any], style_context: dict[str, str]) -> str:
    status = solver.get("solver_status") or solver.get("status")
    objective = solver.get("objective_value")
    passed = correctness.get("is_correct")
    if style_context.get("verbosity_level") == "concise":
        return f"Solver status `{status}`, objective `{_format_number(objective)}`, objective comparison `{passed}`."
    if style_context.get("verbosity_level") == "detailed":
        return (
            f"The generated model solved with status `{status}`. The objective value is `{_format_number(objective)}`. "
            f"The objective comparison passed: `{passed}`. Absolute error is `{_format_number(correctness.get('abs_error'))}` "
            f"and relative error is `{_format_number(correctness.get('rel_error'))}`."
        )
    return (
        f"The generated model solved with status `{status}`. "
        f"The objective value is `{_format_number(objective)}`. "
        f"The objective comparison passed: `{passed}`."
    )


def _check_rendered_text(
    text: str,
    doc_type: str,
    *,
    numeric_policy: dict[str, Any] | None = None,
    generator_id: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    policy = _numeric_policy(numeric_policy)
    if policy["enabled"]:
        text_for_numeric_check = text if policy["apply_to_code_blocks"] else _remove_code_blocks(text)
        long_decimal_count = count_long_decimals(text_for_numeric_check, max_decimal_places=policy["max_decimal_places"])
        if long_decimal_count:
            issues.append(
                f"EXCESSIVE_DECIMAL_PRECISION:count={long_decimal_count}:max_decimal_places={policy['max_decimal_places']}"
            )
    lowered = text.lower()
    for phrase in FORBIDDEN_RENDER_PHRASES:
        if phrase in lowered:
            issues.append(f"FORBIDDEN_PHRASE:{phrase}")
    for issue in semantic_coherence_issues(str(generator_id or ""), text):
        issues.append(f"SEMANTIC_COHERENCE:{issue}")
    for section in ("## Problem", "## Model Formulation", "## Implementation", "## Validation"):
        if text.startswith("# Optimization Modeling Report") and section not in text:
            issues.append(f"MISSING_SECTION:{section}")
    if doc_type == "solver_validation_note":
        for section in ("## Problem Snapshot", "## Solver Result", "## Objective Check", "## Feasibility Signals"):
            if section not in text:
                issues.append(f"MISSING_SECTION:{section}")
    for section in ("## Optimal Solution", "## Feasibility Check"):
        if section not in text:
            issues.append(f"MISSING_SECTION:{section}")
    if len(text.strip()) < 400:
        issues.append("TOO_SHORT")
    if "```python" not in text:
        issues.append("MISSING_CODE_BLOCK")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues}


def _normalize_rendered_numeric_precision(text: str, numeric_policy: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    policy = _numeric_policy(numeric_policy)
    if not policy["enabled"]:
        return text, {"enabled": False, "normalized_count": 0, "max_decimal_places": policy["max_decimal_places"]}
    normalized_count = 0

    def replace_number(match: re.Match[str]) -> str:
        nonlocal normalized_count
        value = match.group(0)
        decimals = value.split(".", 1)[1]
        if len(decimals.rstrip("`),.;:")) <= policy["max_decimal_places"]:
            return value
        normalized_count += 1
        trailing = ""
        while value and value[-1] in "`),.;:":
            trailing = value[-1] + trailing
            value = value[:-1]
        return _format_number(value, decimal_places=policy["max_decimal_places"]) + trailing

    number_pattern = re.compile(r"(?<![\w.])-?\d+\.\d{3,}(?![\w.])")
    if policy["apply_to_code_blocks"]:
        return number_pattern.sub(replace_number, text), {
            "enabled": True,
            "normalized_count": normalized_count,
            "max_decimal_places": policy["max_decimal_places"],
            "apply_to_code_blocks": True,
        }

    parts = re.split(r"(```[\s\S]*?```)", text)
    normalized_parts = [
        part if part.startswith("```") else number_pattern.sub(replace_number, part)
        for part in parts
    ]
    return "".join(normalized_parts), {
        "enabled": True,
        "normalized_count": normalized_count,
        "max_decimal_places": policy["max_decimal_places"],
        "apply_to_code_blocks": False,
    }


def _numeric_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    raw = policy or {}
    try:
        max_decimal_places = int(raw.get("max_decimal_places", 2))
    except (TypeError, ValueError):
        max_decimal_places = 2
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_decimal_places": max(0, max_decimal_places),
        "apply_to_code_blocks": bool(raw.get("apply_to_code_blocks", False)),
    }


def _remove_code_blocks(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", text)


def _approx_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


def _short_hash(*parts: str) -> str:
    joined = ":".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _render_optimal_solution(solver: dict[str, Any], *, max_rows: int | None = None) -> str:
    solution = solver.get("solution") or {}
    objective_value = solution.get("objective_value")
    if objective_value is None:
        objective_value = solver.get("objective_value")
    values = solution.get("nonzero_variable_values") or solution.get("variable_values") or {}
    lines = [
        "## Optimal Solution",
        "",
        f"The optimal objective value reported by the solver is `{_format_number(objective_value)}`.",
        "",
    ]
    if not values:
        lines.append("No nonzero decision-variable values were available in the structured solver result.")
        return "\n".join(lines)

    lines.extend(
        [
            "Nonzero decision variables:",
            "",
            "| Variable | Value |",
            "| --- | ---: |",
        ]
    )
    row_limit = max_rows if max_rows is not None else _MAX_RENDERED_SOLUTION_ROWS
    for index, (name, value) in enumerate(sorted(values.items(), key=lambda item: _natural_sort_key(str(item[0])))):
        if index >= row_limit:
            remaining = max(0, len(values) - row_limit)
            lines.append(f"| ... | {remaining} additional nonzero values omitted from the rendered table |")
            break
        lines.append(f"| `{_escape_table_cell(str(name))}` | {_format_number(value)} |")
    if solution.get("nonzero_variable_values_truncated"):
        lines.append("")
        lines.append("The stored nonzero variable list was truncated by the solver-result size limit.")
    return "\n".join(lines)


def _render_feasibility_check(solver: dict[str, Any], *, verbosity_level: str = "standard") -> str:
    solution = solver.get("solution") or {}
    feasibility = solution.get("feasibility") or {}
    lines = [
        "## Feasibility Check",
        "",
        "| Check | Value |",
        "| --- | ---: |",
        f"| Constraint violation | {_format_number(feasibility.get('constraint_violation'))} |",
        f"| Bound violation | {_format_number(feasibility.get('bound_violation'))} |",
        f"| Integer violation | {_format_number(feasibility.get('integer_violation'))} |",
        f"| Maximum violation | {_format_number(feasibility.get('max_violation'))} |",
        "",
    ]
    if verbosity_level == "concise":
        return "\n".join(lines).strip()
    if feasibility.get("is_feasible_within_tolerance") is False:
        lines.append("The solver reported a nonzero feasibility violation, so this sample should be reviewed before use.")
    else:
        lines.append("The solver quality metrics indicate that the solution is feasible within the configured numerical tolerance.")
    return "\n".join(lines)


def _render_objective_recalculation(solver: dict[str, Any]) -> str:
    solution = solver.get("solution") or {}
    objective_value = solution.get("objective_value")
    if objective_value is None:
        objective_value = solver.get("objective_value")
    objective_recomputed = solution.get("objective_recomputed")
    return "\n".join(
        [
            "## Objective Recalculation",
            "",
            (
                "The objective value recomputed from the solved model expression is "
                f"`{_format_number(objective_recomputed)}`. The solver objective is "
                f"`{_format_number(objective_value)}`."
            ),
        ]
    )


def _render_business_to_model_mapping(answer: dict[str, Any]) -> str:
    explanation = str(answer.get("modeling_explanation") or "").strip()
    if explanation:
        return explanation
    return "\n".join(
        [
            "The business entities define the model's sets and indices. Numeric quantities from the problem become parameters, and controllable decisions become variables. The objective encodes the stated business goal, while each constraint family enforces a capacity, demand, assignment, balance, or bound requirement from the scenario.",
        ]
    )


def _renderable_modeling_answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = dict(row.get("modeling_answer") or {})
    fallback_explanation = (
        "The model maps the business entities to indexed decision variables, preserves the stated objective, "
        "and enforces the capacity, demand, balance, assignment, or bound constraints described in the problem."
    )
    fallback_math = "The mathematical formulation is represented by the executable Gurobi model below."
    answer["modeling_explanation"] = _strip_repair_audit_language(
        answer.get("modeling_explanation"),
        fallback=fallback_explanation,
    )
    answer["math_model"] = _strip_repair_audit_language(
        answer.get("math_model"),
        fallback=fallback_math,
    )
    return answer


def _strip_repair_audit_language(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if not _contains_repair_audit_language(text):
        return text

    kept: list[str] = []
    # Keep headings and safe sentences, but drop sentences that describe failed
    # forward attempts or repair actions. Those are pipeline audit details, not
    # OR modeling content for CPT training.
    for paragraph in re.split(r"\n{2,}", text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not _contains_repair_audit_language(stripped):
            kept.append(stripped)
            continue
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        safe_parts = [part.strip() for part in parts if part.strip() and not _contains_repair_audit_language(part)]
        if safe_parts:
            kept.append(" ".join(safe_parts))

    cleaned = "\n\n".join(kept).strip()
    if len(cleaned) < 80 or _contains_repair_audit_language(cleaned):
        return fallback
    return cleaned


def _contains_repair_audit_language(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REPAIR_AUDIT_MARKERS)


def _render_code_to_model_mapping(answer: dict[str, Any]) -> str:
    code = str(answer.get("gurobipy_code") or "")
    hints: list[str] = []
    if "addVar" in code or "addVars" in code:
        hints.append("The `addVar` / `addVars` calls create the decision variables from the mathematical formulation.")
    if "setObjective" in code:
        hints.append("The `setObjective` call implements the objective function and optimization direction.")
    if "addConstr" in code or "addConstrs" in code:
        hints.append("The `addConstr` / `addConstrs` calls encode the constraint families from the formulation.")
    if "optimize()" in code:
        hints.append("The `optimize()` call solves the constructed model before validation.")
    if not hints:
        hints.append("The executable code implements the formulation above and is independently solved during validation.")
    return "\n".join(f"- {hint}" for hint in hints)


def _render_signature_summary(signature: dict[str, Any]) -> str:
    if not signature:
        return "No structured canonical signature was available for this generator family."
    lines = ["| Field | Value |", "| --- | --- |"]
    for key in ("sets", "parameters", "variables", "objective", "constraints", "forbidden_changes"):
        value = signature.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"| `{_escape_table_cell(str(key))}` | {_escape_table_cell(rendered)} |")
    return "\n".join(lines)


def _format_number(value: Any, *, decimal_places: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
    if abs(numeric - round(numeric)) <= 1e-9:
        return str(int(round(numeric)))
    places = max(0, int(decimal_places))
    return f"{numeric:.{places}f}".rstrip("0").rstrip(".")


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _natural_sort_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]
