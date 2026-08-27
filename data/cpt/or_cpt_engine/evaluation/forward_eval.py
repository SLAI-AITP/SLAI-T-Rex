from __future__ import annotations

from pathlib import Path
from typing import Any

from collections import Counter

from or_cpt_engine.evaluation.rejection_classifier import classify_forward_eval_rejection, normalize_gurobi_status
from or_cpt_engine.forward_modeling.code_quality import classify_execution_failure, prepare_forward_code
from or_cpt_engine.schemas.common import AcceptedPair, ObjectiveComparison
from or_cpt_engine.solver.gurobi_executor import execute_gurobi_code
from or_cpt_engine.solver.objective_comparator import compare_objective
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text


def evaluate_forward_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 60,
    threads_per_process: int = 1,
    abs_tolerance: float = 1e-4,
    rel_tolerance: float = 1e-4,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "accepted_pairs.jsonl"
    rejected_path = output / "rejected_pairs.jsonl"
    accepted_count = 0
    rejected_count = 0
    fine_reasons: Counter[str] = Counter()

    for row in (iter_jsonl(input_path) or []):
        accepted, record = evaluate_forward_output_row(
            row,
            timeout_seconds=timeout_seconds,
            threads_per_process=threads_per_process,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        if accepted:
            append_jsonl(accepted_path, record)
            accepted_count += 1
        else:
            append_jsonl(rejected_path, record)
            rejected_count += 1
            if isinstance(record, dict):
                fine_reasons[str(record.get("fine_rejection_reason") or record.get("rejection_reason") or "unknown")] += 1

    write_text(output / "forward_eval_report.md", render_forward_eval_report(accepted_count, rejected_count, fine_reasons))
    write_text(
        output / "forward_eval_rejection_dashboard.md",
        render_forward_eval_rejection_dashboard(accepted_count, rejected_count, fine_reasons),
    )
    return {"accepted": accepted_count, "rejected": rejected_count}


def evaluate_forward_output_row(
    row: dict[str, Any],
    *,
    timeout_seconds: int = 60,
    threads_per_process: int = 1,
    abs_tolerance: float = 1e-4,
    rel_tolerance: float = 1e-4,
) -> tuple[bool, AcceptedPair | dict[str, Any]]:
    try:
        code_path = row.get("code_path")
        if not code_path:
            raise ValueError("missing code_path")
        code = Path(code_path).read_text(encoding="utf-8")
        code_preparation = prepare_forward_code(code)
        if not code_preparation.passed:
            rejection_reason = "FORWARD_CODE_STATIC_CHECK_FAILED:" + "; ".join(code_preparation.issues)
            return False, {
                **row,
                "forward_eval": {
                    "code_preparation": code_preparation.to_dict(),
                },
                "rejection_stage": "forward_eval",
                "rejection_reason": rejection_reason,
                **classify_forward_eval_rejection(
                    row=row,
                    rejection_reason=rejection_reason,
                    code_preparation=code_preparation.to_dict(),
                ),
            }
        solver_result = execute_gurobi_code(
            code_preparation.code,
            timeout_seconds=timeout_seconds,
            threads_per_process=threads_per_process,
            log_context={"instance_id": row.get("instance_id", ""), "stage": "08"},
        )
        reference_objective = _reference_objective(row)
        generated_objective = solver_result.get("objective_value")
        correctness = compare_objective(
            reference_objective,
            generated_objective,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        if solver_result.get("execution_status") == "SUCCESS" and correctness.is_correct:
            return True, AcceptedPair(
                pair_id=f"pair_{row['fm_id']}",
                fm_id=row["fm_id"],
                bt_id=row["bt_id"],
                instance_id=row["instance_id"],
                generator_id=row.get("generator_id", ""),
                problem_statement=row.get("problem_statement") or "",
                reference=row.get("reference_answer") or {},
                generated={"solver_result": solver_result, "code_preparation": code_preparation.to_dict()},
                source_compact_data=row.get("source_compact_data") or {},
                generator_profile_version=row.get("generator_profile_version"),
                difficulty_level=row.get("difficulty_level"),
                sub_family=row.get("sub_family"),
                canonical_math_signature=row.get("canonical_math_signature") or {},
                concept_tags=list(row.get("concept_tags") or []),
                variant_id=row.get("variant_id"),
                source_metadata=row.get("source_metadata") or {},
                family_contract_version=row.get("family_contract_version"),
                family_contract_id=row.get("family_contract_id"),
                family_contract=row.get("family_contract") or {},
                correctness=ObjectiveComparison.model_validate(correctness.model_dump(mode="json")),
                modeling_answer=row.get("generated_answer") or {},
                forward_repair=row.get("forward_repair") or {},
            )
        rejection_reason = _rejection_reason(solver_result, correctness)
        return False, {
            **row,
            "forward_eval": {
                "solver_result": solver_result,
                "correctness": correctness.model_dump(mode="json"),
                "code_preparation": code_preparation.to_dict(),
            },
            "rejection_stage": "forward_eval",
            "rejection_reason": rejection_reason,
            **classify_forward_eval_rejection(
                row=row,
                rejection_reason=rejection_reason,
                solver_result=solver_result,
                correctness=correctness,
                code_preparation=code_preparation.to_dict(),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        rejection_reason = f"{exc.__class__.__name__}: {exc}"
        return False, {
            **row,
            "rejection_stage": "forward_eval",
            "rejection_reason": rejection_reason,
            **classify_forward_eval_rejection(
                row=row,
                rejection_reason=rejection_reason,
                exception=exc,
            ),
        }


def render_forward_eval_report(
    accepted_count: int,
    rejected_count: int,
    fine_reasons: dict[str, int] | Counter[str] | None = None,
) -> str:
    lines = [
        "# Forward Evaluation Report",
        "",
        f"- Accepted pairs: {accepted_count}",
        f"- Rejected pairs: {rejected_count}",
        "",
    ]
    if fine_reasons:
        lines.extend(["## Fine-Grained Rejection Reasons", "", "| Reason | Count |", "| --- | ---: |"])
        for reason, count in sorted(dict(fine_reasons).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_forward_eval_rejection_dashboard(
    accepted_count: int,
    rejected_count: int,
    fine_reasons: dict[str, int] | Counter[str] | None = None,
) -> str:
    total = accepted_count + rejected_count
    rejection_rate = (rejected_count / total) if total else 0.0
    lines = [
        "# Forward Eval Rejection Dashboard",
        "",
        f"- Accepted pairs: {accepted_count}",
        f"- Rejected pairs: {rejected_count}",
        f"- Rejection rate: {rejection_rate:.2%}",
        "",
        "## Fine-Grained Rejection Reasons",
        "",
        "| Reason | Count | Share of rejected |",
        "| --- | ---: | ---: |",
    ]
    reasons = dict(fine_reasons or {})
    if not reasons:
        lines.append("| n/a | 0 | 0.00% |")
    else:
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            share = count / rejected_count if rejected_count else 0.0
            lines.append(f"| `{reason}` | {count} | {share:.2%} |")
    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- High `execution_key_error_or_invalid_schema` usually points to invalid index-set or schema guessing in 07 forward modeling.",
            "- High `solver_infeasible` usually points to missing or contradictory core constraints.",
            "- High `objective_mismatch_likely_missing_fixed_cost` usually points to omitted objective terms or family-contract drift.",
            "- High static gate failures should be handled before adding LLM reviewer/repair because they are cheap to catch deterministically.",
            "",
        ]
    )
    return "\n".join(lines)


def _reference_objective(row: dict[str, Any]) -> float | None:
    reference = row.get("reference_answer") or {}
    value = reference.get("objective_value")
    if value is None:
        value = reference.get("reference_objective")
    if value is None:
        return None
    return float(value)


def _rejection_reason(solver_result: dict[str, Any], correctness) -> str:
    normalized_status = normalize_gurobi_status(solver_result.get("status") or solver_result.get("execution_status"))
    if normalized_status in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "TIME_LIMIT", "SUBOPTIMAL", "ITERATION_LIMIT"}:
        return f"FORWARD_SOLVER_NOT_OPTIMAL:{normalized_status}"
    if solver_result.get("execution_status") != "SUCCESS":
        failure_class = classify_execution_failure(solver_result)
        status = normalized_status or solver_result.get("status") or solver_result.get("execution_status")
        return f"FORWARD_CODE_EXECUTION_FAILED:{status}:{failure_class}"
    if normalized_status != "OPTIMAL":
        return f"FORWARD_SOLVER_NOT_OPTIMAL:{normalized_status}"
    if not correctness.is_correct:
        return "OBJECTIVE_MISMATCH"
    return "UNKNOWN_FORWARD_EVAL_REJECTION"
