from __future__ import annotations

from typing import Any

from or_cpt_engine.forward_modeling.code_quality import classify_execution_failure

_GUROBI_STATUS_MAP = {
    "2": "OPTIMAL",
    "3": "INFEASIBLE",
    "4": "INF_OR_UNBD",
    "5": "UNBOUNDED",
    "7": "ITERATION_LIMIT",
    "9": "TIME_LIMIT",
    "13": "SUBOPTIMAL",
}


def classify_forward_eval_rejection(
    *,
    row: dict[str, Any],
    rejection_reason: str,
    solver_result: dict[str, Any] | None = None,
    correctness: Any | None = None,
    code_preparation: dict[str, Any] | None = None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    """Return an actionable, fine-grained rejection diagnosis for 08 forward eval."""
    solver_result = solver_result or {}
    code_preparation = code_preparation or {}
    static_issues = [str(issue) for issue in code_preparation.get("issues") or []]
    raw_status = str(solver_result.get("status") or solver_result.get("execution_status") or "")
    status = normalize_gurobi_status(raw_status)
    execution_status = str(solver_result.get("execution_status") or "")
    failure_class = classify_execution_failure(solver_result) if solver_result else ""

    if static_issues:
        fine_reason = _static_issue_reason(static_issues)
        return _payload(
            base_reason=rejection_reason,
            fine_reason=fine_reason,
            failure_family="static_code_gate",
            repairable=fine_reason not in {"external_dependency", "placeholder_or_incomplete_code"},
            row=row,
            solver_result=solver_result,
            correctness=correctness,
            static_issues=static_issues,
            failure_class=failure_class,
        )

    if exception is not None:
        return _payload(
            base_reason=rejection_reason,
            fine_reason="forward_eval_exception",
            failure_family="pipeline_exception",
            repairable=False,
            row=row,
            solver_result=solver_result,
            correctness=correctness,
            static_issues=static_issues,
            failure_class=exception.__class__.__name__,
        )

    if status and status not in {"OPTIMAL", "EXECUTION_ERROR", "EXEC_ERROR", "RESULT_PARSE_ERROR", "EMPTY_CODE", "EMPTY_LP", "IMPORT_ERROR", "MODEL_NOT_FOUND", "TIMEOUT"}:
        return _payload(
            base_reason=base_reason_for_status(rejection_reason, status),
            fine_reason=_solver_status_reason(status),
            failure_family="solver_status_failure",
            repairable=status in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"},
            row=row,
            solver_result={**solver_result, "status_normalized": status},
            correctness=correctness,
            static_issues=static_issues,
            failure_class=failure_class,
        )

    if execution_status and execution_status != "SUCCESS":
        return _payload(
            base_reason=rejection_reason,
            fine_reason=_execution_failure_reason(failure_class, status),
            failure_family="execution_failure",
            repairable=failure_class in {"KEY_ERROR", "INDEX_ERROR", "NAME_ERROR", "ATTRIBUTE_ERROR", "TYPE_ERROR", "MODEL_NOT_FOUND"},
            row=row,
            solver_result=solver_result,
            correctness=correctness,
            static_issues=static_issues,
            failure_class=failure_class,
        )

    if status and status != "OPTIMAL":
        return _payload(
            base_reason=rejection_reason,
            fine_reason=_solver_status_reason(status),
            failure_family="solver_status_failure",
            repairable=status in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"},
            row=row,
            solver_result=solver_result,
            correctness=correctness,
            static_issues=static_issues,
            failure_class=failure_class,
        )

    if getattr(correctness, "is_correct", None) is False:
        return _payload(
            base_reason=rejection_reason,
            fine_reason=_objective_mismatch_reason(row, correctness),
            failure_family="objective_mismatch",
            repairable=True,
            row=row,
            solver_result=solver_result,
            correctness=correctness,
            static_issues=static_issues,
            failure_class=failure_class,
        )

    return _payload(
        base_reason=rejection_reason,
        fine_reason="unknown_forward_eval_rejection",
        failure_family="unknown",
        repairable=False,
        row=row,
        solver_result=solver_result,
        correctness=correctness,
        static_issues=static_issues,
        failure_class=failure_class,
    )


def _static_issue_reason(static_issues: list[str]) -> str:
    issue_blob = " ".join(static_issues)
    if "EXTERNAL_" in issue_blob:
        return "external_dependency"
    if "PLACEHOLDER" in issue_blob or "ANGLE_BRACKET_PLACEHOLDER" in issue_blob:
        return "placeholder_or_incomplete_code"
    if "MISSING_SET_OBJECTIVE" in issue_blob:
        return "missing_objective"
    if "MISSING_OPTIMIZE" in issue_blob:
        return "missing_optimize_call"
    if "MISSING_GUROBIPY" in issue_blob or "MISSING_MODEL" in issue_blob:
        return "missing_gurobi_model"
    return "static_code_quality_failure"


def _execution_failure_reason(failure_class: str, status: str) -> str:
    mapping = {
        "KEY_ERROR": "execution_key_error_or_invalid_schema",
        "INDEX_ERROR": "execution_index_error_or_invalid_index_set",
        "NAME_ERROR": "execution_name_error",
        "ATTRIBUTE_ERROR": "execution_attribute_error",
        "TYPE_ERROR": "execution_type_error",
        "SYNTAX_ERROR": "execution_syntax_error",
        "MODEL_NOT_FOUND": "execution_missing_model_object",
        "TIMEOUT": "execution_timeout",
        "RESULT_PARSE_ERROR": "execution_result_parse_error",
        "GUROBI_SIZE_LIMIT": "gurobi_size_limited_license",
        "GUROBI_LICENSE_ERROR": "gurobi_license_error",
    }
    if failure_class in mapping:
        return mapping[failure_class]
    if status == "TIMEOUT":
        return "execution_timeout"
    return "execution_unclassified_error"


def _solver_status_reason(status: str) -> str:
    normalized = normalize_gurobi_status(status).upper()
    if normalized == "INFEASIBLE":
        return "solver_infeasible"
    if normalized == "INF_OR_UNBD":
        return "solver_infeasible_or_unbounded"
    if normalized == "UNBOUNDED":
        return "solver_unbounded"
    if normalized in {"TIME_LIMIT", "SUBOPTIMAL", "ITERATION_LIMIT"}:
        return "solver_not_optimal_limit"
    return "solver_not_optimal_other"


def normalize_gurobi_status(status: Any) -> str:
    text = str(status or "").strip()
    return _GUROBI_STATUS_MAP.get(text, text.upper())


def base_reason_for_status(original_reason: str, status: str) -> str:
    if original_reason.startswith("FORWARD_SOLVER_NOT_OPTIMAL"):
        return original_reason
    return f"FORWARD_SOLVER_NOT_OPTIMAL:{status}"


def _objective_mismatch_reason(row: dict[str, Any], correctness: Any) -> str:
    concept_tags = {str(value).lower() for value in row.get("concept_tags") or []}
    family_contract = row.get("family_contract") or {}
    contract_tags = {str(value).lower() for value in family_contract.get("contract_tags") or []}
    signature = row.get("canonical_math_signature") or {}
    required = {str(value).lower() for value in signature.get("core_constraints") or []}
    rel_error = getattr(correctness, "rel_error", None)
    abs_error = getattr(correctness, "abs_error", None)
    if "fixed_cost" in concept_tags or any("fixed" in tag for tag in contract_tags):
        return "objective_mismatch_likely_missing_fixed_cost"
    if "flow_balance" in required or any("balance" in tag or "flow" in tag for tag in contract_tags):
        return "objective_mismatch_likely_balance_or_flow_error"
    if rel_error is not None and abs(rel_error) > 0.25:
        return "objective_mismatch_large_relative_error"
    if abs_error is not None and abs(abs_error) <= 1e-2:
        return "objective_mismatch_tolerance_edge"
    return "objective_mismatch_general"


def _payload(
    *,
    base_reason: str,
    fine_reason: str,
    failure_family: str,
    repairable: bool,
    row: dict[str, Any],
    solver_result: dict[str, Any],
    correctness: Any | None,
    static_issues: list[str],
    failure_class: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "solver_status": solver_result.get("status"),
        "execution_status": solver_result.get("execution_status"),
        "failure_class": failure_class,
        "static_issues": static_issues,
        "generator_id": row.get("generator_id"),
        "family_contract_id": row.get("family_contract_id"),
    }
    if correctness is not None:
        evidence["objective_abs_error"] = getattr(correctness, "abs_error", None)
        evidence["objective_rel_error"] = getattr(correctness, "rel_error", None)
    return {
        "rejection_reason_base": base_reason,
        "fine_rejection_reason": fine_reason,
        "failure_family": failure_family,
        "repairable": repairable,
        "diagnostic_evidence": evidence,
    }
