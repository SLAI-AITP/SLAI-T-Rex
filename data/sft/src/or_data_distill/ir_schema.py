from __future__ import annotations

from typing import Any


PROBLEM_MODES = {"DP", "DT", "DPS"}
DATA_INTERFACES = {"inline_text", "markdown_table", "attached_files"}
ANSWER_STYLES = {"math_model", "math_model_with_code", "code_only"}

DOMAINS = {
    "logistics",
    "production",
    "energy",
    "healthcare",
    "finance",
    "agriculture",
    "education",
    "environment",
    "computing",
    "generic",
}

STRUCTURES = {
    "assignment",
    "scheduling",
    "routing",
    "transportation",
    "blending",
    "capacity_planning",
    "facility_location",
    "production_planning",
    "portfolio",
    "generic_lp_mip",
}

DIFFICULTIES = {"small", "medium", "large", "industrial"}


def normalize_mode(value: str | None) -> str:
    value = (value or "").strip().upper()
    return value if value in PROBLEM_MODES else "DP"


def infer_data_interface(mode: str, problem: str = "") -> str:
    if mode == "DT" or "|" in problem:
        return "markdown_table"
    if mode == "DPS":
        return "attached_files"
    return "inline_text"


def validate_ir(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if row.get("mode") not in PROBLEM_MODES:
        issues.append("invalid_mode")
    if row.get("domain") not in DOMAINS:
        issues.append("invalid_domain")
    if row.get("structure") not in STRUCTURES:
        issues.append("invalid_structure")
    if row.get("difficulty") not in DIFFICULTIES:
        issues.append("invalid_difficulty")
    if row.get("data_interface") not in DATA_INTERFACES:
        issues.append("invalid_data_interface")
    if row.get("answer_style") not in ANSWER_STYLES:
        issues.append("invalid_answer_style")
    if not str(row.get("objective") or "").strip():
        issues.append("missing_objective")
    if not row.get("decision_variables"):
        issues.append("missing_decision_variables")
    if not row.get("constraints"):
        issues.append("missing_constraints")
    return issues

