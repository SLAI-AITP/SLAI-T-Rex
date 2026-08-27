from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForwardCodePreparation:
    code: str
    issues: list[str] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "issues": list(self.issues),
            "repairs_applied": list(self.repairs_applied),
        }


_PLACEHOLDER_PATTERNS = (
    (re.compile(r"\bTODO\b", re.IGNORECASE), "PLACEHOLDER_TODO"),
    (re.compile(r"\bplaceholder\b", re.IGNORECASE), "PLACEHOLDER_TEXT"),
    (re.compile(r"\bperm\s*\?", re.IGNORECASE), "PLACEHOLDER_PERM_QUESTION"),
    (re.compile(r"adjust\s+if\s+needed", re.IGNORECASE), "PLACEHOLDER_ADJUST_IF_NEEDED"),
    (re.compile(r"replace\s+with", re.IGNORECASE), "PLACEHOLDER_REPLACE_WITH"),
    (re.compile(r"path/to/|your_file|filename\.|data\.csv", re.IGNORECASE), "PLACEHOLDER_FILE_PATH"),
    (re.compile(r"<\s*(?:replace|insert|fill|todo|your|path|file|value|data)[^>\n]{0,80}>", re.IGNORECASE), "ANGLE_BRACKET_PLACEHOLDER"),
)

_EXTERNAL_READ_PATTERNS = (
    (re.compile(r"\bopen\s*\([^,\n]+,\s*['\"]r", re.IGNORECASE), "EXTERNAL_FILE_OPEN_READ"),
    (re.compile(r"\b(?:pd|pandas)\.read_(?:csv|excel|json|parquet|table)\s*\(", re.IGNORECASE), "EXTERNAL_PANDAS_READ"),
    (re.compile(r"\bnp\.load(?:txt)?\s*\(", re.IGNORECASE), "EXTERNAL_NUMPY_LOAD"),
    (re.compile(r"\bjson\.load\s*\(", re.IGNORECASE), "EXTERNAL_JSON_LOAD"),
    (re.compile(r"\bPath\s*\([^)]*\)\.read_(?:text|bytes)\s*\(", re.IGNORECASE), "EXTERNAL_PATH_READ"),
    (re.compile(r"\bpathlib\.Path\s*\([^)]*\)\.read_(?:text|bytes)\s*\(", re.IGNORECASE), "EXTERNAL_PATHLIB_READ"),
)


def prepare_forward_code(code: str) -> ForwardCodePreparation:
    normalized = _normalize_code(code)
    repaired, repairs = _apply_safe_repairs(normalized)
    issues = _static_code_issues(repaired)
    return ForwardCodePreparation(code=repaired, issues=issues, repairs_applied=repairs)


def classify_execution_failure(solver_result: dict) -> str:
    status = str(solver_result.get("status") or solver_result.get("execution_status") or "UNKNOWN")
    execution_status = str(solver_result.get("execution_status") or "")
    if execution_status == "SUCCESS":
        return ""
    stderr = str(solver_result.get("stderr") or "")
    stdout = str(solver_result.get("stdout") or "")
    combined = f"{stderr}\n{stdout}"
    if "size-limited license" in combined or "Model too large for size-limited license" in combined:
        return "GUROBI_SIZE_LIMIT"
    if "license" in combined.lower() and "gurobi" in combined.lower():
        return "GUROBI_LICENSE_ERROR"
    for marker in (
        "SyntaxError",
        "IndentationError",
        "NameError",
        "KeyError",
        "TypeError",
        "AttributeError",
        "IndexError",
        "ValueError",
        "ModuleNotFoundError",
        "ImportError",
    ):
        if marker in combined:
            return marker.upper().replace("ERROR", "_ERROR")
    if "No gurobipy.Model instance found" in combined or status == "MODEL_NOT_FOUND":
        return "MODEL_NOT_FOUND"
    if status == "TIMEOUT":
        return "TIMEOUT"
    if status == "RESULT_PARSE_ERROR":
        return "RESULT_PARSE_ERROR"
    if status in {"IMPORT_ERROR", "EXEC_ERROR", "EXECUTION_ERROR", "EMPTY_CODE"}:
        return status
    return "UNCLASSIFIED_EXECUTION_ERROR"


def _normalize_code(code: str) -> str:
    return str(code or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _apply_safe_repairs(code: str) -> tuple[str, list[str]]:
    repaired = code
    repairs: list[str] = []

    if "GRB." in repaired and "from gurobipy import GRB" not in repaired:
        lines = repaired.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import gurobipy") or stripped.startswith("from gurobipy"):
                insert_at = index + 1
        lines.insert(insert_at, "from gurobipy import GRB")
        repaired = "\n".join(lines)
        repairs.append("ADD_GRB_IMPORT")

    updated = re.sub(r"\bmodel\.status\b", "model.Status", repaired)
    if updated != repaired:
        repaired = updated
        repairs.append("NORMALIZE_MODEL_STATUS_CASE")

    updated = re.sub(r"\bmodel\.objVal\b", "model.ObjVal", repaired)
    updated = re.sub(r"\bmodel\.objval\b", "model.ObjVal", updated)
    if updated != repaired:
        repaired = updated
        repairs.append("NORMALIZE_MODEL_OBJVAL_CASE")

    updated, indent_repairs = _repair_unexpected_top_level_indents(repaired)
    if updated != repaired:
        repaired = updated
        repairs.extend(indent_repairs)

    return repaired, repairs


def _repair_unexpected_top_level_indents(code: str) -> tuple[str, list[str]]:
    """Strip accidental top-level indentation only when Python pinpoints it.

    LLMs occasionally emit a single leading space before an otherwise top-level
    assignment, which later fails with ``IndentationError: unexpected indent``.
    We only accept the repair if the resulting code parses successfully, so
    legitimate block indentation is not guessed at or flattened.
    """
    try:
        ast.parse(code)
        return code, []
    except IndentationError as exc:
        if "unexpected indent" not in str(exc).lower() or not exc.lineno:
            return code, []
    except SyntaxError:
        return code, []

    lines = code.splitlines()
    repaired_lines = list(lines)
    changed = False
    for _ in range(20):
        try:
            ast.parse("\n".join(repaired_lines))
            return "\n".join(repaired_lines), ["STRIP_UNEXPECTED_TOP_LEVEL_INDENT"] if changed else []
        except IndentationError as exc:
            if "unexpected indent" not in str(exc).lower() or not exc.lineno:
                break
            line_index = exc.lineno - 1
            if line_index < 0 or line_index >= len(repaired_lines):
                break
            line = repaired_lines[line_index]
            if not line.startswith((" ", "\t")):
                break
            repaired_lines[line_index] = line.lstrip()
            changed = True
        except SyntaxError:
            break

    if changed:
        updated = "\n".join(repaired_lines)
        try:
            ast.parse(updated)
            return updated, ["STRIP_UNEXPECTED_TOP_LEVEL_INDENT"]
        except SyntaxError:
            return code, []
    return code, []


def _static_code_issues(code: str) -> list[str]:
    lowered = code.lower()
    issues: list[str] = []

    if not _contains_gurobi_import(code):
        issues.append("MISSING_GUROBIPY_IMPORT")
    if not re.search(r"\b(?:gp|gurobipy)\.Model\s*\(", code):
        if not _uses_embedded_lp_reader(code):
            issues.append("MISSING_MODEL_CREATION")
    if not _has_explicit_objective(code):
        if not _uses_embedded_lp_reader(code):
            issues.append("MISSING_SET_OBJECTIVE_CALL")
    if ".optimize(" not in code and "optimize()" not in lowered:
        issues.append("MISSING_OPTIMIZE_CALL")

    for pattern, reason in _PLACEHOLDER_PATTERNS:
        if pattern.search(code):
            issues.append(reason)

    for pattern, reason in _EXTERNAL_READ_PATTERNS:
        if pattern.search(code):
            issues.append(reason)

    if re.search(r"\bgp\.read\s*\(", code) and not _uses_embedded_lp_reader(code):
        issues.append("EXTERNAL_GUROBI_READ")

    return _dedupe_keep_order(issues)


def _contains_gurobi_import(code: str) -> bool:
    return bool(re.search(r"^\s*import\s+gurobipy\b|^\s*from\s+gurobipy\s+import\b", code, re.MULTILINE))


def _has_explicit_objective(code: str) -> bool:
    lowered = code.lower()
    if ".setobjective" in lowered:
        return True
    if "modelsense" in lowered and re.search(r"\bobj\s*=", code):
        return True
    return False


def _uses_embedded_lp_reader(code: str) -> bool:
    lowered = code.lower()
    return "gp.read" in lowered and "write_text" in lowered and "embedded_model.lp" in lowered


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
