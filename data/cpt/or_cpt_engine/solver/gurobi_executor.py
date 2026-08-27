from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cpt_cleaner.utils.code_utils import extract_executable_code

RESULT_SENTINEL = "__OR_CPT_SOLVER_RESULT__"
_SOLVER_LOGGER = logging.getLogger("or_cpt_engine.solver")
_MAX_VARIABLE_VALUES = 500
_MAX_NONZERO_VARIABLE_VALUES = 200
_MAX_VARIABLE_DIAGNOSTICS = 800
_MAX_CONSTRAINT_DIAGNOSTICS = 800


def _log_solver_exec(result: dict[str, Any], log_context: dict[str, str] | None) -> None:
    ctx = log_context or {}
    parts = [
        f"status={result.get('status')}",
    ]
    for key in ("instance_id", "stage"):
        if ctx.get(key):
            parts.append(f"{key}={ctx[key]}")
    if result.get("objective_value") is not None:
        parts.append(f"objective_value={result['objective_value']}")
    if result.get("runtime_sec") is not None:
        parts.append(f"runtime_sec={result['runtime_sec']:.3f}")
    error = result.get("stderr") or result.get("error")
    if error and result.get("status") not in ("OPTIMAL",):
        parts.append(f"error={str(error)[:300]}")
    _SOLVER_LOGGER.info("solver_exec %s", " ".join(parts))


def execute_gurobi_code(
    code: str,
    *,
    timeout_seconds: int = 60,
    threads_per_process: int = 1,
    stdout_limit: int = 16000,
    stderr_limit: int = 16000,
    log_context: dict[str, str] | None = None,
) -> dict:
    executable_code = extract_executable_code(code)
    if not executable_code:
        result: dict[str, Any] = {"execution_status": "FAILED", "status": "EMPTY_CODE", "objective_value": None, "stdout": "", "stderr": "empty code"}
        _log_solver_exec(result, log_context)
        return result
    script = executable_code.rstrip() + "\n\n" + _helper_script(threads_per_process=threads_per_process)
    with tempfile.TemporaryDirectory(prefix="or_cpt_solver_") as temp_dir:
        script_path = Path(temp_dir) / "solver_check.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "execution_status": "TIMEOUT",
                "status": "TIMEOUT",
                "objective_value": None,
                "stdout": (exc.stdout or "")[:stdout_limit],
                "stderr": "Solver execution timed out",
                "runtime_sec": timeout_seconds,
            }
            _log_solver_exec(result, log_context)
            return result

    raw_stdout = completed.stdout
    raw_stderr = completed.stderr
    stdout = raw_stdout[:stdout_limit]
    stderr = raw_stderr[:stderr_limit]
    payload = _extract_payload(raw_stdout)
    if payload is None:
        result = {
            "execution_status": "FAILED" if completed.returncode else "FAILED",
            "status": "EXECUTION_ERROR" if completed.returncode else "RESULT_PARSE_ERROR",
            "objective_value": None,
            "stdout": stdout,
            "stderr": stderr,
            "runtime_sec": None,
        }
        _log_solver_exec(result, log_context)
        return result
    payload["stdout"] = stdout
    payload["stderr"] = stderr
    payload["execution_status"] = "SUCCESS" if payload.get("status") == "OPTIMAL" else "FAILED"
    _log_solver_exec(payload, log_context)
    return payload


def execute_gurobi_lp(
    lp_text: str,
    *,
    timeout_seconds: int = 60,
    threads_per_process: int = 1,
    stdout_limit: int = 16000,
    stderr_limit: int = 16000,
    log_context: dict[str, str] | None = None,
) -> dict:
    if not lp_text.strip():
        result: dict[str, Any] = {"execution_status": "FAILED", "status": "EMPTY_LP", "objective_value": None, "stdout": "", "stderr": "empty lp text"}
        _log_solver_exec(result, log_context)
        return result
    with tempfile.TemporaryDirectory(prefix="or_cpt_lp_solver_") as temp_dir:
        temp_path = Path(temp_dir)
        lp_path = temp_path / "model.lp"
        script_path = temp_path / "lp_solver_check.py"
        lp_path.write_text(lp_text, encoding="utf-8")
        script_path.write_text(_lp_reader_script(threads_per_process=threads_per_process), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "execution_status": "TIMEOUT",
                "status": "TIMEOUT",
                "objective_value": None,
                "stdout": (exc.stdout or "")[:stdout_limit],
                "stderr": "LP solver execution timed out",
                "runtime_sec": timeout_seconds,
            }
            _log_solver_exec(result, log_context)
            return result

    raw_stdout = completed.stdout
    raw_stderr = completed.stderr
    stdout = raw_stdout[:stdout_limit]
    stderr = raw_stderr[:stderr_limit]
    payload = _extract_payload(raw_stdout)
    if payload is None:
        result = {
            "execution_status": "FAILED",
            "status": "EXECUTION_ERROR" if completed.returncode else "RESULT_PARSE_ERROR",
            "objective_value": None,
            "stdout": stdout,
            "stderr": stderr,
            "runtime_sec": None,
        }
        _log_solver_exec(result, log_context)
        return result
    payload["stdout"] = stdout
    payload["stderr"] = stderr
    payload["execution_status"] = "SUCCESS" if payload.get("status") == "OPTIMAL" else "FAILED"
    _log_solver_exec(payload, log_context)
    return payload


def _extract_payload(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            try:
                return json.loads(line[len(RESULT_SENTINEL) :])
            except json.JSONDecodeError:
                return None
    return None


def _helper_script(*, threads_per_process: int = 1) -> str:
    threads = max(1, int(threads_per_process))
    return f"""
import json

def _emit(payload):
    print("{RESULT_SENTINEL}" + json.dumps(payload, ensure_ascii=True))

def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

def _safe_model_attr(model, attr):
    try:
        return _safe_float(getattr(model, attr))
    except Exception:
        try:
            return _safe_float(model.getAttr(attr))
        except Exception:
            return None

def _safe_bound(value):
    numeric = _safe_float(value)
    if numeric is None or abs(numeric) >= 1e50:
        return None
    return numeric

def _extract_solution(model):
    solution = {{
        "objective_value": _safe_float(model.ObjVal) if model.SolCount > 0 else None,
        "objective_recomputed": None,
        "variable_count": int(model.NumVars),
        "nonzero_variable_count": 0,
        "variable_values": {{}},
        "nonzero_variable_values": {{}},
        "variable_values_truncated": False,
        "nonzero_variable_values_truncated": False,
        "feasibility": {{}},
        "variable_diagnostics": [],
        "constraint_diagnostics": [],
        "variable_diagnostics_truncated": False,
        "constraint_diagnostics_truncated": False,
        "solution_summary": {{}},
    }}
    if model.SolCount <= 0:
        return solution

    try:
        solution["objective_recomputed"] = _safe_float(model.getObjective().getValue())
    except Exception:
        solution["objective_recomputed"] = solution["objective_value"]

    variable_values = {{}}
    nonzero_variable_values = {{}}
    variable_diagnostics = []
    nonzero_count = 0
    non_fixed_count = 0
    non_fixed_nonzero_count = 0
    non_fixed_at_lb_count = 0
    non_fixed_at_ub_count = 0
    finite_lb_count = 0
    finite_ub_count = 0
    binary_count = 0
    binary_active_count = 0
    for var in model.getVars():
        value = _safe_float(var.X)
        if value is None:
            continue
        lb = _safe_bound(getattr(var, "LB", None))
        ub = _safe_bound(getattr(var, "UB", None))
        vtype = str(getattr(var, "VType", ""))
        obj = _safe_float(getattr(var, "Obj", None))
        at_lb = lb is not None and abs(value - lb) <= 1e-6
        at_ub = ub is not None and abs(value - ub) <= 1e-6
        is_fixed = lb is not None and ub is not None and abs(ub - lb) <= 1e-9
        if lb is not None:
            finite_lb_count += 1
        if ub is not None:
            finite_ub_count += 1
        if vtype in ("B", "I"):
            binary_count += 1
            if abs(value) > 1e-6:
                binary_active_count += 1
        if not is_fixed:
            non_fixed_count += 1
            if abs(value) > 1e-9:
                non_fixed_nonzero_count += 1
            if at_lb:
                non_fixed_at_lb_count += 1
            if at_ub:
                non_fixed_at_ub_count += 1
        if len(variable_values) < {_MAX_VARIABLE_VALUES}:
            variable_values[var.VarName] = value
        elif not solution["variable_values_truncated"]:
            solution["variable_values_truncated"] = True
        if abs(value) > 1e-9:
            nonzero_count += 1
            if len(nonzero_variable_values) < {_MAX_NONZERO_VARIABLE_VALUES}:
                nonzero_variable_values[var.VarName] = value
            elif not solution["nonzero_variable_values_truncated"]:
                solution["nonzero_variable_values_truncated"] = True
        if len(variable_diagnostics) < {_MAX_VARIABLE_DIAGNOSTICS}:
            variable_diagnostics.append({{
                "name": var.VarName,
                "value": value,
                "lb": lb,
                "ub": ub,
                "vtype": vtype,
                "obj": obj,
                "at_lb": at_lb,
                "at_ub": at_ub,
                "is_nonzero": abs(value) > 1e-9,
                "is_fixed": is_fixed,
            }})
        elif not solution["variable_diagnostics_truncated"]:
            solution["variable_diagnostics_truncated"] = True
    solution["variable_values"] = variable_values
    solution["nonzero_variable_count"] = nonzero_count
    solution["nonzero_variable_values"] = nonzero_variable_values
    solution["variable_diagnostics"] = variable_diagnostics

    constraint_diagnostics = []
    binding_constraint_count = 0
    for constr in model.getConstrs():
        slack = _safe_float(getattr(constr, "Slack", None))
        is_binding = slack is not None and abs(slack) <= 1e-6
        if is_binding:
            binding_constraint_count += 1
        if len(constraint_diagnostics) < {_MAX_CONSTRAINT_DIAGNOSTICS}:
            constraint_diagnostics.append({{
                "name": getattr(constr, "ConstrName", ""),
                "sense": str(getattr(constr, "Sense", "")),
                "rhs": _safe_float(getattr(constr, "RHS", None)),
                "slack": slack,
                "is_binding": is_binding,
            }})
        elif not solution["constraint_diagnostics_truncated"]:
            solution["constraint_diagnostics_truncated"] = True
    solution["constraint_diagnostics"] = constraint_diagnostics

    feasibility = {{
        "constraint_violation": _safe_model_attr(model, "ConstrVio"),
        "bound_violation": _safe_model_attr(model, "BoundVio"),
        "integer_violation": _safe_model_attr(model, "IntVio"),
    }}
    numeric_violations = [value for value in feasibility.values() if value is not None]
    feasibility["max_violation"] = max(numeric_violations) if numeric_violations else None
    feasibility["is_feasible_within_tolerance"] = (
        feasibility["max_violation"] is None or feasibility["max_violation"] <= 1e-6
    )
    solution["feasibility"] = feasibility
    solution["solution_summary"] = {{
        "non_fixed_variable_count": non_fixed_count,
        "non_fixed_nonzero_variable_count": non_fixed_nonzero_count,
        "finite_lb_variable_count": finite_lb_count,
        "finite_ub_variable_count": finite_ub_count,
        "non_fixed_at_lb_count": non_fixed_at_lb_count,
        "non_fixed_at_ub_count": non_fixed_at_ub_count,
        "all_vars_at_lb": non_fixed_count > 0 and non_fixed_at_lb_count == non_fixed_count,
        "all_vars_at_ub": non_fixed_count > 0 and non_fixed_at_ub_count == non_fixed_count,
        "all_non_fixed_vars_zero": non_fixed_count > 0 and non_fixed_nonzero_count == 0,
        "binary_variable_count": binary_count,
        "binary_active_count": binary_active_count,
        "binding_constraint_count": binding_constraint_count,
        "constraint_count": int(model.NumConstrs),
        "binding_constraint_ratio": (binding_constraint_count / int(model.NumConstrs)) if int(model.NumConstrs) else None,
    }}
    return solution

try:
    import gurobipy as _gp
except Exception as exc:
    _emit({{"status": "IMPORT_ERROR", "objective_value": None, "stderr": str(exc)}})
    raise

_models = [value for value in globals().values() if isinstance(value, _gp.Model)]
if not _models:
    _emit({{"status": "MODEL_NOT_FOUND", "objective_value": None, "stderr": "No gurobipy.Model instance found"}})
else:
    _model = _models[0]
    try:
        try:
            _model.Params.Threads = {threads}
        except Exception:
            pass
        if _model.Status == _gp.GRB.LOADED:
            _model.optimize()
        _status = _model.Status
        _emit({{
            "status": "OPTIMAL" if _status == _gp.GRB.OPTIMAL else str(_status),
            "objective_value": float(_model.ObjVal) if _model.SolCount > 0 else None,
            "runtime_sec": float(_model.Runtime),
            "solution": _extract_solution(_model),
        }})
    except Exception as exc:
        _emit({{"status": "EXEC_ERROR", "objective_value": None, "stderr": str(exc)}})
        raise
"""


def _lp_reader_script(*, threads_per_process: int = 1) -> str:
    threads = max(1, int(threads_per_process))
    return f"""
import json

def _emit(payload):
    print("{RESULT_SENTINEL}" + json.dumps(payload, ensure_ascii=True))

def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

def _safe_model_attr(model, attr):
    try:
        return _safe_float(getattr(model, attr))
    except Exception:
        try:
            return _safe_float(model.getAttr(attr))
        except Exception:
            return None

def _safe_bound(value):
    numeric = _safe_float(value)
    if numeric is None or abs(numeric) >= 1e50:
        return None
    return numeric

def _extract_solution(model):
    solution = {{
        "objective_value": _safe_float(model.ObjVal) if model.SolCount > 0 else None,
        "objective_recomputed": None,
        "variable_count": int(model.NumVars),
        "nonzero_variable_count": 0,
        "variable_values": {{}},
        "nonzero_variable_values": {{}},
        "variable_values_truncated": False,
        "nonzero_variable_values_truncated": False,
        "feasibility": {{}},
        "variable_diagnostics": [],
        "constraint_diagnostics": [],
        "variable_diagnostics_truncated": False,
        "constraint_diagnostics_truncated": False,
        "solution_summary": {{}},
    }}
    if model.SolCount <= 0:
        return solution
    try:
        solution["objective_recomputed"] = _safe_float(model.getObjective().getValue())
    except Exception:
        solution["objective_recomputed"] = solution["objective_value"]
    variable_values = {{}}
    nonzero_variable_values = {{}}
    variable_diagnostics = []
    nonzero_count = 0
    non_fixed_count = 0
    non_fixed_nonzero_count = 0
    non_fixed_at_lb_count = 0
    non_fixed_at_ub_count = 0
    finite_lb_count = 0
    finite_ub_count = 0
    binary_count = 0
    binary_active_count = 0
    for var in model.getVars():
        value = _safe_float(var.X)
        if value is None:
            continue
        lb = _safe_bound(getattr(var, "LB", None))
        ub = _safe_bound(getattr(var, "UB", None))
        vtype = str(getattr(var, "VType", ""))
        obj = _safe_float(getattr(var, "Obj", None))
        at_lb = lb is not None and abs(value - lb) <= 1e-6
        at_ub = ub is not None and abs(value - ub) <= 1e-6
        is_fixed = lb is not None and ub is not None and abs(ub - lb) <= 1e-9
        if lb is not None:
            finite_lb_count += 1
        if ub is not None:
            finite_ub_count += 1
        if vtype in ("B", "I"):
            binary_count += 1
            if abs(value) > 1e-6:
                binary_active_count += 1
        if not is_fixed:
            non_fixed_count += 1
            if abs(value) > 1e-9:
                non_fixed_nonzero_count += 1
            if at_lb:
                non_fixed_at_lb_count += 1
            if at_ub:
                non_fixed_at_ub_count += 1
        if len(variable_values) < {_MAX_VARIABLE_VALUES}:
            variable_values[var.VarName] = value
        elif not solution["variable_values_truncated"]:
            solution["variable_values_truncated"] = True
        if abs(value) > 1e-9:
            nonzero_count += 1
            if len(nonzero_variable_values) < {_MAX_NONZERO_VARIABLE_VALUES}:
                nonzero_variable_values[var.VarName] = value
            elif not solution["nonzero_variable_values_truncated"]:
                solution["nonzero_variable_values_truncated"] = True
        if len(variable_diagnostics) < {_MAX_VARIABLE_DIAGNOSTICS}:
            variable_diagnostics.append({{
                "name": var.VarName,
                "value": value,
                "lb": lb,
                "ub": ub,
                "vtype": vtype,
                "obj": obj,
                "at_lb": at_lb,
                "at_ub": at_ub,
                "is_nonzero": abs(value) > 1e-9,
                "is_fixed": is_fixed,
            }})
        elif not solution["variable_diagnostics_truncated"]:
            solution["variable_diagnostics_truncated"] = True
    solution["variable_values"] = variable_values
    solution["nonzero_variable_count"] = nonzero_count
    solution["nonzero_variable_values"] = nonzero_variable_values
    solution["variable_diagnostics"] = variable_diagnostics

    constraint_diagnostics = []
    binding_constraint_count = 0
    for constr in model.getConstrs():
        slack = _safe_float(getattr(constr, "Slack", None))
        is_binding = slack is not None and abs(slack) <= 1e-6
        if is_binding:
            binding_constraint_count += 1
        if len(constraint_diagnostics) < {_MAX_CONSTRAINT_DIAGNOSTICS}:
            constraint_diagnostics.append({{
                "name": getattr(constr, "ConstrName", ""),
                "sense": str(getattr(constr, "Sense", "")),
                "rhs": _safe_float(getattr(constr, "RHS", None)),
                "slack": slack,
                "is_binding": is_binding,
            }})
        elif not solution["constraint_diagnostics_truncated"]:
            solution["constraint_diagnostics_truncated"] = True
    solution["constraint_diagnostics"] = constraint_diagnostics

    feasibility = {{
        "constraint_violation": _safe_model_attr(model, "ConstrVio"),
        "bound_violation": _safe_model_attr(model, "BoundVio"),
        "integer_violation": _safe_model_attr(model, "IntVio"),
    }}
    numeric_violations = [value for value in feasibility.values() if value is not None]
    feasibility["max_violation"] = max(numeric_violations) if numeric_violations else None
    feasibility["is_feasible_within_tolerance"] = (
        feasibility["max_violation"] is None or feasibility["max_violation"] <= 1e-6
    )
    solution["feasibility"] = feasibility
    solution["solution_summary"] = {{
        "non_fixed_variable_count": non_fixed_count,
        "non_fixed_nonzero_variable_count": non_fixed_nonzero_count,
        "finite_lb_variable_count": finite_lb_count,
        "finite_ub_variable_count": finite_ub_count,
        "non_fixed_at_lb_count": non_fixed_at_lb_count,
        "non_fixed_at_ub_count": non_fixed_at_ub_count,
        "all_vars_at_lb": non_fixed_count > 0 and non_fixed_at_lb_count == non_fixed_count,
        "all_vars_at_ub": non_fixed_count > 0 and non_fixed_at_ub_count == non_fixed_count,
        "all_non_fixed_vars_zero": non_fixed_count > 0 and non_fixed_nonzero_count == 0,
        "binary_variable_count": binary_count,
        "binary_active_count": binary_active_count,
        "binding_constraint_count": binding_constraint_count,
        "constraint_count": int(model.NumConstrs),
        "binding_constraint_ratio": (binding_constraint_count / int(model.NumConstrs)) if int(model.NumConstrs) else None,
    }}
    return solution

try:
    import gurobipy as gp
except Exception as exc:
    _emit({{"status": "IMPORT_ERROR", "objective_value": None, "stderr": str(exc)}})
    raise

try:
    model = gp.read("model.lp")
    model.Params.OutputFlag = 0
    model.Params.Threads = {threads}
    model.optimize()
    status = model.Status
    _emit({{
        "status": "OPTIMAL" if status == gp.GRB.OPTIMAL else str(status),
        "objective_value": float(model.ObjVal) if model.SolCount > 0 else None,
        "runtime_sec": float(model.Runtime),
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs),
        "solution": _extract_solution(model),
    }})
except Exception as exc:
    _emit({{"status": "EXEC_ERROR", "objective_value": None, "stderr": str(exc)}})
    raise
"""
