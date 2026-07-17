"""基于 solver 的 OptiBench 评测辅助函数。"""

import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from evaluation.bench4opt.utils.lp_utils import add_gurobi_imports, extract_python_code


def get_optimal_value_from_model(
    code: str,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 360,
) -> Tuple[Optional[float], str]:
    import subprocess
    import sys

    temp_dir = tempfile.mkdtemp()

    if data:
        data_file_path = os.path.join(temp_dir, "data.json")
        with open(data_file_path, "w") as file:
            json.dump(data, file)
        code = code.replace("data.json", data_file_path)

    code = add_gurobi_imports(code)
    user_code_repr = repr(code)

    wrapper_template = """import sys
import json
import os

os.environ['GRB_LICENSE_FILE'] = os.environ.get('GRB_LICENSE_FILE', '')

try:
    import gurobipy as gp
    from gurobipy import GRB

    user_code = {user_code_repr}
    exec(user_code, globals(), locals())

    model_obj = None
    for var_name in ['model', 'm']:
        if var_name in locals():
            obj = locals()[var_name]
            if hasattr(obj, 'optimize') and hasattr(obj, 'ObjVal'):
                model_obj = obj
                break
        if var_name in globals():
            obj = globals()[var_name]
            if hasattr(obj, 'optimize') and hasattr(obj, 'ObjVal'):
                model_obj = obj
                break

    if model_obj is None:
        result = {{"status": -1, "optimal_value": None, "status_name": "MODEL_NOT_FOUND"}}
        print(json.dumps(result))
        sys.exit(1)

    try:
        model_obj.setParam("OutputFlag", 0)
    except Exception:
        pass

    try:
        if not hasattr(model_obj, 'status') or model_obj.status == GRB.LOADED:
            model_obj.optimize()

        if model_obj.status == GRB.OPTIMAL:
            result = {{"status": 2, "optimal_value": model_obj.ObjVal, "status_name": "OPTIMAL"}}
        elif model_obj.status == GRB.INFEASIBLE:
            result = {{"status": 3, "optimal_value": None, "status_name": "INFEASIBLE"}}
        elif model_obj.status == GRB.UNBOUNDED:
            result = {{"status": 5, "optimal_value": None, "status_name": "UNBOUNDED"}}
        elif model_obj.status == GRB.INF_OR_UNBD:
            result = {{"status": 4, "optimal_value": None, "status_name": "INF_OR_UNBD"}}
        else:
            result = {{"status": model_obj.status, "optimal_value": None, "status_name": "UNKNOWN"}}

        print(json.dumps(result))
    except Exception as exc:
        result = {{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(exc)}}
        print(json.dumps(result))
        sys.exit(1)
except Exception as exc:
    result = {{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(exc)}}
    print(json.dumps(result))
    sys.exit(1)
"""

    wrapper_code = wrapper_template.format(user_code_repr=user_code_repr)
    code_file = os.path.join(temp_dir, "model_code.py")
    with open(code_file, "w") as file:
        file.write(wrapper_code)

    try:
        try:
            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
            )

            output = result.stdout.strip()
            if not output:
                error_msg = result.stderr if result.stderr else "No output"
                return None, f"Code execution failed: {error_msg}"

            try:
                result_dict = json.loads(output)
            except json.JSONDecodeError:
                if '{"status":' in output:
                    json_start = output.rfind('{"status":')
                    result_dict = json.loads(output[json_start:])
                else:
                    return None, f"Could not parse output: {output}"

            status = result_dict.get("status")
            optimal_value = result_dict.get("optimal_value")
            status_name = result_dict.get("status_name", "UNKNOWN")

            if status == 2 and optimal_value is not None:
                return optimal_value, "Optimal solution found"
            return None, f"Optimization failed: {status_name}"
        except subprocess.TimeoutExpired:
            return None, f"Code execution timeout ({timeout}s)"
        except Exception as exc:
            return None, f"Error executing code: {exc}"
    finally:
        import shutil

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def get_optimal_value_from_lp(lp_path: str) -> Tuple[Optional[float], str]:
    try:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.read(lp_path, env)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            optimal_value = model.ObjVal
            env.close()
            return optimal_value, "Optimal solution found"
        if model.status == GRB.INF_OR_UNBD:
            env.close()
            return None, "Model is infeasible or unbounded"
        if model.status == GRB.INFEASIBLE:
            env.close()
            return None, "Model is infeasible"
        if model.status == GRB.UNBOUNDED:
            env.close()
            return None, "Model is unbounded"

        env.close()
        return None, f"Optimization failed with status: {model.status}"
    except Exception as exc:
        return None, f"Error reading/solving LP file: {exc}"


def evaluate_with_solver(
    code: str,
    gt_optimal_value: float,
    data: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-6,
    timeout: int = 360,
    verbose: bool = False,
) -> Dict[str, Any]:
    extracted_code = extract_python_code(code)

    if verbose:
        print(f"Extracted code length: {len(extracted_code)}")

    predicted_optimal_value, message = get_optimal_value_from_model(
        extracted_code,
        data,
        timeout=timeout,
    )

    result = {
        "predicted_optimal_value": predicted_optimal_value,
        "gt_optimal_value": gt_optimal_value,
        "message": message,
    }

    if predicted_optimal_value is None:
        result["success"] = False
        result["error"] = None
        return result

    error = abs(predicted_optimal_value - gt_optimal_value)
    result["error"] = error
    result["success"] = error <= tolerance

    if verbose:
        print(f"Predicted optimal value: {predicted_optimal_value}")
        print(f"GT optimal value: {gt_optimal_value}")
        print(f"Error: {error}")
        print(f"Tolerance: {tolerance}")
        print(f"Success: {result['success']}")

    return result


__all__ = [
    "evaluate_with_solver",
    "get_optimal_value_from_lp",
    "get_optimal_value_from_model",
]
