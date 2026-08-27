"""
基于solver的评估代码：使用gurobi计算optimal value并与ground truth对比
"""
import os
import json
import tempfile
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any, Optional, Tuple
from evaluation.bench4opt.utils.lp_utils import extract_python_code, process_code_for_lp, add_gurobi_imports
from evaluation.bench4opt.utils.python_executor import PythonExecutor


def get_optimal_value_from_model(code: str, data: Optional[Dict[str, Any]] = None, timeout: int = 360) -> Tuple[Optional[float], str]:
    """
    执行代码并获取optimal value
    
    Args:
        code: Python代码字符串
        data: 问题数据字典
        timeout: 超时时间（秒）
        
    Returns:
        (optimal_value, message): optimal value和消息
            optimal_value为None表示求解失败
    """
    import subprocess
    import sys
    
    temp_dir = tempfile.mkdtemp()
    
    # 如果有data字段，保存到文件并替换代码中的data.json路径
    if data:
        data_file_path = os.path.join(temp_dir, "data.json")
        # 保存数据到文件
        with open(data_file_path, "w") as f:
            json.dump(data, f)
        # 处理代码：替换data.json路径
        code = code.replace("data.json", data_file_path)
    
    # 确保代码有必要的导入
    code = add_gurobi_imports(code)
    
    # 在代码末尾添加获取optimal value的代码
    # 确保代码执行optimize()后返回optimal value
    # 注意：使用字符串格式化时，需要正确处理大括号
    # 使用exec()执行用户代码，避免缩进问题
    # 将代码转换为字符串，用三引号包裹，然后在exec中执行
    user_code_repr = repr(code)
    
    wrapper_template = """import sys
import json
import os

# 设置Gurobi环境变量以关闭日志输出（在导入gurobipy之前）
os.environ['GRB_LICENSE_FILE'] = os.environ.get('GRB_LICENSE_FILE', '')

try:
    import gurobipy as gp
    from gurobipy import GRB
    
    # 执行用户代码（使用exec避免缩进问题）
    user_code = {user_code_repr}
    exec(user_code, globals(), locals())
    
    # 获取模型对象（在用户代码执行后）
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
    
    # 关闭模型的输出（在执行optimize之前）
    try:
        model_obj.setParam("OutputFlag", 0)
    except:
        pass
    
    # 确保已经优化
    try:
        if not hasattr(model_obj, 'status') or model_obj.status == GRB.LOADED:
            model_obj.optimize()
        
        # 获取结果
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
        
        # 输出JSON（这是唯一的输出）
        print(json.dumps(result))
    except Exception as e:
        result = {{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(e)}}
        print(json.dumps(result))
        sys.exit(1)
except Exception as e:
    import traceback
    result = {{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(e)}}
    print(json.dumps(result))
    sys.exit(1)
"""
    
    wrapper_code = wrapper_template.format(user_code_repr=user_code_repr)
    
    # 创建临时文件保存代码
    code_file = os.path.join(temp_dir, "model_code.py")
    with open(code_file, "w") as f:
        f.write(wrapper_code)
    
    try:
        # 使用subprocess执行代码并设置超时
        try:
            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir
            )
            
            # 解析输出
            output = result.stdout.strip()
            if not output:
                error_msg = result.stderr if result.stderr else "No output"
                return None, f"Code execution failed: {error_msg}"
            
            try:
                result_dict = json.loads(output)
                status = result_dict.get("status")
                optimal_value = result_dict.get("optimal_value")
                status_name = result_dict.get("status_name", "UNKNOWN")
                
                # GRB.OPTIMAL == 2
                if status == 2 and optimal_value is not None:
                    return optimal_value, "Optimal solution found"
                else:
                    return None, f"Optimization failed: {status_name}"
            except json.JSONDecodeError:
                # 尝试从输出中提取JSON（可能在最后）
                if '{"status":' in output:
                    json_start = output.rfind('{"status":')
                    try:
                        json_str = output[json_start:]
                        result_dict = json.loads(json_str)
                        status = result_dict.get("status")
                        optimal_value = result_dict.get("optimal_value")
                        status_name = result_dict.get("status_name", "UNKNOWN")
                        if status == 2 and optimal_value is not None:
                            return optimal_value, "Optimal solution found"
                        else:
                            return None, f"Optimization failed: {status_name}"
                    except json.JSONDecodeError:
                        pass
                return None, f"Could not parse output: {output}"
                
        except subprocess.TimeoutExpired:
            return None, f"Code execution timeout ({timeout}s)"
        except Exception as e:
            return None, f"Error executing code: {str(e)}"
            
    finally:
        # 清理
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def get_optimal_value_from_lp(lp_path: str) -> Tuple[Optional[float], str]:
    """
    从LP文件获取optimal value
    
    Args:
        lp_path: LP文件路径
        
    Returns:
        (optimal_value, message): optimal value和消息
    """
    try:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.read(lp_path, env)
        
        # 优化模型
        model.optimize()
        
        # 检查状态并获取optimal value
        if model.status == GRB.OPTIMAL:
            optimal_value = model.ObjVal
            env.close()
            return optimal_value, "Optimal solution found"
        elif model.status == GRB.INF_OR_UNBD:
            env.close()
            return None, "Model is infeasible or unbounded"
        elif model.status == GRB.INFEASIBLE:
            env.close()
            return None, "Model is infeasible"
        elif model.status == GRB.UNBOUNDED:
            env.close()
            return None, "Model is unbounded"
        else:
            env.close()
            return None, f"Optimization failed with status: {model.status}"
    except Exception as e:
        return None, f"Error reading/solving LP file: {str(e)}"


def evaluate_with_solver(
    code: str,
    gt_optimal_value: float,
    data: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-6,
    timeout: int = 360,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    基于solver的评估：执行代码获取optimal value并与ground truth对比
    
    Args:
        code: Python代码字符串
        gt_optimal_value: ground truth的optimal value
        data: 问题数据字典（可选，如果代码需要从data.json读取数据）
        tolerance: 精度容差（默认1e-6）
        timeout: 超时时间（秒）
        verbose: 是否打印详细信息
        
    Returns:
        评估结果字典：
        {
            "success": bool,  # 是否成功（optimal value匹配）
            "predicted_optimal_value": float or None,  # 预测的optimal value
            "gt_optimal_value": float,  # ground truth的optimal value
            "error": float or None,  # 误差
            "message": str,  # 消息
        }
    """
    # 提取Python代码
    extracted_code = extract_python_code(code)
    
    if verbose:
        print(f"Extracted code length: {len(extracted_code)}")
    
    # 执行代码获取optimal value
    predicted_optimal_value, message = get_optimal_value_from_model(
        extracted_code, data, timeout=timeout
    )
    
    result = {
        "predicted_optimal_value": predicted_optimal_value,
        "gt_optimal_value": gt_optimal_value,
        "message": message,
    }
    
    # 如果预测失败，返回失败结果
    if predicted_optimal_value is None:
        result["success"] = False
        result["error"] = None
        return result
    
    # 计算误差
    error = abs(predicted_optimal_value - gt_optimal_value)
    result["error"] = error
    
    # 判断是否匹配（精度为tolerance）
    if error <= tolerance:
        result["success"] = True
    else:
        result["success"] = False
    
    if verbose:
        print(f"Predicted optimal value: {predicted_optimal_value}")
        print(f"GT optimal value: {gt_optimal_value}")
        print(f"Error: {error}")
        print(f"Tolerance: {tolerance}")
        print(f"Success: {result['success']}")
    
    return result

