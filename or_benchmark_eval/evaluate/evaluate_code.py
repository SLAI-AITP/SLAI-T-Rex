import json
import os
import tempfile
import gurobipy as gp
import time
import networkx as nx
import numpy as np
from typing import Optional

from evaluate._utils import (
    wltest_coloring_two,
    graph_generator,
    check_color_equivalence, derive_adjacency
)
from evaluate.python_executor import PythonExecutor

# import SD checking tools
from evaluate._utils import check_symmetric_decomposable, check_wl_determinable

# import lp generation tools
from evaluate._lp_utils import ensure_imports, process_code_for_lp

import evaluate.equivalence_check as equivcheck


def read_sample(problem_id):
    with open(f"data/sample_{problem_id}.json", "r") as f:
        data = json.load(f)
    return data


def convert_to_lp(
    model_name: str, data_dir: str, code: str, data_path: str, problem_id: Optional[int] = None, verbose: bool = False
):
    """
    将Gurobi Python代码转换为LP文件

    Args:
        code: Gurobi Python代码
        data_path: 样本数据JSON文件路径
        problem_id: 样本ID，用于生成文件名
        verbose: 是否打印详细信息
    Returns:
        tuple: (是否成功, LP文件内容或错误信息, LP文件路径)
    """
    # 创建临时目录存储LP文件
    data_dir_name = data_dir.split("/")[-1]
    temp_dir = f"temp_lp/{model_name}_{data_dir_name}_lp/"
    os.makedirs(temp_dir, exist_ok=True)
    if problem_id is not None:
        lp_file_name = f"{problem_id}_model_{data_path.split('/')[-1].split('.')[0]}.lp"
        lp_file_path = os.path.join(temp_dir, lp_file_name)
    else:
        lp_file_path = os.path.join(temp_dir, "model_test.lp")
    
    # 如果lp文件存在，删除它
    if os.path.exists(lp_file_path):
        os.remove(lp_file_path)

    if verbose:
        print(f"Processing code for to write LP file at: {lp_file_path}")


    # 首先提取Python代码
    from evaluate._lp_utils import extract_python_code
    extracted_code = extract_python_code(code)
    code_for_lp = process_code_for_lp(extracted_code, data_path, lp_file_path)
    if verbose:
        print(f"*" * 10, "Processed code", "*" * 10)
        #print(code_for_lp)
    # This is not necessary for RL training
    code_for_lp = ensure_imports(code_for_lp)

    os.makedirs(f"temp_code/{model_name}/", exist_ok=True)
    with open(f"temp_code/{model_name}/temp_exec_{problem_id}.py", "w", encoding="utf-8") as f:
        f.write(code_for_lp)

    import subprocess
    # 使用conda环境中的python
    result = subprocess.run(["conda", "run", "-n", "vllm", "python", f"temp_code/{model_name}/temp_exec_{problem_id}.py"], capture_output=True, text=True)
    #print("Output:", result.stdout)
    #print("Error:", result.stderr)  # 查看错误信息

    if os.path.exists(lp_file_path):
        with open(lp_file_path, "r") as f:
            lp_content = f.read()
        return True, lp_content, lp_file_path
    else:
        # 如果LP文件不存在，返回stderr错误信息
        error_msg = result.stderr.strip() if result.stderr.strip() else "Unknown error"
        return False, "LP file was not created due to error: " + error_msg, None




def store_sample(
    code,
    problem_id,
    reference_lp_path,
    problem_class,
    domain,
    is_equivalent,
    equivalence_error,
    output_dir="./outputs/bench4opt_samples",
):
    """
    存储优化问题样本数据为JSON格式

    Args:
        code: Gurobi代码内容
        problem_id: 样本ID (如: LP_001, MILP_002)
        path: 样本文件路径
        problem_class: 问题类别 (如: Network Flow)
        domain: 应用领域 (如: Health Care, Transportation)
        output_dir: 输出目录，默认为'data_output'
    """
    # 创建样本数据字典
    sample_data = {
        "sample_id": problem_id,
        "sample_code": code,
        "sample_path": f"data/sample_{problem_id}.json",
        "reference_lp_path": reference_lp_path,
        "problem_type": problem_id.split("_")[0],  # 从ID提取问题类型(LP/MILP)
        "is_equivalent": is_equivalent,
        "equivalence_error": equivalence_error,
        "problem_class": problem_class,
        "domain": domain,
        # "code":,
        # "reward":,
        # "verification":
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件路径，与read_sample函数的读取路径保持一致
    output_file = os.path.join(output_dir, f"sample_{problem_id}.json")

    # 保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

    print(f"Sample {problem_id} saved to {output_file}")




def evaluate_code_re(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    评估LLM生成的Gurobi Python代码，使用equivalence_check.py中的check_lp_equivalence进行判定

    Args:
        code: LLM生成的Gurobi Python代码
        data_path: 样本数据路径
        reference_lp_path: 参考LP文件路径
        verbose: 是否打印详细信息
        problem_id: 可选的问题ID，用于生成文件名

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: 代码执行结果字典
            equivalence_eval_result: 等价性检查结果字典
    """
    # 检查参考LP文件是否存在
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 转换代码到LP文件

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 使用equivalence_check.py中的check_lp_equivalence进行等价性检查
    start_time = time.time()
    is_equivalent, error_msg,time_info = equivcheck.check_lp_equivalence(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time,
        "step_time": time_info
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    
    equivalence_eval_result = {
        "success": is_equivalent,
        "message": error_msg
    }

    return code_eval_result, equivalence_eval_result, equivalence_check_time



def evaluate_code_solver(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    评估LLM生成的Gurobi Python代码，使用solver求解行判定

    Args:
        code: LLM生成的Gurobi Python代码
        data_path: 样本数据路径
        reference_lp_path: 参考LP文件路径
        verbose: 是否打印详细信息
        problem_id: 可选的问题ID，用于生成文件名

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: 代码执行结果字典
            equivalence_eval_result: 等价性检查结果字典
    """
    # 检查参考LP文件是否存在
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 转换代码到LP文件

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 使用equivalence_check.py中的check_lp_equivalence进行等价性检查
    start_time = time.time()
    is_equivalent,msg,eval_time = solver_evaluation(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    
    equivalence_eval_result = {
        "success": is_equivalent,
        "message": msg
    }

    return code_eval_result, equivalence_eval_result, equivalence_check_time


def solver_evaluation(lp_path1, lp_path2):
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    try:
        model1 = gp.read(lp_path1,env)
        model2 = gp.read(lp_path2,env)
    except Exception as e:
        return False, f"Error reading LP file: {e}", 0
        
    start_time = time.time()
    try:
        model1.optimize()
    except Exception as e:
        return False, f"Error optimizing model: {e}", 0

    try:
        model2.optimize()
    except Exception as e:
        return False, f"Error optimizing model: {e}", 0
    
    # 状态映射字典
    status_map = {
        gp.GRB.LOADED: "Model loaded but not solved",
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.INF_OR_UNBD: "infeasible or unbounded",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.CUTOFF: "cutoff",
        gp.GRB.ITERATION_LIMIT: "iteration limit",
        gp.GRB.NODE_LIMIT: "node limit",
        gp.GRB.TIME_LIMIT: "time limit",
        gp.GRB.SOLUTION_LIMIT: "solution limit",
        gp.GRB.INTERRUPTED: "interrupted",
        gp.GRB.NUMERIC: "numeric",
        gp.GRB.SUBOPTIMAL: "suboptimal",
        gp.GRB.INPROGRESS: "in progress",
        gp.GRB.USER_OBJ_LIMIT: "user objective limit"
    }
    
    # 检查参考模型状态
    if model1.status != gp.GRB.OPTIMAL:
        return False, f"reference model status: {status_map.get(model1.status, 'unknown status')}", 0
    
    try:
        obj1 = model1.getAttr("ObjVal")
    except Exception as e:
        return False, f"cannot get reference model objective value: {e}", 0
    
    # 检查待评估模型状态
    if model2.status == gp.GRB.OPTIMAL:
        try:
            obj2 = model2.getAttr("ObjVal")
            eval_result = abs(obj1 - obj2) < 1e-6
            msg = "both models are optimal"
        except Exception as e:
            return False, f"optimal but cannot get objective value: {e}", 0
    else:
        eval_result = False
        msg = f"evaluated model status: {status_map.get(model2.status, 'unknown status')}"
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    env.dispose()
    return eval_result, msg, eval_time



def evaluate_code_ged(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    评估LLM生成的Gurobi Python代码，使用solver求解行判定

    Args:
        code: LLM生成的Gurobi Python代码
        data_path: 样本数据路径
        reference_lp_path: 参考LP文件路径
        verbose: 是否打印详细信息
        problem_id: 可选的问题ID，用于生成文件名

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: 代码执行结果字典
            equivalence_eval_result: 等价性检查结果字典
    """
    # 检查参考LP文件是否存在
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 转换代码到LP文件

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # 使用equivalence_check.py中的check_lp_equivalence进行等价性检查
    start_time = time.time()
    ged = calculate_code_ged(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    


    return code_eval_result, ged, equivalence_check_time


def calculate_ged(G1, G2):
    """
    计算两个图之间的归一化图编辑距离
    
    Args:
        G1, G2: networkx图对象，具有节点和边的特征
        
    Returns:
        float: 归一化后的图编辑距离
    """
    def node_subst_cost(node1_data, node2_data):
        """节点替换成本：特征向量的L1距离"""
        return np.sum(np.abs(node1_data['feature'] - node2_data['feature']))
    
    def edge_subst_cost(edge1_data, edge2_data):
        """边替换成本：特征向量的L1距离"""
        return np.sum(np.abs(edge1_data['feature'] - edge2_data['feature']))
    
    def node_del_cost(node_data):
        """节点删除成本：固定为1"""
        return 1.0
    
    def edge_del_cost(edge_data):
        """边删除成本：固定为1"""
        return 1.0
    
    def node_ins_cost(node_data):
        """节点插入成本：固定为1"""
        return 1.0
    
    def edge_ins_cost(edge_data):
        """边插入成本：固定为1"""
        return 1.0
    
    # 计算原始图编辑距离
    ged = nx.graph_edit_distance(G1, G2,
                               node_subst_cost=node_subst_cost,
                               edge_subst_cost=edge_subst_cost,
                               node_del_cost=node_del_cost,
                               edge_del_cost=edge_del_cost,
                               node_ins_cost=node_ins_cost,
                               edge_ins_cost=edge_ins_cost)
    
    # 归一化处理：使用两个图的最大规模进行归一化
    max_size = max(len(G1.nodes) + len(G1.edges), len(G2.nodes) + len(G2.edges))
    normalized_ged = ged / max_size if max_size > 0 else 0
    
    return 1-normalized_ged

def calculate_code_ged(lp_path1,lp_path2):
    with gp.Env(empty=True) as env:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model1 = gp.read(lp_path1,env)
        model2 = gp.read(lp_path2,env)
        
        # 统一处理上下界约束
        model1 = equivcheck.convert_boundary(model1)
        model1.write("model1_converted.lp")
        model2 = equivcheck.convert_boundary(model2)
        model2.write("model2_converted.lp")


        # Generate bipartite graphs
        G1 = equivcheck.generate_bipartite(model1)
        G2 = equivcheck.generate_bipartite(model2)
        # Calculate Graph Edit Distance
        dist = calculate_ged(G1, G2)
        #print(dist)
    env.close()
    return dist
