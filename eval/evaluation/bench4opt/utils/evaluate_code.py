import json
import os
import gurobipy as gp
import numpy as np
from gurobipy import GRB
import networkx as nx
import hashlib
from itertools import chain
import time


def float_array_to_str(arr, precision=6):
    """
    将浮点数组转为字符串, 保留一定小数位, 例如 [0.123456, 0.98765] -> "0.1235,0.9877"
    """
    return ",".join([f"{x:.{precision}f}" for x in arr.flatten()])

def hash_multiset(color, color_pairs):
    """
    对多重集合 {(x1,y1), (x2,y2), ...} 做哈希。
    - 顺序无关(排序)
    - 重复计数敏感(相同元素出现多次就会重复出现)
    """
    # 每个 (color, edge_attr) -> "color|edge_str"
    pair_strs = []
    for (c, e) in color_pairs:
        pair_strs.append(f"{c}|{e}")
    # 排序后拼接
    pair_strs.sort()
    combined = "|".join(pair_strs)
    combined = f"{color}|{combined}"
    return hashlib.md5(combined.encode()).hexdigest()


def wl_test_nx(G):
    """
    对networkx形式的图进行WL-Test
    
    Args:
        G: networkx图，节点特征为长度为2的numpy数组，边特征为长度为1的numpy数组
        
    Returns:
        colors: 最终的节点颜色字典
        n_iter: 实际迭代次数
    """

    max_iter = len(G.nodes)
    
    # 初始化节点颜色，将节点特征转换为字符串作为初始颜色
    for node, feat in nx.get_node_attributes(G, 'feature').items():
        G.nodes[node]['color'] = float_array_to_str(feat)
    
    # 初始化边颜色，将边权重转换为字符串
    for u, v, data in G.edges(data=True):
        G[u][v]['color'] = float_array_to_str(data['feature'])
    
    # 记录每轮迭代后的颜色分布
    prev_partition = {}
    n_iter = 0
    
    # 迭代直到收敛或达到最大迭代次数
    for i in range(max_iter):
        new_partition = {}
        # 对每个节点
        for node in G.nodes:
            # 获取邻居的颜色和边特征对
            neighbor_color_pairs = []
            for neighbor in G.neighbors(node):
                # 获取邻居颜色和边颜色
                neighbor_color = G.nodes[neighbor]['color']
                edge_color = G[node][neighbor]['color']
                # 将邻居颜色和边颜色组合
                neighbor_color_pairs.append((neighbor_color, edge_color))
            
            # 使用hash_multiset计算新的颜色
            node_color = G.nodes[node]['color']
            hashed_color = hash_multiset(node_color,neighbor_color_pairs)
            if hashed_color not in new_partition.keys():
                new_partition[hashed_color] = [node]
            else:
                new_partition[hashed_color].append(node)
        
        # 将颜色更新到图中
        for hashed_color, nodes in new_partition.items():
            for node in nodes:
                # 更新节点的颜色属性
                G.nodes[node]['color'] = hashed_color
        
        # 检查是否收敛(颜色类别数量不再增加)
        if len(new_partition) == len(prev_partition):
            # 如果相同，直接返回
            return new_partition, n_iter
        else:
            # 如果不同，更新 prev_partition
            prev_partition = new_partition.copy()
            n_iter += 1
        
    return new_partition, n_iter

def convert_cons2boundary(model):
    nvars = model.NumVars
    vars = model.getVars()
    existing_constrs = model.getConstrs()
    cons_to_remove = []
    for n in range(nvars):
        var = vars[n]
        #取出只包括var的cons
        existing_constrs_only_var = []
        for constr in existing_constrs:
            other_vars = [v for v in vars if v is not var and model.getCoeff(constr, v) != 0]
            if not other_vars and model.getCoeff(constr, var) != 0:
                existing_constrs_only_var.append(constr)
        #print(f'existing_constrs_only_var: {existing_constrs_only_var}')
        var_lb =[]
        var_ub=[]
        for constr in existing_constrs_only_var:
            if constr.Sense == GRB.GREATER_EQUAL:
                var_lb.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            elif constr.Sense == GRB.LESS_EQUAL:
                var_ub.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            elif constr.Sense == GRB.EQUAL:
                var_lb.append(constr.RHS/model.getCoeff(constr, var))
                var_ub.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            else:
                raise NotImplementedError('Only implemented >=, <=, and =')
        print(f'var_lb: {var_lb}, var_ub: {var_ub}')
        # 如果var_lb和var_ub存在，则将var的LB和UB设置为var_lb和var_ub中的最小值和最大值
        if len(var_ub) > 0:
            var.setAttr(GRB.Attr.UB, min(var_ub))
        else:
            var.ub = var.ub
        if len(var_lb) > 0:
            var.setAttr(GRB.Attr.LB, max(var_lb))
        else:
            var.lb = var.lb
        
    for constr in cons_to_remove:
        model.remove(constr)
    model.update()

    return model

def convert_boundary(model):
    nvars = model.NumVars
    vars = model.getVars()
    # Extract variable names from vars
    var_names = [var.VarName for var in vars]
    
    for n in range(nvars):
        var = vars[n]
        obj_coeff = var.Obj
        lb = var.LB
        ub = var.UB
        
        existing_constrs = model.getConstrs()
        has_lb_constr = any(
            abs(model.getCoeff(constr, var) - 1.0) < 1e-6 and 
            abs(constr.RHS - lb) < 1e-6 and 
            constr.Sense == ">" 
            for constr in existing_constrs
        )
        has_ub_constr = any(
            abs(model.getCoeff(constr, var) - 1.0) < 1e-6 and 
            abs(constr.RHS - ub) < 1e-6 and 
            constr.Sense == "<" 
            for constr in existing_constrs
        )
        
        # 只有在不存在相同约束时才添加新约束
        if not has_lb_constr:
            model.addConstr(var >= lb)
        if ub != float('inf'):
            model.addConstr(var <= ub)
            
        model.update()

    # 删除 Gurobi 变量的默认上下界
    for var in vars:
        var.setAttr(GRB.Attr.LB, -GRB.INFINITY)
        var.setAttr(GRB.Attr.UB, GRB.INFINITY)

    model.update()
    return model

def generate_bipartite(model):
    # 创建图
    G = nx.Graph()

    # 创建变量节点，提取变量节点特征
    nvars = model.NumVars
    vars = model.getVars()
    var_names = [var.VarName for var in vars]
    for n, var in enumerate(vars):
        var = vars[n]
        obj_coeff = var.Obj
        if var.VType == GRB.BINARY:
            bin = 1.0
        elif var.VType==GRB.INTEGER:
            bin = 2.0
        else:
            bin = 0.0
        # 增加节点
        feature = [float(obj_coeff), bin, var.LB, var.UB]
        G.add_node(n, feature = np.array(feature, dtype=np.float32))
    
    # 创建约束节点，创建边，提取约束节点特征与边特征
    ncons = model.NumConstrs
    cons = model.getConstrs()

    for m, con in enumerate(cons):
        sense = con.Sense
        rhs = con.RHS
        
        if sense == GRB.EQUAL:
            cons_sense = 0.0
        elif sense == GRB.LESS_EQUAL:
            cons_sense = -1.0
        elif sense == GRB.GREATER_EQUAL:
            cons_sense = 1.0
        else:
            raise NotImplementedError('Only implemented feature >=,<=, and, =')
        
        feature = [float(rhs), cons_sense]
        G.add_node(nvars + m, feature = np.array(feature, dtype=np.float32))
        
        # 获取约束中的所有变量
        row = model.getRow(con)
        
        # 遍历约束中的变量
        for i in range(row.size()):
            # 获取变量在vars列表中的索引
            var_idx = var_names.index(row.getVar(i).varName)
            # 获取系数
            coeff = row.getCoeff(i)
            # 添加边,边的特征为系数
            G.add_edge(var_idx, nvars + m, feature=np.array([coeff], dtype=np.float32))

    return G


def check_wl_determinable(same_color_partition):
    """
    检查是否满足WL-determinable条件
    """
    unique_set = [v[0] for k, v in same_color_partition.items() if len(v) == 1]
    if len(unique_set) == len(same_color_partition.keys()):
        return True
    else:
        return False



def find_group(node, partition):
    for group, nodes in partition.items():
        if node in nodes:
            return group  # Return the key of the group
    return f'{node} not found in {partition}!'  # Return None if node is not found

def get_nbhd(node_index_list, unique_set, Adj):
    """获取节点列表的邻居节点"""
    nbhd = []
    # 对于每个节点
    for i in node_index_list:
        # 获取该节点的所有邻居（Adj[i]中值为1的位置）
        neighbors = np.where(Adj[i] > 0)[0]
        for j in neighbors:
            if j not in unique_set:
                nbhd.append(int(j))
    return nbhd

def get_cluster(same_color_partition,unique_set,partition_index,sub_graph_nodes,Adj):
    labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
    for i in same_color_partition[partition_index]:
        if i not in labeled_nodes:
            sub_graph_i = []
            sub_graph_i.append(i)
            sub_graph_nodes[i] = list(set(sub_graph_i))
        else:
            continue
    previous_sum = 0
    
    while sum([len(v) for k,v in sub_graph_nodes.items() if k in same_color_partition[partition_index]]) -previous_sum >0:
        previous_sum = sum([len(v) for k,v in sub_graph_nodes.items()if k in same_color_partition[partition_index]])
        for i in same_color_partition[partition_index]:
            if i not in labeled_nodes:
                sub_graph_nodes[i] = sub_graph_nodes[i]+get_nbhd(sub_graph_nodes[i],unique_set,Adj)
                sub_graph_nodes[i] = set(sub_graph_nodes[i])
                sub_graph_nodes[i] = list(set(sub_graph_nodes[i]))
    return sub_graph_nodes

def derive_subgraphs(same_color_partition,Adj):
    """
    根据节点颜色划分子图
    """
    unique_set = [v[0] for k, v in same_color_partition.items() if len(v) == 1]
    same_color_partition = {k: v for k, v in same_color_partition.items() if len(v) > 1}
    sub_graph_nodes = {}
    multiplicity_nodes = [item for item in list(range(len(Adj))) if item not in unique_set]
    condition = True
    state = 'uncheck'
    while condition:
        labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
        for i in multiplicity_nodes:
            if i not in labeled_nodes:
                partition_index = find_group(i,same_color_partition)
                sub_graph_nodes = get_cluster(same_color_partition,unique_set,partition_index,sub_graph_nodes,Adj)
        if sum([len(v) for v in sub_graph_nodes.values()]) < len(Adj)-len(unique_set):
            condition = True
            state = 'continue'
        elif sum([len(v) for v in sub_graph_nodes.values()]) == len(Adj) - len(unique_set):
            condition = False
            state = 'complete'
        else:
            condition = False
            state = 'error'
        
    for i in list(sub_graph_nodes.keys()):
        sub_graph_nodes[i] = set(sub_graph_nodes[i])
    return sub_graph_nodes,state


def check_distinct(sub_graph_nodes,stable_partition):
    for k,v in sub_graph_nodes.items():
        for p,q in stable_partition.items():
            if len(v.intersection(q)) > 1:
                print(f'{v}intersection{q} greater than 1')
                return False
    return True


def check_graph_disjoint(sub_graph_nodes):
    for i in sub_graph_nodes:
        for j in sub_graph_nodes:
            if i != j:
                if len(sub_graph_nodes[i].intersection(sub_graph_nodes[j])) != 0:
                    return False
    return True

def connectivity_check(dict1,dict2,Adj):
    for i in dict1:
        for j in dict2:
            if Adj[i][j] != 0:
                return True
    return False

def check_graph_disconnect(sub_graph_nodes,Adj):
    for i in sub_graph_nodes:
        for j in sub_graph_nodes:
            if i != j:
                if connectivity_check(sub_graph_nodes[i],sub_graph_nodes[j],Adj):
                    return False
    return True

def check_symmetric_decomposable(same_color_partition,Adj):
    """
    检查是否满足对称可分解条件
    """
    sub_graph_nodes,state = derive_subgraphs(same_color_partition,Adj)
    same_color_partition = {k: v for k, v in same_color_partition.items() if len(v) > 1}

    if state == 'error':
        #print('False: Graph is not symmetric decomposable due to error.')
        return False, "Due to decompose error"
    if not check_distinct(sub_graph_nodes,same_color_partition):
        return False, "Due to distinctness."
    elif check_graph_disjoint(sub_graph_nodes):
        if check_graph_disconnect(sub_graph_nodes,Adj):
            #print('True: Graph is symmetric decomposable.')
            return True, 'Symmetric Decomposable.'
        else:
            #print('False: Graph is not symmetric decomposable due to connectivity.')
            return False, "Due to connectivity."
    else:
        #print('False: Graph is not symmetric decomposable due to disjointness.')
        return False, "Due to disjointness."

def check_sufficient_conditions(same_color_partition,Adj):
    """
    检查是否满足充分条件
    """
    if check_wl_determinable(same_color_partition):
        sufficient_cond = True
        msg = "WL-determinable"
    else:
        check_sym_decom,msg = check_symmetric_decomposable(same_color_partition,Adj)
        if check_sym_decom:
            sufficient_cond = True
        else:
            sufficient_cond = False
    return sufficient_cond, msg

# 评估两个样本是否满足sufficient conditions
def check_sufficiency_two(same_color_partition1,Adj1, same_color_partition2,Adj2):
    check_sufficient1,msg1 = check_sufficient_conditions(same_color_partition1,Adj1)
    check_sufficient2,msg2 = check_sufficient_conditions(same_color_partition2,Adj2)
    msg = {'msg1':msg1, 'msg2':msg2}
    if check_sufficient1 and check_sufficient2:
        sufficient = True
    else:
        sufficient = False
    return sufficient, msg

def check_lp_equivalence(lp_path1, lp_path2, verbose: bool = False):
    """
    检查两个LP文件是否等价

    Args:
        lp_path1: 第一个LP文件路径
        lp_path2: 第二个LP文件路径
        verbose: 是否打印详细信息
    Returns:
        tuple: (是否等价, 错误信息)
    """

    info = {}
    time_info = {}

    with gp.Env(empty=True) as env:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model1 = gp.read(lp_path1,env)
        try:
            model2 = gp.read(lp_path2,env)
        except Exception as e:
            return False, f"Error reading LP file: {e}",time_info
        
        # 统一处理上下界约束
        model1 = convert_cons2boundary(model1)
        model1.write('model_ref.lp')
        model2 = convert_cons2boundary(model2)
        model2.write('model_test.lp')

        # 检查基本属性
        if model1.NumVars != model2.NumVars:
            info['var_num_check'] = False
            return False, info, time_info
        else:
            info['var_num_check'] = True
            
        if model1.NumConstrs != model2.NumConstrs:
            info['cons_num_check'] = False
            return False, info, time_info
        else:
            info['cons_num_check'] = True

        # 生成图并做WL-Test
        start_time = time.time()
        G1 = generate_bipartite(model1)
        partition1, iter1 = wl_test_nx(G1)        
        G2 = generate_bipartite(model2)
        partition2, iter2 = wl_test_nx(G2)
        end_time = time.time()
        wl_coloring_time = end_time - start_time
        time_info['wl_coloring_time'] = wl_coloring_time
        if not (partition1.keys() == partition2.keys()):
            info['wl_check'] = False
            return False, info, time_info
        else:
            # 检查partition内节点数量是否一致
            #start_time = time.time()
            #for key in partition1.keys():
            #    if len(partition1[key]) != len(partition2[key]):
            #        info['color match'] = False
            #        return False, info
            #    else:
            #        continue
            #end_time = time.time()
            #coloring_mathing_time = end_time - start_time
            #time_info['coloring_mathing_time'] = coloring_mathing_time
            #检查是否满足充分条件
            start_time = time.time()
            Adj1 = nx.adjacency_matrix(G1).toarray()
            Adj2 = nx.adjacency_matrix(G2).toarray()
            check_sufficiency, sufficient_msg = check_sufficiency_two(partition1, Adj1,partition2,Adj2)
            end_time = time.time()
            sufficient_check_time = end_time - start_time
            time_info['sufficient_check_time'] = sufficient_check_time
            if check_sufficiency:
                info['sufficient_check'] = True
                info['sufficient_msg'] = sufficient_msg
                return True, info, time_info
            else:
                info['sufficient_check'] = False
                info['sufficient_msg'] = sufficient_msg

                return False, info, time_info
                
                

        # 仅测试编译情况
        env.close()
        return True, "No other check", time_info


def evaluate_code(
    data,
    code,
    reference_lp,
    save_path=None,
    model_name=None,
    ensure_imports=False,
    problem_id=None,
    data_source=None,
    verbose=False,
):
    """
    评估LLM生成的Gurobi Python代码
    
    Args:
        data: 问题数据字典
        code: LLM生成的Python代码
        reference_lp: 参考LP文件路径或LP文件内容字符串
        save_path: 保存生成的LP文件的路径
        model_name: 模型名称
        ensure_imports: 是否确保导入语句
        problem_id: 问题ID
        data_source: 数据来源
        verbose: 是否打印详细信息
        
    Returns:
        tuple: (code_eval_result, wl_eval_result, equivalence_check_time)
            code_eval_result: 代码执行结果字典 {"success": bool, "message": str, "lp_file_path": str}
            wl_eval_result: WL等价性检查结果字典 {"success": bool, "message": str}
            equivalence_check_time: 等价性检查时间信息
    """
    import tempfile
    import json
    from evaluation.bench4opt.utils.lp_utils import process_code_for_lp, add_gurobi_imports
    from evaluation.bench4opt.utils.python_executor import PythonExecutor
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        lp_file_path = save_path if save_path.endswith('.lp') else f"{save_path}.lp"
    else:
        lp_file_path = os.path.join(temp_dir, "model_test.lp")
    
    # 准备数据文件路径
    data_file_path = os.path.join(temp_dir, "data.json")
    with open(data_file_path, "w") as f:
        json.dump(data, f)
    
    # 处理代码以生成LP文件
    try:
        code_for_lp = process_code_for_lp(code, data_file_path, lp_file_path)
        if ensure_imports:
            code_for_lp = add_gurobi_imports(code_for_lp)
    except ValueError as e:
        # Handle empty code error
        code_eval_result = {
            "success": False,
            "message": f"Code extraction failed: {str(e)}",
            "lp_file_path": "",
        }
        wl_eval_result = {"success": False, "message": "Code execution failed"}
        return code_eval_result, wl_eval_result, {"total_time": 0.0}
    
    # 执行代码生成LP文件
    try:
        python_executor = PythonExecutor(timeout_length=360, get_answer_from_stdout=True)
        result, report = python_executor.apply(code_for_lp)
        
        if os.path.exists(lp_file_path):
            code_eval_result = {
                "success": True,
                "message": "LP file generated successfully",
                "lp_file_path": lp_file_path,
            }
        else:
            code_eval_result = {
                "success": False,
                "message": f"LP file was not created: {report}",
                "lp_file_path": "",
            }
            wl_eval_result = {"success": False, "message": "Code execution failed"}
            return code_eval_result, wl_eval_result, {"total_time": 0.0}
    except Exception as e:
        code_eval_result = {
            "success": False,
            "message": f"Error executing code: {str(e)}",
            "lp_file_path": "",
        }
        wl_eval_result = {"success": False, "message": "Code execution failed"}
        return code_eval_result, wl_eval_result, {"total_time": 0.0}
    
    # 准备参考LP文件
    if isinstance(reference_lp, str) and os.path.exists(reference_lp):
        reference_lp_path = reference_lp
    elif isinstance(reference_lp, str):
        # 如果reference_lp是LP文件内容，保存到临时文件
        reference_lp_path = os.path.join(temp_dir, "reference.lp")
        with open(reference_lp_path, "w") as f:
            f.write(reference_lp)
    else:
        wl_eval_result = {"success": False, "message": "Invalid reference LP"}
        return code_eval_result, wl_eval_result, {"total_time": 0.0}
    
    # 进行等价性检查
    start_time = time.time()
    is_equivalent, error_msg_or_info, time_info = check_lp_equivalence(
        reference_lp_path, lp_file_path, verbose=verbose
    )
    end_time = time.time()
    total_time = end_time - start_time
    
    equivalence_check_time = {
        "total_time": total_time,
        **time_info
    }
    
    # 处理错误信息，可能是字符串或字典
    if isinstance(error_msg_or_info, dict):
        error_message = json.dumps(error_msg_or_info, ensure_ascii=False)
    else:
        error_message = str(error_msg_or_info)
    
    wl_eval_result = {
        "success": is_equivalent,
        "message": error_message
    }
    
    return code_eval_result, wl_eval_result, equivalence_check_time

