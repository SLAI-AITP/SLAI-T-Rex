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
        #print(f'var_lb: {var_lb}, var_ub: {var_ub}')
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

def transform_constriant_with_0_rhs_ineq(model):
    all_cons = model.getConstrs()

    for cons in all_cons:
        if cons.RHS == 0 and cons.Sense == GRB.LESS_EQUAL:
            # 改成 >=
            cons.setAttr("Sense", GRB.GREATER_EQUAL)
            cons.setAttr("RHS", 0.0)

            row = model.getRow(cons)   # 这在 Gurobi 9 里是可用的！
            rhs = cons.RHS

            # 删除旧的
            model.remove(cons)

            # 构造新的
            new_expr = gp.LinExpr()
            for i in range(row.size()):
                v = row.getVar(i)
                c = row.getCoeff(i)
                new_expr.addTerms(-c, v)

            new_cons = model.addConstr(new_expr >= -rhs)
            # 循环完统一更新一次
            model.update()
        else:
            continue
    return model

def transform_constriant_with_0_rhs_eq(model, save_path):
    """
    标准化 Gurobi 模型中 rhs=0 的等式约束。
    
    规则：
    1. 对每个 rhs=0 的等式约束：
       - 取出变量和系数
       - 如果系数和 > 0：保持
       - 如果系数和 < 0：所有系数取相反数
    2. 如果某个约束系数和 == 0，则保存模型到 save_path（如果给定）。

    参数
    ----
    model : gurobipy.Model
        输入模型
    save_path : str
        如果存在特殊情况（sum(coeffs)=0），保存模型的路径（.lp 或 .mps）
    
    返回
    ----
    model : gurobipy.Model
        更新后的模型（in-place 修改）
    """
    needs_save = False

    for constr in model.getConstrs():
        # 只处理 equality 且 rhs=0
        if constr.Sense == '=' and abs(constr.RHS) < 1e-9:
            expr = model.getRow(constr)
            coeffs = [expr.getCoeff(i) for i in range(expr.size())]
            vars_ = [expr.getVar(i) for i in range(expr.size())]

            coeff_sum = sum(coeffs)

            if abs(coeff_sum) < 1e-12:
                needs_save = True
                continue  # 不改动，但标记保存
            
            # 如果和 < 0，就翻转符号
            if coeff_sum < 0:
                coeffs = [-c for c in coeffs]

            # 删除旧约束，添加新约束
            model.remove(constr)
            model.addConstr(sum(c * v for c, v in zip(coeffs, vars_)) == 0)

    model.update()
    #model.write("transformed_model_eq.lp")
    if needs_save and save_path is not None:
        model.write(save_path)

    return model


def check_lp_equivalence_legacy(lp_path1, lp_path2, verbose: bool = False):
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
        model1 = transform_constriant_with_0_rhs_ineq(model1)
        model1 = transform_constriant_with_0_rhs_eq(model1,'failure/'+lp_path1.split('/')[-1]+'failure.lp')
        model1.write('model_ref.lp')
        model2 = convert_cons2boundary(model2)
        model2 = transform_constriant_with_0_rhs_ineq(model2)
        model2 = transform_constriant_with_0_rhs_eq(model2,'failure/'+lp_path2.split('/')[-1]+'failure.lp')
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


def _format_feature(values, precision=6):
    return ",".join(f"{np.float32(value):.{precision}f}" for value in values)


def _update_bound_maps(lb_updates, ub_updates, var_name, coeff, sense, rhs):
    if abs(coeff) < 1e-12:
        return

    bound = rhs / coeff
    if sense == GRB.GREATER_EQUAL:
        lb_updates[var_name] = max(lb_updates.get(var_name, float("-inf")), bound)
    elif sense == GRB.LESS_EQUAL:
        ub_updates[var_name] = min(ub_updates.get(var_name, float("inf")), bound)
    elif sense == GRB.EQUAL:
        lb_updates[var_name] = max(lb_updates.get(var_name, float("-inf")), bound)
        ub_updates[var_name] = min(ub_updates.get(var_name, float("inf")), bound)
    else:
        raise NotImplementedError("Only implemented >=, <=, and =")


def convert_cons2boundary_fast(model):
    vars_ = model.getVars()
    var_by_name = {var.VarName: var for var in vars_}
    lb_updates = {}
    ub_updates = {}
    cons_to_remove = []

    for constr in model.getConstrs():
        row = model.getRow(constr)
        if row.size() != 1:
            continue

        coeff = row.getCoeff(0)
        if abs(coeff) < 1e-12:
            continue

        var = row.getVar(0)
        _update_bound_maps(lb_updates, ub_updates, var.VarName, coeff, constr.Sense, constr.RHS)
        cons_to_remove.append(constr)

    for var_name, lower_bound in lb_updates.items():
        var_by_name[var_name].setAttr(GRB.Attr.LB, lower_bound)
    for var_name, upper_bound in ub_updates.items():
        var_by_name[var_name].setAttr(GRB.Attr.UB, upper_bound)

    if cons_to_remove:
        model.remove(cons_to_remove)
    model.update()
    return model


def transform_constriant_with_0_rhs_ineq_fast(model):
    cons_to_remove = []
    replacements = []

    for cons in model.getConstrs():
        if cons.Sense != GRB.LESS_EQUAL or abs(cons.RHS) >= 1e-9:
            continue

        row = model.getRow(cons)
        vars_ = []
        coeffs = []
        for i in range(row.size()):
            vars_.append(row.getVar(i))
            coeffs.append(-row.getCoeff(i))
        cons_to_remove.append(cons)
        replacements.append((vars_, coeffs))

    if cons_to_remove:
        model.remove(cons_to_remove)
        model.update()

    for vars_, coeffs in replacements:
        expr = gp.LinExpr(coeffs, vars_)
        model.addConstr(expr >= 0.0)

    if replacements:
        model.update()
    return model


def transform_constriant_with_0_rhs_eq_fast(model, save_path):
    needs_save = False
    cons_to_remove = []
    replacements = []

    for constr in model.getConstrs():
        if constr.Sense != GRB.EQUAL or abs(constr.RHS) >= 1e-9:
            continue

        row = model.getRow(constr)
        vars_ = []
        coeffs = []
        for i in range(row.size()):
            vars_.append(row.getVar(i))
            coeffs.append(row.getCoeff(i))

        coeff_sum = sum(coeffs)
        if abs(coeff_sum) < 1e-12:
            needs_save = True
            continue

        if coeff_sum < 0:
            cons_to_remove.append(constr)
            replacements.append((vars_, [-coeff for coeff in coeffs]))

    if cons_to_remove:
        model.remove(cons_to_remove)
        model.update()

    for vars_, coeffs in replacements:
        expr = gp.LinExpr(coeffs, vars_)
        model.addConstr(expr == 0.0)

    if replacements:
        model.update()

    if needs_save and save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.write(save_path)

    return model


def generate_bipartite_fast(model):
    nvars = model.NumVars
    vars_ = model.getVars()
    cons = model.getConstrs()
    total_nodes = nvars + len(cons)

    initial_colors = [""] * total_nodes
    weighted_neighbors = [[] for _ in range(total_nodes)]
    plain_neighbors = [[] for _ in range(total_nodes)]
    var_name_to_index = {var.VarName: idx for idx, var in enumerate(vars_)}

    for idx, var in enumerate(vars_):
        if var.VType == GRB.BINARY:
            var_type = 1.0
        elif var.VType == GRB.INTEGER:
            var_type = 2.0
        else:
            var_type = 0.0
        initial_colors[idx] = _format_feature([float(var.Obj), var_type, var.LB, var.UB])

    for cons_idx, con in enumerate(cons):
        node_idx = nvars + cons_idx
        if con.Sense == GRB.EQUAL:
            cons_sense = 0.0
        elif con.Sense == GRB.LESS_EQUAL:
            cons_sense = -1.0
        elif con.Sense == GRB.GREATER_EQUAL:
            cons_sense = 1.0
        else:
            raise NotImplementedError("Only implemented feature >=,<=, and, =")

        initial_colors[node_idx] = _format_feature([float(con.RHS), cons_sense])

        row = model.getRow(con)
        for i in range(row.size()):
            var_idx = var_name_to_index[row.getVar(i).VarName]
            edge_color = _format_feature([row.getCoeff(i)])
            weighted_neighbors[var_idx].append((node_idx, edge_color))
            weighted_neighbors[node_idx].append((var_idx, edge_color))
            plain_neighbors[var_idx].append(node_idx)
            plain_neighbors[node_idx].append(var_idx)

    return initial_colors, weighted_neighbors, [tuple(neighbors) for neighbors in plain_neighbors]


def wl_test_fast(initial_colors, weighted_neighbors):
    colors = list(initial_colors)
    prev_partition_size = 0
    n_iter = 0

    for _ in range(len(colors)):
        new_partition = {}
        new_colors = [None] * len(colors)

        for node_idx, node_color in enumerate(colors):
            neighbor_color_pairs = [
                (colors[neighbor_idx], edge_color) for neighbor_idx, edge_color in weighted_neighbors[node_idx]
            ]
            hashed_color = hash_multiset(node_color, neighbor_color_pairs)
            new_colors[node_idx] = hashed_color
            new_partition.setdefault(hashed_color, []).append(node_idx)

        if len(new_partition) == prev_partition_size:
            return new_partition, n_iter

        colors = new_colors
        prev_partition_size = len(new_partition)
        n_iter += 1

    return new_partition, n_iter


def get_nbhd_fast(node_index_list, unique_set, neighbors):
    unique_lookup = unique_set if isinstance(unique_set, set) else set(unique_set)
    nbhd = []
    for node_idx in node_index_list:
        for neighbor_idx in neighbors[node_idx]:
            if neighbor_idx not in unique_lookup:
                nbhd.append(int(neighbor_idx))
    return nbhd


def get_cluster_fast(same_color_partition, unique_set, partition_index, sub_graph_nodes, neighbors):
    labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
    for node_idx in same_color_partition[partition_index]:
        if node_idx not in labeled_nodes:
            sub_graph_nodes[node_idx] = [node_idx]
        else:
            continue

    previous_sum = 0
    while sum(
        len(value)
        for key, value in sub_graph_nodes.items()
        if key in same_color_partition[partition_index]
    ) - previous_sum > 0:
        previous_sum = sum(
            len(value)
            for key, value in sub_graph_nodes.items()
            if key in same_color_partition[partition_index]
        )
        for node_idx in same_color_partition[partition_index]:
            if node_idx not in labeled_nodes:
                sub_graph_nodes[node_idx] = sub_graph_nodes[node_idx] + get_nbhd_fast(
                    sub_graph_nodes[node_idx], unique_set, neighbors
                )
                sub_graph_nodes[node_idx] = list(set(sub_graph_nodes[node_idx]))
    return sub_graph_nodes


def derive_subgraphs_fast(same_color_partition, neighbors):
    unique_set = [value[0] for value in same_color_partition.values() if len(value) == 1]
    unique_lookup = set(unique_set)
    same_color_partition = {key: value for key, value in same_color_partition.items() if len(value) > 1}
    sub_graph_nodes = {}
    multiplicity_nodes = [node_idx for node_idx in range(len(neighbors)) if node_idx not in unique_lookup]
    condition = True
    state = "uncheck"

    while condition:
        labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
        for node_idx in multiplicity_nodes:
            if node_idx not in labeled_nodes:
                partition_index = find_group(node_idx, same_color_partition)
                sub_graph_nodes = get_cluster_fast(
                    same_color_partition,
                    unique_lookup,
                    partition_index,
                    sub_graph_nodes,
                    neighbors,
                )

        covered_nodes = sum(len(value) for value in sub_graph_nodes.values())
        target_nodes = len(neighbors) - len(unique_lookup)
        if covered_nodes < target_nodes:
            condition = True
            state = "continue"
        elif covered_nodes == target_nodes:
            condition = False
            state = "complete"
        else:
            condition = False
            state = "error"

    for key in list(sub_graph_nodes.keys()):
        sub_graph_nodes[key] = set(sub_graph_nodes[key])
    return sub_graph_nodes, state


def connectivity_check_fast(dict1, dict2, neighbors):
    target_nodes = dict2 if isinstance(dict2, set) else set(dict2)
    for node_idx in dict1:
        for neighbor_idx in neighbors[node_idx]:
            if neighbor_idx in target_nodes:
                return True
    return False


def check_graph_disconnect_fast(sub_graph_nodes, neighbors):
    for key_i in sub_graph_nodes:
        for key_j in sub_graph_nodes:
            if key_i != key_j and connectivity_check_fast(sub_graph_nodes[key_i], sub_graph_nodes[key_j], neighbors):
                return False
    return True


def _iter_ambiguous_components(same_color_partition, neighbors):
    unique_nodes = {value[0] for value in same_color_partition.values() if len(value) == 1}
    ambiguous_partitions = {key: value for key, value in same_color_partition.items() if len(value) > 1}

    node_to_group = {}
    ambiguous_nodes = []
    for group_key, nodes in ambiguous_partitions.items():
        for node_idx in nodes:
            node_to_group[node_idx] = group_key
            ambiguous_nodes.append(node_idx)

    ambiguous_lookup = set(ambiguous_nodes)
    visited = set()

    for seed in ambiguous_nodes:
        if seed in visited:
            continue

        component_nodes = []
        stack = [seed]
        visited.add(seed)

        while stack:
            node_idx = stack.pop()
            component_nodes.append(node_idx)
            for neighbor_idx in neighbors[node_idx]:
                if neighbor_idx in unique_nodes or neighbor_idx not in ambiguous_lookup:
                    continue
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                stack.append(neighbor_idx)

        yield component_nodes, node_to_group


def check_symmetric_decomposable_fast(same_color_partition, neighbors):
    has_ambiguous_partition = False
    for nodes in same_color_partition.values():
        if len(nodes) > 1:
            has_ambiguous_partition = True
            break

    if not has_ambiguous_partition:
        return True, "Symmetric Decomposable."

    covered_nodes = 0
    for component_nodes, node_to_group in _iter_ambiguous_components(same_color_partition, neighbors):
        covered_nodes += len(component_nodes)
        seen_groups = set()
        for node_idx in component_nodes:
            group_key = node_to_group[node_idx]
            if group_key in seen_groups:
                return False, "Due to distinctness."
            seen_groups.add(group_key)

    target_nodes = sum(len(nodes) for nodes in same_color_partition.values() if len(nodes) > 1)
    if covered_nodes != target_nodes:
        return False, "Due to decompose error"

    return True, "Symmetric Decomposable."


def check_sufficient_conditions_fast(same_color_partition, neighbors):
    if check_wl_determinable(same_color_partition):
        return True, "WL-determinable"

    check_sym_decom, msg = check_symmetric_decomposable_fast(same_color_partition, neighbors)
    if check_sym_decom:
        return True, msg
    return False, msg


def check_sufficiency_two_fast(same_color_partition1, neighbors1, same_color_partition2, neighbors2):
    check_sufficient1, msg1 = check_sufficient_conditions_fast(same_color_partition1, neighbors1)
    check_sufficient2, msg2 = check_sufficient_conditions_fast(same_color_partition2, neighbors2)
    msg = {"msg1": msg1, "msg2": msg2}
    return check_sufficient1 and check_sufficient2, msg


def check_lp_equivalence_fast(lp_path1, lp_path2, verbose: bool = False):
    info = {}
    time_info = {}

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()

    try:
        model1 = gp.read(lp_path1, env)
        try:
            model2 = gp.read(lp_path2, env)
        except Exception as exc:
            return False, f"Error reading LP file: {exc}", time_info

        start_time = time.time()
        model1 = convert_cons2boundary_fast(model1)
        model1 = transform_constriant_with_0_rhs_ineq_fast(model1)
        model1 = transform_constriant_with_0_rhs_eq_fast(
            model1, f"failure/{lp_path1.split('/')[-1]}failure.lp"
        )

        model2 = convert_cons2boundary_fast(model2)
        model2 = transform_constriant_with_0_rhs_ineq_fast(model2)
        model2 = transform_constriant_with_0_rhs_eq_fast(
            model2, f"failure/{lp_path2.split('/')[-1]}failure.lp"
        )
        time_info["normalization_time"] = time.time() - start_time

        if verbose:
            model1.write("model_ref.lp")
            model2.write("model_test.lp")

        if model1.NumVars != model2.NumVars:
            info["var_num_check"] = False
            return False, info, time_info
        info["var_num_check"] = True

        if model1.NumConstrs != model2.NumConstrs:
            info["cons_num_check"] = False
            return False, info, time_info
        info["cons_num_check"] = True

        start_time = time.time()
        initial_colors1, weighted_neighbors1, neighbors1 = generate_bipartite_fast(model1)
        initial_colors2, weighted_neighbors2, neighbors2 = generate_bipartite_fast(model2)
        time_info["graph_build_time"] = time.time() - start_time

        start_time = time.time()
        partition1, _ = wl_test_fast(initial_colors1, weighted_neighbors1)
        partition2, _ = wl_test_fast(initial_colors2, weighted_neighbors2)
        time_info["wl_coloring_time"] = time.time() - start_time

        if partition1.keys() != partition2.keys():
            info["wl_check"] = False
            return False, info, time_info

        start_time = time.time()
        check_sufficiency, sufficient_msg = check_sufficiency_two_fast(
            partition1,
            neighbors1,
            partition2,
            neighbors2,
        )
        time_info["sufficient_check_time"] = time.time() - start_time

        if check_sufficiency:
            info["sufficient_check"] = True
            info["sufficient_msg"] = sufficient_msg
            return True, info, time_info

        info["sufficient_check"] = False
        info["sufficient_msg"] = sufficient_msg
        return False, info, time_info
    finally:
        env.close()


def check_lp_equivalence_no_sufficiency(lp_path1, lp_path2, verbose: bool = False):
    info = {}
    time_info = {}

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()

    try:
        model1 = gp.read(lp_path1, env)
        try:
            model2 = gp.read(lp_path2, env)
        except Exception as exc:
            return False, f"Error reading LP file: {exc}", time_info

        start_time = time.time()
        model1 = convert_cons2boundary_fast(model1)
        model1 = transform_constriant_with_0_rhs_ineq_fast(model1)
        model1 = transform_constriant_with_0_rhs_eq_fast(
            model1, f"failure/{lp_path1.split('/')[-1]}failure.lp"
        )

        model2 = convert_cons2boundary_fast(model2)
        model2 = transform_constriant_with_0_rhs_ineq_fast(model2)
        model2 = transform_constriant_with_0_rhs_eq_fast(
            model2, f"failure/{lp_path2.split('/')[-1]}failure.lp"
        )
        time_info["normalization_time"] = time.time() - start_time

        if verbose:
            model1.write("model_ref.lp")
            model2.write("model_test.lp")

        if model1.NumVars != model2.NumVars:
            info["var_num_check"] = False
            return False, info, time_info
        info["var_num_check"] = True

        if model1.NumConstrs != model2.NumConstrs:
            info["cons_num_check"] = False
            return False, info, time_info
        info["cons_num_check"] = True

        start_time = time.time()
        initial_colors1, weighted_neighbors1, _ = generate_bipartite_fast(model1)
        initial_colors2, weighted_neighbors2, _ = generate_bipartite_fast(model2)
        time_info["graph_build_time"] = time.time() - start_time

        start_time = time.time()
        partition1, _ = wl_test_fast(initial_colors1, weighted_neighbors1)
        partition2, _ = wl_test_fast(initial_colors2, weighted_neighbors2)
        time_info["wl_coloring_time"] = time.time() - start_time

        if partition1.keys() != partition2.keys():
            info["wl_check"] = False
            return False, info, time_info

        info["wl_check"] = True
        info["sufficient_check"] = True
        info["sufficient_msg"] = "SKIPPED_ASSUME_TRUE"
        time_info["sufficient_check_time"] = 0.0
        return True, info, time_info
    finally:
        env.close()


check_lp_equivalence = check_lp_equivalence_fast

