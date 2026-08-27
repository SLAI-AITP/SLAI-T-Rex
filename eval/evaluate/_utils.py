import gurobipy as gp
from gurobipy import GRB
import torch
import numpy as np
import math 
import json
import os
import argparse



def graph_generator(ins_name):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    # Create a Gurobi model
    m = gp.read(ins_name,env)
    nvars = m.NumVars
    # Gurobi handles variables and constraints slightly differently
    mvars = m.getVars()
    # Sort variables by name (Gurobi does not guarantee order)
    mvars = sorted(mvars, key=lambda v: v.VarName)
    var_feature = torch.zeros((nvars, 2))

    
    # Obtain variable features
    for n in range(nvars):
        var = mvars[n]
        obj_coeff = var.Obj
        lb = var.LB
        ub = var.UB
        if ub != float('inf'):
            m.addConstr(var <= float('inf'))
        if lb != 0:
            m.addConstr(var >= 0)
        
        m.addConstr(var >= lb)
        m.addConstr(var <= ub)
        m.update()
        bin = 1 if var.VType == GRB.BINARY else 0
        feature = [obj_coeff, bin]
        var_feature[n] = torch.tensor(feature)

    ncons = m.NumConstrs
    mcons = m.getConstrs()
    cons_feature = torch.zeros((ncons, 2))
    A = torch.zeros((ncons, nvars))
    for i in range(ncons):
        cons = mcons[i]
        sense = cons.Sense
        rhs = cons.RHS
        # lhs = -np.inf  # In Gurobi, the lower bound can be implicitly negative infinity
        b = rhs
        if sense == GRB.EQUAL:
            
            cons_sense = 0.0
        elif sense == GRB.LESS_EQUAL:
            
            cons_sense = -1.0
        elif sense == GRB.GREATER_EQUAL:
            
            cons_sense = 1.0
        else:
            raise NotImplementedError('Only implemented feature >=,<=, and, =')
        
        feature = [b, cons_sense]
        cons_feature[i] = torch.tensor(feature)
        for j in range(nvars):
            var = mvars[j]
            coeff = m.getCoeff(cons, var)  # Gurobi-specific function for constraint coefficients
            #print(f"Constraint {i}, Variable {j}, Coefficient: {coeff}")
            #if abs(coeff) > 1000000:
            #    A[i, j] = np.sign(coeff)*1000000
            #else:
            #    A[i, j] = coeff
            A[i, j] = torch.tensor(coeff, dtype=torch.float32)
    return A, var_feature, cons_feature

def hash_coloring(feature):
    '''
    This function takes in the constraint and variable features and returns the one-hot encoding of the features.
    '''
    if isinstance(feature, torch.Tensor):
        feature = feature
    else:
        feature = torch.tensor(feature)

    hash = {}
    num_nodes = len(feature)
    copy = torch.zeros(num_nodes)
    k = 0
    for i in range(num_nodes):
        key_i = tuple(feature[i].tolist())
        if key_i not in hash.keys():
            hash[key_i] = k
            k+=1
        copy[i] = hash[key_i]
    copy = copy.to(torch.int64)
    coding = torch.nn.functional.one_hot(copy)
    color = coding.to(torch.float)
    return color


def concate_color(color_cons,color_var):
    import torch.nn.functional as F
    color = torch.zeros(len(color_cons)+len(color_var),len(color_cons[0])+len(color_var[0]))
    for i in range(len(color_cons)):
        color[i] = F.pad(color_cons[i], (0,len(color_var[0])), "constant", 0)
    for j in range(len(color_cons),len(color_cons)+len(color_var)):
        color[j] = F.pad(color_var[j-len(color_cons)], (len(color_cons[0]),0), "constant", 0)
    return color


def wltest_coloring(c1,f1,A1):
    color_cons = hash_coloring(c1)
    color_var = hash_coloring(f1)
    color = concate_color(color_cons,color_var)

    L = len(c1)+len(f1)
    for i in range(2*L):
        aggr_cons = A1@color_var
        conc_concate = torch.cat((color_cons, aggr_cons), -1)
        cons = hash_coloring(conc_concate)

        aggr_var = A1.T@cons
        var_concate = torch.cat((color_var, aggr_var), -1)
        var = hash_coloring(var_concate)
        color_cons = cons
        color_var = var

    color = concate_color(color_cons,color_var)
    return color,color_cons,color_var


def wltest_coloring_two(c1,f1,A1,c2,f2,A2):
    c  = torch.cat((c1, c2), dim=0)
    f  = torch.cat((f1, f2), dim=0)
    A = torch.block_diag(A1, A2)
    color_cons = hash_coloring(c)
    color_var = hash_coloring(f)

    L = len(c)+len(f)
    for i in range(2*L):
        aggr_cons = A@color_var
        conc_concate = torch.cat((color_cons, aggr_cons), -1)
        cons = hash_coloring(conc_concate)

        aggr_var = A.T@cons
        var_concate = torch.cat((color_var, aggr_var), -1)
        var = hash_coloring(var_concate)
        color_cons = cons
        color_var = var

    #color = concate_color(color_cons,color_var)
    color_cons1 = color_cons[:len(c1)]
    color_var1 = color_var[:len(f1)]
    color_cons2 = color_cons[len(c1):]
    color_var2 = color_var[len(f1):]
    color1 = concate_color(color_cons1,color_var1)
    color2 = concate_color(color_cons2,color_var2)
    return color1,color_cons1,color_var1,color2,color_cons2,color_var2

def stable_partition(color):
    samecolor_cluster = {}
    for i,color_i in enumerate(color):
        color_index = torch.argmax(color_i).item()
        if color_index not in samecolor_cluster.keys():
            samecolor_cluster[color_index] = [i]
        else:
            samecolor_cluster[color_index].append(i)
    return samecolor_cluster



def derive_adjacency(A1):
    n_var = len(A1[0])
    n_cons = len(A1)
    zeros_upper = torch.zeros(n_var,n_var)
    zeros_lower = torch.zeros(n_cons,n_cons)
    #Adj = torch.cat((torch.cat((zeros_upper,A1.T),dim=1),torch.cat((A1,zeros_lower),dim=1)),dim=0)
    Adj = torch.cat((torch.cat((zeros_lower,A1),dim=1),torch.cat((A1.T,zeros_upper),dim=1)),dim=0)
    return Adj

def check_wl_determinable(color):
    same_color_partition =stable_partition(color)
    unique_set = [v[0] for k, v in same_color_partition.items() if len(v) == 1]
    if len(unique_set) == len(color[0]):
        return True
    else:
        return False


def get_nbhd(node_index_list,unique_set,Adj):
    Adj = Adj.to_sparse()
    col_indices = Adj.indices()[1]
    nbhd = []
    for i in node_index_list:
        row_filter = Adj.indices()[0] == i
        neighbors = col_indices[row_filter]
        for j in neighbors:
            if j not in unique_set:
                nbhd.append(int(j))
    return nbhd
from itertools import chain

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

def derive_subgraphs(color,Adj):
    same_color_partition =stable_partition(color)
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

def find_group(node, partition):
    for group, nodes in partition.items():
        if node in nodes:
            return group  # Return the key of the group
    return f'{node} not found in {partition}!'  # Return None if node is not found

def check_symmetric_decomposable(color,Adj):
    sub_graph_nodes,state = derive_subgraphs(color,Adj)
    same_color_partition =stable_partition(color)
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

def check_color_equivalence(color1,color2):
    samecolor_cluster1 = stable_partition(color1)
    samecolor_cluster2 = stable_partition(color2)
    for i in samecolor_cluster1.keys():
        if i not in samecolor_cluster2.keys():
            print('-------------An unique color occurs in color1 but not in color2-----------')
            return False
        else:
            if len(samecolor_cluster1[i]) != len(samecolor_cluster2[i]):
                print('-----------An unmatched cluster occurs!---------')
                return False
    return True