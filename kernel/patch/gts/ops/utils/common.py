# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import sys
import torch
import numpy as np
torch.set_printoptions(linewidth=140, profile="full")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_input_name(op_id, node_name, shape, dtype_desc, data_range):
    str_shape = '_'.join(str(int(x)) for x in shape)
    return f'{op_id}_{node_name}_{str_shape}_{dtype_desc}_{data_range[0]}_{data_range[1]}.bin'

def get_input_name_norange(op_id, node_name, shape, dtype_desc):
    str_shape = '_'.join(str(int(x)) for x in shape)
    return f'{op_id}_{node_name}_{str_shape}_{dtype_desc}.bin'

def get_tensor(shape, range, dtype_desc):
    torch_type = desc_dtype_dic_torch[dtype_desc]
    np_type = desc_dtype_dic_numpy[dtype_desc]
    tensor = torch.empty(tuple(shape), dtype=torch_type)
    if (dtype_desc.upper().startswith("INT")):
        if "NAN" == range[0].upper():
            a = torch.log(torch.tensor([-0.1]))
            tensor[:] = a
        elif "NULL" == range[0].upper():
            tensor = torch.empty(tuple(shape), dtype=torch_type)
        elif "0" == range[0]:
            tensor[:] = 0
        elif "MAX" == range[0]:
            tensor[:] = torch.tensor(np.iinfo(np_type).max, dtype=torch_type)
        elif "MIN" == range[0]:
            tensor[:] = torch.tensor(np.iinfo(np_type).min, dtype=torch_type)
        elif range[0].upper() == range[1].upper():
            tensor[:] = int(range[0])
        else:
            tensor_np = np.random.uniform(float(range[0]), float(range[1]), tuple(shape)).astype(np_type)
            tensor = torch.tensor(tensor_np, dtype=torch_type)
    else:
        if "INF" == range[0].upper():
            a = float("inf")
            tensor[:] = a
        elif "NAN" == range[0].upper():
            a = torch.log(torch.tensor([-0.1]))
            tensor[:] = a
        elif "NULL" == range[0].upper():
            tensor = torch.empty(tuple(shape), dtype=torch_type)
        elif "N_INF" == range[0].upper():
            a = float("-inf")
            tensor[:] = a
        elif "0" == range[0]:
            tensor[:] = 0
        elif "MAX" == range[0]:
            tensor[:] = torch.tensor(np.finfo(np_type).max, dtype=torch_type)
        elif "MIN" == range[0]:
            tensor[:] = torch.tensor(np.finfo(np_type).min, dtype=torch_type)
        elif range[0].upper() == range[1].upper():
            tensor[:] = float(range[0])
        else:
            # -2E-9 as '@2E@9'
            minValue = float((range[0]).replace('@', '-'))
            maxValue = float((range[1]).replace('@', '-'))
            tensor_np = np.random.uniform(minValue, maxValue, tuple(shape)).astype(np_type)
            tensor = torch.tensor(tensor_np, dtype=torch_type)

    return tensor.type(torch_type)

def get_output_name(op_name, op_id, node_name, shape, dtype_desc):
    str_shape = '_'.join(str(int(x)) for x in shape)
    return f'{op_name}_{op_id}_{node_name}_{str_shape}_{dtype_desc}.bin'

def debug_print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    print(*args, **kwargs)


desc_dtype_dic_torch = {
    'BF16': torch.bfloat16,
    'FP16': torch.float16,
    'FP32': torch.float32,
    'FP64': torch.float64,
    'INT8': torch.int8,
    'INT16': torch.int16,
    'INT32': torch.int32,
    'INT64': torch.int64,
    'UINT8': torch.uint8,
    # 'UINT16': torch.uint16,  # 不存在
    # 'UINT32': torch.uint32,  # 不存在
    # 'UINT64': torch.uint64,  # 不存在
    'STRING': str,
    'COMPLEX64': torch.complex64,
    'COMPLEX128': torch.complex128,
    'BOOL': bool,
}

desc_dtype_dic_torch_upgrade = {
    'BF16': torch.float32,
    'FP16': torch.float32,
    'FP32': torch.float64,
    'FP64': torch.float64,
    'INT8': torch.int8,
    'INT16': torch.int16,
    'INT32': torch.int32,
    'INT64': torch.int64,
    'UINT8': torch.uint8,
    # 'UINT16': torch.uint16,  # 不存在
    # 'UINT32': torch.uint32,  # 不存在
    # 'UINT64': torch.uint64,  # 不存在
    'STRING': str,
    'COMPLEX64': torch.complex64,
    'COMPLEX128': torch.complex128,
    'BOOL': bool,
}

desc_dtype_dic_numpy = {
    'BF16': np.float32,
    'FP16': np.float16,
    'FP32': np.float32,
    'FP64': np.float64,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'STRING': str,
    'COMPLEX64': np.complex64,
    'COMPLEX128': np.complex128,
    'BOOL': bool,
}

desc_dtype_dic_numpy_upgrade = {
    'BF16': np.float32,
    'FP16': np.float32,
    'FP32': np.float64,
    'FP64': np.float64,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'STRING': str,
    'COMPLEX64': np.complex64,
    'COMPLEX128': np.complex128,
    'bool': bool,
}
desc_dtype_dic_numpy_cmp = {
    'BF16': np.float32,
    'FP16': np.float32,
    'FP32': np.float64,
    'FP64': np.float64,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'STRING': str,
    'COMPLEX64': np.complex64,
    'COMPLEX128': np.complex128,
    'BOOL': np.int32,
}

desc_atol_dic = {
    'BF16': 10 ** (-3),
    'FP16': 10 ** (-3),
    'FP32': 10 ** (-4),
    'FP64': 10 ** (-4),
    'INT8': 0,
    'INT16': 0,
    'INT32': 0,
    'INT64': 0,
    'UINT8': 0,
    'UINT16': 0,
    'UINT32': 0,
    'UINT64': 0,
    # 'STRING': '',  # 无法比较吧, 暂不处理
    'COMPLEX64': 0,
    'COMPLEX128': 0,
    'BOOL': 0,
}

def get_atol_new(dtype):
    if (dtype == torch.bfloat16):
        return 2 ** (-7)
    elif (dtype == torch.float16):
        return 2 ** (-8)
    elif (dtype == torch.float32):
        return 2 ** (-11)
    return 0

def get_atol(dtype, new_standard=False):
    if new_standard:
        return get_atol_new(dtype)
    if (dtype == torch.bfloat16):
        return 10 ** (-3)
    elif (dtype == torch.float16):
        return 10 ** (-3)
    elif (dtype == torch.float32):
        return 10 ** (-4)
    return 0

dtype_desc_dic_torch = {desc: dtype for dtype, desc in desc_dtype_dic_torch.items()}
dtype_desc_dic_numpy = {desc: dtype for dtype, desc in desc_dtype_dic_numpy.items()}


