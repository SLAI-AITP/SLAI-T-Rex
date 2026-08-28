# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import torch
import numpy as np
import pandas as pd

from utils.common import debug_print
from utils.compare_extra import cmp_apply

cerror = "\033[1;31m[error]\033[0m"
cpass = "\033[1;32m[pass]\033[0m"
def compare(cpu_output, npu_output, dtype, op_name = "", show_detail_count=10, sort_mask=0, show_all=False, new_standard=False, mean_equal_return=True):
    if mean_equal_return == True:
        try:
            cpu_mean = cpu_output.mean()
            npu_mean = npu_output.mean()
        except:
            # mean() does not support int16
            cpu_mean = cpu_output.float().mean()
            npu_mean = npu_output.float().mean()
        debug_print('[debug]', f"cpu_output mean is {cpu_mean}, npu_output mean is {npu_mean}")
        if cpu_mean == npu_mean:
            print("mean is same")
            debug_print(cpass, 'Success to compare the npu_output and cpu_output!!!')
            return True

    try:
        real_data = npu_output.detach().numpy().flatten()
        golden_data = cpu_output.detach().numpy().flatten()
    except:
        # numpy does not support bf16
        real_data = npu_output.detach().to(torch.float32).numpy().flatten()
        golden_data = cpu_output.detach().to(torch.float32).numpy().flatten()
    if real_data.size != golden_data.size:
        debug_print('[%s]' % cerror, 'Error,the size of npu_output[%s] and cpu_output[%s] is not equal.' % (real_data.size, golden_data.size))
        return False
    if real_data.size == 0:
        debug_print('[%s]' % cerror, 'The npu_output is [],and it is same as cpu_output, the result of data_compare is \"Pass\"')
        return True

    df = pd.DataFrame(dict(golden=golden_data, output=real_data))
    ret, df_error, df, rate = cmp_apply(op_name, df, dtype, new_standard)

    if df_error.size == 0:
        show_all = True
        show_detail_count = df.size

    show_df = df if show_all else df_error
    if show_detail_count > 0:
        show_df = show_df[:show_detail_count]
    if df_error.size == 0:
        show_df = show_df[:10]

    if sort_mask:
        show_df = show_df.sort_values(['mask'])
        if sort_mask > 0:  # 默认就是, 正序的, 非倒序是 1
            show_df = show_df[::-1]

    if show_detail_count:
        debug_print(show_df)

    if ret==False:
        debug_print(cerror, 'Failed to compare the npu_output and cpu_output!!!')
    else:
        debug_print(cpass, 'Success to compare the npu_output and cpu_output!!!')
    return ret

# Check if completely same
def compare_tensors(tensor1, tensor2):
    # Check types
    type_check = tensor1.dtype == tensor2.dtype
    print(f"Type Check: {'Same' if type_check else 'Different'} {tensor1.dtype} {tensor2.dtype}")

    # Check shapes
    shape_check = tensor1.shape == tensor2.shape
    print(f"Shape Check: {'Same' if shape_check else 'Different'} {tensor1.shape} {tensor2.shape}")

    if type_check and shape_check:
        # Compare values
        value_check = torch.equal(tensor1, tensor2)
        print(f"Value Check: {'Same' if value_check else 'Different'}")

        if not value_check:
            # Calculate differences
            differences = tensor1 - tensor2
            print(f"tensor1:\n{tensor1}")
            print(f"tensor2:\n{tensor2}")
            print(f"Differences:\n{differences}")
    else:
        print("Cannot compare values due to different types or shapes.")