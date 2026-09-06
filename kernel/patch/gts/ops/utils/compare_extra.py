# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import sys
import numpy as np
import pandas as pd
import psutil
from pandarallel import pandarallel
from utils.common import debug_print, get_atol


# def cmp_matmul(node_name, df, dtype_desc, atol):
#     return cmp_default(node_name, df, dtype_desc, 0.04)


# def cmp_adaptiveavgpool2d(node_name, df, dtype_desc, atol):
#     return cmp_default(node_name, df, dtype_desc, 0.005)


def util(se):
    golden = se.golden
    output = se.output
    if pd.isna(golden) and pd.isna(output):
        return 0
    elif np.isinf(golden) and np.isinf(output):
        return 0
    else:
        # print(golden, output, abs(golden - output))
        return np.abs(golden - output)
            

def cmp_default(df, atol, new_standard = False):
    nb_workers = int(psutil.cpu_count(logical=True)/4)
    if len(df) > 1024 * nb_workers:
        pandarallel.initialize(nb_workers=nb_workers, progress_bar=False)
        df['diff'] = df[['output', 'golden']].parallel_apply(util, axis=1)
    else:
        df['diff'] = df[['output', 'golden']].apply(util, axis=1)

    if new_standard:
        df['maxv'] = df.golden.where(df.golden.abs() > 1, 1)
        df['mask'] = df['diff'] / df['maxv']
    else:
        df['mask'] = df['diff'] / df.golden.abs()
    df_error = df[df['mask'] > atol]
    error_rate = len(df_error) / len(df)
    ret = (error_rate <= atol)
    debug_print('[debug] all_count:', len(df))
    debug_print('[debug] error_count:', len(df_error))
    debug_print('[debug] error_rate:', error_rate, f'(need <= {atol})')
    return ret, df_error, df, 1 - error_rate

def cmp_apply(op_name, df, dtype, new_standard = False):
    func = globals().get(f'cmp_{op_name}', cmp_default)
    if new_standard:
        return func(df, get_atol(dtype, True), True)
    else:
        return func(df, get_atol(dtype))
