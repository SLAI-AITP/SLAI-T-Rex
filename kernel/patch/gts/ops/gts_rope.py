# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import apply_rotary_emb
from mindspeed.gts.ops.gts_custom_op import gts_scale_rope, is_enable_op

# [Add by GTS-AILab] [feature GTSOps] scale-rope
def gts_rope_forward(x, freqs_cis, rope_head_dim, scale=None, inverse=False, need_transpose=True):
    """RoPE 变化点隔离：融合算子优先，开关关 / 算子未编译时回退原生复数式 RoPE（逐元素等价）。
    forward 主流程每处只留一行调用，不感知开关。
      scale:          q 传 rsqrt（外露），kv/o 传 None。
      inverse:        o 段传 True（乘 conj，sin 翻号），q/kv 传 False。
      need_transpose: q/kv 在 [s,b,..] 上接、需函数内 transpose 到 [b,s,..] 再转回；
                      o 已在外部 transpose 到 [b,s,..]，传 False（函数内不再 transpose）。
    """
    out = gts_scale_rope(x, freqs_cis, rope_head_dim, scale=scale, inverse=inverse) if is_enable_op("rope") else None
    if out is not None:
        return out
    # ---- 回退：原生复数式 RoPE（与改造前逐元素一致）----
    if scale is not None:
        x = x * scale
    if need_transpose:
        x = x.transpose(0, 1)
        x = x.clone()
        x[..., -rope_head_dim:] = apply_rotary_emb(x[..., -rope_head_dim:], freqs_cis, inverse)
        return x.transpose(0, 1)
    x[..., -rope_head_dim:] = apply_rotary_emb(x[..., -rope_head_dim:], freqs_cis, inverse)
    return x