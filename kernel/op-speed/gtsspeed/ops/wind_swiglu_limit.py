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
import gtsops

# [Add by GTS-AILab] [feature GTSOps] swiglu_limit 前向+反向（融合 clamp+swiglu，含 clamp 直通掩码）
class NpuSwiGluLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, limit, y=None, dim=-1, backend="v2"):
        # 仅支持合并输入（y=None）；与训练用法一致。backend 默认 v2 融合算子。
        outputs = gtsops.npu_swiglu_limit(x, y, dim, float(limit), backend)
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.limit = float(limit)
        ctx.backend = backend   # 存 backend，反向走同版本，前后向成对一致
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        (x,) = ctx.saved_tensors
        # x_grad 与 x 同形（grad_a|grad_b 拼接）。前向 backend → 反向同 backend。
        # probs 用 kwarg 传，避免与新增的 probs 入参位置冲突；此处无 probs(=None)。
        grad_x = gtsops.npu_swiglu_limit_backward(grad_outputs.contiguous(), x, ctx.dim, ctx.limit, backend=ctx.backend)
        # 对应 forward(x, limit, y, dim, backend) 的 5 个入参：只有 x 需要梯度
        return grad_x, None, None, None, None


wind_swiglu_limit = NpuSwiGluLimit.apply


# MoE 专家路径：swiglu_limit + probs 加权（仅反向融合 probs；正向 probs 乘法在外部做）。
# 正向：out = swiglu_limit(x) * probs  —— 正向算子无 probs 输入，probs 乘法是外部 Mul([T,1])，正向不融。
# 反向：grad_x = swiglu_back(grad_out * probs)  —— grad_out*probs 进反向算子（消除外部 Mul([*,1])+Cast）。
#       grad_probs = sum(grad_out * h, -1)  —— 轻型方案：由调用方/上层 routing 反向负责，此函数不返回。
# 注意：forward 内的 *probs 在 autograd.Function 黑盒内，反向全由本 backward 负责，不会 double-count。
class NpuSwiGluLimitProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, probs, limit, dim=-1, highprecision=True, hasprobs=True, backend="v3"):
        # x[bf16, Mx2H], probs[float, Mx1] -> output[bf16, MxH]
        outputs = gtsops.npu_swiglu_limit_with_probs(x, y, probs, dim, float(limit), highprecision, hasprobs, backend)
        ctx.save_for_backward(x, probs)
        ctx.dim = dim
        ctx.limit = float(limit)
        ctx.backend = backend
        return outputs

    @staticmethod
    def backward(ctx, grad_without_probs, grad_with_probs):
        x, probs = ctx.saved_tensors
        if grad_with_probs is None:
            return None, None, None, None, None, None, None, None

        # 算子要求 gradout 与 self(x) 同 dtype；上游 grad 可能是 fp32，进算子前 cast 对齐 x.dtype。
        grad_outputs = grad_with_probs.contiguous().to(x.dtype)
        # 重型方案：算子一次出 (grad_x, grad_probs)。grad_x=swiglu_back(grad_out*probs)，
        #   grad_probs=sum_H(grad_out*h) 由算子内逐行 reduce 算（消除外部重算 h + Mul + ReduceSum）。
        # routing 反向需要 probs 梯度（现场 permuted_probs_inputs_detach.grad），返回算子出的 grad_probs。
        grad_x, grad_probs = gtsops.npu_swiglu_limit_backward_probs(
            grad_outputs, x, ctx.dim, ctx.limit, probs, backend=ctx.backend)
        grad_probs = grad_probs.to(probs.dtype)
        if not ctx.needs_input_grad[2]:
            grad_probs = None
        return grad_x, None, grad_probs, None, None, None, None, None


wind_swiglu_limit_probs = NpuSwiGluLimitProbs.apply
