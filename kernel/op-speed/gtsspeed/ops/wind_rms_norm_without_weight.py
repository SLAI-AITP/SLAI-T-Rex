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


def _dtype_to_int(dtype):
    mapping = {
        torch.float32: 6,   # 假设使用 ACL_FLOAT
        torch.float16: 5,
        torch.bfloat16: 15,
    }
    return mapping.get(dtype, 0)


class WindRmsNormWithoutWeight(torch.autograd.Function):
    """
    RMSNorm without weight: computes rsqrt(mean(x^2) + eps) per row.

    Forward: res = rsqrt(mean(x^2) + eps), output shape: (*, 1)
    Backward: grad_x = -grad_res * res^3 / D * x
    """

    @staticmethod
    def forward(ctx, x, epsilon, out_dtype):
        if out_dtype is None:
            out_dtype = x.dtype
        res = gtsops.npu_rms_norm_without_weight(x, epsilon, _dtype_to_int(out_dtype))
        ctx.save_for_backward(x, res)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        x, res = ctx.saved_tensors
        grad_x = gtsops.npu_rms_norm_without_weight_backward(grad_res, x, res)
        return grad_x, None, None


class WindRmsNormWithoutWeightBackward(torch.autograd.Function):
    """
    RMSNorm without weight: computes rsqrt(mean(x^2) + eps) per row.

    Forward: res = rsqrt(mean(x^2) + eps), output shape: (*, 1)
    Backward: grad_x = -grad_res * res^3 / D * x
    """

    @staticmethod
    def forward(ctx, grad_res, x, res):
        grad_x = gtsops.npu_rms_norm_without_weight_backward(grad_res, x, res)
        return grad_x


class WindRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, epsilon=1e-5):
        outputs = gtsops.npu_rms_norm(x, gamma, epsilon)
        ctx.save_for_backward(x, outputs)
        ctx.gamma = gamma
        ctx.epsilon = epsilon
        return outputs

    @staticmethod
    def backward(ctx, grad_y):
        x, y = ctx.saved_tensors
        gamma = ctx.gamma
        # RMSNorm backward: grad_x = (y / rms) * (grad_y - mean(grad_y * y * gamma) / d)
        # For simplicity, fall back to PyTorch
        d = x.shape[-1]
        rms = (x.square().mean(dim=-1, keepdim=True) + ctx.epsilon).sqrt()
        grad_x = (gamma / rms) * (grad_y - (grad_y * y * gamma).sum(dim=-1, keepdim=True) * gamma / (d * rms))
        grad_gamma = (grad_y * y / rms).sum(dim=tuple(range(x.dim() - 1)))
        return grad_x, grad_gamma, None


wind_rms_norm_without_weight = WindRmsNormWithoutWeight.apply
wind_rms_norm_without_weight_backward = WindRmsNormWithoutWeightBackward.apply