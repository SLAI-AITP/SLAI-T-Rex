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


class NpuWindFusedOverlapTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score, compress_ratio=4, head_dim=128, kv_value=0.0, score_value=0.0):
        ctx.compress_ratio = compress_ratio
        ctx.head_dim = head_dim
        kv_out, score_out = gtsops.npu_wind_fused_overlap_transform(kv, score, compress_ratio,
                                                                    head_dim, kv_value, score_value)
        return kv_out.contiguous(), score_out.contiguous()

    @staticmethod
    def backward(ctx, grad_kv_out, grad_score_out):
        grad_kv, grad_score = gtsops.npu_wind_fused_overlap_transform_backward(grad_kv_out.contiguous(),
                                                                               grad_score_out.contiguous(),
                                                                               ctx.compress_ratio, ctx.head_dim, False)
        return grad_kv, grad_score, None, None, None, None


wind_fused_overlap_transform = NpuWindFusedOverlapTransform.apply
