# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import torch
import gtsops

# GTS_ROPE_NO_SINBACK=1 强制 backward 走 (-sin) 路径(忽略 host 传的 sin_back), 用于定位问题。
_NO_SINBACK = os.getenv('GTS_ROPE_NO_SINBACK', '0') == '1'


class WindScaleRope(torch.autograd.Function):
    """forward: u = x*scale; out[...,:-rope]=u; out[...,-rope:]=fwd_rope(u[...,-rope:])。
    x:[*,D] bf16; scale:[*,1] bf16; cos/sin:[S,ropeDim] fp32(host预展开:cos每复数2份,sin=sinSigned偶位取负)。

    backward(grad_x 复用前向算子, grad_scale 用 torch reduce):
      因 scale 标量、rope 线性 => rope(v*scale)=rope(v)*scale, forward 与 backward 算子结构相同,
      只需传 inverse 版 sin(符号翻)。
        grad_u[...,:-rope] = grad_out[...,:-rope]
        grad_u[...,-rope:] = inv_rope(grad_out[...,-rope:])   <- 用同一算子 + cos/sin_bwd
        grad_x = grad_u * scale  <- 算子内的 *scale 一并完成
      即: grad_x = WindScaleRope(grad_out, scale, cos, sin_bwd), sin_bwd = -sin_fwd。

      grad_scale = sum(grad_u * x, -1) = grad_rsqrt, 必须算(rsqrt=f(q), 回传给
        WindRmsNormWithoutWeightBackward; 见接入点 g2_attention.py:303-304)。
      [0623 profiling 对账] 接入后本 backward 替代现场 q*rsqrt 反向的 eager 链:
        - 消除: slice-assign 散射(ZerosLike+ViewCopy+Add ~1900us) + Mul742(中间搬运) ≈ 2.7ms/次
        - grad_x 包含 Mul483(grad_out*rsqrt) + inverse RoPE
        - grad_scale 即 ReduceSum300(grad_rsqrt), 我们算后喂给 WindRmsNormBwd(385us, 固有不动)
      grad_scale 实测 2.285ms/次 ≈ 现场 eager 2.24ms(持平, 不劣化)。
    """
    @staticmethod
    def forward(ctx, x, scale, cos, sin, sin_back=None):
        # sin_back = 反向用的 sinSigned(= -sin, host 侧已算好), 避免 backward 在图里做 Neg。
        #   兼容旧调用(不传 sin_back): backward 退回 (-sin)。
        if _NO_SINBACK:
            sin_back = None   # 调试: 强制旧 (-sin) 路径
        ctx.save_for_backward(x, scale, cos, sin if sin_back is None else sin_back)
        ctx.has_sin_back = sin_back is not None
        return gtsops.npu_wind_scale_rope(x, scale, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        x, scale, cos, sin_or_back = ctx.saved_tensors
        # inverse RoPE 反向: 乘 conj(freqs) => sinSigned 取反。
        #   新路径: host 已传 sin_back(=-sin), 直接用(无图上 Neg); 旧路径: 在图里 (-sin)。
        sin_bwd = sin_or_back if ctx.has_sin_back else (-sin_or_back).contiguous()
        # [反向算子 WindScaleRopeGrad] 一次出 grad_x + grad_scale, 和现场 q*rsqrt 反向链一致:
        #   grad_u[:passDim]=grad_out, grad_u[passDim:]=inv_rope(grad_out[passDim:])  (grad_u 共享)
        #   grad_x = grad_u * scale;  grad_scale = sum(grad_u * x, -1)  (reduce 进 kernel, 单趟 HBM)
        grad_x, grad_scale = gtsops.npu_wind_scale_rope_grad(
            grad_out.contiguous(), x, scale, cos, sin_bwd)
        # grad_scale(=grad_rsqrt, fp32) 喂给上游 WindRmsNormWithoutWeightBackward(rsqrt=f(q) 反向)。
        return grad_x, grad_scale.to(scale.dtype), None, None, None


wind_scale_rope = WindScaleRope.apply
