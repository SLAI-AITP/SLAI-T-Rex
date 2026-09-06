/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 *   前向: out = rope(x*scale); 反向给 grad_out 求 grad_x, grad_scale。
 *   和现场 q*rsqrt 反向链一致(grad_u 共享):
 *     grad_u[:passDim]  = grad_out[:passDim]              # 直通段
 *     grad_u[passDim:]  = inv_rope(grad_out[passDim:])    # inverse RoPE(乘 conj, 即 sin 翻号)
 *     grad_x            = grad_u * scale                  # = 现场 Mul483
 *     grad_scale        = ReduceSum(grad_u * x, -1)       # = 现场 Mul742+ReduceSum, 进 kernel 单趟
 *   grad_scale 喂给 WindRmsNormWithoutWeightBackward(rsqrt=f(q) 的反向, 不归本算子)。
 * Create: 2026-06-23
 */
#include <register/op_def_registry.h>

namespace ops {
// inputs=[grad_out; x; scale; cos; sin]; outputs=[grad_x; grad_scale]
//   grad_out:[T,D]      BF16  (上游梯度, 对 out)
//   x:       [T,D]      BF16  (前向输入 q, 用于 grad_scale=sum(grad_u*x,-1))
//   scale:   [T,1]      BF16 或 FP32 (= rsqrt, 两档 dtype 与前向 WindScaleRope 一一对应:
//                                     档0 BF16 <- rms0 算子; 档1 FP32 <- npu_rms_norm/torch.rsqrt)
//   cos:     [S,ropeDim] FP32 (host 预展开, 与前向同)
//   sin:     [S,ropeDim] FP32 (host 预展开 sinSigned; 反向【inverse】需调用方传 -sin)
//   grad_x:  [T,D]      BF16
//   grad_scale:[T,1]    FP32  (grad_rsqrt, 喂给 rmsnorm 反向; 两档都固定 fp32 保精度,
//                              由 python 侧 .to(scale.dtype) 还原成 scale 的 dtype)
class WindScaleRopeGrad : public OpDef {
public:
    explicit WindScaleRopeGrad(const char* name) : OpDef(name)
    {
        std::vector<ge::Format> fmt = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("grad_out").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(fmt);
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(fmt);
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT}).Format(fmt);
        this->Input("cos").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(fmt);
        this->Input("sin").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(fmt);
        this->Output("grad_x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(fmt);
        this->Output("grad_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(fmt);

        OpAICoreConfig aicore_conf;
        aicore_conf.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("prebuildPattern.value", "Opaque")
                .ExtendCfgInfo("coreType.value", "AiCore")
                .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_conf);
        this->AICore().AddConfig("ascend910_93", aicore_conf);
    }
};

OP_ADD(WindScaleRopeGrad);
} // namespace ops
