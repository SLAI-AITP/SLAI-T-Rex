/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <register/op_def_registry.h>

namespace ops {
// inputs=[x; scale; cos; sin]; outputs=[out]
//   x:    [T, D]      BF16  (T = B*S*N 摊平; D = head_dim = 512)
//   scale:[T, 1]      BF16  (rmsnorm 的 rstd, 每行一个)
//   cos:  [S, ropeDim] FP32 (host 预展开: 每复数 cos 重复2份到相邻两 lane, S 维索引)
//   sin:  [S, ropeDim] FP32 (host 预展开: sinSigned, 偶位=-sin 奇位=+sin, 供 out=u*cos+rotate(u)*sin)
//   out:  [T, D]      BF16
class WindScaleRope : public OpDef {
public:
    explicit WindScaleRope(const char* name) : OpDef(name)
    {
        std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(formats);
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(formats);
        this->Input("cos").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(formats);
        this->Input("sin").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(formats);
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format(formats);

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

OP_ADD(WindScaleRope);
} // namespace ops
