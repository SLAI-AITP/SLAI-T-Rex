/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"
#include "xxglu_limit_tiling_host.h"
#include "xxglu_limit_tiling_host_single.h"
#include "default_infer_shape.h"

namespace optiling {
inline ge::graphStatus Tiling4WindSwiGluLimitBackward(gert::TilingContext* context)
{
    ge::graphStatus result = ge::GRAPH_FAILED;
    // 第2个tensor为null则表示单输入，否则双输入
    bool isSingleInput = (context->GetOptionalInputShape(2) == nullptr);
    if (isSingleInput) {
        result = optilingLimitBackwardSingle::Tiling4Glu<optilingLimitBackwardSingle::GLU_FLAG::SWIGLU_BACKWARD>(context);
    } else {
        result = optilingLimitBackward::Tiling4Glu<optilingLimitBackward::GLU_FLAG::SWIGLU_BACKWARD>(context);
    }
    return result;
}
}

namespace ops {
struct WindSwiGluLimitBackward : public OpDef {
public:
    explicit WindSwiGluLimitBackward(const char* name) : OpDef(name)
    {
        this->Input("gradout")
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("self")
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // probs: MoE per-token routing 权重 [T,1] fp32（OPTIONAL）。
        // 反向把 grad_h = gradout * probs 收进算子（消除外部独立 Mul([*,1])+Cast）。null 时与现状逐字节等价。
        // 注：probs 排在 other(idx2) 之后 = idx3，不破坏现有 GetOptionalInputShape(2) 判单/双输入的逻辑。
        this->Input("probs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out1")
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("limit").AttrType(OPTIONAL).Float(0.0f);   // limit 固定 attr index1(dim 仍 index0)

        this->SetInferShape(ge::XxGluInferShape<true>);
        this->AICore().SetTiling(optiling::Tiling4WindSwiGluLimitBackward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(WindSwiGluLimitBackward);
}
