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
#include "xxglu_limit_tiling_host.h"        // [Add by GTS-AILab] [feature GTSOps]
#include "xxglu_limit_tiling_host_single.h" // [Add by GTS-AILab] [feature GTSOps]
#include "default_infer_shape.h"

namespace optiling {
    inline ge::graphStatus Tiling4WindSwiGluLimit(gert::TilingContext* context)   // [Add by GTS-AILab] [feature GTSOps]
    {
        ge::graphStatus result = ge::GRAPH_FAILED;
        // 第1个tensor为null则表示单输入，否则双输入
        bool isSingleInput = (context->GetOptionalInputShape(1) == nullptr);
        if (isSingleInput) {
            result = optilingLimitSingle::Tiling4Glu<optilingLimitSingle::GLU_FLAG::SWIGLU>(context);
        } else {
            result = optilingLimit::Tiling4Glu<optilingLimit::GLU_FLAG::SWIGLU>(context);
        }
        return result;
    }
}

namespace ops {

struct WindSwiGluLimit : public OpDef {   // [Add by GTS-AILab] [feature GTSOps]
public:
    explicit WindSwiGluLimit(const char* name) : OpDef(name)   // [Add by GTS-AILab] [feature GTSOps]
    {
        this->Input("self")
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("other")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("probs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Output("outputProbs") // 新增probs输出
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("limit").AttrType(OPTIONAL).Float(0.0f);   // [Add by GTS-AILab] [feature GTSOps] limit 固定 attr index1(dim 仍 index0)
        this->Attr("highPrecision").AttrType(OPTIONAL).Bool(true);   // [Add by GTS-AILab] [feature GTSOps] limit 插入后 highPrecision 顺移到 index2
        this->Attr("hasProbs").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::XxGluInferShape);
        this->SetInferDataType(ge::XxGluInferDataType);
        this->AICore().SetTiling(optiling::Tiling4WindSwiGluLimit);   // [Add by GTS-AILab] [feature GTSOps]
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(WindSwiGluLimit);   // [Add by GTS-AILab] [feature GTSOps]
}
