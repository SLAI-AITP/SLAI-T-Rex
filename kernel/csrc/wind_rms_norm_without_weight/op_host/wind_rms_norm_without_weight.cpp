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
#include "wind_rms_norm_without_weight_tiling.hpp"
#include "utils/op_math.h"
namespace ge {
    static ge::graphStatus WindRmsNormWithoutWeightInferShape(gert::InferShapeContext *context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape *xShape = context->GetInputShape(0);
        if (xShape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        gert::Shape *resShape = context->GetOutputShape(0);
        *resShape = *xShape;

        size_t x_dim_num = xShape->GetDimNum();
        // 将最后一个dim设为1
        if (x_dim_num > 0) {
            resShape->SetDim(x_dim_num - 1, 1);
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus WindRmsNormWithoutWeightInferDataType(gert::InferDataTypeContext *context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto input_dtype = context->GetInputDataType(0);
        auto attrs = context->GetAttrs();
        auto outDtypePtr = attrs->GetInt(1);
        ge::DataType outDataType = ConvertAtDtypeToGeDtype(*outDtypePtr, input_dtype);
        context->SetOutputDataType(0, outDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
class WindRmsNormWithoutWeight : public OpDef {
public:
    explicit WindRmsNormWithoutWeight(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("outDtype").AttrType(OPTIONAL).Int(0); // 输出的dtype类型

        this->SetInferShape(ge::WindRmsNormWithoutWeightInferShape)
            .SetInferDataType(ge::WindRmsNormWithoutWeightInferDataType);

        this->AICore().SetTiling(optiling::Tiling4RmsNormWithoutWeight);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_without_bf16;
        config_without_bf16.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_without_bf16.Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_without_bf16.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend310p", config_without_bf16);
        this->AICore().AddConfig("ascend910", config_without_bf16);
    }
};

OP_ADD(WindRmsNormWithoutWeight);
}
