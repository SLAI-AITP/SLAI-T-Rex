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
#include "wind_rms_norm_without_weight_backward_tiling.hpp"

namespace ge {
    static ge::graphStatus WindRmsNormWithoutWeightBackwardInferShape(gert::InferShapeContext *context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape *gradResShape = context->GetInputShape(0);
        const gert::Shape *xShape = context->GetInputShape(1);
        if (gradResShape == nullptr || xShape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        gert::Shape *gradXShape = context->GetOutputShape(0);
        *gradXShape = *xShape;

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus WindRmsNormWithoutWeightBackwardInferDataType(gert::InferDataTypeContext *context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(0, context->GetInputDataType(1));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
class WindRmsNormWithoutWeightBackward : public OpDef {
public:
    explicit WindRmsNormWithoutWeightBackward(const char* name) : OpDef(name)
    {
        this->Input("grad_res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::WindRmsNormWithoutWeightBackwardInferShape)
            .SetInferDataType(ge::WindRmsNormWithoutWeightBackwardInferDataType);

        this->AICore().SetTiling(optiling::Tiling4RmsNormWithoutWeightBackward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_without_bf16;
        config_without_bf16.Input("grad_res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        config_without_bf16.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_without_bf16.Input("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        config_without_bf16.Output("grad_x")
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

    OP_ADD(WindRmsNormWithoutWeightBackward);
}