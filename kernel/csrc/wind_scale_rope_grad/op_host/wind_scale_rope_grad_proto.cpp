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
#include "log/ops_log.h"

namespace ge {
inline ge::graphStatus InferShape4WindScaleRopeGrad(gert::InferShapeContext* context)
{
    // grad_x 形状 = grad_out(= x) 形状; grad_scale 形状 = scale 形状
    const auto gradOutShape = context->GetInputShape(0);   // grad_out [T,D]
    const auto scaleShape = context->GetInputShape(2);     // scale    [T,1]
    auto gradXShape = context->GetOutputShape(0);
    auto gradScaleShape = context->GetOutputShape(1);
    gradXShape->SetDimNum(gradOutShape->GetDimNum());
    for (size_t i = 0; i < gradOutShape->GetDimNum(); ++i) {
        gradXShape->SetDim(i, gradOutShape->GetDim(i));
    }
    gradScaleShape->SetDimNum(scaleShape->GetDimNum());
    for (size_t i = 0; i < scaleShape->GetDimNum(); ++i) {
        gradScaleShape->SetDim(i, scaleShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4WindScaleRopeGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));  // grad_x = grad_out dtype(bf16)
    context->SetOutputDataType(1, ge::DT_FLOAT);                  // grad_scale = fp32
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(WindScaleRopeGrad)
    .InferShape(InferShape4WindScaleRopeGrad)
    .InferDataType(InferDataType4WindScaleRopeGrad);
}
