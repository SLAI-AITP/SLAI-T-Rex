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
inline ge::graphStatus InferShape4WindScaleRope(gert::InferShapeContext* context)
{
    // out 形状与 x 完全一致(elementwise + 后rope维原地旋转)
    const auto x_shape = context->GetInputShape(0);
    auto out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(x_shape->GetDimNum());
    for (size_t i = 0; i < x_shape->GetDimNum(); ++i) {
        out_shape->SetDim(i, x_shape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4WindScaleRope(gert::InferDataTypeContext* context)
{
    auto x_dtype = context->GetInputDataType(0); // 与 x(bf16)一致
    context->SetOutputDataType(0, x_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(WindScaleRope)
    .InferShape(InferShape4WindScaleRope)
    .InferDataType(InferDataType4WindScaleRope);
}
