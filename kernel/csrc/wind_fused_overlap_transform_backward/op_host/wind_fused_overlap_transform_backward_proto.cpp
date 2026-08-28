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
#include "log/log.h"
#include "utils/op_math.h"

namespace ge {
inline ge::graphStatus InferShape4WindFusedOverlapTransformBackward(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto isKvScoreComb = context->GetAttrs()->GetBool(2U); // kv_score_comb
    const auto grad_kv_out_shape = context->GetInputShape(0); // get grad_kv_out shape
    uint32_t B = 1;
    uint32_t R;
    uint32_t D;
    if (grad_kv_out_shape->GetDimNum() == 4U) {
        if (grad_kv_out_shape->GetDim(0) != 1U) {
            OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] grad_kv_out [B,S,2R,D], B:%d, B must be 1.",
                grad_kv_out_shape->GetDim(0));
            return ge::GRAPH_FAILED;
        }
        if (MOD(grad_kv_out_shape->GetDim(2U), 2U) != 0) {
            OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] kv [B,S,2R,D], dim[2]:%d must be 2R.", grad_kv_out_shape->GetDim(2U));
            return ge::GRAPH_FAILED;
        }
        R = grad_kv_out_shape->GetDim(2U) / 2U;
        D = grad_kv_out_shape->GetDim(3U);
    } else if (grad_kv_out_shape->GetDimNum() == 3U) {
        if (MOD(grad_kv_out_shape->GetDim(1U), 2U) != 0) {
            OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] kv [S,2R,D], dim[1]:%d must be 2R.", grad_kv_out_shape->GetDim(1U));
            return ge::GRAPH_FAILED;
        }
        R = grad_kv_out_shape->GetDim(1) / 2U;
        D = grad_kv_out_shape->GetDim(2U);
    } else {
        OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] kv only support [B,S,2R,D] or [S,2R,D], dims:%d.",
                grad_kv_out_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    // grad_score_out
    const auto grad_score_out_shape = context->GetInputShape(1); // get score_out_grad shape
    if (grad_score_out_shape == nullptr || grad_score_out_shape->GetDimNum() == 0) {
        OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] score_out_grad is null.");
        return false;
    }
    // socre的shape必须与kv一样
    if (grad_score_out_shape->GetDimNum() != grad_kv_out_shape->GetDimNum()) {
        OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] score dims:%d must be same with kv dims:%d.",
                grad_score_out_shape->GetDimNum(), grad_kv_out_shape->GetDimNum());
        return false;
    }

    for (uint32_t i = 0; i < grad_score_out_shape->GetDimNum(); i++) {
        if (grad_score_out_shape->GetDim(i) != grad_kv_out_shape->GetDim(i)) {
            OP_LOGE("wind_fused_overlap_transform_backward", "[ERROR] score dim[%d]:%d must be same with kv dim[%d]:%d.",
                    i, grad_score_out_shape->GetDim(i), i, grad_kv_out_shape->GetDim(i));
            return false;
        }
    }

    auto grad_kv_shape = context->GetOutputShape(0); // get kv_grad shape
    *grad_kv_shape = *grad_kv_out_shape;
    // [B, S, 2R, D] -> [B, S, R, 2D]
    if (isKvScoreComb) {
        grad_kv_shape->SetDim(0, 2U);
    } else {
        grad_kv_shape->SetDim(0, 1);
    }
    int lastDimIdx = grad_kv_shape->GetDimNum() - 1;
    grad_kv_shape->SetDim((lastDimIdx - 1), R);
    grad_kv_shape->SetDim(lastDimIdx, 2U * D);

    auto grad_score_shape = context->GetOutputShape(1); // get score_grad shape
    if (isKvScoreComb) {
        grad_score_shape->SetDimNum(0); // 空tensor
    } else {
        // [B, S, 2R, D] -> [B, S, R, 2D]
        *grad_score_shape = *grad_kv_shape;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4WindFusedOverlapTransformBackward(gert::InferDataTypeContext* context)
{
    auto kv_out_grad_dtype = context->GetInputDataType(0); // get kv_out_grad datatype
    context->SetOutputDataType(0, kv_out_grad_dtype); // set kv_grad datatype
    auto score_out_grad_dtype = context->GetInputDataType(0); // get score_out_grad datatype
    context->SetOutputDataType(1, score_out_grad_dtype); // set score_grad datatype
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(WindFusedOverlapTransformBackward)
    .InferShape(InferShape4WindFusedOverlapTransformBackward)
    .InferDataType(InferDataType4WindFusedOverlapTransformBackward);
}
