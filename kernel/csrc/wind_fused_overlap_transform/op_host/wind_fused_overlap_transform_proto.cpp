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
inline ge::graphStatus InferShape4WindFusedOverlapTransform(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto score_shape = context->GetOptionalInputShape(1); // get optional score shape
    bool isKvScoreComb = score_shape == nullptr || score_shape->GetDimNum() == 0; // kv和score合并为一个tensor，B=2，前面一个B为kv，后面一个B为score

    uint32_t B = 1;
    uint32_t R;
    uint32_t D;
    const auto kv_shape = context->GetInputShape(0); // get kv shape
    if (kv_shape->GetDimNum() == 4U) {
        if ((isKvScoreComb && kv_shape->GetDim(0) != 2U) ||
            (!isKvScoreComb && kv_shape->GetDim(0) != 1)) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv [B,S,R,2D], B:%d, kv_score_comb:%d, "
                "B must be 2 when kv_score_comb is true, or B must be 1 when kv_score_comb is false",
                kv_shape->GetDim(0), isKvScoreComb);
            return ge::GRAPH_FAILED;
        }
        if (MOD(kv_shape->GetDim(3U), 2U) != 0) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv [B,S,R,2D], dim[3]:%d must be 2D.",
                    kv_shape->GetDim(3U));
            return ge::GRAPH_FAILED;
        }
        R = kv_shape->GetDim(2U);
        D = kv_shape->GetDim(3U) / 2U;
    } else if (kv_shape->GetDimNum() == 3U) {
        if (isKvScoreComb) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv must be [B,S,R,2D], when kv_score_comb is true.dims:%d",
                    kv_shape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (MOD(kv_shape->GetDim(2U), 2U) != 0) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv [S,R,2D], dim[2]:%d must be 2D.",
                    kv_shape->GetDim(2U));
            return ge::GRAPH_FAILED;
        }
        R = kv_shape->GetDim(1);
        D = kv_shape->GetDim(2U) / 2U;
    } else {
        OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv only support [B,S,R,2D] or [S,R,2D], dims:%d.",
                kv_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (!isKvScoreComb) {
        // socre的shape必须与kv一样
        if (score_shape->GetDimNum() != kv_shape->GetDimNum()) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] score dims:%d must be same with kv dims:%d.",
                    score_shape->GetDimNum(), kv_shape->GetDimNum());
            return false;
        }

        for (uint32_t i = 0; i < score_shape->GetDimNum(); i++) {
            if (score_shape->GetDim(i) != kv_shape->GetDim(i)) {
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] score dim[%d]:%d must be same with kv dim[%d]:%d.",
                        i, score_shape->GetDim(i), i, kv_shape->GetDim(i));
                return false;
            }
        }
    }

    auto kv_out_shape = context->GetOutputShape(0); // get out shape
    *kv_out_shape = *kv_shape;
    // [B, S, R, 2D] -> [B, S, 2R, D]
    if (isKvScoreComb) {
        kv_out_shape->SetDim(0, B);
    }
    int lastDimIdx = kv_out_shape->GetDimNum() - 1;
    kv_out_shape->SetDim((lastDimIdx - 1), 2U * R);
    kv_out_shape->SetDim(lastDimIdx, D);

    auto score_out_shape = context->GetOutputShape(1); // get out shape
    // [B, S, R, 2D] -> [B, S, 2R, D]
    *score_out_shape = *kv_out_shape;

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4WindFusedOverlapTransform(gert::InferDataTypeContext* context)
{
    auto kv_dtype = context->GetInputDataType(0); // get kv datatype
    context->SetOutputDataType(0, kv_dtype); // set kv_out datatype
    context->SetOutputDataType(1, kv_dtype); // set score_out datatype
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(WindFusedOverlapTransform)
    .InferShape(InferShape4WindFusedOverlapTransform)
    .InferDataType(InferDataType4WindFusedOverlapTransform);
}
