/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <kernel_operator.h>
#include "wind_fused_overlap_transform_backward.hpp"

extern "C" __global__ __aicore__ void wind_fused_overlap_transform_backward(GM_ADDR kv_out_gradGm, GM_ADDR score_out_gradGm,
                                                                            GM_ADDR grad_kvGm, GM_ADDR grad_scoreGm, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(WindFusedOverlapTransformBackwardTilingData, gmTilingData, tiling);
    if (TILING_KEY_IS(0)) {
        AscendC::WindFusedOverlapTransformBackwardKernel<false> op(gmTilingData, pipe);
        op.Process(kv_out_gradGm, score_out_gradGm, grad_kvGm, grad_scoreGm);
    } else if (TILING_KEY_IS(1)) {
        AscendC::WindFusedOverlapTransformBackwardKernel<true> op(gmTilingData, pipe);
        op.Process(kv_out_gradGm, score_out_gradGm, grad_kvGm, grad_scoreGm);
    }
}