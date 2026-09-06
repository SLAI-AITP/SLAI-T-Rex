/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "swi_glu_limit_backward_float.hpp"
#include "swi_glu_limit_backward_bf16.hpp"
#include "swi_glu_limit_backward_single.inc"
#include "swi_glu_limit_backward_split.inc"

extern "C" __global__ __aicore__ void wind_swi_glu_limit_backward(GM_ADDR gradout_gm, GM_ADDR self_gm, GM_ADDR other_gm,
                                                          GM_ADDR probs_gm,   // MoE per-token 权重(OPTIONAL，null=不乘)
                                                          GM_ADDR out1_gm, GM_ADDR out2_gm,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(XxGluLimitBackwardTilingData, tilingDataIn, tiling);
        const XxGluLimitBackwardTilingData* tempTilingGm = &tilingDataIn;
        if (tempTilingGm->isDoubleBuffer == 1) {
            INVOKE_SWI_GLU_SPLIT_IMPL(2)
        } else {
            INVOKE_SWI_GLU_SPLIT_IMPL(1)
        }
    } else if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(XxGluLimitBackwardSingleTilingData, tilingDataIn, tiling);
        const XxGluLimitBackwardSingleTilingData* tempTilingGm = &tilingDataIn;
        if (tempTilingGm->isDoubleBuffer == 1) {
            INVOKE_SWI_GLU_SINGLE_IMPL(2)
        } else {
            INVOKE_SWI_GLU_SINGLE_IMPL(1)
        }
    }
}
