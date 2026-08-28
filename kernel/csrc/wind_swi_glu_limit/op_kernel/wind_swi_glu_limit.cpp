/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "swi_glu_float.hpp"   // [Add by GTS-AILab] [feature GTSOps]
#include "swi_glu_bf16.hpp"    // [Add by GTS-AILab] [feature GTSOps]
#include "swi_glu_bf16_probs.hpp"
#include "swi_glu_single.inc"  // [Add by GTS-AILab] [feature GTSOps]
#include "swi_glu_split.inc"   // [Add by GTS-AILab] [feature GTSOps]
#include "swi_glu_single_probs.inc"

extern "C" __global__ __aicore__ void wind_swi_glu_limit(GM_ADDR self_gm, GM_ADDR other_gm, GM_ADDR probs_gm, GM_ADDR output_gm, GM_ADDR outputProbs_gm,   // [Add by GTS-AILab] [feature GTSOps]
                                                 GM_ADDR workspace, GM_ADDR tiling) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        // [feature] [multi tiling data] 获取1：显式指定 limit 专用 tiling 结构，避免和通用 tiling 结构重名
        GET_TILING_DATA_WITH_STRUCT(XxGluLimitTilingData, tilingDataIn, tiling);
        const XxGluLimitTilingData* tempTilingGm = &tilingDataIn;
        if (tempTilingGm->isDoubleBuffer == 1) {
            INVOKE_SWI_GLU_SPLIT_IMPL(2, self_gm, other_gm, output_gm, tempTilingGm, pipe)
        } else {
            INVOKE_SWI_GLU_SPLIT_IMPL(1, self_gm, other_gm, output_gm, tempTilingGm, pipe)
        }
    } else if (TILING_KEY_IS(1)) {
        // [feature] [multi tiling data] 获取2：非缺省结构必须带Tiling结构体名来获取
        GET_TILING_DATA_WITH_STRUCT(XxGluLimitSingleTilingData, tilingDataIn, tiling);
        const XxGluLimitSingleTilingData* tempTilingGm = &tilingDataIn;
        if (tempTilingGm->isDoubleBuffer == 1) {
            if (tempTilingGm->highPrecision) {
                if (tempTilingGm->hasProbs) {
                    INVOKE_SWI_GLU_SINGLE_PROBS_IMPL(2, true, true)
                } else {
                    INVOKE_SWI_GLU_SINGLE_IMPL(2, true)
                }
            } else {
                if (tempTilingGm->hasProbs) {
                    INVOKE_SWI_GLU_SINGLE_PROBS_IMPL(2, false, true)
                } else {
                    INVOKE_SWI_GLU_SINGLE_IMPL(2, false)
                }
            }
        } else {
            if (tempTilingGm->highPrecision) {
                if (tempTilingGm->hasProbs) {
                    INVOKE_SWI_GLU_SINGLE_PROBS_IMPL(1, true, true)
                } else {
                    INVOKE_SWI_GLU_SINGLE_IMPL(1, true)
                }
            } else {
                if (tempTilingGm->hasProbs) {
                    INVOKE_SWI_GLU_SINGLE_PROBS_IMPL(1, false, true)
                } else {
                    INVOKE_SWI_GLU_SINGLE_IMPL(1, false)
                }
            }
        }
    }
}

