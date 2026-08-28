/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if ASCENDC_CPU_DEBUG
#define DTYPE_X half
#endif

#include "wind_rms_norm_without_weight.hpp"

extern "C" __global__ __aicore__ void wind_rms_norm_without_weight(GM_ADDR xGm, GM_ADDR resGm, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tempTilingGm, tiling);
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)
    if (TILING_KEY_IS(0)) {
        WindRmsNormWithoutWeight op(tempTilingGm, pipe);
        op.Init(xGm, resGm);
        op.Process();
    }
#endif
}
