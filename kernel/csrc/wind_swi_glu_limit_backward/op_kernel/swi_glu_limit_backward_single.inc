/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "glu_limit_backward_single.hpp"

#define INVOKE_SWI_GLU_SINGLE_IMPL(bufferNum)                                                                        \
if constexpr (std::is_same_v<float, DTYPE_SELF>) {                                                                   \
    GluLimitBackwardSingle<SwiGluLimitBackwardVector<float, float, float, float, float, bufferNum>, float, float> op; \
    op.Init(gradout_gm, self_gm, probs_gm, out1_gm, out2_gm, tempTilingGm, pipe);  /* probs_gm + out2_gm(grad_probs) 透传 */ \
    op.Process();                                                                                                    \
} else {                                                                                                             \
    GluLimitBackwardSingle<SwiGluLimitBackwardBF16<DTYPE_SELF, float, DTYPE_SELF, bufferNum>, DTYPE_SELF, DTYPE_SELF> op; \
    op.Init(gradout_gm, self_gm, probs_gm, out1_gm, out2_gm, tempTilingGm, pipe);  /* probs_gm + out2_gm(grad_probs) 透传 */ \
    op.Process();                                                                                                    \
}
