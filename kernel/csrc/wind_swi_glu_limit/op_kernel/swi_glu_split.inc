/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "glu_split.hpp"

#define INVOKE_SWI_GLU_SPLIT_IMPL(bufferNum, self_gm, other_gm, output_gm, tiling, pipe)                      \
if constexpr (std::is_same_v<float, DTYPE_SELF>) {                                                            \
    GluSplit<SwigluVector<float, float, bufferNum>, float, float> op;                                         \
    op.Init(self_gm, other_gm, output_gm, tiling, pipe);                                                      \
    op.Process();                                                                                             \
} else {                                                                                                      \
    GluSplit<SwigluVectorBF16<DTYPE_SELF, float, DTYPE_SELF, bufferNum, true>, DTYPE_SELF, DTYPE_SELF> op;    \
    op.Init(self_gm, other_gm, output_gm, tiling, pipe);                                                      \
    op.Process();                                                                                             \
}
