/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inc/aclnn_common.h"
#include "pybind.h"

at::Tensor npu_wind_make_chunk_sort_map(const at::Tensor &splitSizes,
                                        const at::Tensor &sortedIndices,
                                        int64_t numTokens)
{
    TORCH_CHECK(numTokens >= 0, "numTokens must be non-negative");
    at::Tensor result = at::empty({numTokens}, splitSizes.options().dtype(at::kInt));
    if (numTokens == 0) {
        return result;
    }
    ACLNN_CMD(aclnnWindMakeChunkSortMap, splitSizes, sortedIndices, numTokens, result);
    return result;
}
