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
#include "wind_make_chunk_sort_map.hpp"

extern "C" __global__ __aicore__ void wind_make_chunk_sort_map(GM_ADDR splitSizesGm,
                                                               GM_ADDR sortedIndicesGm,
                                                               GM_ADDR rowIdMapGm,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(WindMakeChunkSortMapTilingData, gmTilingData, tiling);
    AscendC::WindMakeChunkSortMap<DTYPE_SPLIT_SIZES, DTYPE_SORTED_INDICES> op(gmTilingData, pipe);
    if (TILING_KEY_IS(0)) {
        op.Process(splitSizesGm, sortedIndicesGm, rowIdMapGm);
    }
}
