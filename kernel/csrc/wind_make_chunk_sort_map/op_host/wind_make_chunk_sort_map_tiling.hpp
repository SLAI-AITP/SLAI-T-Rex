/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_WIND_MAKE_CHUNK_SORT_MAP_TILING_HPP
#define GTSOPS_WIND_MAKE_CHUNK_SORT_MAP_TILING_HPP

#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(WindMakeChunkSortMapTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numSplits);
    TILING_DATA_FIELD_DEF(uint32_t, usedCore);
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, outTileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindMakeChunkSortMap, WindMakeChunkSortMapTilingData);
} // namespace optiling

#endif // GTSOPS_WIND_MAKE_CHUNK_SORT_MAP_TILING_HPP
