/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_WindScaleRope_TILING_HPP
#define GTSOPS_WindScaleRope_TILING_HPP
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(WindScaleRopeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, T);           // 总行数 = B*S*N
    TILING_DATA_FIELD_DEF(uint32_t, S);           // seq 长度(cos/sin 的索引维)
    TILING_DATA_FIELD_DEF(uint32_t, N);           // head 数(保留)
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerSeq);  // = T/S = B*N。seq = row/rowsPerSeq(摊平顺序 (s,b,n))
    TILING_DATA_FIELD_DEF(uint32_t, D);           // head_dim = 512
    TILING_DATA_FIELD_DEF(uint32_t, ropeDim);     // rope_head_dim = 64
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore); // 每核处理行数(尾核可能少)
    TILING_DATA_FIELD_DEF(uint32_t, usedCore);    // 实际用核数
    TILING_DATA_FIELD_DEF(uint32_t, rowTileLen);  // 每次 tile 的行数(double buffer)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindScaleRope, WindScaleRopeTilingData);
}
#endif // GTSOPS_WindScaleRope_TILING_HPP
