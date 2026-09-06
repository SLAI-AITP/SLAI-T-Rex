/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_WindFusedOverlapTransform_TILING_HPP
#define GTSOPS_WindFusedOverlapTransform_TILING_HPP
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(WindFusedOverlapTransformTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, R);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    TILING_DATA_FIELD_DEF(float, kvValue);
    TILING_DATA_FIELD_DEF(float, scoreValue);
    TILING_DATA_FIELD_DEF(uint32_t, tBaseLen);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindFusedOverlapTransform, WindFusedOverlapTransformTilingData);
}
#endif // GTSOPS_WindFusedOverlapTransform_TILING_HPP
