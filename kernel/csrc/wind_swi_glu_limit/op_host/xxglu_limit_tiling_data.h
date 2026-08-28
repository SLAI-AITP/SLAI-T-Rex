/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_SWIGLU_LIMIT_V2_TILING_DATA_H   // [Add by GTS-AILab] [feature GTSOps]
#define GTSOPS_SWIGLU_LIMIT_V2_TILING_DATA_H   // [Add by GTS-AILab] [feature GTSOps]
#include <register/tilingdata_base.h>
#include "utils/op_math.h"

const uint32_t MAX_CORE_NUMBER = 64;
const uint32_t NUMBER_OF_DOUBLE = 2;

namespace optiling {
BEGIN_TILING_DATA_DEF(XxGluLimitTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);       // number of vector core. Don't move, must be in the first
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);    // is double buffer on?
    TILING_DATA_FIELD_DEF(uint64_t, maxTileLen);        // number of calculations in one tile On each core. Unit(number)

    // vector dimension on each core
    // Apply for memory based on the maximum number of vector cores, but use the memory based on usedCoreNum
    // BlockLen on each core may be different. Unit: number of elements
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUMBER, singleCoreBlockLens);
    TILING_DATA_FIELD_DEF(float, limit);                // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindSwiGluLimit, XxGluLimitTilingData)      // [Add by GTS-AILab] [feature GTSOps] 注册1：必须有一个缺省的注册

// tiling for single input tensor, the memory of the two input tenors
BEGIN_TILING_DATA_DEF(XxGluLimitSingleTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);      // Is 32-byte aligned for split colLen?
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);    // is double buffer on?
    // eg: for shape:[8192,1,2*3904], totalRowLen is 8192
    TILING_DATA_FIELD_DEF(uint64_t, rowLen);            // row length for split vector, Unit:element
    // eg: for shape:[8192,1,2*3904], totalColLen is 3904
    TILING_DATA_FIELD_DEF(uint64_t, colLen);            // column length for split vector, Unit:element
    TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);        // for one tile in one core, Unit:element
    TILING_DATA_FIELD_DEF(uint32_t, baseColLen);        // for one tile in one core, Unit:element
    TILING_DATA_FIELD_DEF(bool, highPrecision);         // default high precision
    TILING_DATA_FIELD_DEF(float, limit);                // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪
    TILING_DATA_FIELD_DEF(bool, hasProbs);                // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindSwiGluLimit_1, XxGluLimitSingleTilingData)   // [Add by GTS-AILab] [feature GTSOps] 注册2：其余结构算子名带后缀，后缀内容不限
};

#endif // GTSOPS_SWIGLU_LIMIT_V2_TILING_DATA_H   // [Add by GTS-AILab] [feature GTSOps]
