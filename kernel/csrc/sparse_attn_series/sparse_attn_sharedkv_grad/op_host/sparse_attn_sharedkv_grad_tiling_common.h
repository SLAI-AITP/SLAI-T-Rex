/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_grad_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>
#include "err/ops_err.h"
#include "log/log.h"

namespace optiling {
namespace sasg {

constexpr const char *BSND_STR = "BSND";
constexpr const char *TND_STR = "TND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;

enum class InputLayout : uint32_t {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

enum class InputIndex : uint32_t {
    QUERY = 0,
    ORI_KV,
    CMP_KV,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    LSE,
    ORI_SPARSE_INDICES,
    CMP_SPARSE_INDICES,
    CU_SEQLENS_Q,
    CU_SEQLENS_ORI_KV,
    CU_SEQLENS_CMP_KV,
    SINKS
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DORI_KV,
    DCMP_KV,
    DSINKS
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    CMP_RATIO,
    ORI_MASK_MODE,
    CMP_MASK_MODE,
    ORI_WIN_LEFT,
    ORI_WIN_RIGHT,
    INPUT_LAYOUT
};

struct SparseAttnSharedkvGradCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
};

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b)
{
    if (b == 0U) {
        return a;
    }
    return (a + b - 1U) / b * b;
}

} // namespace sasg
} // namespace optiling
