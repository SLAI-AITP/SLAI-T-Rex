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
 * \file sparse_attn_sharedkv_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "sparse_attn_sharedkv_grad_tiling_common.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SparseAttnSharedkvGradSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseM);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseN);
TILING_DATA_FIELD_DEF(int64_t, s1BasicSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvGradSplitCoreParamsOp, SparseAttnSharedkvGradSplitCoreParams)

BEGIN_TILING_DATA_DEF(SasgPostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkSpaceOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SasgPostParamsOp, SasgPostParams)

BEGIN_TILING_DATA_DEF(SparseAttnSharedkvGradBasicParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, S3);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm12WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
TILING_DATA_FIELD_DEF(int64_t, cmpRatio);
TILING_DATA_FIELD_DEF(int64_t, oriWinLeft);
TILING_DATA_FIELD_DEF(int64_t, oriWinRight);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvGradBasicParamsOp, SparseAttnSharedkvGradBasicParams)

BEGIN_TILING_DATA_DEF(SparseAttnSharedkvGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseAttnSharedkvGradBasicParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SparseAttnSharedkvGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SasgPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvGrad, SparseAttnSharedkvGradTilingData)
} // namespace optiling
