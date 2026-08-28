/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_SWIGLU_LIMIT_V2_TILING_KERNEL_H   // [Add by GTS-AILab] [feature GTSOps]
#define OP_SWIGLU_LIMIT_V2_TILING_KERNEL_H   // [Add by GTS-AILab] [feature GTSOps]
#include "utils/op_math.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220 && !ASCENDC_CPU_DEBUG
using bfloat16_t = half;
#endif

const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t HEAD_AND_TAIL = 2; // pack unit in cache 512B
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B

// tiling for XxGlu Vector on one VectorCore
struct XxgluTilingKernel {
    float limit = 0.0f; // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪
    uint64_t blockLength = 0; // number of calculations on this core
    uint32_t tileLength = 0;  // number of calculations in one tile
    uint64_t tileLoopNum = 0; // number of tile(tileLength per tile) on this core, don't include tailTile
    uint32_t tailTileLen = 0; // number of calculations in tail tile

    uint64_t gmOffset = 0;

    bool warmup;
    uint64_t headElemNum;
    uint64_t tailElemNum;

    bool isTile32BAlign = true;
    bool isTailTile32BAlign = true;

    // calc tiling data
    __aicore__ void GetTilingAndOffset(const XxGluLimitTilingData* tempTilingGm, uint32_t inputDTypeLen)
    {
        // Pre-condition:The singleCoreBlockLens[i] is multiples of 32Bytes,and the maxTileLen is multiples of 512Bytes.
        // Ensured by the tiling algorithm on the host
        limit = tempTilingGm->limit; // [Add by GTS-AILab] [feature GTSOps] 从 tiling 读出 clamp 阈值
        blockLength = tempTilingGm->singleCoreBlockLens[AscendC::GetBlockIdx()];
        tileLength = tempTilingGm->maxTileLen;
        isTile32BAlign = MOD(tileLength * inputDTypeLen, DEFAULT_ALIGN_SIZE) == 0;
        // number of tile(tileLength per tile) on this core, don't include tailTile
        tileLoopNum = OpDiv(blockLength, tileLength, blockLength);
        tailTileLen = blockLength - tileLength * tileLoopNum;
        isTailTile32BAlign = MOD(tileLength * inputDTypeLen, DEFAULT_ALIGN_SIZE) == 0;

        gmOffset = 0;
        for (uint32_t i = 0; i < AscendC::GetBlockIdx(); i++) {
            gmOffset += tempTilingGm->singleCoreBlockLens[i];
        }

        uint64_t packElemNum = (inputDTypeLen == 0) ? PACK_SIZE : (PACK_SIZE / inputDTypeLen);
        warmup = tileLoopNum > 1 && tileLength >= packElemNum * HEAD_AND_TAIL;
        if (warmup) {
            headElemNum = ((tileLength / HEAD_AND_TAIL) + packElemNum - 1) / packElemNum * packElemNum;
            tailElemNum = tileLength - headElemNum;
        }
    }
};

#define XXGLU_PROCESS(beginIndex, kernelTiling) \
do { \
    for (uint64_t i = beginIndex; i < kernelTiling.tileLoopNum; i++) { \
        CopyIn(i * kernelTiling.tileLength, kernelTiling.tileLength); \
        this->Compute(kernelTiling.tileLength); \
        CopyOut(i * kernelTiling.tileLength, kernelTiling.tileLength); \
    } \
    if (kernelTiling.tailTileLen > 0) { \
        CopyIn(kernelTiling.tileLoopNum * kernelTiling.tileLength, kernelTiling.tailTileLen); \
        this->Compute(kernelTiling.tailTileLen); \
        CopyOut(kernelTiling.tileLoopNum * kernelTiling.tileLength, kernelTiling.tailTileLen); \
    } \
} while (0)

#endif // OP_SWIGLU_LIMIT_V2_TILING_KERNEL_H   // [Add by GTS-AILab] [feature GTSOps]
