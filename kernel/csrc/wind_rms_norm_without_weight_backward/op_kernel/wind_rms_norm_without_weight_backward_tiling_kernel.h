/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_TILING_KERNEL_H
#define GTSOPS_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_TILING_KERNEL_H

#include "kernel_operator.h"
#include "utils/op_math.h"

const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B

struct RmsNormWithoutWeightBackwardTilingKernel {
    uint64_t blockRowNum = 0;
    uint64_t blockLength = 0;
    uint32_t rowEachCore = 0;
    uint64_t tileLoopNum = 0;
    uint32_t tailTileRow = 0;
    uint64_t colNum = 0;
    uint64_t rowNum = 0;
    uint64_t gmOffset = 0;
    float avg_factor = 1.0;
    uint64_t blockRowNumAliagn512 = 0;
    uint64_t blockRowNumAliagn32 = 0;

    // 长shape模式下的参数
    uint64_t colNumPerUB = 0;
    uint64_t colNumPerUBAlignDown32 = 0;
    uint64_t colLoopNum = 0;
    uint64_t colTileNum = 0;
    uint64_t colTileNumAlignUp32 = 0;

    __aicore__ void GetTilingAndOffset(GM_ADDR tiling_gm, uint32_t inputDTypeLen)
    {
        GET_TILING_DATA(tilingDataIn, tiling_gm);
        const RmsNormWithoutWeightBackwardTilingData* tempTilingGm = &tilingDataIn;

        if (AscendC::GetBlockIdx() < tempTilingGm->baseBlockRowIndex) {
            blockRowNum = tempTilingGm->tailBlockRowNums + 1;
        } else {
            blockRowNum = tempTilingGm->tailBlockRowNums;
        }
        colNum = tempTilingGm->colNum;
        rowNum = tempTilingGm->rowNum;
        blockLength = blockRowNum * colNum;
        rowEachCore = tempTilingGm->maxTileRow;
        avg_factor = tempTilingGm->avgFactor;

        tileLoopNum = OpDiv(blockRowNum, rowEachCore);
        tailTileRow = blockRowNum - rowEachCore * tileLoopNum;

        // 在长shape场景下赋值
        if (tempTilingGm->longMode == 1) {
            colNumPerUB = tempTilingGm->colNumPerUB;
            colNumPerUBAlignDown32 = tempTilingGm->colNumPerUBAlignDown32;
            colLoopNum = tempTilingGm->colLoopNum;
            colTileNum = tempTilingGm->colTileNum;
            colTileNumAlignUp32 = tempTilingGm->colTileNumAlignUp32;
        }

        blockRowNumAliagn512 = (blockRowNum + tempTilingGm->packLen - 1) / tempTilingGm->packLen * tempTilingGm->packLen;
        blockRowNumAliagn32 = (blockRowNum + tempTilingGm->alignNum - 1) / tempTilingGm->alignNum * tempTilingGm->alignNum;
        if (AscendC::GetBlockIdx() <= tempTilingGm->baseBlockRowIndex) {
            gmOffset = (tempTilingGm->tailBlockRowNums + 1) * colNum * AscendC::GetBlockIdx();
        } else {
            gmOffset = (tempTilingGm->tailBlockRowNums + 1) * colNum * tempTilingGm->baseBlockRowIndex
                + tempTilingGm->tailBlockRowNums * colNum * (AscendC::GetBlockIdx() - tempTilingGm->baseBlockRowIndex);
        }
    }
};

#endif
