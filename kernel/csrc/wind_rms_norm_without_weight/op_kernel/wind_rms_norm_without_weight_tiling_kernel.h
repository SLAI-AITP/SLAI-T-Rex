/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_RMS_NORM_WITHOUT_WEIGHT_TILING_KERNEL_H
#define GTSOPS_RMS_NORM_WITHOUT_WEIGHT_TILING_KERNEL_H

#include "kernel_operator.h"
#include "utils/op_math.h"

struct RmsNormWithoutWeightTilingKernel {
    uint64_t blockRowNum = 0;
    uint64_t blockRowNumAlign512 = 0;
    uint64_t blockRowNumAlign32 = 0;
    uint64_t blockLength = 0;
    uint32_t rowEachCore = 0;
    uint32_t rowEachCoreAlign512 = 0;
    uint32_t rowEachCoreAlign32 = 0;
    uint64_t tileLoopNum = 0;
    uint32_t tailTileRow = 0;
    uint32_t tailTileRowAlign32 = 0;
    uint32_t tailTileRowAlignDown32 = 0;
    uint64_t colNum = 0;
    uint64_t rowNum = 0;
    uint64_t alignNum = 0;
    float avg_factor = 1.0;
    float epsilon = 1e-5;
    uint64_t gmXOffset = 0;
    uint64_t gmResOffset = 0;

    __aicore__ void GetTilingAndOffset(GM_ADDR tiling_gm, uint32_t inputDTypeLen)
    {
        GET_TILING_DATA(tilingDataIn, tiling_gm);
        const RmsNormWithoutWeightTilingData* tempTilingGm = &tilingDataIn;

        if (AscendC::GetBlockIdx() < tempTilingGm->baseBlockRowIndex) {
            blockRowNum = tempTilingGm->tailBlockRowNums + 1;
        } else {
            blockRowNum = tempTilingGm->tailBlockRowNums;
        }
        blockRowNumAlign512 = (blockRowNum + tempTilingGm->packLen - 1) / tempTilingGm->packLen * tempTilingGm->packLen;
        blockRowNumAlign32 = (blockRowNum + tempTilingGm->alignNum - 1) / tempTilingGm->alignNum * tempTilingGm->alignNum;
        colNum = tempTilingGm->colNum;
        rowNum = tempTilingGm->rowNum;
        alignNum = tempTilingGm->alignNum;
        blockLength = blockRowNum * colNum;
        avg_factor = tempTilingGm->avgFactor;
        rowEachCore = tempTilingGm->maxTileRow;
        rowEachCoreAlign512 = (rowEachCore + tempTilingGm->packLen - 1) / tempTilingGm->packLen * tempTilingGm->packLen;
        rowEachCoreAlign32 = (rowEachCore + tempTilingGm->alignNum - 1) / tempTilingGm->alignNum * tempTilingGm->alignNum;
        epsilon = tempTilingGm->epsilon;

        tileLoopNum = OpDiv(blockRowNum, rowEachCore);
        tailTileRow = blockRowNum - rowEachCore * tileLoopNum;
        tailTileRowAlign32 = (tailTileRow + tempTilingGm->alignNum - 1) / tempTilingGm->alignNum * tempTilingGm->alignNum;
        tailTileRowAlignDown32 = tailTileRow / tempTilingGm->alignNum * tempTilingGm->alignNum;

        if (AscendC::GetBlockIdx() <= tempTilingGm->baseBlockRowIndex) {
            gmXOffset = (tempTilingGm->tailBlockRowNums + 1) * colNum * AscendC::GetBlockIdx();
            gmResOffset = (tempTilingGm->tailBlockRowNums + 1) * AscendC::GetBlockIdx();
        } else {
            gmXOffset = (tempTilingGm->tailBlockRowNums + 1) * colNum * tempTilingGm->baseBlockRowIndex
                + tempTilingGm->tailBlockRowNums * colNum * (AscendC::GetBlockIdx() - tempTilingGm->baseBlockRowIndex);
            gmResOffset = (tempTilingGm->tailBlockRowNums + 1) * tempTilingGm->baseBlockRowIndex
                        + tempTilingGm->tailBlockRowNums * (AscendC::GetBlockIdx() - tempTilingGm->baseBlockRowIndex);
        }
    }
};

#endif
