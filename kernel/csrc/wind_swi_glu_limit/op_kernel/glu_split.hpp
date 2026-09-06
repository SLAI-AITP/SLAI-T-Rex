/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_SWI_GLU_LIMIT_SPLIT_HPP   // [Add by GTS-AILab] [feature GTSOps]
#define CANN_SWI_GLU_LIMIT_SPLIT_HPP   // [Add by GTS-AILab] [feature GTSOps]
#include "xxglu_limit_tiling_kernel.h"   // [Add by GTS-AILab] [feature GTSOps]

template<typename ParentClass, typename inType, typename outType>
class GluSplit : public ParentClass {
public:
    __aicore__ inline GluSplit() = default;
    __aicore__ inline ~GluSplit() = default;

    __aicore__ inline void Init(GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR c_gm, const XxGluLimitTilingData* tiling_gm, TPipe& pipe)
    {
        tiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(a_gm, b_gm, c_gm);
        this->InitUbBuffer(pipe, tiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        uint64_t beginIndex = 0;
        if (tiling.warmup) {
            beginIndex = 1;

            CopyIn(0, tiling.headElemNum);
            this->Compute(tiling.headElemNum);
            CopyOut(0, tiling.headElemNum);
        }

        XXGLU_PROCESS(beginIndex, tiling);

        if (tiling.warmup) {
            CopyIn(tiling.headElemNum, tiling.tailElemNum);
            this->Compute(tiling.tailElemNum);
            CopyOut(tiling.headElemNum, tiling.tailElemNum);
        }
    }

protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR c_gm)
    {
        this->limit = tiling.limit; // [Add by GTS-AILab] [feature GTSOps] 透传 clamp 阈值给 Compute 类
        // get start index for current core, core parallel
        this->aGm.SetGlobalBuffer((__gm__ inType*)a_gm + tiling.gmOffset, tiling.blockLength);
        this->bGm.SetGlobalBuffer((__gm__ inType*)b_gm + tiling.gmOffset, tiling.blockLength);
        this->cGm.SetGlobalBuffer((__gm__ outType*)c_gm + tiling.gmOffset, tiling.blockLength);
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint64_t curTileLen)
    {
        // Copy A
        LocalTensor<inType> aLocal_ = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal_, this->aGm[offset], curTileLen);
        this->inQueueA.template EnQue(aLocal_);
        // Copy B
        LocalTensor<inType> bLocal_ = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal_, this->bGm[offset], curTileLen);
        this->inQueueB.template EnQue(bLocal_);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t curTileLen)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<outType> cLocal = this->outQueueC.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->cGm[offset], cLocal, curTileLen);
        // free output tensor for reuse
        this->outQueueC.template FreeTensor(cLocal);
    }

private:
    XxgluTilingKernel tiling;
};
#endif  // CANN_SWI_GLU_LIMIT_SPLIT_HPP   // [Add by GTS-AILab] [feature GTSOps]
