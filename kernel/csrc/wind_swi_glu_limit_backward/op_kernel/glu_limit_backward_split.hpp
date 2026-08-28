/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_XX_GLU_LIMIT_BACKWARD_SPLIT_HPP
#define CANN_XX_GLU_LIMIT_BACKWARD_SPLIT_HPP
#include "xxglu_limit_tiling_kernel.h"

template<typename ParentClass, typename inType, typename outType>
class GluLimitBackwardSplit : public ParentClass {
public:
    __aicore__ inline GluLimitBackwardSplit() = default;
    __aicore__ inline ~GluLimitBackwardSplit() = default;
    __aicore__ inline void Init(GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR l_gm, GM_ADDR m_gm, GM_ADDR n_gm,
                                const XxGluLimitBackwardTilingData* tiling_gm, TPipe& pipe)
    {
        tiling.GetTilingAndOffset(tiling_gm, sizeof(inType));

        InitGmBuffer(a_gm, b_gm, l_gm, m_gm, n_gm);
        this->InitUbBuffer(pipe, tiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        XXGLU_PROCESS(0, tiling);
    }

protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR l_gm, GM_ADDR m_gm, GM_ADDR n_gm)
    {
        this->limit = tiling.limit; // 透传 clamp 阈值给 Compute 类
        this-> lGm.SetGlobalBuffer((__gm__ inType*)l_gm + tiling.gmOffset, tiling.blockLength);
        this-> aGm.SetGlobalBuffer((__gm__ inType*)a_gm + tiling.gmOffset, tiling.blockLength);
        this-> bGm.SetGlobalBuffer((__gm__ inType*)b_gm + tiling.gmOffset, tiling.blockLength);

        this->mGm.SetGlobalBuffer((__gm__ outType*)m_gm + tiling.gmOffset, tiling.blockLength);
        this->nGm.SetGlobalBuffer((__gm__ outType*)n_gm + tiling.gmOffset, tiling.blockLength);
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint64_t curTileLen)
    {
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        ::DataCopy(lLocal, this->lGm[offset], curTileLen);
        this->inQueueL.template EnQue(lLocal);
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        ::DataCopy(aLocal, this->aGm[offset], curTileLen);
        this->inQueueA.template EnQue(aLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        ::DataCopy(bLocal, this->bGm[offset], curTileLen);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t tileLength)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        ::DataCopy(this->mGm[offset], mLocal, tileLength);
        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        ::DataCopy(this->nGm[offset], nLocal, tileLength);
        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);
    }

private:
    XxgluTilingKernel tiling;
};
#endif  // CANN_XX_GLU_LIMIT_BACKWARD_SPLIT_HPP
