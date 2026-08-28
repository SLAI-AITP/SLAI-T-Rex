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
 * \file sparse_lightning_indexer_grad_kl_loss_vector2.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_grad_kl_loss_common.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT> 
class SLIKLLossVector2Service {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM3_OUT_T = T;

    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;

    __aicore__ inline SLIKLLossVector2Service(){};
    __aicore__ inline void InitParams(const struct SLIGradKLLossConstInfo &vecConstInfo,
        const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // =============== vector 2 functions ==============
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM3_OUT_T> &bmm3Res, const GlobalTensor<int32_t> &topK,
        const GlobalTensor<OUT_T> &dKeyIndexGm, GlobalTensor<int64_t> &actualSeqLengthsQ,
        GlobalTensor<int64_t> &actualSeqLengthsKV);
    __aicore__ inline void ProcessVector2();

private:
    // =============== vector 2 functions ==============

    // =============== vector2 variable ==============
    SLIGradKLLossConstInfo constInfo;
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData;

    // global tensor
    GlobalTensor<MM3_OUT_T> bmm3ResGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<int64_t> actualSeqLengthsQGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<OUT_T> dKeyIndexGm;

    // local tensor
    TBuf<> mm3TBuf;         // 64K, 64 * 128 * 4 * 2(DB)
    TBuf<> topKTBuf;        // 16KB, 2048 * 4 * 2(DB)
    TBuf<> castOutTBuf;     // 32KB, 64 * 128 * 4 * 2(DB)
    LocalTensor<MM3_OUT_T> mm3ResUb;
    LocalTensor<int32_t> topKUb;
    LocalTensor<OUT_T> castOutUb;

    static constexpr uint64_t SYNC_SCATTER_BUF_FLAG = 0;
    static constexpr uint64_t SYNC_SCATTER_BUF_PONG_FLAG = 1;
    static constexpr int64_t UB_ROW_SIZE = 64;
};

template <typename SLIT> 
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitParams(const struct SLIGradKLLossConstInfo &vecConstInfo,
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitVector2GM(const GlobalTensor<MM3_OUT_T> &bmm3Res,
    const GlobalTensor<int32_t> &topK, const GlobalTensor<OUT_T> &dKeyIndexGm,
    GlobalTensor<int64_t> &actualSeqLengthsQ, GlobalTensor<int64_t> &actualSeqLengthsKV)
{
    this->bmm3ResGm = bmm3Res;
    this->topKGm = topK;
    this->actualSeqLengthsQGm = actualSeqLengthsQ;
    this->actualSeqLengthsKVGm = actualSeqLengthsKV;
    this->dKeyIndexGm = dKeyIndexGm;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(mm3TBuf, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K * 2);  // 2:pingpong
    pipe->InitBuffer(castOutTBuf, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K * 2);  // 2:pingpong

    mm3ResUb = mm3TBuf.Get<MM3_OUT_T>();
    castOutUb = castOutTBuf.Get<OUT_T>();
}

template <typename SLIT> __aicore__ inline void SLIKLLossVector2Service<SLIT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_PONG_FLAG);
}

template <typename SLIT> __aicore__ inline void SLIKLLossVector2Service<SLIT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_PONG_FLAG);
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVector2Service<SLIT>::ProcessVector2()
{
    // 输入数据[T2,Nidx2,D]
    // T2切多核，执行前需要做全核同步

    int64_t totalCost = 0;
    if constexpr (KV_LAYOUT_T == SLILayout::TND) {
        totalCost = actualSeqLengthsKVGm.GetValue(constInfo.bSize - 1);
    } else {
        totalCost = constInfo.bSize * constInfo.s2Size;
    }

    int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
    int64_t avgCost = CeilDiv(totalCost, totalCoreNum);

    int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
    int32_t t2End = Min(t2Start + avgCost, totalCost);

    int32_t t2ProcessSize = UB_ROW_SIZE;
    int32_t tailSize = (t2End - t2Start) % UB_ROW_SIZE;
    int32_t t2TailSize = (!tailSize) ? UB_ROW_SIZE : (tailSize);
    int32_t pingPongIdx = 0;

    LocalTensor<MM3_OUT_T> copyInUb;
    LocalTensor<OUT_T> copyOutUb;

    for (int32_t t2Idx = t2Start; t2Idx < t2End; t2Idx += UB_ROW_SIZE) {
        if (t2Idx + UB_ROW_SIZE >= t2End) {
            t2ProcessSize = t2TailSize;
        }

        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        copyInUb = mm3ResUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
        DataCopy(copyInUb, bmm3ResGm[t2Idx * constInfo.dSizeQueryIndex], t2ProcessSize * constInfo.dSizeQueryIndex);

        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        copyOutUb = castOutUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
        Cast(copyOutUb, copyInUb, RoundMode::CAST_ROUND, t2ProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);       // 释放copyInUb

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);

        DataCopy(dKeyIndexGm[t2Idx * constInfo.dSizeQueryIndex], copyOutUb, t2ProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);       // 释放copyOutUb

        pingPongIdx = 1 - pingPongIdx;
    }
}

#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H