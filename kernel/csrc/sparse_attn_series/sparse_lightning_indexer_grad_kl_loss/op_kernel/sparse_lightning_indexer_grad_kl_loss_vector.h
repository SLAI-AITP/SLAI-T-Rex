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
 * \file sparse_lightning_indexer_grad_kl_loss_vector.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_grad_kl_loss_common.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"

using namespace AscendC;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT> 
class SLIKLLossVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM5_OUT_T = T;

    static constexpr uint32_t topKSize = static_cast<uint32_t>(SLIT::topKRange);
    static constexpr bool isTopkLess2k = (topKSize <= SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_2K);
    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;
    static constexpr UBAllocPolicy<isTopkLess2k> ubAllocPolicy;

    __aicore__ inline SLIKLLossVectorService(){};
    __aicore__ inline void InitParams(const SLIGradKLLossConstInfo &vecConstInfo,
        const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // =============== vector 0 functions ==============
    __aicore__ inline void InitVector0GM(GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope,
        GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
        GlobalTensor<int64_t> &actualSeqLengthsQ, GlobalTensor<int64_t> &actualSeqLengthsKV,
        GlobalTensor<KV_T> &gatherPRes, GlobalTensor<KV_T> &gatherSYRes);
    __aicore__ inline void ProcessVector0(SLIGradKLLossRunInfo &runInfo);

    // =============== vector 1 functions ==============
    __aicore__ inline void InitVector1GM(const GlobalTensor<MM12_OUT_T> &bmm1Res,
                                    const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum,
                                    const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<Q_T> &weight,
                                    const GlobalTensor<T> psySync, GlobalTensor<T> &loss,
                                    GlobalTensor<OUT_T> &dWeight, GlobalTensor<T> &reluGm,
                                    GlobalTensor<KV_T> &reluGradRes, GlobalTensor<int64_t> &actualSeqLengthsQueryGm,
                                    GlobalTensor<int64_t> &actualSeqLengthsKeyGm);
    __aicore__ inline void ProcessVector1(SLIGradKLLossRunInfo &runInfo);

    // =============== vector 2 functions ==============
    __aicore__ inline void ProcessVector2(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM5_OUT_T> &bmm5Res, const GlobalTensor<int32_t> &topK,
        GlobalTensor<T> &scatterAddRes);
    __aicore__ inline void ProcessDeterVector2(SLIGradKLLossRunInfo &runInfo);

private:
    // =============== vector 0 functions ==============
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void MergeKv(const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_, 
        GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    template <bool gatherRope>
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
        int64_t keyBNBOffset,int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, int32_t dValue);
    template <bool gatherRope>
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
        int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset, int64_t s2IdxOffset, int64_t mergeMte3Idx,
        const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_, GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t s2IdxOffset, int64_t &realS2Idx, const SLIGradKLLossRunInfo &runInfo);

    // =============== vector 1 functions ==============
    __aicore__ inline void VectorP(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void VectorPWithOutSoftMaxSum(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void PreloadWeight(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void VectorSy(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &subResTensor,
        LocalTensor<T> &maskUb, int32_t kRealSizeAlign);
    template <uint32_t range>
    __aicore__ inline void VectorDwDqDk(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorDwDqDkMoreThan2k(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorDwDqDkLess2k(SLIGradKLLossRunInfo &runInfo);
    template <uint32_t range>
    __aicore__ inline void VectorLoss(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorLossMoreThan2k(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorLossLess2k(SLIGradKLLossRunInfo &runInfo);
    // =============== vector 2 functions ==============
    __aicore__ inline void ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb, int64_t keyBNBOffset);
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, 
        SLILayout layout, int64_t &accumLen);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, int64_t cmpRatio, SLISparseMode sparseMode);
    __aicore__ inline void GetRunInfo(int64_t taskId,  int64_t bIdx, int64_t s1Idx, SLIGradKLLossRunInfo &runInfo);

    // =============== vector common variable ==============
    SLIGradKLLossConstInfo constInfo;
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData;

    // global tensor
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyIndexGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<MM12_OUT_T> bmm1ResGm;
    GlobalTensor<T> softmaxMaxGm; 
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<MM12_OUT_T> bmm2ResGm;
    GlobalTensor<Q_T> weightGm;
    GlobalTensor<T> psySyncGm, psySyncOtherGm;
    GlobalTensor<T> tempGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<OUT_T> dWeightGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<KV_T> reluGradResGm;

    GlobalTensor<MM5_OUT_T> bmm5ResGm;

    GlobalTensor<int64_t> actualSeqLengthsQueryGm;
    GlobalTensor<int64_t> actualSeqLengthsKeyGm;

    // local tensor
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf;         // 复用 -> mm4 scatterAdd reluGrad
    TBuf<> sharedTBuf;
    TBuf<> resPSYTBuf;
    TBuf<> reduceSumDwTBuf;
    TBuf<> v1TmpTBuf;

    TBuf<> weightTBuf;
    TBuf<> weightInTBuf;
    TBuf<> lossSumTBuf;
    TBuf<> dwTBuf;
    TBuf<> maskBuf;
    TBuf<> scatterAddBuf;

    // =============== vector0 variable ==============
    GlobalTensor<KV_T> gatherPResGm;
    GlobalTensor<KV_T> gatherSYResGm;

    TBuf<> gatherTbuf;
    
    LocalTensor<KV_T> gatherKeyUb;
    LocalTensor<KV_T> gatherKeyIndexUb;
    static constexpr int32_t blockBytes = 32;

    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> srcTensor;
    GlobalTensor<KV_T> srcRopeTensor;

    GatherParams gatherParams;

    static constexpr int64_t UB_ROW_SIZE = 8;
    // =============== vector1 variable ==============
    LocalTensor<T> weightUb;
    LocalTensor<KV_T> weightInUb;

    LocalTensor<T> reluResUb;
    LocalTensor<T> reduceSumYResTmpBuffer;
    LocalTensor<T> reduceSumYResUb;
    LocalTensor<T> reduceSumPResUb;
    LocalTensor<uint8_t> reduceSumTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpBuffer;
    LocalTensor<T> reduceSumDwResUb;

    // =============== vector2 variable ==============
    GlobalTensor<T> scatterAddResGm;
    LocalTensor<T> scatterAddUb;
    int32_t scatterAddPingpong = 0;

    // dwdqdk计算
    LocalTensor<T> v1TmpUb;
    LocalTensor<T> pingBuf;
    LocalTensor<T> pongBuf;
    LocalTensor<KV_T> reluGradpingBuf;
    LocalTensor<KV_T> reluGradpongBuf;
    LocalTensor<T> weightTensor;
    LocalTensor<uint8_t> tmpUb;
    LocalTensor<KV_T> dwOutTensor;

    // =============== Event_id variable ==============
    static constexpr event_t EVENT_ID0 = (event_t)0;
    static constexpr event_t EVENT_ID1 = (event_t)1;
    static constexpr event_t EVENT_ID2 = (event_t)2;
    static constexpr event_t EVENT_ID3 = (event_t)3;
    static constexpr event_t EVENT_ID4 = (event_t)4;
    static constexpr event_t EVENT_ID5 = (event_t)5;
    static constexpr event_t EVENT_ID6 = (event_t)6;
    static constexpr event_t EVENT_ID7 = (event_t)7;

    // MTE2_V
    event_t eventIdMte2ToV4SY;
    event_t eventIdMte2ToVInnerVecP;
    event_t eventIdMte2ToVInnerPreW;
    event_t eventIdMte2ToVInnerDwDqDk;
    event_t eventIdMte2ToVInnerLoss;
    // V_MTE2
    event_t eventIdVToMte24SY;
    event_t eventIdVToMte2P[2];
    event_t eventIdVToMte2Weight[2];
    event_t eventIdvToMte2DwDqDkPingPong[2];
    // MTE3_V
    event_t eventIdMte3ToVTmp;
    event_t eventIdMte3ToVDwDqDkPingPong[2];
    // V_MTE3
    event_t eventVToMte3InnerVecP;
    event_t eventVToMte3InnerSy;
    event_t eventIdVToMte3DwDqDk;
    event_t eventIdVToMte3InnerVecLoss;
    // MTE3_MTE2
    event_t eventIdVec0Inner0;
    event_t eventIdVec0Inner1;
    event_t eventIdVec0Inner2;
    event_t eventIdVec0Inner3;
    event_t eventIdmte3ToMte2DwDqDkPingPong[2];
    // event_t eventIdVToV4SY; // 加一行性能会变好
    event_t eventIdScatterAdd;
    event_t eventIdScatterAddPong;
    // MTE2_MTE3
    event_t eventIdMergeRes;
};

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitParams(const SLIGradKLLossConstInfo &vecConstInfo,
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector0GM(
    GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope, GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
    GlobalTensor<int64_t> &actualSeqLengthsQ, GlobalTensor<int64_t> &actualSeqLengthsKV,
    GlobalTensor<KV_T> &gatherPRes, GlobalTensor<KV_T> &gatherSYRes)
{
    this->keyGm = key;
    this->keyRopeGm = keyRope;
    this->keyIndexGm = keyIndex;
    this->topKGm = topK;

    this->gatherPResGm = gatherPRes;
    this->gatherSYResGm = gatherSYRes;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector1GM(const GlobalTensor<MM12_OUT_T> &bmm1Res,
                                                        const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum,
                                                        const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<Q_T> &weight,
                                                        const GlobalTensor<T> psySync, GlobalTensor<T> &loss,
                                                        GlobalTensor<OUT_T> &dWeight, GlobalTensor<T> &reluGm,
                                                        GlobalTensor<KV_T> &reluGradRes, GlobalTensor<int64_t> &actualSeqLengthsQueryGm,
                                                        GlobalTensor<int64_t> &actualSeqLengthsKeyGm)
{
    this->bmm1ResGm = bmm1Res;
    this->softmaxMaxGm = softmaxMax;
    this->softmaxSumGm = softmaxSum;
    this->bmm2ResGm = bmm2Res;
    this->weightGm = weight;
    this->psySyncGm = psySync[constInfo.subBlockIdx * topKSize * 2];
    this->psySyncOtherGm = psySync[!constInfo.subBlockIdx * topKSize * 2];
    
    this->lossGm = loss;
    this->dWeightGm = dWeight;
    this->reluGm = reluGm;
    this->reluGradResGm = reluGradRes;
    this->actualSeqLengthsQueryGm = actualSeqLengthsQueryGm;
    this->actualSeqLengthsKeyGm = actualSeqLengthsKeyGm;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector2GM(const GlobalTensor<MM5_OUT_T> &bmm5Res,
    const GlobalTensor<int32_t> &topK, GlobalTensor<T> &scatterAddRes)
{
    this->bmm5ResGm = bmm5Res;
    this->topKGm = topK;
    this->scatterAddResGm = scatterAddRes;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitBuffers(TPipe *pipe)
{
    // 可选输入的buffer空间，保持和stage1处理的size一致
    pipe->InitBuffer(this->gatherTbuf, ubAllocPolicy.gatherUbSize);
    pipe->InitBuffer(this->mm1Tbuf, ubAllocPolicy.mm1UbSize);
    pipe->InitBuffer(this->mm2TBuf, ubAllocPolicy.mm2UbSize);
    pipe->InitBuffer(this->sharedTBuf, ubAllocPolicy.sharedUbSize);
    pipe->InitBuffer(this->resPSYTBuf, ubAllocPolicy.resPSYUbSize);
    pipe->InitBuffer(this->reduceSumDwTBuf, ubAllocPolicy.reduceSumDwUbSize);
    pipe->InitBuffer(this->v1TmpTBuf, ubAllocPolicy.v1TmpUbSize);
    pipe->InitBuffer(this->lossSumTBuf, ubAllocPolicy.lossSumUbSize);
    pipe->InitBuffer(this->weightTBuf, ubAllocPolicy.weightUbSize);
    pipe->InitBuffer(this->weightInTBuf, ubAllocPolicy.weightInUbSize);
    pipe->InitBuffer(this->dwTBuf, ubAllocPolicy.dwUbSize);
    pipe->InitBuffer(this->maskBuf, ubAllocPolicy.maskUbSize);
    pipe->InitBuffer(this->scatterAddBuf, ubAllocPolicy.scatterAddUbSize);

    gatherKeyUb = gatherTbuf.Get<KV_T>();
    gatherKeyIndexUb = gatherKeyUb[SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_9K * 2];

    weightUb = weightTBuf.Get<T>();
    weightInUb = weightInTBuf.Get<KV_T>();
    reluResUb = mm2TBuf.Get<T>();
    reduceSumTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 8*1024);
    reduceSumYResTmpBuffer = sharedTBuf.GetWithOffset<T>(topKSize, 2*8*1024);
    reduceSumYResUb = resPSYTBuf.Get<T>();
    reduceSumPResUb = resPSYTBuf.template Get<T>();
    reduceSumDwResUb = reduceSumDwTBuf.Get<T>();
    softmaxTmpBuffer = sharedTBuf.Get<uint8_t>();
    scatterAddUb = scatterAddBuf.Get<T>();

    v1TmpUb = v1TmpTBuf.template Get<T>();
    pingBuf = mm1Tbuf.template Get<T>();
    pongBuf = mm2TBuf.template Get<T>();
    reluGradpingBuf = mm1Tbuf.template Get<KV_T>();
    reluGradpongBuf = mm2TBuf.template Get<KV_T>();
    weightTensor = weightTBuf.template Get<T>();
    tmpUb = sharedTBuf.template Get<uint8_t>();
    dwOutTensor = this->dwTBuf.template Get<KV_T>();
}

template <typename SLIT> __aicore__ inline void SLIKLLossVectorService<SLIT>::AllocEventID()
{
    // MTE2_To_V
    eventIdMte2ToV4SY = EVENT_ID0;
    eventIdMte2ToVInnerVecP = EVENT_ID1;
    eventIdMte2ToVInnerPreW = EVENT_ID2;
    eventIdMte2ToVInnerDwDqDk = EVENT_ID3;
    eventIdMte2ToVInnerLoss = EVENT_ID4;
    // V_To_MTE2
    eventIdVToMte2Weight[0] = EVENT_ID0;
    eventIdVToMte2Weight[1] = EVENT_ID1;
    eventIdVToMte2P[0] = EVENT_ID3;
    eventIdVToMte2P[1] = EVENT_ID4;
    eventIdVToMte24SY = EVENT_ID5;
    eventIdvToMte2DwDqDkPingPong[0] = EVENT_ID6;
    eventIdvToMte2DwDqDkPingPong[1] = EVENT_ID7;
    // MTE3_To_V
    eventIdMte3ToVTmp = EVENT_ID0;
    eventIdMte3ToVDwDqDkPingPong[0] = EVENT_ID1;
    eventIdMte3ToVDwDqDkPingPong[1] = EVENT_ID2;
    // V_To_MTE3
    eventVToMte3InnerVecP = EVENT_ID0;
    eventVToMte3InnerSy = EVENT_ID1;
    eventIdVToMte3DwDqDk = EVENT_ID2;
    eventIdVToMte3InnerVecLoss = EVENT_ID3;
    // MTE3_MTE2
    eventIdVec0Inner0 = EVENT_ID0;
    eventIdVec0Inner1 = EVENT_ID1;
    eventIdVec0Inner2 = EVENT_ID2;
    eventIdVec0Inner3 = EVENT_ID3;
    eventIdmte3ToMte2DwDqDkPingPong[0] = EVENT_ID4;
    eventIdmte3ToMte2DwDqDkPingPong[1] = EVENT_ID5;
    eventIdScatterAdd = EVENT_ID6;
    eventIdScatterAddPong = EVENT_ID7;
    // MTE2_MTE3
    eventIdMergeRes = EVENT_ID0;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT> __aicore__ inline void SLIKLLossVectorService<SLIT>::FreeEventID()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector0(SLIGradKLLossRunInfo &runInfo)
{
    gatherParams.dValue = constInfo.dSizeQuery;
    gatherParams.dRopeValue = constInfo.dSizeRope;
    gatherParams.gatherResGmEleSize = constInfo.gatherKeySize;

    this->kvMergUb_ = gatherKeyUb;
    this->ropeMergUb_ = gatherKeyUb[UB_ROW_SIZE * gatherParams.dValue * 2];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner1);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner2);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner3);

    for (int32_t i = 0; i < runInfo.s2LoopTimes; ++i) {
        runInfo.s2Idx = i;
	
        gatherParams.s2ProcessSize = (i >= runInfo.s2LoopTimes - 1) ? runInfo.s2TailSize : constInfo.s2BaseSize;
        MergeKv<EVENT_ID0, SLIT::hasRope>(runInfo, gatherPResGm, keyGm, keyRopeGm);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_P_FLAG[i & 1]);
    }

    gatherParams.dValue = constInfo.dSizeQueryIndex;
    gatherParams.dRopeValue = 0;
    gatherParams.gatherResGmEleSize = constInfo.gatherKeyIndexSize;
    this->kvMergUb_ = gatherKeyIndexUb;

    for (int32_t i = 0; i < runInfo.s2LoopTimes; ++i) {
        runInfo.s2Idx = i;
        gatherParams.s2ProcessSize = (i >= runInfo.s2LoopTimes - 1) ? runInfo.s2TailSize : constInfo.s2BaseSize;
        MergeKv<EVENT_ID2, false>(runInfo, gatherSYResGm, keyIndexGm, keyRopeGm);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_SY_FLAG[i & 1]);
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner1);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner2);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner3);
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void
SLIKLLossVectorService<SLIT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                       int64_t keyBNBOffset, int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo, 
                                       GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams(1, validS2Count * gatherParams.dValue * sizeof(KV_T), 0, 0, 0);
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
                srcTensor[keyBNBOffset * gatherParams.dValue], intriParams, padParams);

    if constexpr (gatherRope) {
        intriParams.blockLen = validS2Count * gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue + (mte2Size - mte3Size) * gatherParams.dRopeValue],
                    srcRopeTensor[keyBNBOffset * gatherParams.dRopeValue], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyRopeGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo)
{
    // (accumS2Idx*dRope + realS2Idx*dRope)/dRope 恒等于 accumS2Idx+realS2Idx，
    // 两项都是dRope的整数倍，整除无截断；省去每token一次64位乘法+一次昂贵的64位整数除法
    return runInfo.accumS2Idx + realS2Idx;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, int32_t dValue)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    (void)dValue;   // 保留签名兼容
    // (accumS2Idx*dValue + realS2Idx*dValue)/dValue 恒等于 accumS2Idx+realS2Idx，
    // 两项都是dValue的整数倍，整除无截断；省去每token一次64位乘法+一次昂贵的64位整数除法(标量热点)
    return runInfo.accumS2Idx + realS2Idx;
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t realS2Idx1, int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor)
{
    int64_t s2IdLimit = runInfo.s2SparseLen;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, gatherParams.dValue);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, gatherParams.dValue);
    bool keyOffset1Pass = (keyOffset1 >= 0);
    bool keyOffset2Pass = (keyOffset2 >= 0);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                    (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * gatherParams.dValue * sizeof(KV_T);
    if constexpr (gatherRope) {
        // keyRopeOffset == accumS2Idx+realS2Idx，差值里accumS2Idx抵消 -> 直接用realS2Idx算abs差值，
        // 省去每token两次GetKeyRopeGmOffset调用。注意不能复用keySrcStride里clamp过的keyOffset，
        // 用未clamp的realS2Idx才能保持含多batch在内的bit-exact。
        keyRopeSrcStride = ((realS2Idx1 > realS2Idx2 ? (realS2Idx1 - realS2Idx2) :
                            (realS2Idx2 - realS2Idx1)) - static_cast<int64_t>(constInfo.sparseBlockSize)) *
                                gatherParams.dRopeValue * sizeof(KV_T);
    }

    bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);
    bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0) ||
        (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0);
    bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);

    if (key1LessThankey2 || strideInvalid || copyOutOfRange) {
        // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo, srcTensor, srcRopeTensor);
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo, srcTensor, srcRopeTensor);
    } else {
        DataCopyExtParams intriParams(
            keyOffset1Pass + keyOffset2Pass, constInfo.sparseBlockSize * gatherParams.dValue * sizeof(KV_T), keySrcStride, 0, 0
        );
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1Pass ? keyOffset1 : keyOffset2;
        if (keyOffset2Pass && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
                    srcTensor[startGmOffset * gatherParams.dValue], intriParams, padParams);

        if constexpr (gatherRope) {
            intriParams.blockLen = constInfo.sparseBlockSize * gatherParams.dRopeValue * sizeof(KV_T);
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue + (mte2Size - mte3Size) * gatherParams.dRopeValue],
                        srcRopeTensor[startGmOffset * gatherParams.dRopeValue], intriParams, padParams);
        }

        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
    int64_t s2GmStartOffset, int64_t s2IdxOffset, int64_t mergeMte3Idx, const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_, 
    GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMergeRes);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMergeRes);

    DataCopyExtParams dataCopyParams(mte2Size - mte3Size, gatherParams.dValue * sizeof(KV_T), 0, 0, 0);

    int64_t gmStartOffset = runInfo.taskIdMod2 * gatherParams.gatherResGmEleSize +
        s2IdxOffset * (gatherParams.dValue + gatherParams.dRopeValue);

    DataCopyPad(kvMergeGm_[gmStartOffset + (s2GmStartOffset + mte3Size) * gatherParams.dValue],
                kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue], dataCopyParams);

    if constexpr (gatherRope) {
        dataCopyParams.blockLen = gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[gmStartOffset + constInfo.s2BaseSize * gatherParams.dValue + (s2GmStartOffset + mte3Size) *
            gatherParams.dRopeValue], ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue], dataCopyParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::GetRealS2Idx(int64_t s2GmOffset, int64_t s2IdxOffset, int64_t &realS2Idx,
    const SLIGradKLLossRunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + s2IdxOffset) / constInfo.sparseBlockSize;
    realS2Idx = topKGm.GetValue(runInfo.topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + s2IdxOffset) % constInfo.sparseBlockSize);
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::MergeKv(const SLIGradKLLossRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_, 
                                                             GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor)
{
    int64_t s2Pair = CeilDiv(gatherParams.s2ProcessSize, 2L * constInfo.sparseBlockSize);

    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    bool subBlockZero = (GetSubBlockIdx() == 0);
    int64_t s2GmStartOffset = subBlockZero ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = subBlockZero ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: gatherParams.s2ProcessSize;
    if (s2GmLimit > gatherParams.s2ProcessSize) {
        s2GmLimit = gatherParams.s2ProcessSize;
    }
    int64_t s2IdxOffset = runInfo.s2Idx * constInfo.s2BaseSize;
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxOffset, s2IdxArray0, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, s2IdxOffset, mergeMte3Idx, runInfo, kvMergeGm_, srcTensor, srcRopeTensor);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxOffset, s2IdxArray1, runInfo);
        CopyInKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, srcTensor, srcRopeTensor);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > UB_ROW_SIZE) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, s2IdxOffset, mergeMte3Idx, runInfo, kvMergeGm_, srcTensor, srcRopeTensor);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector1(SLIGradKLLossRunInfo &runInfo)
{
    CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[runInfo.taskIdMod2]);
    CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[runInfo.taskIdMod2]);
    PreloadWeight(runInfo);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    // 保证 reduceN 轴的S'Y在同一个Vec核内 将P和S'Y任务分割
    if (runInfo.calcP) {
        if (constInfo.hasSoftmaxInput) {
            VectorP(runInfo);
        } else {
            VectorPWithOutSoftMaxSum(runInfo);
        }
    } else {
        VectorSy(runInfo); // 计算S'Y部分，一直到Softmax
    }
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag<0x1, PIPE_MTE2>(0x8);
    for (int32_t kLoopIdx = 0; kLoopIdx < runInfo.kLoopTimes; ++kLoopIdx) {
        VectorDwDqDk<topKSize>(runInfo, kLoopIdx); // Sub + Mul + Mul + ReluGrad
        if (kLoopIdx >= runInfo.kLoopTimes - 1) {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[runInfo.taskIdMod2]);
        }
        VectorLoss<topKSize>(runInfo, kLoopIdx);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
}

template <typename SLIT>
__aicore__ inline void
SLIKLLossVectorService<SLIT>::ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb, int64_t keyBNBOffset)
{
    if (keyBNBOffset < 0) {
        return;
    }
    LocalTensor<T> srcTmpUb = srcUb.template ReinterpretCast<T>();
    DataCopy(scatterAddResGm[keyBNBOffset * constInfo.dSizeQueryIndex], srcTmpUb, constInfo.dSizeQueryIndex);
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetActualSeqLens(int32_t bIdx,
    int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, SLILayout layout, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }
    
    if (layout == SLILayout::TND) {
        if (bIdx == 0) {
            accumLen = 0;
            return actualSeqLensGm.GetValue(0);
        } else {
            accumLen = actualSeqLensGm.GetValue(bIdx - 1);
            return (actualSeqLensGm.GetValue(bIdx) - accumLen);
        }
    } else {
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetS2SparseLen(int32_t s1Idx,
    int32_t actualSeqLensQ, int32_t actualSeqLensK, int64_t cmpRatio, SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        if (cmpRatio == 4) { 
            return (actualSeqLensK * constInfo.cmpRatio - actualSeqLensQ + s1Idx + 1) / constInfo.cmpRatio;
        } else {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
    } else {
        return 0;
    }
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::GetRunInfo(int64_t taskId,  int64_t bIdx, int64_t s1Idx, SLIGradKLLossRunInfo &runInfo)
{
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    if constexpr (LAYOUT_T == SLILayout::TND) {
        int32_t actualSeqLensQ = GetActualSeqLens(runInfo.bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, runInfo.accumS1Idx);
        int32_t actualSeqLensK = GetActualSeqLens(runInfo.bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, runInfo.accumS2Idx);
        runInfo.actS1Size = actualSeqLensQ;
        runInfo.actS2Size = actualSeqLensK;
        runInfo.accumS1Idx += s1Idx;
    } else if constexpr (LAYOUT_T == SLILayout::BSND) {
        runInfo.actS1Size = constInfo.s1Size;
        runInfo.actS2Size = constInfo.s2Size;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + s1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size, constInfo.cmpRatio, constInfo.sparseMode);
    runInfo.s2RealSize = Min(topKSize, runInfo.s2SparseLen);

    if (constInfo.cmpRatio != 0) {
        runInfo.s2RealSize = Max(1, runInfo.s2RealSize);
    }

    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.topkGmBaseOffset = runInfo.accumS1Idx * topKSize;
    } else {
        runInfo.topkGmBaseOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * topKSize;
    }
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessDeterVector2(SLIGradKLLossRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[runInfo.taskIdMod2]);

    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);

    int32_t coreNum = GetBlockNum();
    for (int32_t idx = 0; idx < coreNum; idx++) {

        //重新获取b和S1的值
        int64_t bS1StartIndex = tilingData->multiCoreParams.bS1Index[idx];
        int64_t bS1EndIndex = idx + 1 < optiling::MAX_CORE_NUM ?
            tilingData->multiCoreParams.bS1Index[idx + 1] : tilingData->multiCoreParams.totalSize;
        int64_t bIdx, s1Idx;

        int64_t bS1Index = bS1StartIndex + runInfo.taskId;
        if (bS1Index >= bS1EndIndex) {
            continue;
        }
        int64_t actualSum = 0;
        int64_t actualLenQ = 0;
        int64_t seqLenQ = 0;
        int64_t actualLenK = 0;
        int64_t seqLenK = 0;
        if constexpr (LAYOUT_T == SLILayout::TND) {
            for (int index = 0; index < constInfo.bSize; index++) {
                int64_t actualLenQ = this->actualSeqLengthsQueryGm.GetValue(index);
                int64_t actualLenK = this->actualSeqLengthsKeyGm.GetValue(index);
                if (bS1Index < actualLenQ) {
                    bIdx = index;
                    seqLenQ = (index == 0) ? actualLenQ 
                        : this->actualSeqLengthsQueryGm.GetValue(index) - this->actualSeqLengthsQueryGm.GetValue(index - 1);
                    seqLenK = (index == 0) ? actualLenK 
                        : this->actualSeqLengthsKeyGm.GetValue(index) - this->actualSeqLengthsKeyGm.GetValue(index - 1);
                    break;
                }
                actualSum = actualLenQ;
            }
            s1Idx = bS1Index - actualSum;
        } else {
            bIdx = bS1Index / constInfo.s1Size;
            s1Idx = bS1Index - bIdx * constInfo.s1Size;
        }

        if (constInfo.cmpRatio != 0) {
            if (constInfo.cmpRatio * seqLenK - seqLenQ + s1Idx + 1 <= 0) {
                continue;
            }
        }
 
        GetRunInfo(runInfo.taskId, bIdx, s1Idx, runInfo);
        
        int32_t v0RealKSize, v1RealKSize, vRealKSize;
        int32_t perCoreKSize, tailCoreKSize, curCoreKSize; 
        int64_t coreKOffset, vCoreKOffset;

        perCoreKSize = CeilDiv(runInfo.kRealSize, coreNum);
        perCoreKSize = SLIGAlign(perCoreKSize, 4);
        int32_t usedCoreNum = CeilDiv(runInfo.kRealSize, perCoreKSize);
        if (constInfo.aicIdx >= usedCoreNum) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            continue;
        }
        tailCoreKSize = runInfo.kRealSize - (usedCoreNum - 1) * perCoreKSize;
        curCoreKSize = constInfo.aicIdx == usedCoreNum - 1 ? tailCoreKSize : perCoreKSize;
        v0RealKSize = CeilDiv(curCoreKSize, 2);
        v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), curCoreKSize);
        v1RealKSize = curCoreKSize - v0RealKSize;

        coreKOffset = constInfo.aicIdx * perCoreKSize;
        if (constInfo.subBlockIdx == 0) {
            vRealKSize = v0RealKSize;
            vCoreKOffset = coreKOffset;
        } else {
            vRealKSize = v1RealKSize;
            vCoreKOffset = coreKOffset + v0RealKSize;
        }
        if (vRealKSize <= 0) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            continue;
        }

        int64_t srcOffset = idx * topKSize * constInfo.dSizeQueryIndex * 2;
        GlobalTensor<MM5_OUT_T> srcGm = bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + vCoreKOffset) * constInfo.dSizeQueryIndex];
        LocalTensor<MM5_OUT_T> scatterAddTmpUb;
        int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / (2 * sizeof(T) * constInfo.dSizeQueryIndex);
        int32_t tailkSize = vRealKSize % kSplitSize;
        int32_t kTailSize = (!tailkSize) ? kSplitSize : tailkSize;
        int32_t kProcessSize = kSplitSize;
        int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
        int64_t realS2Idx1, realS2Idx2, s2GmOffset;
        event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
        runInfo.s2Idx = 0;
        int64_t s2IdxOffset = runInfo.s2Idx * constInfo.s2BaseSize;

        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
        SetAtomicAdd<T>();

        for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
            if (kLoopIdx >= kLoopTimes - 1) {
                kProcessSize = kTailSize;
            }
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
            scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex].template ReinterpretCast<MM5_OUT_T>();
            DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex], kProcessSize * constInfo.dSizeQueryIndex);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
            constexpr int32_t topKSplitSize = 2;
            int32_t TailSize = kProcessSize % topKSplitSize;
            int32_t topKTailSize = ( !TailSize ) ? topKSplitSize : TailSize;
            int32_t topKProcessSize = topKSplitSize;
            int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
            for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
                if (topKIdx >= topKLoopTimes - 1) {
                    topKProcessSize = topKTailSize;
                }
                s2GmOffset = vCoreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
                GetRealS2Idx(s2GmOffset, s2IdxOffset, realS2Idx1, runInfo);
                GetRealS2Idx(s2GmOffset + 1, s2IdxOffset, realS2Idx2, runInfo);

                int64_t s2IdLimit = runInfo.s2SparseLen;
                int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
                int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
                bool keyOffset1Pass = (keyOffset1 >= 0);
                bool keyOffset2Pass = (keyOffset2 >= 0);
                if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                    SetAtomicNone();
                    break;
                }

                int64_t keySrcStride = 0;
                keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                                (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.dSizeQueryIndex * sizeof(T);

                bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
                bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                    realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
                bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

                int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
                int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

                if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                    // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1);
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2);
                } else {
                    DataCopyExtParams dataCopyParams(
                        keyOffset1Pass + keyOffset1Pass, constInfo.dSizeQueryIndex * sizeof(T), 0, keySrcStride, 0
                    );
                    int64_t keyStartOffset = (keyOffset1Pass) ? keyOffset1 : keyOffset2;
                    int64_t ubStartOffset = (keyOffset1Pass) ? ub1Offset : ub2Offset;
                    LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                    DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset], dataCopyParams);
                }
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
            scatterAddPingpong = 1 - scatterAddPingpong;
        }
        SetAtomicNone();
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
    }
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector2(SLIGradKLLossRunInfo &runInfo)
{
    CrossCoreWaitFlag(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);

    int32_t v0RealKSize, v1RealKSize, vRealKSize;
    int64_t coreKOffset;
    v0RealKSize = CeilDiv(runInfo.kRealSize, 2);
    v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), runInfo.kRealSize);
    v1RealKSize = runInfo.kRealSize - v0RealKSize;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        coreKOffset = 0;
    } else {
        vRealKSize = v1RealKSize;
        coreKOffset = v0RealKSize;
    }
    if (vRealKSize <= 0) {
        return;
    }

    int srcOffset = constInfo.aicIdx * topKSize * constInfo.dSizeQueryIndex * 2;
    GlobalTensor<MM5_OUT_T> srcGm = bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + coreKOffset) * constInfo.dSizeQueryIndex];
    LocalTensor<MM5_OUT_T> scatterAddTmpUb;
    int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / (2 * sizeof(T) * constInfo.dSizeQueryIndex);
    int32_t tailSize = vRealKSize % kSplitSize;
    int32_t kTailSize = (!tailSize) ? kSplitSize : tailSize;
    int32_t kProcessSize = kSplitSize;
    int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
    int64_t realS2Idx1, realS2Idx2, s2GmOffset;
    event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
    runInfo.s2Idx = 0;
    int64_t s2IdxOffset = runInfo.s2Idx * constInfo.s2BaseSize;

    SetAtomicAdd<T>();
    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx >= kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex].template ReinterpretCast<MM5_OUT_T>();
        DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex], kProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);

        constexpr int32_t topKSplitSize = 2;
        int32_t tailSize = kProcessSize % topKSplitSize;
        int32_t topKTailSize = (!tailSize) ? topKSplitSize : tailSize;
        int32_t topKProcessSize = topKSplitSize;
        int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
        for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
            if (topKIdx >= topKLoopTimes - 1) {
                topKProcessSize = topKTailSize;
            }
            s2GmOffset = coreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
            GetRealS2Idx(s2GmOffset, s2IdxOffset, realS2Idx1, runInfo);
            GetRealS2Idx(s2GmOffset + 1, s2IdxOffset, realS2Idx2, runInfo);

            int64_t s2IdLimit = runInfo.s2SparseLen;
            int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            bool keyOffset1Pass = (keyOffset1 >= 0);
            bool keyOffset2Pass = (keyOffset2 >= 0);
            if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                break;
            }

            int64_t keySrcStride = 0;
            keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                            (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.dSizeQueryIndex * sizeof(T);

            bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
            bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
            bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

            int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
            int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

            if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1);
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2);
            } else {
                DataCopyExtParams dataCopyParams(
                    keyOffset1Pass + keyOffset1Pass, constInfo.dSizeQueryIndex * sizeof(T), 0, keySrcStride, 0
                );
                int64_t keyStartOffset = (keyOffset1Pass) ? keyOffset1 : keyOffset2;
                int64_t ubStartOffset = (keyOffset1Pass) ? ub1Offset : ub2Offset;
                LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset], dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
    SetAtomicNone();
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorP(SLIGradKLLossRunInfo &runInfo)
{
    int64_t bmm1Offset = runInfo.taskIdMod2 * constInfo.gSizeQuery * topKSize;
    int64_t nLoops = CeilDiv(constInfo.gSizeQuery, runInfo.nBaseSizeP);
    int64_t kLoops = CeilDiv(runInfo.kRealSize, K_BASE_SIZE);

    LocalTensor<T> mulsResUb[2] = {mm1Tbuf.Get<T>(), mm2TBuf.Get<T>()};
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(16 * 1024, 0);
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(K_BASE_SIZE, 16 * 1024);

    int64_t softmaxInputSize = CeilDiv(runInfo.nBaseSizeP, 8) * 8;
    LocalTensor<T> softmaxMaxTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 0), sharedTBuf.GetWithOffset<T>(softmaxInputSize, softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxSumTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 2 * softmaxInputSize * sizeof(T)), sharedTBuf.GetWithOffset<T>(softmaxInputSize, 3 * softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxMaxUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 1024);
    LocalTensor<T> softmaxSumUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 2048);

    uint8_t pingpong = 0;
    for (int64_t n = 0; n < nLoops; n++) { // nLoops=4
        int64_t curNSize = (n == nLoops - 1) ? constInfo.gSizeQuery - (runInfo.nBaseSizeP * n) : runInfo.nBaseSizeP;
        for (int64_t k = 0; k < kLoops; k++) {
            int64_t curKOffset = K_BASE_SIZE * k;
            int64_t curKSize = (k == kLoops - 1) ? runInfo.kRealSize - curKOffset : K_BASE_SIZE;
            int64_t curKSizeAlign8 = CeilDiv(curKSize, 8) * 8;
            int64_t bmm1NOffset = n * runInfo.nBaseSizeP * topKSize + k * K_BASE_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            DataCopyExtParams copyParams(   static_cast<uint16_t>(curNSize),
                                            static_cast<uint32_t>(curKSize * sizeof(T)),
                                            static_cast<uint32_t>((topKSize - curKSize) * sizeof(T)), 0, 0);
            DataCopyPadExtParams<T> copyPadParams(false, 0, 0, 0);
            AscendC::DataCopyPad(mulsResUb[pingpong], bmm1ResGm[bmm1Offset + bmm1NOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::Muls(mulsResUb[pingpong], mulsResUb[pingpong], constInfo.scaleValue, curNSize * curKSizeAlign8);
            PipeBarrier<PIPE_V>();

            DataCopyExtParams softmaxCopyParams(    static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(runInfo.nBaseSizeP * sizeof(T)), // 8 * 4 = 32Bytes
                                                    static_cast<uint32_t>(0), 0, 0);
            DataCopyPadExtParams<T> softmaxCopyPadParams(false, 0, 0, 0);

            int64_t softmaxInputOffset = 0;
            if constexpr (LAYOUT_T == SLILayout::BSND) {
                softmaxInputOffset = runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQuery) +
                    runInfo.s1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            } else if constexpr (LAYOUT_T == SLILayout::TND) {
                softmaxInputOffset = runInfo.accumS1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            }
            AscendC::DataCopyPad(softmaxMaxTmp[pingpong], softmaxMaxGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::DataCopyPad(softmaxSumTmp[pingpong], softmaxSumGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::Brcb(softmaxMaxUb, softmaxMaxTmp[pingpong], curNSize, {1, 8});
            AscendC::Brcb(softmaxSumUb, softmaxSumTmp[pingpong], curNSize, {1, 8});
            PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> softmaxTmpUb = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 0);
            AscendC::SimpleSoftMax<T>(mulsResUb[pingpong], softmaxSumUb, softmaxMaxUb, mulsResUb[pingpong],
                                      softmaxTmpUb, constInfo.simpleSoftMaxTilingInfo, 
                                      {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8),
                                      static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSize)});
            PipeBarrier<PIPE_V>();
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8)};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(reduceSumPTmpBuffer, mulsResUb[pingpong], sharedTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            if (n == 0) {
                AscendC::DataCopy(reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            } else {
                AscendC::Add(reduceSumPResUb[curKOffset], reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            }
            PipeBarrier<PIPE_V>();
            pingpong = 1 - pingpong;
        }
    }
    PipeBarrier<PIPE_V>();
    float gRec = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    AscendC::Muls(reduceSumPResUb, reduceSumPResUb, gRec, topKSize);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopy(psySyncGm, reduceSumPResUb, topKSize);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorPWithOutSoftMaxSum(SLIGradKLLossRunInfo &runInfo)
{
    int64_t bmm1Offset = runInfo.taskIdMod2 * constInfo.gSizeQuery * topKSize;
    int64_t kLoops = CeilDiv(runInfo.kRealSize, K_BASE_SIZE);

    // Case-adaptive row tiling: mulsResUb (mm1Tbuf=32KB) must hold nBase * tileKalign8 floats.
    // When the whole K fits one chunk (kLoops==1), the chunk width is align8(kRealSize) (<=2048),
    // so we can pack more rows per iteration -> fewer iterations -> fewer PIPE_V barrier stalls.
    // When kLoops>1, chunks are full K_BASE_SIZE=2048 wide, keep the original 4-row tile (safe).
    int64_t nBaseLocal = runInfo.nBaseSizeP; // default 4
    if (kLoops == 1) {
        int64_t kAlign8 = CeilDiv(runInfo.kRealSize, (int64_t)8) * 8;
        int64_t maxRows = (32 * 1024) / (kAlign8 * (int64_t)sizeof(T)); // floor(32KB / (Kalign8*4))
        if (maxRows < 1) { maxRows = 1; }
        if (maxRows > (int64_t)constInfo.gSizeQuery) { maxRows = constInfo.gSizeQuery; }
        if (maxRows > nBaseLocal) { nBaseLocal = maxRows; }
    }
    int64_t nLoops = CeilDiv(constInfo.gSizeQuery, nBaseLocal);

    LocalTensor<T> mulsResUb[2] = {mm1Tbuf.Get<T>(), mm2TBuf.Get<T>()};
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(16 * 1024, 0);
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(K_BASE_SIZE, 16 * 1024);

    uint8_t pingpong = 0;
    for (int64_t n = 0; n < nLoops; n++) {
        int64_t curNSize = (n == nLoops - 1) ? constInfo.gSizeQuery - (nBaseLocal * n) : nBaseLocal;
        for (int64_t k = 0; k < kLoops; k++) {
            int64_t curKOffset = K_BASE_SIZE * k;
            int64_t curKSize = (k == kLoops - 1) ? runInfo.kRealSize - curKOffset : K_BASE_SIZE;
            int64_t curKSizeAlign8 = CeilDiv(curKSize, 8) * 8;
            int64_t bmm1NOffset = n * nBaseLocal * topKSize + k * K_BASE_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            DataCopyExtParams copyParams(   static_cast<uint16_t>(curNSize),
                                            static_cast<uint32_t>(curKSize * sizeof(T)),
                                            static_cast<uint32_t>((topKSize - curKSize) * sizeof(T)), 0, 0);
            DataCopyPadExtParams<T> copyPadParams(false, 0, 0, 0);
            AscendC::DataCopyPad(mulsResUb[pingpong], bmm1ResGm[bmm1Offset + bmm1NOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            // scaleValue==1.0 -> Muls is identity; skip it (and its PIPE_V barrier, since then no
            // vector op precedes SoftMax and the MTE2->V wait already guards the data). Bit-exact.
            if (constInfo.scaleValue != 1.0f) {
                AscendC::Muls(mulsResUb[pingpong], mulsResUb[pingpong], constInfo.scaleValue, curNSize * curKSizeAlign8);
                PipeBarrier<PIPE_V>();
            }

            LocalTensor<uint8_t> softmaxTmpUb = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 0);
            AscendC::SoftMax<T, false, true>(mulsResUb[pingpong], mulsResUb[pingpong], softmaxTmpUb, constInfo.simpleSoftMaxTilingInfo,
                                {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8),
                                 static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSize)});

            PipeBarrier<PIPE_V>();
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8)};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(reduceSumPTmpBuffer, mulsResUb[pingpong], sharedTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            if (n == 0) {
                AscendC::DataCopy(reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            } else {
                AscendC::Add(reduceSumPResUb[curKOffset], reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            }
            PipeBarrier<PIPE_V>();
            pingpong = 1 - pingpong;
        }
    }
    PipeBarrier<PIPE_V>();
    float gRec = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    AscendC::Muls(reduceSumPResUb, reduceSumPResUb, gRec, topKSize);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopy(psySyncGm, reduceSumPResUb, topKSize);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::PreloadWeight(SLIGradKLLossRunInfo &runInfo)
{
    int64_t weightOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        weightOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * constInfo.gSizeQueryIndex;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        weightOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex;
    }
    int64_t weightDBOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16;
    // weight 可以常驻, 所以直接搬运, 减少搬运切片

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    AscendC::DataCopy(weightInUb[weightDBOffset], weightGm[weightOffset], constInfo.gSizeQueryIndexAlign16);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerPreW);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerPreW);
    AscendC::Cast(weightUb[weightDBOffset], weightInUb[weightDBOffset], AscendC::RoundMode::CAST_NONE, constInfo.gSizeQueryIndex);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    PipeBarrier<PIPE_V>();
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorSy(SLIGradKLLossRunInfo &runInfo)
{
    {
        // weight UB cast To weight.GetValue and release eventID
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    }
    // 核内保留一份BS1Idx
    int64_t realKSize = runInfo.s2RealSize; // int64_t &realKSize = runInfo.s2RealSize;
    int64_t realKSizeAlign = CeilDiv(realKSize, 8) * 8; 
 
    // 切B, S1做循环, bmm2 单次存储 [N1_INDEX, TopK] 大小的数据块
    // 为了softmax K轴尽量不切分, 只能在N1方向上切
    int64_t bmm2ResSize = constInfo.gSizeQueryIndex * topKSize; //  sizeof(float)
    int64_t bmm2ResOffset = runInfo.taskIdMod2 * bmm2ResSize;
    // relu无法复用LocalTensor 所以再 8/ 2 = 4行
    int64_t nSplitSize = ubAllocPolicy.mm2UbSize / (topKSize * sizeof(float));
    // 一个Cube 做 G 行, 分给两个Vector, 每个做 G / 2 的大小
    int64_t nLoopSize = CeilDiv(constInfo.gSizeQueryIndex, nSplitSize);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    for (int64_t n = 0; n < nLoopSize; n++) {
        int64_t bmm2Offset = n * nSplitSize * topKSize; // 4 * 2048
        DataCopyExtParams copyBmm2Params(static_cast<uint16_t>(nSplitSize), static_cast<uint32_t>(realKSize * sizeof(T)), static_cast<uint32_t>((topKSize - realKSize) * sizeof(T)), 0, 0);
        DataCopyPadExtParams<T> copyBmm2PadParams(true, 0, (uint8_t)(realKSizeAlign - realKSize), 0.0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        AscendC::DataCopyPad<T>(reluResUb, bmm2ResGm[bmm2ResOffset + bmm2Offset], copyBmm2Params, copyBmm2PadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        int64_t weightOffset = !constInfo.subBlockIdx * constInfo.gSizeQueryIndexAlign16;
        // 先一次性把本n块的nSplitSize个权重标量读出，再做“一次”S->V同步，最后统一Muls；
        // 把原来每个i一次的S->V往返同步(nSplitSize次)降到1次，减少标量-向量串行开销。
        // reluResUb各行(mulOffset=i*realKSizeAlign)互不相交，权重值不变 -> bit-exact。
        constexpr int32_t MAX_N_SPLIT_SIZE = 64;   // nSplitSize <= gSizeQueryIndex(<=64)
        float weightValues[MAX_N_SPLIT_SIZE];
        for (int32_t i = 0; i < nSplitSize; i++) {
            weightValues[i] = weightUb[weightOffset].GetValue(n * nSplitSize + i);
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        for (int32_t i = 0; i < nSplitSize; i++) {
            int64_t mulOffset = i * realKSizeAlign;
            AscendC::Muls<T>(reluResUb[mulOffset], reluResUb[mulOffset], weightValues[i], realKSizeAlign);
        }
        PipeBarrier<PIPE_V>();
        
        LocalTensor<T> reduceUb = nSplitSize > 1 ? reduceSumYResTmpBuffer : reluResUb;
        if (nSplitSize > 1) {
            // 使用高阶API做自选维度的reduce 不支持源操作数与目的操作数地址重叠。不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
            uint32_t reduceShape[] = { static_cast<uint32_t>(nSplitSize), static_cast<uint32_t>(realKSizeAlign) };
            constexpr bool isReuse = true;
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, isReuse>(reduceUb, reluResUb, reduceSumTmpBuffer, reduceShape, true);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
            PipeBarrier<PIPE_V>();
        }

        if (n == 0) {
            AscendC::DataCopy(reduceSumYResUb, reduceUb, realKSizeAlign);
        } else {
            AscendC::Add(reduceSumYResUb, reduceSumYResUb, reduceUb, realKSizeAlign);
        }

        if (nSplitSize == 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        }
        PipeBarrier<PIPE_V>();
    }

    SoftMaxShapeInfo softmaxShapeInfoData = {
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSizeAlign),
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSize),
    };
    AscendC::SoftMax<T>(reduceSumYResUb, reduceSumYResUb, softmaxTmpBuffer, constInfo.tilingInfo,
                        {static_cast<uint32_t>(1), static_cast<uint32_t>(realKSizeAlign),
                         static_cast<uint32_t>(1), static_cast<uint32_t>(realKSize)});
    PipeBarrier<PIPE_V>();
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(realKSize * sizeof(float)), 0, 0, 0); 
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::DataCopyPad(psySyncGm[topKSize], reduceSumYResUb, copyParams);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &subResTensor,
                                                              LocalTensor<T> &maskUb, int32_t kRealSizeAlign)
{
    // 根据mask选择subResTensor中可用数据
    Select(subResTensor, maskUb, subResTensor, static_cast<T>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, kRealSizeAlign);
    PipeBarrier<PIPE_V>();
    Cast(reluGradOutTensor, subResTensor, AscendC::RoundMode::CAST_ROUND, kRealSizeAlign);
    PipeBarrier<PIPE_V>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDkLess2k(SLIGradKLLossRunInfo &runInfo)
{
    LocalTensor<T> reluResUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> mulLeftUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> reduceSumPUbSingleK, reduceSumYUbSingleK;
    LocalTensor<KV_T> reluGradUb[2] = {reluGradpingBuf, reluGradpongBuf};
    
    // maskTensor需要创建 目前一次传输32k数据对比需要8k个数，需要8k/8=1k大小空间（按bit存储）
    LocalTensor<T> maskUb = this->dwTBuf.template Get<T>()[256]; 
    // dwTBuf理论上最大只需要512bytes，一个元素4bytes，一共最大64个元素，开DB一共256*2=512
    LocalTensor<T> reduceSumResTensor = this->dwTBuf.template Get<T>();

    int32_t kRealSize = runInfo.kTailSize;
    int32_t kRealSizeAlign16 = BlockAlign<KV_T>(kRealSize); // 按照KV_T类型对齐，datacopy才不会有问题

    SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    if (runInfo.calcP) {
        reduceSumPUbSingleK = resPSYTBuf.Get<T>();
        reduceSumYUbSingleK = mm2TBuf.template Get<T>()[6144];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize], kRealSizeAlign16);
    } else {
        reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
        reduceSumYUbSingleK = resPSYTBuf.Get<T>();
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopy(reduceSumPUbSingleK, psySyncOtherGm, kRealSizeAlign16);
    }

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

    Sub(v1TmpUb, reduceSumYUbSingleK, reduceSumPUbSingleK, kRealSize);
    PipeBarrier<PIPE_V>();

    // k方向单次处理最大长度为2048， g方向为4，故一次性为 4*2048*4 = 32k
    uint32_t gSizePerVec = constInfo.gSizeQueryIndex / 2;
    constexpr uint32_t gSizeInner = 4;
    int32_t gLoopTimes = gSizePerVec / gSizeInner;

    SumParams sumParams = {static_cast<uint32_t>(gSizeInner), static_cast<uint32_t>(kRealSizeAlign16), static_cast<uint32_t>(kRealSize)};
    DataCopyParams dataCopyReluResParams(gSizeInner, kRealSizeAlign16 * sizeof(T), (topKSize - kRealSizeAlign16) * sizeof(T), 0);
    DataCopyPadParams dataCopyReluResPadParams(false, 0, 0, 0);
    DataCopyParams dataCopyReluGradParams(gSizeInner, kRealSize * sizeof(KV_T), 0, (topKSize - kRealSize) * sizeof(KV_T));

    int64_t subBlockGQueryIndexOffset = constInfo.subBlockIdx * gSizePerVec;
    int64_t reluGmOffset = (runInfo.taskIdMod2 * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset) * topKSize;
    int64_t weightOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16 + subBlockGQueryIndexOffset;
    
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    for (size_t nIdx = 0; nIdx < gLoopTimes; nIdx++) {
        uint8_t pingPong = nIdx & 1;
        maskUb = maskUb[pingPong * 256]; // 相差256个数正好1k
        WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]);

        // 获取输入的relu数据和weights数据偏差每gSizePerVec中gSizeInner个
        int64_t reluResOffset = reluGmOffset + nIdx * gSizeInner * topKSize; 
        // 每次搬运gSizeInner个kRealSizeAlign16 | reluResUb shape-> (gSizeInner, kRealSizeAlign16)
        DataCopyPad(reluResUb[pingPong], reluGm[reluResOffset], dataCopyReluResParams, dataCopyReluResPadParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

        CompareScalar(maskUb, reluResUb[pingPong], static_cast<T>(0.0), AscendC::CMPMODE::GT, gSizeInner * kRealSizeAlign16);
        PipeBarrier<PIPE_V>();

        // Disjoint-output, shared read-only v1TmpUb: no inter-iter hazard -> issue back-to-back.
        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            Mul(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb, reluResUb[pingPong][gInnerIdx * kRealSizeAlign16], kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        // 每一块相同g轴方向的k需要Add相加，第一次可直接输出到最后的蔬菜，后面需要叠加到第一块输出上
        // 只有一次计算直接输出 Sum每一次得到gSizeInner个数
        AscendC::Sum(reduceSumResTensor[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);

        // Disjoint-output, shared read-only v1TmpUb: no inter-iter hazard -> issue back-to-back.
        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            Muls(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb, weightTensor.GetValue(nIdx * gSizeInner + gInnerIdx + weightOffset), kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
        ReLUGrad(reluGradUb[pingPong], mulLeftUb[pingPong], maskUb, gSizeInner * kRealSizeAlign16);
        SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        DataCopyPad(reluGradResGm[reluResOffset], reluGradUb[pingPong], dataCopyReluGradParams);

        SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]); // reluGradUb、reluResUb用的同一块buf，reluResUb获取Gm数据前需等待这里mte3搬运完
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    // 只有一轮计算，计算完直接可以拷贝到GM输出
    PipeBarrier<PIPE_V>();

    if (runInfo.s2SparseLen <= 0) {
        Duplicate(dwOutTensor, (KV_T)0.0, gSizePerVec);
    } else {
        Cast(dwOutTensor, reduceSumResTensor, AscendC::RoundMode::CAST_ROUND, gSizePerVec);
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);

    // 拷出到GM
    DataCopyParams dataCopyParams(1, gSizePerVec * sizeof(KV_T), 0, 1);
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        DataCopyPad(dWeightGm[runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
            runInfo.s1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        DataCopyPad(dWeightGm[runInfo.accumS1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDkMoreThan2k(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    LocalTensor<T> reluResUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> mulLeftUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> reduceSumPUbSingleK, reduceSumYUbSingleK;
    LocalTensor<KV_T> reluGradUb[2] = {reluGradpingBuf, reluGradpongBuf};
    
    // maskTensor需要创建 目前一次传输32k数据对比需要8k个数，需要8k/8=1k大小空间（按bit存储）
    LocalTensor<T> maskUb = this->dwTBuf.template Get<T>()[256]; 
    // dwTBuf理论上最大只需要512bytes，一个元素4bytes，一共最大64个元素，开DB一共256*2=512
    LocalTensor<T> reduceSumResTensor = this->dwTBuf.template Get<T>();
    int64_t kLoopOffset = kLoopIdx * K_BASE_SIZE;
    int32_t kRealSize = (kLoopIdx == runInfo.kLoopTimes - 1) ? runInfo.kTailSize : K_BASE_SIZE;
    int32_t kRealSizeAlign16 = BlockAlign<KV_T>(kRealSize); // 按照KV_T类型对齐，datacopy才不会有问题

    SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    if (runInfo.calcP) {
        reduceSumPUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
        reduceSumYUbSingleK = mm2TBuf.template Get<T>()[6144];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize + kLoopOffset], kRealSizeAlign16);
    } else {
        reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
        reduceSumYUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopy(reduceSumPUbSingleK, psySyncOtherGm[kLoopOffset], kRealSizeAlign16);
    }

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

    Sub(v1TmpUb, reduceSumYUbSingleK, reduceSumPUbSingleK, kRealSize);
    PipeBarrier<PIPE_V>();

    // k方向单次处理最大长度为2048， g方向为4，故一次性为 4*2048*4 = 32k
    uint32_t gSizePerVec = constInfo.gSizeQueryIndex / 2;
    constexpr uint32_t gSizeInner = 4;
    int32_t gLoopTimes = gSizePerVec / gSizeInner;

    SumParams sumParams = {static_cast<uint32_t>(gSizeInner), static_cast<uint32_t>(kRealSizeAlign16), static_cast<uint32_t>(kRealSize)};
    DataCopyParams dataCopyReluResParams(gSizeInner, kRealSizeAlign16 * sizeof(T), (topKSize - kRealSizeAlign16) * sizeof(T), 0);
    DataCopyPadParams dataCopyReluResPadParams(false, 0, 0, 0);

    DataCopyParams dataCopyReluGradParams(gSizeInner, kRealSize * sizeof(KV_T), 0, (topKSize - kRealSize) * sizeof(KV_T));

    int64_t subBlockGQueryIndexOffset = constInfo.subBlockIdx * gSizePerVec;
    int64_t reluGmOffset = (runInfo.taskIdMod2 * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset) * topKSize + kLoopOffset;
    int64_t weightOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16 + subBlockGQueryIndexOffset;
    
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    for (size_t nIdx = 0; nIdx < gLoopTimes; nIdx++) {
        uint8_t pingPong = nIdx & 1;
        maskUb = maskUb[pingPong * 256]; // 相差256个数正好1k
        WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]);

        // 获取输入的relu数据和weights数据偏差每gSizePerVec中gSizeInner个
        int64_t reluResOffset = reluGmOffset + nIdx * gSizeInner * topKSize; 
        // 每次搬运gSizeInner个kRealSizeAlign16 | reluResUb shape-> (gSizeInner, kRealSizeAlign16)
        DataCopyPad(reluResUb[pingPong], reluGm[reluResOffset], dataCopyReluResParams, dataCopyReluResPadParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

        CompareScalar(maskUb, reluResUb[pingPong], static_cast<T>(0.0), AscendC::CMPMODE::GT, gSizeInner * kRealSizeAlign16);
        PipeBarrier<PIPE_V>();

        // Disjoint-output, shared read-only v1TmpUb: no inter-iter hazard -> issue back-to-back.
        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            Mul(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb, reluResUb[pingPong][gInnerIdx * kRealSizeAlign16], kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        // 每一块相同g轴方向的k需要Add相加，第一次可直接输出到最后的蔬菜，后面需要叠加到第一块输出上
        if (kLoopIdx == 0) {
            // 首轮直接输出 Sum每一次得到gSizeInner个数
            AscendC::Sum(reduceSumResTensor[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);
        } else {
            AscendC::Sum(reduceSumDwResUb[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);
        }
        
        // Disjoint-output, shared read-only v1TmpUb: no inter-iter hazard -> issue back-to-back.
        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            Muls(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb, weightTensor.GetValue(nIdx * gSizeInner + gInnerIdx + weightOffset), kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
        ReLUGrad(reluGradUb[pingPong], mulLeftUb[pingPong], maskUb, gSizeInner * kRealSizeAlign16);
        SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        DataCopyPad(reluGradResGm[reluResOffset], reluGradUb[pingPong], dataCopyReluGradParams);

        SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]); // reluGradUb、reluResUb用的同一块buf，reluResUb获取Gm数据前需等待这里mte3搬运完
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    if (kLoopIdx > 0) {
        PipeBarrier<PIPE_V>();
        Add(reduceSumResTensor, reduceSumResTensor, reduceSumDwResUb, gSizePerVec);
    }

    if (kLoopIdx == runInfo.kLoopTimes - 1) {
        PipeBarrier<PIPE_V>();
        if (runInfo.s2SparseLen <= 0) {
            Duplicate(dwOutTensor, (KV_T)0.0, gSizePerVec);
        } else {
            Cast(dwOutTensor, reduceSumResTensor, AscendC::RoundMode::CAST_ROUND, gSizePerVec);
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);

        // 拷出到GM
        DataCopyParams dataCopyParams(1, gSizePerVec * sizeof(KV_T), 0, 1);
        if constexpr (LAYOUT_T == SLILayout::BSND) {
            DataCopyPad(dWeightGm[runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
                runInfo.s1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
        } else if constexpr (LAYOUT_T == SLILayout::TND) {
            DataCopyPad(dWeightGm[runInfo.accumS1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
        }
    }
}

template <typename SLIT>
template <uint32_t range>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDk(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    if constexpr (range <= SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_2K){
        VectorDwDqDkLess2k(runInfo);
    } else {
        VectorDwDqDkMoreThan2k(runInfo, kLoopIdx);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorLossLess2k(SLIGradKLLossRunInfo &runInfo)
{
    int64_t realKSize = runInfo.kTailSize;
    int32_t realKSizeAlign = BlockAlign<T>(realKSize);
    int32_t v0RealKSizeAlign = BlockAlign<T>(CeilDiv(realKSize, 2));
    int32_t v0RealKSize = realKSize;
    if (realKSize > 8) {
        v0RealKSize = v0RealKSizeAlign;
    }
    int32_t v1RealKSize = Max(realKSize - v0RealKSizeAlign, 0);
    int32_t v1RealKSizeAlign = BlockAlign<T>(v1RealKSize);

    int32_t vRealKSize, vRealKSizeAlign;
    int64_t kLoopOffset = 0;
    int64_t coreOffset = 0;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        vRealKSizeAlign = v0RealKSizeAlign;
    } else {
        vRealKSize = v1RealKSize;
        vRealKSizeAlign = v1RealKSizeAlign;
        coreOffset += v0RealKSizeAlign;
    }

    int64_t calcSize = vRealKSize;
    int64_t calcSizeAlign = vRealKSizeAlign;
    const float MIN_VALUE = 1e-8;
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(4 * 1024, 0);
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8 * 1024, 4 * 1024);
    LocalTensor<T> vecTmpUb;
    LocalTensor<T> lossSumUb;
    if (constInfo.subBlockIdx) {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 4 * 1024);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 32);
    } else {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 0);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 0);
    }

    if (vRealKSize != 0) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        LocalTensor<T> reduceSumPTmpUb, reduceSumYResTmpUb;
        if (runInfo.calcP) {  // 当前核做的是VectorP
            reduceSumPTmpUb = resPSYTBuf.Get<T>();
            reduceSumPTmpUb = reduceSumPTmpUb[kLoopOffset + coreOffset];
            reduceSumYResTmpUb = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumYResTmpUb, psySyncOtherGm[topKSize + kLoopOffset + coreOffset], realKSizeAlign);
        } else {  // 当前核做的是VectorSY
            reduceSumYResTmpUb = resPSYTBuf.Get<T>();
            reduceSumYResTmpUb = reduceSumYResTmpUb[kLoopOffset + coreOffset];
            reduceSumPTmpUb = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumPTmpUb, psySyncOtherGm[kLoopOffset + coreOffset], realKSizeAlign);
        }

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerLoss);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerLoss);

        // Two Maxs write disjoint tensors after an MTE2->V sync (no prior PIPE_V op): drop the
        // leading and inter-Maxs barriers; keep the one before Log (RAW on Maxs outputs).
        AscendC::Maxs<T>(reduceSumPTmpUb, reduceSumPTmpUb, MIN_VALUE, calcSize);
        AscendC::Maxs<T>(reduceSumYResTmpUb, reduceSumYResTmpUb, MIN_VALUE, calcSize);
        PipeBarrier<PIPE_V>();

        AscendC::Log(reduceSumPTmpBuffer, reduceSumPTmpUb, calcSize);
        AscendC::Log(reduceSumYResTmpUb, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(vecTmpUb, reduceSumPTmpBuffer, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(vecTmpUb, vecTmpUb, reduceSumPTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Sum(lossSumUb, vecTmpUb, sharedTmpBuffer,
                {static_cast<uint32_t>(1), static_cast<uint32_t>(calcSizeAlign), static_cast<uint32_t>(calcSize)});
        PipeBarrier<PIPE_V>();
    }

    if (vRealKSize != 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3InnerVecLoss);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3InnerVecLoss);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopyPad(lossGm, lossSumUb,
                            {static_cast<uint32_t>(1), static_cast<uint32_t>(4),
                            static_cast<uint32_t>(0), static_cast<uint32_t>(0)});
        SetAtomicNone();
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorLossMoreThan2k(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    bool isLastBasicBlock = kLoopIdx == runInfo.kLoopTimes - 1;
    int64_t realKSize = isLastBasicBlock ? runInfo.kTailSize : K_BASE_SIZE;
    int32_t realKSizeAlign = BlockAlign<T>(realKSize);
    int32_t v0RealKSizeAlign = BlockAlign<T>(CeilDiv(realKSize, 2));
    int32_t v0RealKSize = realKSize;
    if (realKSize > 8) {
        v0RealKSize = v0RealKSizeAlign;
    }
    int32_t v1RealKSize = Max(realKSize - v0RealKSizeAlign, 0);
    int32_t v1RealKSizeAlign = BlockAlign<T>(v1RealKSize);

    int32_t vRealKSize, vRealKSizeAlign;
    int64_t kLoopOffset = kLoopIdx * K_BASE_SIZE;
    int64_t coreOffset = 0;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        vRealKSizeAlign = v0RealKSizeAlign;
    } else {
        vRealKSize = v1RealKSize;
        vRealKSizeAlign = v1RealKSizeAlign;
        coreOffset += v0RealKSizeAlign;
    }

    int64_t calcSize = vRealKSize;
    int64_t calcSizeAlign = vRealKSizeAlign;
    const float MIN_VALUE = 1e-8;
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(4 * 1024, 0);
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8 * 1024, 4 * 1024);
    LocalTensor<T> vecTmpUb;
    LocalTensor<T> lossSumUb;
    LocalTensor<T> tmpLossUb;
    if (constInfo.subBlockIdx) {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 4 * 1024);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 32);
        tmpLossUb = sharedTBuf.GetWithOffset<T>(8, 12 * 1024 + 32);
    } else {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 0);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 0);
        tmpLossUb = sharedTBuf.GetWithOffset<T>(8, 12 * 1024);
    }

    if (vRealKSize != 0) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        LocalTensor<T> reduceSumPTmpUb, reduceSumYResTmpUb;
        if (runInfo.calcP) {  // 当前核做的是VectorP
            reduceSumPTmpUb = resPSYTBuf.Get<T>();
            reduceSumPTmpUb = reduceSumPTmpUb[kLoopOffset + coreOffset];
            reduceSumYResTmpUb = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumYResTmpUb, psySyncOtherGm[topKSize + kLoopOffset + coreOffset], realKSizeAlign);
        } else {  // 当前核做的是VectorSY
            reduceSumYResTmpUb = resPSYTBuf.Get<T>();
            reduceSumYResTmpUb = reduceSumYResTmpUb[kLoopOffset + coreOffset];
            reduceSumPTmpUb = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumPTmpUb, psySyncOtherGm[kLoopOffset + coreOffset], realKSizeAlign);
        }

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerLoss);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerLoss);

        // Two Maxs write disjoint tensors after an MTE2->V sync (no prior PIPE_V op): drop the
        // leading and inter-Maxs barriers; keep the one before Log (RAW on Maxs outputs).
        AscendC::Maxs<T>(reduceSumPTmpUb, reduceSumPTmpUb, MIN_VALUE, calcSize);
        AscendC::Maxs<T>(reduceSumYResTmpUb, reduceSumYResTmpUb, MIN_VALUE, calcSize);
        PipeBarrier<PIPE_V>();

        AscendC::Log(reduceSumPTmpBuffer, reduceSumPTmpUb, calcSize);
        AscendC::Log(reduceSumYResTmpUb, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(vecTmpUb, reduceSumPTmpBuffer, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(vecTmpUb, vecTmpUb, reduceSumPTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        if (kLoopIdx == 0) {
            AscendC::Sum(lossSumUb, vecTmpUb, sharedTmpBuffer,
                    {static_cast<uint32_t>(1), static_cast<uint32_t>(calcSizeAlign), static_cast<uint32_t>(calcSize)});
            PipeBarrier<PIPE_V>();
        } else {
            AscendC::Sum(tmpLossUb, vecTmpUb, sharedTmpBuffer,
                    {static_cast<uint32_t>(1), static_cast<uint32_t>(calcSizeAlign), static_cast<uint32_t>(calcSize)});
            PipeBarrier<PIPE_V>();
            //与之前基本块的loss相加
            AscendC::Add(lossSumUb, lossSumUb, tmpLossUb, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    if (isLastBasicBlock && (runInfo.kLoopTimes > 1 || vRealKSize != 0)) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3InnerVecLoss);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3InnerVecLoss);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopyPad(lossGm, lossSumUb,
                            {static_cast<uint32_t>(1), static_cast<uint32_t>(4),
                            static_cast<uint32_t>(0), static_cast<uint32_t>(0)});
        SetAtomicNone();
    }
}

template <typename SLIT>
template <uint32_t range>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorLoss(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    if (runInfo.s2SparseLen > 0) {
        if constexpr (range <= SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_2K) {
            VectorLossLess2k(runInfo);
        } else {
            VectorLossMoreThan2k(runInfo, kLoopIdx);
        }
    }
}
#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
