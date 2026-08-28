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
 * \file sparse_lightning_indexer_grad_kl_loss_service_cube.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_grad_kl_loss_common.h"

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose = false;
    bool isRightTranspose = false;
    bool isOutKFisrt = true;
    bool isFixOut = true;
    bool isL0CAccum = false;
};

template <typename SLIT> class SLITMatmulService {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using MM_OUT_T = T;

    __aicore__ inline SLITMatmulService(){};
    __aicore__ inline void InitParams(const SLIGradKLLossConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &keyGatherWithRopeGm,
                                             GlobalTensor<Q_T> &qRopeGm,
                                             GlobalTensor<int64_t> &actualSeqLengthsQueryGm,
                                             GlobalTensor<int64_t> &actualSeqLengthsKeyGm,
                                             GlobalTensor<MM_OUT_T> &bmm1Res,
                                             GM_ADDR mm1Res);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<Q_T> &queryIndex, GlobalTensor<KV_T> &keyIndexGather, GlobalTensor<T> &mm2Res);
    __aicore__ inline void InitMm5GlobalTensor(GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &queryIndexGm,
                                             GlobalTensor<MM_OUT_T> &bmm5Res, GlobalTensor<int32_t> &topKIndex);
    __aicore__ inline void InitMm6GlobalTensor(GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &keyIndexGm,
                                             GlobalTensor<OUT_T> &bmm6Res);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const SLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm2(const SLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm5(const SLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm6(const SLIGradKLLossRunInfo &info);

private:
    static constexpr bool HAS_ROPE = SLIT::hasRope;
    static constexpr SLILayout INPUT_LAYOUT = SLIT::inputQLayout;
    static constexpr uint32_t topKSize = static_cast<uint32_t>(SLIT::topKRange);
    static constexpr bool IS_RELUGRAD_REUSE = topKSize <= SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_2K;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_SPLIT_SIZE = 128;     // k方向切分
    static constexpr uint32_t RELU_GRAD_SPLIT_SIZE = 2048; // reluGrad切分

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // AIC 和 AIV间同步
    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    static constexpr uint32_t RELU_GRAD_EVENT = EVENT_ID1;
    static constexpr uint32_t SYNC_MM_RELU_GRAD_EVENT = EVENT_ID4;
    static constexpr uint32_t SYNC_MM5_MM6_EVENT = EVENT_ID5;
    static constexpr uint32_t QUERY_INDEX_EVENT[2] = {EVENT_ID6, EVENT_ID7};

    static constexpr uint32_t SYNC_MTE21_FLAG[2] = {L1_EVENT0, L1_EVENT1};
    static constexpr uint32_t SYNC_MTE1MM_FLAG[2] = {L1_EVENT2, L1_EVENT3};
    static constexpr uint32_t SYNC_MMFIX_FLAG[2] = {L1_EVENT4, L1_EVENT5};

    SLIGradKLLossConstInfo constInfo{};

    GlobalTensor<int64_t> actualSeqLengthsQueryGm, actualSeqLengthsKeyGm;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGatherWithRopeGm[2];
    GlobalTensor<MM_OUT_T> mm1ResGm[2];

    // mm2
    GlobalTensor<Q_T> queryIndexGm;
    GlobalTensor<KV_T> keyIndexGatherGm;
    GlobalTensor<MM_OUT_T> mm2ResGm[2];

    // mm5
    GlobalTensor<int32_t> topKIndexGm;
    GlobalTensor<MM_OUT_T> mm5ResGm;
    // mm6
    GlobalTensor<OUT_T> mm6ResGm;
    GlobalTensor<Q_T> reluGradRes[2];

    TBuf<TPosition::A1> queryBuf;
    TBuf<TPosition::A1> queryIndexBuf[2];
    TBuf<TPosition::A1> keyIndexBuf[2];
    TBuf<TPosition::A1> reluGradOutBuf;

    TBuf<TPosition::A2> tmpBufL0A[2];
    TBuf<TPosition::B2> tmpBufL0B[2];
    TBuf<TPosition::CO1> tmpBufL0C[2];

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<KV_T> l1KVTensor[2];
    LocalTensor<Q_T> l1QIndexTensor[2];
    LocalTensor<Q_T> aL0TensorPingPong[2];
    LocalTensor<KV_T> bL0TensorPingPong[2];
    LocalTensor<MM_OUT_T> cL0TensorPingPong[2];
    LocalTensor<Q_T> l1ReLuGradTensor;

    uint8_t keyGatherResPingPoingFlag = 0;
    uint8_t gatherPingPongFlag = 0;
    uint8_t indexPingPongFlag = 0;
    uint8_t mm2ResPingPongFlag = 0;
    uint8_t l0abPingPongFlag = 0;
    uint8_t l0cPingPongFlag = 0;
    uint8_t mm5ResPingPongFlag = 0;
    uint8_t mm6ResPingPongFlag = 0;

    MMParam mmParam;

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const SLIGradKLLossRunInfo &info, uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T> &aL1Tensor, const SLIGradKLLossRunInfo &info, uint32_t mSizeAct);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T> &l1Tensor, uint64_t gatherOffset,
                                          struct MMParam &mmParam, uint32_t headOffset);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info,
                                                                     uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T> &l1Tensor, uint64_t gatherOffset,
                                                                    uint32_t mSizeAct, uint32_t realDSize, uint32_t headOffset);
    __aicore__ inline void MmadInner(LocalTensor<MM_OUT_T> &l0cTensor, LocalTensor<Q_T> &l1QPTensor, LocalTensor<KV_T> &kTensor,
                                     struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmA(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmAWithTranspose(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmB(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, struct MMParam &mmParam);
    __aicore__ inline void CopyInMm5AToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info, uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void ScatterAdd(GlobalTensor<MM_OUT_T> &resGm, LocalTensor<MM_OUT_T> &l0cTensor, struct MMParam &mmParam, const SLIGradKLLossRunInfo &info, int64_t scatterOffset);
};

template <typename SLIT> __aicore__ inline void SLITMatmulService<SLIT>::InitParams(const SLIGradKLLossConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename SLIT>
__aicore__ inline void
SLITMatmulService<SLIT>::InitMm1GlobalTensor(GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &keyGatherWithRopeGm,
                                             GlobalTensor<Q_T> &qRopeGm,
                                             GlobalTensor<int64_t> &actualSeqLengthsQueryGm,
                                             GlobalTensor<int64_t> &actualSeqLengthsKeyGm,
                                             GlobalTensor<MM_OUT_T> &bmm1Res, GM_ADDR mm1Res)
{
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGatherWithRopeGm[0] = keyGatherWithRopeGm;
    this->keyGatherWithRopeGm[1] = keyGatherWithRopeGm[constInfo.gatherKeySize];
    this->actualSeqLengthsQueryGm = actualSeqLengthsQueryGm;
    this->actualSeqLengthsKeyGm = actualSeqLengthsKeyGm;

    this->mm1ResGm[0] = bmm1Res;
    this->mm1ResGm[1] = bmm1Res[constInfo.gSizeQuery * topKSize];
}

template <typename SLIT>
__aicore__ inline void
SLITMatmulService<SLIT>::InitMm2GlobalTensor(GlobalTensor<Q_T> &queryIndex, GlobalTensor<KV_T> &keyIndexGather, GlobalTensor<T> &mm2Res)
{
    queryIndexGm = queryIndex;
    keyIndexGatherGm = keyIndexGather;
    mm2ResGm[0] = mm2Res;
    mm2ResGm[1] = mm2Res[topKSize * constInfo.gSizeQueryIndex];
}

template <typename SLIT>
__aicore__ inline void
SLITMatmulService<SLIT>::InitMm5GlobalTensor(GlobalTensor<Q_T> &reluGradRes, GlobalTensor<KV_T> &queryIndexGm,
                                             GlobalTensor<MM_OUT_T> &bmm5Res, GlobalTensor<int32_t> &topKIndex)
{
    this->reluGradRes[0] = reluGradRes;
    this->reluGradRes[1] = reluGradRes[topKSize * constInfo.gSizeQueryIndex];
    this->queryIndexGm = queryIndexGm;
    this->topKIndexGm = topKIndex;
    this->mm5ResGm = bmm5Res;
}

template <typename SLIT>
__aicore__ inline void
SLITMatmulService<SLIT>::InitMm6GlobalTensor(GlobalTensor<Q_T> &reluGradRes, GlobalTensor<KV_T> &keyIndexGm,
                                             GlobalTensor<OUT_T> &bmm6Res)
{
    this->reluGradRes[0] = reluGradRes;
    this->reluGradRes[1] = reluGradRes[topKSize * constInfo.gSizeQueryIndex];
    this->keyIndexGatherGm = keyIndexGm;
    this->mm6ResGm = bmm6Res;
}      

template <typename SLIT> __aicore__ inline void SLITMatmulService<SLIT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[1]);
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(RELU_GRAD_EVENT);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[1]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);
}

template <typename SLIT> __aicore__ inline void SLITMatmulService<SLIT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[1]);
    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(RELU_GRAD_EVENT);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[1]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);
}


template <typename SLIT> __aicore__ inline void SLITMatmulService<SLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(queryBuf, 576 * 128 * sizeof(Q_T));
    l1QPTensor = queryBuf.Get<Q_T>();
    pipe->InitBuffer(queryIndexBuf[0], 128 * 64 * sizeof(Q_T));
    pipe->InitBuffer(queryIndexBuf[1], 128 * 64 * sizeof(Q_T));
    l1QIndexTensor[0] = queryIndexBuf[0].Get<Q_T>();
    l1QIndexTensor[1] = queryIndexBuf[1].Get<Q_T>();

    pipe->InitBuffer(keyIndexBuf[0], 128 * 128 * sizeof(KV_T));
    pipe->InitBuffer(keyIndexBuf[1], 128 * 128 * sizeof(KV_T));
    l1KVTensor[0] = keyIndexBuf[0].Get<Q_T>();
    l1KVTensor[1] = keyIndexBuf[1].Get<Q_T>();

    pipe->InitBuffer(reluGradOutBuf, 64 * 2048 * sizeof(KV_T));
    l1ReLuGradTensor = reluGradOutBuf.Get<Q_T>();
    
    // L0A
    pipe->InitBuffer(tmpBufL0A[0], L0A_PP_SIZE); // 32K
    aL0TensorPingPong[0] = tmpBufL0A[0].Get<KV_T>();
    pipe->InitBuffer(tmpBufL0A[1], L0A_PP_SIZE); // 32K
    aL0TensorPingPong[1] = tmpBufL0A[1].Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B[0], L0B_PP_SIZE); // 32K
    bL0TensorPingPong[0] = tmpBufL0B[0].Get<KV_T>();
    pipe->InitBuffer(tmpBufL0B[1], L0B_PP_SIZE); // 32K
    bL0TensorPingPong[1] = tmpBufL0B[1].Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C[0], L0C_PP_SIZE); // 64K
    cL0TensorPingPong[0] = tmpBufL0C[0].Get<MM_OUT_T>();
    pipe->InitBuffer(tmpBufL0C[1], L0C_PP_SIZE); // 64K
    cL0TensorPingPong[1] = tmpBufL0C[1].Get<MM_OUT_T>();
}


template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                                GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                                                uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info,
                                                                     uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryGm[info.queryTensorOffset + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.dSizeQuery);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyInMm1ARopeToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info, uint32_t mSizeAct)
{
    auto srcGm = qRopeGm[info.queryRopeTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, constInfo.dSizeQueryRope, constInfo.dSizeQueryRope);
}

template <typename SLIT>
__aicore__ inline void
SLITMatmulService<SLIT>::CopyInMm1BToL1(LocalTensor<KV_T> &l1Tensor, uint64_t gatherOffset, struct MMParam &mmParam, uint32_t headOffset)
{
    auto srcGm = keyGatherWithRopeGm[keyGatherResPingPoingFlag & 1][gatherOffset];
    CopyGmToL1(l1Tensor, srcGm, mmParam.singleN, mmParam.singleK, headOffset);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyInMm5AToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info,
                                                                     uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = reluGradRes[info.taskIdMod2][headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, topKSize);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::LoadDataMmA(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, struct MMParam &mmParam)
{
    uint32_t alignM = AlignTo(mmParam.isLeftTranspose ? mmParam.singleK : mmParam.singleM, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignK = AlignTo(mmParam.isLeftTranspose ? mmParam.singleM : mmParam.singleK, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignK / C0_SIZE;
    loadData2DParams.srcStride = alignM / C0_SIZE;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = mmParam.isLeftTranspose;
    for (int32_t i = 0; i < alignM / C0_SIZE; i++) {
        LoadData(aL0Tensor[i * alignK * C0_SIZE], aL1Tensor[i * 256], loadData2DParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::LoadDataMmAWithTranspose(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, struct MMParam &mmParam)
{
    uint32_t mloop = (mmParam.singleM + 15) / 16;
    uint32_t kloop = (mmParam.singleK + 15) / 16;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = mloop * kloop;
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = mmParam.isLeftTranspose;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::LoadDataMmB(LocalTensor<KV_T> &l0Tensor,
                                                             LocalTensor<KV_T> &l1Tensor,
                                                             struct MMParam &mmParam)
{
    uint32_t alignN = AlignTo(mmParam.singleN, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignK = AlignTo(mmParam.singleK, static_cast<uint32_t>(C0_SIZE));
    int64_t l1_offset = mmParam.isRightTranspose ? 256 : alignN * C0_SIZE;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignN / C0_SIZE;
    loadData2DParams.srcStride = mmParam.isRightTranspose ? alignK / C0_SIZE : 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = mmParam.isRightTranspose;
    for (int32_t i = 0; i < alignK / C0_SIZE; i++) {
        LoadData(l0Tensor[i * alignN * C0_SIZE], l1Tensor[i * l1_offset], loadData2DParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyInMm2AToL1(LocalTensor<KV_T> &l1Tensor, const SLIGradKLLossRunInfo &info,
                                                                     uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryIndexGm[info.queryIndexTensorOffset + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.dSizeQueryIndex);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::CopyInMm2BToL1(LocalTensor<KV_T> &l1Tensor, uint64_t gatherOffset,
                                                                    uint32_t mSizeAct, uint32_t realDSize, uint32_t headOffset)
{
    auto srcGm = keyIndexGatherGm[gatherOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, realDSize, headOffset);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::MmadInner(LocalTensor<MM_OUT_T> &l0cTensor, LocalTensor<Q_T> &l1QPTensor, LocalTensor<KV_T> &kTensor,
                                                          struct MMParam &mmParam) {
    MmadParams mmadParams;
    mmadParams.m = mmParam.singleM == 1 ? 16 : mmParam.singleM;
    mmadParams.n = mmParam.singleN;
    mmadParams.k = mmParam.singleK;
    mmadParams.cmatrixInitVal = mmParam.isOutKFisrt;
    //左矩阵在L1复用
    LocalTensor<Q_T> l0aTensor = aL0TensorPingPong[l0abPingPongFlag & 1];
    LocalTensor<KV_T> l0bTensor = bL0TensorPingPong[l0abPingPongFlag & 1];

    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]);
    if (mmParam.isLeftTranspose) {
        LoadDataMmAWithTranspose(l0aTensor, l1QPTensor, mmParam);
    } else {
        LoadDataMmA(l0aTensor, l1QPTensor, mmParam);
    }
    LoadDataMmB(l0bTensor, kTensor, mmParam);
    SetFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]); // L0A L0B搬完发给matmul可以运算

    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
    WaitFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]);
    Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
    if (mmParam.isL0CAccum && ((mmadParams.m / 16) * (mmadParams.n / 16) < 10)) {
        PipeBarrier<PIPE_M>();
    } 
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]);
    l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::ScatterAdd(GlobalTensor<MM_OUT_T> &resGm, LocalTensor<MM_OUT_T> &l0cTensor, 
                                                                struct MMParam &mmParam, const SLIGradKLLossRunInfo &info, int64_t scatterOffset)
{
    if (mmParam.isFixOut) {
        SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = mmParam.singleN;
        fixpipeParams.mSize = mmParam.singleM;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = constInfo.dSizeQueryIndex;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        Fixpipe<MM_OUT_T, MM_OUT_T>(resGm, l0cTensor, fixpipeParams);
        SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
    }
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::ComputeMm1(const SLIGradKLLossRunInfo &info)
{
    uint32_t dSizeTotal = HAS_ROPE ? constInfo.dSizeQuery + constInfo.dSizeQueryRope : constInfo.dSizeQuery;
    uint32_t dLoopTimes = (dSizeTotal + 127) / K_SPLIT_SIZE;
    uint32_t kTailLoopSize = N_WORKSPACE_SIZE;
    uint32_t kInnerLoopTimes = N_WORKSPACE_SIZE / N_SPLIT_SIZE;
    uint32_t tailLoopKSize = N_SPLIT_SIZE;
    uint32_t perLoopDSize = K_SPLIT_SIZE;
    uint32_t tailLoopDSize = dSizeTotal - (dLoopTimes - 1) * perLoopDSize;

    int64_t pWorkspaceOffset = 0;
    mmParam.singleM = constInfo.gSizeQuery;
    mmParam.singleN = N_SPLIT_SIZE;
    mmParam.isLeftTranspose = false;
    mmParam.isRightTranspose = false;
    mmParam.isL0CAccum = (dLoopTimes > 1);
    
    for (uint32_t kOuterIdx = 0; kOuterIdx < info.s2LoopTimes; kOuterIdx++) {
        if (kOuterIdx == 0) {
            // 搬运query到L1 128 * 576 * sizeof(fp16)
            CopyInMm1AToL1(l1QPTensor, info, constInfo.gSizeQuery, constInfo.dSizeQuery, 0);
            if constexpr(HAS_ROPE) {
                LocalTensor<Q_T> qRopeTensor = l1QPTensor[AlignTo(constInfo.gSizeQuery, static_cast<uint32_t>(C0_SIZE)) * constInfo.dSizeQuery];
                // 由于L1里面是NZ, 这里q rope的偏移为整块q nope
                CopyInMm1ARopeToL1(qRopeTensor, info, constInfo.gSizeQuery);
            }
        }
        if (kOuterIdx == info.s2LoopTimes - 1) {
            kTailLoopSize = info.kRealSize - kOuterIdx * N_WORKSPACE_SIZE;
            kInnerLoopTimes = AlignTo(kTailLoopSize, N_SPLIT_SIZE) / N_SPLIT_SIZE;
            tailLoopKSize = kTailLoopSize - (kInnerLoopTimes - 1) * N_SPLIT_SIZE;
        }        
        // 此处同步id需要与v0对齐
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_P_FLAG[kOuterIdx & 1]);
        for (uint32_t kInnerIdx = 0; kInnerIdx < kInnerLoopTimes; kInnerIdx++) {
            int64_t workspaceOffset = kOuterIdx * N_WORKSPACE_SIZE * dSizeTotal + kInnerIdx * N_SPLIT_SIZE * constInfo.dSizeQuery;
            mmParam.singleK = perLoopDSize;
            LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];
            int64_t rightStride = constInfo.dSizeQuery;
            if (kInnerIdx == kInnerLoopTimes - 1) {
                mmParam.singleN = tailLoopKSize;
            }
            for (uint32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
                mmParam.isOutKFisrt = dIdx == 0;
                mmParam.isFixOut = dIdx == dLoopTimes - 1;
                LocalTensor<KV_T> kTensor = l1KVTensor[gatherPingPongFlag & 1];
                int64_t gatherWorkspaceOffset = workspaceOffset + dIdx * 128;
                if (HAS_ROPE && (dIdx == dLoopTimes - 1)) {
                    mmParam.singleK = tailLoopDSize;
                    gatherWorkspaceOffset = kOuterIdx * N_WORKSPACE_SIZE * dSizeTotal + N_WORKSPACE_SIZE * constInfo.dSizeQuery + kInnerIdx * K_SPLIT_SIZE * constInfo.dSizeQueryRope;
                    rightStride = constInfo.dSizeQueryRope;
                }
                // 反向同步mte1_mte2
                WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
                CopyInMm1BToL1(kTensor, gatherWorkspaceOffset, mmParam, rightStride);
                SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
                WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
                uint32_t offset = dIdx * AlignTo(constInfo.gSizeQuery, static_cast<uint32_t>(C0_SIZE)) * perLoopDSize;
                LocalTensor<KV_T> qTensor = l1QPTensor[offset];
                GlobalTensor<MM_OUT_T> resGm = mm1ResGm[info.taskIdMod2][pWorkspaceOffset];
                MmadInner(l0cTensor, qTensor, kTensor, mmParam);
                SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]); 
                if (mmParam.isFixOut) {
                    SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
                    WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
                    FixpipeParamsV220 fixpipeParams;
                    fixpipeParams.nSize = mmParam.singleN;
                    fixpipeParams.mSize = mmParam.singleM;
                    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
                    fixpipeParams.dstStride = topKSize;
                    fixpipeParams.ndNum = 1;
                    fixpipeParams.srcNdStride = 0;
                    fixpipeParams.dstNdStride = 0;
                    Fixpipe<MM_OUT_T, MM_OUT_T>(resGm, l0cTensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
                }
                SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);  
                gatherPingPongFlag = (gatherPingPongFlag + 1) & 1;
            }
            l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
            pWorkspaceOffset += N_SPLIT_SIZE;
        }
    }
    // 此处同步id需要与v1对齐
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[info.taskIdMod2]);
    keyGatherResPingPoingFlag = (keyGatherResPingPoingFlag + 1) & 1;
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::ComputeMm2(const SLIGradKLLossRunInfo &info)
{
    uint32_t dSize = constInfo.dSizeQueryIndex;
    uint32_t kTailLoopSize = N_WORKSPACE_SIZE;
    uint32_t kInnerLoopTimes = N_WORKSPACE_SIZE / N_SPLIT_SIZE;
    uint32_t tailLoopKSize = N_SPLIT_SIZE;

    mmParam.singleM = constInfo.gSizeQueryIndex;
    mmParam.singleN = N_SPLIT_SIZE;
    mmParam.singleK = constInfo.dSizeQueryIndex;
    mmParam.isLeftTranspose = false;
    mmParam.isRightTranspose = false;
    mmParam.isOutKFisrt = true;
    mmParam.isFixOut = true;
    mmParam.isL0CAccum = false;

    for (uint32_t kOuterIdx = 0; kOuterIdx < info.s2LoopTimes; kOuterIdx++) {
        if (kOuterIdx == 0) {
            // 搬运query到L1 64 * 128 * sizeof(fp16)
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[indexPingPongFlag & 1]);
            CopyInMm2AToL1(l1QIndexTensor[indexPingPongFlag & 1], info, constInfo.gSizeQueryIndex, dSize, 0);
        }

        if (kOuterIdx == info.s2LoopTimes - 1) {
            kTailLoopSize = info.kRealSize - kOuterIdx * N_WORKSPACE_SIZE;
            kInnerLoopTimes = AlignTo(kTailLoopSize, N_SPLIT_SIZE) / N_SPLIT_SIZE;
            tailLoopKSize = kTailLoopSize - (kInnerLoopTimes - 1) * N_SPLIT_SIZE;
        }        
        
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_SY_FLAG[kOuterIdx & 1]);
        // 此处同步id需要与v0对齐
        for (uint32_t kInnerIdx = 0; kInnerIdx < kInnerLoopTimes; kInnerIdx++) {
            if (kInnerIdx == kInnerLoopTimes - 1) {
                mmParam.singleN = tailLoopKSize;
            }
            int64_t gatherWorkspaceOffset = info.taskIdMod2 * topKSize * dSize + kOuterIdx * N_WORKSPACE_SIZE * dSize + kInnerIdx * N_SPLIT_SIZE * dSize;
            int64_t outWorkspaceOffset = kOuterIdx * N_WORKSPACE_SIZE + kInnerIdx * N_SPLIT_SIZE;
            // 搬运gather到L1 128 * 128 * sizeof(fp16)
            LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];
            LocalTensor<KV_T> kTensor = l1KVTensor[gatherPingPongFlag & 1];
            // 反向同步mte1_mte2
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            CopyInMm2BToL1(kTensor, gatherWorkspaceOffset, mmParam.singleN, dSize, dSize);
            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            GlobalTensor<MM_OUT_T> resGm = mm2ResGm[info.taskIdMod2][outWorkspaceOffset];
            MmadInner(l0cTensor, l1QIndexTensor[indexPingPongFlag & 1], kTensor, mmParam);
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]); 
            if (mmParam.isFixOut) {
                SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
                WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
                FixpipeParamsV220 fixpipeParams;
                fixpipeParams.nSize = mmParam.singleN;
                fixpipeParams.mSize = mmParam.singleM;
                fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
                fixpipeParams.dstStride = topKSize;
                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;
                fixpipeParams.reluEn = true;
                Fixpipe<MM_OUT_T, MM_OUT_T>(resGm, l0cTensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
            }
            SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);               
            l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
            gatherPingPongFlag = (gatherPingPongFlag + 1) & 1;
        }   
    }
    // 此处同步id需要与v1对齐
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_SY_FLAG[info.taskIdMod2]);
    mm2ResPingPongFlag = (mm2ResPingPongFlag + 1) & 1;
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::ComputeMm5(const SLIGradKLLossRunInfo &info)
{
    uint32_t kLoopSize = RELU_GRAD_SPLIT_SIZE;
    uint32_t kInnerLoopTimes = RELU_GRAD_SPLIT_SIZE / K_SPLIT_SIZE;
    uint32_t tailLoopKSize = K_SPLIT_SIZE;

    mmParam.singleM = K_SPLIT_SIZE;
    mmParam.singleN = constInfo.dSizeQueryIndex;
    mmParam.singleK = constInfo.gSizeQueryIndex;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.isOutKFisrt = true;
    mmParam.isFixOut = true;
    mmParam.isL0CAccum = false;

    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V1_TO_C2_DW_FLAG[info.taskIdMod2]);
    int64_t scatterOffset = 0;
    int64_t resOffset = constInfo.aicIdx * topKSize * constInfo.dSizeQueryIndex * 2;
    GlobalTensor<MM_OUT_T> resGm = mm5ResGm[resOffset + info.taskIdMod2 * topKSize * constInfo.dSizeQueryIndex];
    int64_t kOuterStride = info.kBaseSize * constInfo.dSizeQueryIndex;
    int64_t kInnerStride = K_SPLIT_SIZE * constInfo.dSizeQueryIndex;
    for (uint32_t kOuterIdx = 0; kOuterIdx < info.kLoopTimes; kOuterIdx++) {
        if (kOuterIdx == info.kLoopTimes - 1) {
            kLoopSize = info.kRealSize - kOuterIdx * RELU_GRAD_SPLIT_SIZE;
            kInnerLoopTimes = AlignTo(kLoopSize, K_SPLIT_SIZE) / K_SPLIT_SIZE;
            tailLoopKSize = kLoopSize - (kInnerLoopTimes - 1) * K_SPLIT_SIZE;
        }
        if (IS_RELUGRAD_REUSE) {
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(RELU_GRAD_EVENT);
        }
        int64_t reLuGradWorkspaceOffset = kOuterIdx * RELU_GRAD_SPLIT_SIZE;
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);
        CopyInMm5AToL1(l1ReLuGradTensor, info, constInfo.gSizeQueryIndex, kLoopSize, reLuGradWorkspaceOffset);
        SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[indexPingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[indexPingPongFlag & 1]);    
        for (uint32_t kInnerIdx = 0; kInnerIdx < kInnerLoopTimes; kInnerIdx++) {
            int64_t reluGradOffset = kInnerIdx * M_SPLIT_SIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            //搬运 result 128 * 128 * fp16
            LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];
            GlobalTensor<MM_OUT_T> resTmpGm = resGm[kOuterIdx * kOuterStride + kInnerIdx * kInnerStride];
            LocalTensor<Q_T> reLuTensor = l1ReLuGradTensor[reluGradOffset];
            if (kInnerIdx == kInnerLoopTimes - 1){
                mmParam.singleM = tailLoopKSize;
            }
            MmadInner(l0cTensor, reLuTensor, l1QIndexTensor[indexPingPongFlag & 1], mmParam);
            ScatterAdd(resTmpGm, l0cTensor, mmParam, info, scatterOffset);
            l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);
        mm5ResPingPongFlag = 1 - mm5ResPingPongFlag;
    }
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_INDEX_EVENT[indexPingPongFlag & 1]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM5_MM6_EVENT);
    indexPingPongFlag = 1 - indexPingPongFlag;
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[info.taskIdMod2]);
}

template <typename SLIT>
__aicore__ inline void SLITMatmulService<SLIT>::ComputeMm6(const SLIGradKLLossRunInfo &info)
{
    uint32_t kLoopSize = RELU_GRAD_SPLIT_SIZE;
    uint32_t kInnerLoopTimes = RELU_GRAD_SPLIT_SIZE / K_SPLIT_SIZE;
    uint32_t tailLoopKSize = K_SPLIT_SIZE;
    uint32_t dSize = constInfo.dSizeQueryIndex;
    mmParam.singleM = constInfo.gSizeQueryIndex;
    mmParam.singleN = constInfo.dSizeQueryIndex;
    mmParam.singleK = K_SPLIT_SIZE;
    mmParam.isLeftTranspose = false;
    mmParam.isRightTranspose = true;
    mmParam.isFixOut = false;

    LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM5_MM6_EVENT);    
    for (uint32_t kOuterIdx = 0; kOuterIdx < info.kLoopTimes; kOuterIdx++) {
        bool isLastKLoop = kOuterIdx == info.kLoopTimes - 1;
        if (isLastKLoop) {
            kLoopSize = info.kRealSize - kOuterIdx * RELU_GRAD_SPLIT_SIZE;
            kInnerLoopTimes = AlignTo(kLoopSize, K_SPLIT_SIZE) / K_SPLIT_SIZE;
            tailLoopKSize = kLoopSize - (kInnerLoopTimes - 1) * K_SPLIT_SIZE;
        }
        mmParam.isL0CAccum = (kInnerLoopTimes > 1);
        if (!IS_RELUGRAD_REUSE){ // 无法复用reluGrad
            int64_t reLuGradWorkspaceOffset = kOuterIdx * RELU_GRAD_SPLIT_SIZE;
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);            
            CopyInMm5AToL1(l1ReLuGradTensor, info, constInfo.gSizeQueryIndex, kLoopSize, reLuGradWorkspaceOffset);
            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[indexPingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[indexPingPongFlag & 1]);    
        }
        for (uint32_t kInnerIdx = 0; kInnerIdx < kInnerLoopTimes; kInnerIdx++) {
            int64_t gatherWorkspaceOffset = info.taskIdMod2 * topKSize * dSize + kOuterIdx * RELU_GRAD_SPLIT_SIZE * dSize + kInnerIdx * K_SPLIT_SIZE * dSize;
            int64_t reluGradOffset = kInnerIdx * K_SPLIT_SIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            mmParam.isOutKFisrt = (kOuterIdx == 0) && (kInnerIdx == 0);            
            // 搬运gather到L1 128 * 128 * sizeof(fp16)
            LocalTensor<KV_T> kTensor = l1KVTensor[gatherPingPongFlag & 1];
            if (isLastKLoop && (kInnerIdx == kInnerLoopTimes -1)) {
                mmParam.singleK = tailLoopKSize;
                mmParam.isFixOut = true;
            }
            // 反向同步mte1_mte2
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            CopyInMm2BToL1(kTensor, gatherWorkspaceOffset, mmParam.singleK, dSize, dSize);
            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);

            GlobalTensor<OUT_T> resGm = mm6ResGm[info.queryIndexTensorOffset];
            LocalTensor<Q_T> reLuTensor = l1ReLuGradTensor[reluGradOffset];
            MmadInner(l0cTensor, reLuTensor, kTensor, mmParam);
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG[gatherPingPongFlag & 1]);
            if (mmParam.isFixOut) {
                SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
                WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);     
                FixpipeParamsV220 fixpipeParams;
                fixpipeParams.nSize = mmParam.singleN;
                fixpipeParams.mSize = mmParam.singleM;
                fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
                fixpipeParams.dstStride = constInfo.dSizeQueryIndex;
                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;
                if constexpr(std::is_same<OUT_T, half>::value) {
                    fixpipeParams.quantPre = QuantMode_t::F322F16;
                } else {
                    fixpipeParams.quantPre = QuantMode_t::F322BF16;
                }
                Fixpipe<OUT_T, MM_OUT_T>(resGm, l0cTensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
            }
            SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);            
            gatherPingPongFlag = 1 - gatherPingPongFlag;
        }
        if (!IS_RELUGRAD_REUSE){
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MM_RELU_GRAD_EVENT);
        }
    }
    if (IS_RELUGRAD_REUSE) {
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(RELU_GRAD_EVENT);
    }
    mm6ResPingPongFlag = 1 - mm6ResPingPongFlag;
    l0cPingPongFlag = 1 - l0cPingPongFlag;
}

#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_H
