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
 * \file vec_op.h
 * \brief
 */

#pragma once
#include "kernel_operator.h"
#include "common_header.h"

namespace SASG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

template <typename SASGT>
class VecOp {
public:
    using TILING_CLASS = typename SASGT::tiling_class;
    using T1 = typename SASGT::t1;
    static constexpr bool IS_BSND = SASGT::is_bsnd;
    static constexpr uint32_t MODE = SASGT::mode;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks, GM_ADDR dsinks,
                                GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv,
                                GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const RunInfo &runInfo);
    __aicore__ inline void GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const RunInfo &runInfo);
protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData);
    __aicore__ inline void InitGMBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR lse,
                                        GM_ADDR topk_indices, GM_ADDR sinks, GM_ADDR dsinks, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t lseGmOffset, const int32_t processM);
    __aicore__ inline void CalAttenMsk(const int32_t processM, const RunInfo &runInfo, const float maskVal);
    __aicore__ inline void CalScfaOriBandMask(const int32_t processM, const float maskVal);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const RunInfo &runInfo, const int32_t mOffset, const int64_t s1Index);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void CalDsinks(const int32_t loopIdx, const int32_t processM, const RunInfo &runInfo, const int32_t mOffset);

protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    uint32_t subBlockIdx;
    StaticParams params;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> oriKvGm;
    GlobalTensor<T1> cmpKvGm;
    GlobalTensor<float> sinksGm;
    GlobalTensor<float> dSinksGm;

    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * 2 * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * 2 * K * Dv

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseTmp;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<float> dSinkSumTensor;
    LocalTensor<float> sinkTmp;
    LocalTensor<float> sinkTensor;
    LocalTensor<float> sinkTmpBuf;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;
    LocalTensor<T1> gatherTensorPing;
    LocalTensor<T1> gatherTensorPong;
    LocalTensor<float> scatterAddTensorK;
    LocalTensor<float> scatterAddTensorV;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;

    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimS3;
    int64_t dimG;
    int64_t dimD;
    int64_t dimD2;
    int64_t dimDqk; // Dqk = D
    int64_t dimDv;  // Dv = D2
    int64_t dimDAlign;
    int64_t dimD2Align;
    int64_t dimTkv{0};
    int64_t dimTq{0};
    int64_t selectedS2{0};
    int64_t curS1;
    int64_t curS2;
    int64_t lastBlock;
    int64_t s1BasicSize;
    int64_t actualSelS2;
    int64_t actualSelS2Align;
    int64_t curActualLeft;
    int64_t curActualRight;
    // SCFA batched ori band-mask state (set in CalSoftmax, reused by CalSoftmaxGrad).
    // For B_q>1 each query in the batch must attend only its own ori window within the
    // loaded union span; cmp columns (shared representative top-k) stay unmasked.
    int32_t scfaBandL = 0;       // mask ori cols [0, scfaBandL)
    int32_t scfaBandR = 0;       // mask ori cols [scfaBandR, scfaOriEnd)
    int32_t scfaOriEnd = 0;      // ori columns count in this tile (cmp starts here); 0 => no SCFA mask
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int32_t maxSelCnt;
    int64_t dimRope;
    int64_t cmpRatio;
    int64_t dOriKvSize;
    int64_t oriWinLeft;
    int64_t oriWinRight;
    int64_t oriWinLen;
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t selectedVWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte2Pong;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
    event_t mte2WaitMte3Pong;
    event_t mte3WaitMte2Pong;
    event_t mte2WaitV;

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams outParamK;
    uint32_t mergePingPong{0};
    int32_t pingPongIdx = 0;
};

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::Init(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks, GM_ADDR dsinks,
                                          GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv, 
                                          GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData);
    InitGMBuffer(ori_kv, cmp_kv, attention_out, attention_out_grad, lse, topk_indices, sinks, dsinks, workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::InitParams(const TILING_CLASS *__restrict ordTilingData)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimS3 = tilingData->opInfo.S3;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = dimD;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimD;
    dimRope = 0;
    dimDAlign = (dimD + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    s1BasicSize = tilingData->splitCoreParams.s1BasicSize;

    cmpRatio = tilingData->opInfo.cmpRatio;
    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = tilingData->splitCoreParams.singleN;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= tilingData->splitCoreParams.singleN) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    oriWinLeft = tilingData->opInfo.oriWinLeft;
    oriWinRight = tilingData->opInfo.oriWinRight;
    // ori UNION span for a batch of s1BasicSize(B_q) consecutive queries (workspace stride).
    // B_q consecutive sliding windows span winLeft+winRight+B_q blocks. (=winLeft+winRight+1 at B_q=1.)
    oriWinLen = oriWinLeft + oriWinRight + tilingData->splitCoreParams.s1BasicSize;

    if constexpr (IS_BSND == false) {
        dimTq = tilingData->opInfo.S1;
        dimTkv = (tilingData->opInfo.S2 + tilingData->opInfo.S3);
        dOriKvSize = tilingData->opInfo.S2 * dimN2 * dimDqk;
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * (dimS2 + dimS3);
        dOriKvSize = dimB * dimS2 * dimN2 * dimDqk;
    }

    intriParamsKey.blockLen = selectedBlockSize * dimDqk * sizeof(T1);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    outParamK.blockCount = 2 * selectedBlockSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = dimRope * sizeof(T1);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::InitGMBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                                  GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks, GM_ADDR dsinks, GM_ADDR workspace)
{
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    lseGm.SetGlobalBuffer((__gm__ float *)lse);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    oriKvGm.SetGlobalBuffer((__gm__ T1 *)ori_kv);
    cmpKvGm.SetGlobalBuffer((__gm__ T1 *)cmp_kv);
    dSinksGm.SetGlobalBuffer((__gm__ float *)dsinks);
    sinksGm.SetGlobalBuffer((__gm__ float *)sinks);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    // mm1 与 p 复用workspace
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // mm2 与 ds 复用workspace
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen;

    // scatter add
    int64_t mm4ResAddr = usedWorkspaceLen / sizeof(float);
    int64_t mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * dimDAlign * 2;
    usedWorkspaceLen += MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * (dimDAlign + dimDAlign) * 2 * sizeof(float);

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);

    mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
    mm5ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResAddr);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    // topk
    int32_t topkNumber = AlignTo<int32_t>(selectedCountOffset, BLOCK);
    topkIndicesTensor = vecQue.GetWithOffset<int32_t>(topkNumber, ubOffset);
    ubOffset += topkNumber * sizeof(int32_t);
    uint32_t topkUbOffset = ubOffset;

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    lseTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);

    int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
    lseTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    rowsumUbOffset = ubOffset;

    // rowsum cal
    attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1); //
    attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);
    attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1);
    attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);

    // sftGard
    int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
    pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sinkTmpBuf = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    sinkTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    sinkTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    dSinkSumTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
    scatterAddTensorK = vecQue.GetWithOffset<float>(16 * dimDAlign * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimDAlign * 2 * sizeof(float);
    scatterAddTensorV = vecQue.GetWithOffset<float>(16 * dimD2Align * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimD2Align * 2 * sizeof(float);

    maxSelCnt = selectedBlockSize >= 64 ? 1 : 2;
    gatherTensorPing = vecQue.GetWithOffset<T1>(maxSelCnt * selectedBlockSize * dimDqk, topkUbOffset);
    topkUbOffset += maxSelCnt * selectedBlockSize * dimDqk * sizeof(T1);
    gatherTensorPong = vecQue.GetWithOffset<T1>(maxSelCnt * selectedBlockSize * dimDqk, topkUbOffset);
    topkUbOffset += maxSelCnt * selectedBlockSize * dimDqk * sizeof(T1);

    uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(params.sftBaseM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    lseTensor.SetSize(params.sftBaseM * 8);
    sinkTensor.SetSize(params.sftBaseM * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), 8};
    lseTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    sinkTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    mte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::AtomicClean()
{
    int64_t dqSize, dkSize;
    dqSize = dimTq * dimN2 * dimG * dimDAlign;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dqSize = (dqSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    DumpGmZero(dqWorkspaceGm, dqSize);
    if constexpr (MODE != SASG_SCFA_MODE) {
        dkSize *= 2;
    }
    DumpGmZero(dkWorkspaceGm, dkSize);
    DumpGmZero(dSinksGm, dimN2 * dimG);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        InitOutput<float>(gm[vecBlockIdx * perSize], initSize, 0);
    }
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                            int32_t baseN)
{
    constexpr static int32_t subMsk = 64;
    int8_t s2Repeat = baseN / subMsk;
    int8_t s2RepeatTail = baseN % subMsk;

    for (int32_t i = 0; i < baseM; i++) {
        Sub(dstTensor[i * baseN], dstTensor[i * baseN], srcTensor[i * BLOCK_FP32], subMsk, s2Repeat,
            {1, 1, 0, 8, 8, 0});
    }

    if (s2RepeatTail) {
        for (int32_t i = 0; i < baseM; i++) {
            Sub(dstTensor[s2Repeat * subMsk + i * baseN], dstTensor[s2Repeat * subMsk + i * baseN],
                srcTensor[i * BLOCK_FP32], s2RepeatTail, 1, {1, 1, 0, 8, 8, 0});
        }
    }
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t lseGmOffset, const int32_t processM)
{
    uint32_t dataSize = processM * dimDv;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processM * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
    DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    PIPE_BARRIER(PIPE_V);
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);

    // [N2, T1, G] or [B, N2, S1, G] --> [sftBaseM, 1] --> [sftBaseM, 8]
    DataCopyPad(lseTmp, lseGm[lseGmOffset], copyParams2, padParams2);
    SET_FLAG(MTE2, V, vWaitMte2);
    
    WAIT_FLAG(MTE2, V, vWaitMte2);
    Brcb(lseTensor, lseTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalAttenMsk(const int32_t processM, const RunInfo &runInfo, const float maskVal)
{
    if (curActualLeft > 0) {
        // mask: [0, curActualLeft]
        int32_t leftLoop = CeilDiv(curActualLeft, 64);
        for (int32_t loop = 0; loop < leftLoop; loop++) {
            uint64_t mask = (loop == leftLoop - 1) ? curActualLeft - (loop * 64) : 64;
            Duplicate(pTensor[loop * 64], maskVal, mask, processM, 1, (actualSelS2Align * sizeof(float)) / BLOCK);
        }
    }

    if (curActualRight < actualSelS2Align) {
        // mask: [curActualRight, actualSelS2Align]
        int32_t rightMaskLen = actualSelS2Align - curActualRight;
        int32_t rightLoop = CeilDiv(rightMaskLen, 64);
        int32_t rightStartOffset = curActualRight / 8 * 8;
        int32_t attenMskRsv = curActualRight - rightStartOffset;
        int32_t attenMskEnd = Min(actualSelS2Align - rightStartOffset, 64);
        int32_t remain = rightMaskLen - (attenMskEnd - attenMskRsv + 1) % 64 ? 
                        rightMaskLen - (attenMskEnd - attenMskRsv + 1) % 64 :
                        64;
        for (int32_t loop = 0; loop < rightLoop; loop++) {
            if (loop == 0) {
                uint64_t maskEnd = (attenMskEnd == 64) ? UINT64_MAX : (1ULL << attenMskEnd) - 1;
                uint64_t maskRsv = (attenMskRsv == 64) ? UINT64_MAX : (1ULL << attenMskRsv) - 1;
                uint64_t mask[1] = { maskEnd & ~maskRsv };
                Duplicate(pTensor[loop * 64 + rightStartOffset], maskVal, mask, processM, 1, (actualSelS2Align * sizeof(float)) / BLOCK);
            } else {
                uint64_t maskLen = (loop == rightLoop - 1) ? remain : 64;
                Duplicate(pTensor[loop * 64 + rightStartOffset], maskVal, maskLen, processM, 1, (actualSelS2Align * sizeof(float)) / BLOCK);
            }
        }
    }
}

// SCFA batched ori band mask: for ONE query position (all processM rows share its window),
// mask ori cols [0, scfaBandL) and [scfaBandR, scfaOriEnd); cmp cols (>= scfaOriEnd) untouched.
// Masked lengths are <= B_q-1 (small) since the union span = winLen + (B_q-1).
template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalScfaOriBandMask(const int32_t processM, const float maskVal)
{
    if (scfaOriEnd <= 0) {
        return;
    }
    const int32_t repStride = (actualSelS2Align * sizeof(float)) / BLOCK;  // per-row stride (32B blocks)
    int32_t l = scfaBandL < 0 ? 0 : (scfaBandL > scfaOriEnd ? scfaOriEnd : scfaBandL);
    int32_t r = scfaBandR < 0 ? 0 : (scfaBandR > scfaOriEnd ? scfaOriEnd : scfaBandR);
    // left: mask ori cols [0, l). base 0 is 32B-aligned; l <= B_q-1 < 64.
    if (l > 0) {
        Duplicate(pTensor, maskVal, static_cast<uint64_t>(l), processM, 1, repStride);
    }
    // right: mask ori cols [r, scfaOriEnd). VEC needs a 32B(8-fp32)-aligned base, so start at
    // alignBase = r/8*8 and skip the first (r-alignBase) lanes via a bitmask (like CalAttenMsk).
    if (r < scfaOriEnd) {
        int32_t alignBase = (r / 8) * 8;
        int32_t rsv = r - alignBase;                                  // lanes to keep at the aligned start
        int32_t firstEnd = Min(scfaOriEnd - alignBase, 64);           // up to 64 lanes from alignBase
        uint64_t mEnd = (firstEnd >= 64) ? UINT64_MAX : ((1ULL << firstEnd) - 1);
        uint64_t mRsv = (rsv >= 64) ? UINT64_MAX : ((1ULL << rsv) - 1);
        uint64_t mask0[1] = { mEnd & ~mRsv };
        Duplicate(pTensor[alignBase], maskVal, mask0, processM, 1, repStride);
        for (int32_t off = alignBase + 64; off < scfaOriEnd; off += 64) {  // off is 64-aligned
            uint64_t cnt = static_cast<uint64_t>(Min(scfaOriEnd - off, 64));
            Duplicate(pTensor[off], maskVal, cnt, processM, 1, repStride);
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const RunInfo &runInfo, 
                                                const int32_t mOffset, const int64_t s1Index)
{
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    actualSelS2 = runInfo.actualSelCntOffset * selectedBlockSize;
    if constexpr (MODE != SASG_SCFA_MODE) {
        if (MODE == SASG_SWA_MODE || runInfo.isOri) {
            int64_t curOriWinStart = Max(s1Index - oriWinLeft + runInfo.oriWinDiagOffset, 0);
            int64_t curOriWinEnd = Min(s1Index + oriWinRight + 1 + runInfo.oriWinDiagOffset, runInfo.curS2);
            int64_t blkStart = runInfo.oriWinStart + runInfo.blkCntOffset;
            curActualRight = Min(runInfo.oriWinStart + runInfo.blkCntOffset + runInfo.actualSelCntOffset, curOriWinEnd) - blkStart;
            curActualLeft = Max(runInfo.oriWinStart + runInfo.blkCntOffset, curOriWinStart) - blkStart;
            curActualRight = Max(curActualRight, curActualLeft);
        } else {
            curActualLeft = 0;
            curActualRight = Max((runInfo.oriWinDiagOffset + s1Index + 1) / cmpRatio - runInfo.blkCntOffset, 0);
            curActualRight = Min(curActualRight, actualSelS2);
        }
    }
    actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    if constexpr (MODE == SASG_SCFA_MODE) {
        // batched SCFA: ori occupies cols [0, oriS2End) of the selected sequence
        // (oriS2End = union span <= singleN, so only the first tile holds ori). Each query
        // (all processM rows share its window) attends only [winStart, winEnd); cmp cols
        // (shared representative top-k) stay unmasked. At B_q=1 the band == full ori -> no-op.
        int64_t oriS2End = runInfo.oriWinEnd - runInfo.oriWinStart;
        int64_t oriInTile = Min(oriS2End - static_cast<int64_t>(runInfo.blkCntOffset),
                                static_cast<int64_t>(actualSelS2));
        if (oriInTile > 0) {
            int64_t winStartAbs = Max(s1Index - oriWinLeft, static_cast<int64_t>(0));   // BSND: diag=0
            int64_t winEndAbs   = Min(s1Index + oriWinRight + 1, runInfo.curS1);
            int64_t base = runInfo.oriWinStart + runInfo.blkCntOffset;
            scfaBandL  = static_cast<int32_t>(Max(winStartAbs, base) - base);
            scfaBandR  = static_cast<int32_t>(Min(winEndAbs, base + oriInTile) - base);
            scfaOriEnd = static_cast<int32_t>(oriInTile);
        } else {
            scfaOriEnd = 0;  // pure cmp tile -> no SCFA mask
        }
    }

    WAIT_FLAG(V, MTE2, mte2WaitV); // wait sinkTmp
    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(sinkTmp, sinksGm[mOffset], {1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Brcb(sinkTensor, sinkTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (MODE != SASG_SCFA_MODE) {
        CalAttenMsk(processM, runInfo, *((float *)&ATTEN_MASK_SCALE));
        PIPE_BARRIER(PIPE_V);
    } else {
        CalScfaOriBandMask(processM, *((float *)&ATTEN_MASK_SCALE));  // batched ori band (no-op at B_q=1)
        PIPE_BARRIER(PIPE_V);
    }

    // pTensor: [processM, actualSelS2Align]
    // lseTensor: [sftBaseM, 8]
    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2Align - (i * 64) : 64;
        Sub(pTensor[i * 64], pTensor[i * 64], lseTensor, mask, processM, {1, 1, 0, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8)), 1});
        PIPE_BARRIER(PIPE_V);

        Exp(pTensor[i * 64], pTensor[i * 64], mask, processM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
        PIPE_BARRIER(PIPE_V);

        Cast(sftOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    Sub(sinkTensor, sinkTensor, lseTensor, processM * 8);
    PIPE_BARRIER(PIPE_V);
    Exp(sinkTensor, sinkTensor, processM * 8);
    PIPE_BARRIER(PIPE_V);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Mul(dSinkSumTensor, dPTensor, pTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    CalSub(dPTensor, rowSumOutTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (MODE != SASG_SCFA_MODE) {
        CalAttenMsk(processM, runInfo, 0.0);
        PIPE_BARRIER(PIPE_V);
    } else {
        // reuse band state computed by CalSoftmax for this same position (no-op at B_q=1)
        CalScfaOriBandMask(processM, 0.0);
        PIPE_BARRIER(PIPE_V);
    }

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2Align - (i * 64) : 64;
        Cast(sftgOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, params.sftBaseM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::CalDsinks(const int32_t loopIdx, const int32_t processM, const RunInfo &runInfo, const int32_t mOffset)
{
    if (MODE != SASG_SCFA_MODE && curActualRight - curActualLeft == 0) {
        return;
    }
    for (int32_t i = 0; i < processM; ++i) {
        uint64_t tailS2 = actualSelS2Align % 64 ? actualSelS2Align % 64 : 64;
        uint64_t totalLoop = CeilDiv(actualSelS2Align, 64);
        for (int32_t loop = 0; loop < totalLoop; ++loop) {
            uint64_t mask = loop == totalLoop - 1 ? tailS2 : 64;
            Mul(dSinkSumTensor[i * actualSelS2Align + loop * 64], dSinkSumTensor[i * actualSelS2Align + loop * 64], sinkTensor[i * 8], mask, 1, {1, 1, 0, 0, 0, 0});
        }
        PIPE_BARRIER(PIPE_V);

        // ReduceSum
        if constexpr (MODE != SASG_SCFA_MODE) {
            ReduceSum(dSinkSumTensor[i * actualSelS2Align], dSinkSumTensor[i * actualSelS2Align], sinkTmpBuf, curActualRight);
        } else {
            ReduceSum(dSinkSumTensor[i * actualSelS2Align], dSinkSumTensor[i * actualSelS2Align], sinkTmpBuf, actualSelS2);
        }
        PIPE_BARRIER(PIPE_V);

        float dSink = -dSinkSumTensor.GetValue(i * actualSelS2Align);
        sinkTensor.SetValue(i * 8, dSink);
    }

    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    // DataCopy
    SetAtomicAdd<float>();
    DataCopyPad(dSinksGm[mOffset], sinkTensor, {static_cast<uint16_t>(processM), sizeof(float), 0, 0}); // MTE3 read
    SetAtomicNone();
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo)
{
    uint64_t kSelectedWsAddr = runInfo.selectedKGmOffset;
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + runInfo.s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + runInfo.blkCntOffset;

    outParamK.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;

    uint64_t totalD = dimDqk + dimRope;
    event_t mte2WaitMte3EventId;
    event_t mte3WaitMte2EventId;

    SET_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = subBlockIdx == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = subBlockIdx == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outWsOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * totalD;
    uint64_t orgOutWsOffset = outWsOffset;
    uint32_t i;

    bool isLast = runInfo.isLastBasicBlock && subBlockIdx == 1;
    int64_t oriS2End = runInfo.oriWinEnd - runInfo.oriWinStart;

    for (i = curBlk; i < curBlk + curActualSelCntOffset / maxSelCnt * maxSelCnt; i += maxSelCnt) {
        bool isFirstCmp = (i + runInfo.blkCntOffset) >= oriS2End;
        bool isSecCmp = (i + runInfo.blkCntOffset + 1) >= oriS2End;

        int64_t keyOffset1 = isFirstCmp ? topkIndicesGm.GetValue(gmOffset + i - oriS2End) * selectedBlockSize : i + runInfo.blkCntOffset + runInfo.oriWinStart;
        int64_t keyOffset2 = isSecCmp ? topkIndicesGm.GetValue(gmOffset + i + 1 - oriS2End) * selectedBlockSize : i + runInfo.blkCntOffset + runInfo.oriWinStart + 1;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsKey.blockLen = selectedBlockSize * dimDqk * sizeof(T1);

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * dimN2 * dimDqk * sizeof(T1);
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        bool isActualLast = isLast && i >= runInfo.actualSelCntOffset - 2;
        if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast || isFirstCmp != isSecCmp) {
            intriParamsKey.blockCount = 1;
            if (isFirstCmp) {
                DataCopyPad(gatherTensor, cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
            } else {
                DataCopyPad(gatherTensor, oriKvGm[runInfo.oriKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
            }
            if (selectedBlockSize < 64) {
                intriParamsKey.blockLen = isActualLast ? 
                                          runInfo.lastBlockSize * dimDqk * sizeof(T1) : 
                                          selectedBlockSize * dimDqk * sizeof(T1);
                if (isSecCmp) {
                    DataCopyPad(gatherTensor[selectedBlockSize * dimDqk], cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset2 * dimN2 * dimDqk], intriParamsKey, padParams);
                } else {
                    DataCopyPad(gatherTensor[selectedBlockSize * dimDqk], oriKvGm[runInfo.oriKeyGmOffset + keyOffset2 * dimN2 * dimDqk], intriParamsKey, padParams);
                }
            }
        } else {
            if (isFirstCmp && isSecCmp) {
                DataCopyPad(gatherTensor, cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
            } else {
                DataCopyPad(gatherTensor, oriKvGm[runInfo.oriKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
            }
        }
        
        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        // CopyOut
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        outWsOffset += maxSelCnt * selectedBlockSize * totalD;
        mergePingPong = 1 - mergePingPong;
    }
    if (i < curActualSelCntEnd) {
        bool isCmp = (i + runInfo.blkCntOffset) >= oriS2End;
        int64_t keyOffset1 = isCmp ? topkIndicesGm.GetValue(gmOffset + i - oriS2End) * selectedBlockSize : i + runInfo.blkCntOffset + runInfo.oriWinStart;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        // CopyIn
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        intriParamsKey.blockCount = 1;
        if (i == runInfo.actualSelCntOffset - 1 && isLast) {
            intriParamsKey.blockLen = runInfo.lastBlockSize * dimDqk * sizeof(T1);
        }
        if (isCmp) {
            DataCopyPad(gatherTensor, cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
        } else {
            DataCopyPad(gatherTensor, oriKvGm[runInfo.oriKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
        }

        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        outParamK.blockCount = 1 * selectedBlockSize;
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        mergePingPong = 1 - mergePingPong;
    }
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::Process(const RunInfo &runInfo)
{
    int32_t gloop = (dimG + params.sftBaseM - 1) / params.sftBaseM;
    int32_t processM = params.sftBaseM;
    int32_t tailM = dimG % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;
    int64_t dyGmOffset = runInfo.dyGmOffset;
    int64_t lseGmOffset = runInfo.lseGmOffset;
    int64_t mm12Addr = runInfo.mm12GmOffset;
    int64_t mm345Addr = runInfo.mm345GmOffset;
    int32_t mOffset = 0;
    int64_t s1Index = runInfo.s1Index;

    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    SET_FLAG(V, MTE2, mte2WaitV);
    for (int32_t s1Basic = 0; s1Basic < runInfo.curS1Basic; s1Basic++) {
        processM = params.sftBaseM;
        mOffset = 0;
        dataSize = processM * params.sftBaseN;
        for (int32_t i = 0; i < gloop; i++) {
            if (i == gloop - 1 && tailM != 0) {
                processM = tailM;
                dataSize = tailM * params.sftBaseN;
            }
            CalRowsumAndSftCopyIn(dyGmOffset, lseGmOffset, processM);
            dyGmOffset += (processM * dimDv);
            lseGmOffset += processM;

            CalSoftmax(i, processM, mm12Addr, mm345Addr, runInfo, mOffset, s1Index);
            CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr, runInfo);
            CalDsinks(i, processM, runInfo, mOffset);
            SET_FLAG(V, MTE2, mte2WaitV);
            SET_FLAG(MTE3, MTE2, mte2WaitMte3);
            mm12Addr += dataSize;
            mm345Addr += dataSize;
            mOffset += processM;
        }
        s1Index++;
    }
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(V, MTE2, mte2WaitV);
}

template <typename SASGT>
__aicore__ inline void VecOp<SASGT>::ScatterAdd(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;
    int64_t UB_ROW_SIZE = 16;

    GlobalTensor<float> dOriKvOutGm = dkWorkspaceGm[runInfo.oriKeyGmOffset];
    GlobalTensor<float> dCmpKvOutGm = dkWorkspaceGm[runInfo.cmpKeyGmOffset + dOriKvSize];
    int64_t s2RealSize = runInfo.actualSelectedBlockCount;
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = subBlockIdx == 0 ? firstCoreKSize : s2RealSize - firstCoreKSize;
    int64_t curS2Index = subBlockIdx == 0 ? 0 : firstCoreKSize;
    int64_t oriS2End = runInfo.oriWinEnd - runInfo.oriWinStart;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();

    int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
    int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

    int64_t currentDkSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * dimDAlign + cubeBlockIdx * (selectedBlockCount + oriWinLen) * dimDAlign + subBlockIdx * firstCoreKSize * dimDAlign;
    int64_t currentDvSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * dimD2Align + cubeBlockIdx * (selectedBlockCount + oriWinLen) * dimD2Align + subBlockIdx * firstCoreKSize * dimD2Align;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[runInfo.indicesGmOffset];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * (UB_ROW_SIZE * dimDAlign)];
        dvInUb = scatterAddTensorV[pingPongIdx * (UB_ROW_SIZE * dimD2Align)];
        DataCopy(dkInUb, dkSrcGm[loop * UB_ROW_SIZE * dimDAlign], UB_ROW_SIZE * dimDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->opInfo.scaleValue, UB_ROW_SIZE * dimDAlign);
        DataCopy(dvInUb, dvSrcGm[loop * UB_ROW_SIZE * dimD2Align], UB_ROW_SIZE * dimD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
            bool isCmp = curS2Index >= oriS2End;
            int32_t s2Idx = isCmp ? indicesGm.GetValue(curS2Index - oriS2End) : runInfo.oriWinStart + curS2Index;
            if (s2Idx >= 0) {
                Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
                SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                if (isCmp) {
                    DataCopy(dCmpKvOutGm[s2Idx * dimDAlign], dkInUb[row * dimDAlign], dimDAlign);
                } else {
                    DataCopy(dOriKvOutGm[s2Idx * dimDAlign], dkInUb[row * dimDAlign], dimDAlign);
                }
            }
            curS2Index++;
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * (UB_ROW_SIZE * dimDAlign)];
    dvInUb = scatterAddTensorV[pingPongIdx * (UB_ROW_SIZE * dimD2Align)];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * UB_ROW_SIZE * dimDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    Muls(dkInUb, dkInUb, (float)tilingData->opInfo.scaleValue, tailRows * dimDAlign);
    DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * UB_ROW_SIZE * dimD2Align], tailRows * dimD2Align);
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    PipeBarrier<PIPE_V>();
    for (int64_t row = 0; row < tailRows; row++) {
        bool isCmp = curS2Index >= oriS2End;
        int32_t s2Idx = isCmp ? indicesGm.GetValue(curS2Index - oriS2End) : runInfo.oriWinStart + curS2Index;
        if (s2Idx >= 0) {
            Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
            SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            if (isCmp) {
                DataCopy(dCmpKvOutGm[s2Idx * dimDAlign], dkInUb[row * dimDAlign], dimDAlign);
            } else {
                DataCopy(dOriKvOutGm[s2Idx * dimDAlign], dkInUb[row * dimDAlign], dimDAlign);
            }
        }
        curS2Index++;
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;
    SetAtomicNone();

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
}

} // namespace SASG_BASIC
