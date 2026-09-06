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
 * \file cube_op.h
 * \brief
 */
#pragma once
#include "kernel_operator.h"
#include "common_header.h"
#include "matmul.h"
using namespace AscendC;

namespace SASG_BASIC {

constexpr bool HAS_ROPE = false;

template <typename SASGT>
class CubeOp {
    using TILING_CLASS = typename SASGT::tiling_class;
    using T1 = typename SASGT::t1;
    static constexpr bool IS_BSND = SASGT::is_bsnd;
    static constexpr uint32_t MODE = SASGT::mode;

public:
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out_grad, GM_ADDR workspace, 
                                const TILING_CLASS *__restrict tilingData, TPipe *pipe);

    __aicore__ inline __attribute__((always_inline)) void cube12Process(const RunInfo &runInfo,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube345Process(const RunInfo &runInfo,
                                                                         const int32_t blkCntOffset, 
                                                                         const int32_t mmPingPongIdx);
private:
    __aicore__ inline __attribute__((always_inline)) void cube1Process(const int64_t queryGmOffset,
                                                                       const int64_t keyGmOffset,
                                                                       const int64_t outGmOffset, 
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx,
                                                                       const RunInfo &runInfo);

    __aicore__ inline __attribute__((always_inline)) void cube2Process(const int64_t dyGmOffset,
                                                                       const int64_t valueGmOffset,
                                                                       const int64_t outGmOffset, 
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx,
                                                                       const RunInfo &runInfo);

    __aicore__ inline __attribute__((always_inline)) void cube3Process(const int64_t keyGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx,
                                                                       const int64_t lastBlockSize,
                                                                       const bool isLastBasicBlock,
                                                                       const RunInfo &runInfo);

    __aicore__ inline __attribute__((always_inline)) void cube4Process(const int64_t dsGmOffset,
                                                                       const int64_t queryGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx,
                                                                       const RunInfo &runInfo);

    __aicore__ inline __attribute__((always_inline)) void cube5Process(const int64_t pGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx,
                                                                       const RunInfo &runInfo);

    __aicore__ inline void InitGMBuffer(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv,
                                        GM_ADDR attention_out_grad,
                                        GM_ADDR workspace);
    __aicore__ inline void InitLocalTensor();

    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        0,
        0,
        0,
        128,
        128,
        1,
        0
    };
    AscendC::Nd2NzParams cube1Nd2NzParams{
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        128,
        128,
        0,
        0,
        false,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsTrans {
        0,
        128,
        128,
        0,
        0,
        true,
        0
    };
    /* clang-format on */
    // Gm Addr
    GlobalTensor<T1> queryGm;
    GlobalTensor<T1> oriKvGm;
    GlobalTensor<T1> cmpKvGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<T1> dqGm;
    GlobalTensor<T1> dkGm;
    GlobalTensor<T1> dvGm;
    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<T1> selectedVWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * 2 * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * 2 * K * Dv
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t selectedVWorkspaceLen;
    int64_t mm4ResAddr;
    int64_t mm5ResAddr;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_SPLIT_SIZE = 128;     // k方向切分
    static constexpr uint32_t TOTAL_BLOCK_SIZE = 512;     // blocksize * blockcount

    uint32_t ping_pong_flag_l1_query_ = 0;
    uint32_t ping_pong_flag_l1_common_ = 0;
    uint32_t ping_pong_flag_l1_dy_ = 0;
    uint32_t ping_pong_flag_l0a_ = 0;
    uint32_t ping_pong_flag_l0b_ = 0;
    uint32_t ping_pong_flag_l0c_ = 0;
    // L1
    // 128 * 128 * sizeof(2) * 2(DB) = 64k
    LocalTensor<T1> l1_query_tensors[2];
    // 128 * 128 * sizeof(2) * 2(DB) = 64k
    LocalTensor<T1> l1_common_tensors[2];
    // 128 * 512 * sizeof(2) * 2(DB) = 256k
    LocalTensor<T1> l1_dy_tensors[2];
    // 128 * 8 * 64 * sizeof(fp16) = 128k
    LocalTensor<T1> l1_ds_tensor;

    // L0
    LocalTensor<T1> aL0TensorPingPong[2];
    LocalTensor<T1> bL0TensorPingPong[2];
    LocalTensor<float> cL0TensorPingPong[2];

    TBuf<AscendC::TPosition::CO1> L0CBuffer;
    TBuf<AscendC::TPosition::A1> L1Buffer;
    constexpr static int32_t SIZE_16 = 16;
    constexpr static int32_t BLOCK_SIZE = 16 * 16; // 一个分形的数据量
    constexpr static uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;
    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t SINGLE_BLOCK_CNT = 8; // 单次最多处理的selcted_block_cnt
    uint32_t pingPongIdx = 0;
    uint64_t globalBlockOffset = 0;

    int64_t dimB;
    int64_t dimN2;
    int64_t dimG;
    int64_t dimGAlign;
    int64_t dimDqk;
    int64_t dimDTotal;
    int64_t dimDv;
    int64_t dimRope;
    int64_t selectedBlockCount;
    int64_t selectedBlockSize;
    int64_t selectedS2;
    int64_t oriWinLen;
    int32_t selectedCntOffset;
    int32_t processBS1ByCore;
    uint32_t usedCoreNum;
    uint32_t eventIdPing = 4;
    uint32_t eventIdPong = 5;
    uint32_t cBlockIdx;
    uint32_t singleN;
    int64_t s1BasicSize;
    int64_t dOriKvSize;                                                     
};

template <typename SASGT>
__aicore__ inline void CubeOp<SASGT>::Init(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out_grad, GM_ADDR workspace,
                                           const TILING_CLASS *__restrict tilingData, TPipe *pipe)
{
    dimB = tilingData->opInfo.B;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimDqk;
    dimRope = 0;
    dimDTotal = HAS_ROPE ? dimDqk + dimRope : dimDqk;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    selectedS2 = selectedBlockCount * selectedBlockSize;
    dimGAlign = AlignTo<int64_t>(dimG, SIZE_16);
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    // ori UNION span for a batch of s1BasicSize(B_q) queries (workspace stride); =+1 at B_q=1
    oriWinLen = tilingData->opInfo.oriWinLeft + tilingData->opInfo.oriWinRight +
                tilingData->splitCoreParams.s1BasicSize;
    singleN = tilingData->splitCoreParams.singleN;
    s1BasicSize = tilingData->splitCoreParams.s1BasicSize;

    if constexpr (IS_BSND == false) {
        dOriKvSize = tilingData->opInfo.S2 * dimN2 * dimDqk;
    } else {
        dOriKvSize = dimB * tilingData->opInfo.S2 * dimN2 * dimDqk;
    }

    usedCoreNum = tilingData->opInfo.usedCoreNum;
    pipe->InitBuffer(L0CBuffer, HardwareInfo<ArchType::ASCEND_V220>::l0CSize);
    pipe->InitBuffer(L1Buffer, HardwareInfo<ArchType::ASCEND_V220>::l1Size);
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    InitGMBuffer(query, ori_kv, cmp_kv, attention_out_grad, workspace);
    InitLocalTensor();
}

template <typename SASGT>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<SASGT>::cube12Process(const RunInfo &runInfo,
                             const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    selectedCntOffset = runInfo.actualSelCntOffset;
    cube1Process(runInfo.queryGmOffset, runInfo.selectedKGmOffset, runInfo.mm12GmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS[mmPingPongIdx]);
    cube2Process(runInfo.dyGmOffset, runInfo.selectedVGmOffset, runInfo.mm12GmOffset, blkCntOffset, mmPingPongIdx, runInfo);
}

template <typename SASGT>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<SASGT>::cube345Process(const RunInfo &runInfo,
                              const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    selectedCntOffset = runInfo.actualSelCntOffset;
    cube5Process(runInfo.mm345GmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS[mmPingPongIdx]);

    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
    cube4Process(runInfo.mm345GmOffset, runInfo.queryGmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    cube3Process(runInfo.selectedKGmOffset, runInfo.dqOutGmOffset, blkCntOffset, mmPingPongIdx, runInfo.lastBlockSize, runInfo.isLastBasicBlock, runInfo);
    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
}

template <typename SFAGT>
__aicore__ inline void CubeOp<SFAGT>::InitGMBuffer(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out_grad, GM_ADDR workspace)
{
    // 必选输入初始化
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);

    if constexpr (MODE == SASG_SWA_MODE) {
        oriKvGm.SetGlobalBuffer((__gm__ T1 *)ori_kv);
    } else if (MODE == SASG_CFA_MODE) {
        oriKvGm.SetGlobalBuffer((__gm__ T1 *)ori_kv);
        cmpKvGm.SetGlobalBuffer((__gm__ T1 *)cmp_kv);
    }

    int64_t usedWorkspaceLen = 0;
    cBlockIdx = GetBlockIdx();
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    /*
     * mm1 与 p 复用workspace
     */
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    /*
     * mm2 与 ds 复用workspace
     */
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;

    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen;

    if constexpr (MODE == SASG_SCFA_MODE) {
        // scatter add
        mm4ResAddr = usedWorkspaceLen / sizeof(float);
        mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * dimDTotal * 2;
        usedWorkspaceLen += MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * (dimDTotal + dimDTotal) * 2 * sizeof(float);
    } else {
        mm5ResAddr = dkAddr;
        mm4ResAddr = usedWorkspaceLen / sizeof(float);
    }

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

template <typename SFAGT>
__aicore__ inline void CubeOp<SFAGT>::InitLocalTensor()
{
    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;
    /*
     * Init L1 Tensor
     */
    int32_t l1Offset = 0;
    // query
    l1_query_tensors[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 128 * 128 * sizeof(T1);
    l1_query_tensors[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 128 * 128 * sizeof(T1);
    // common
    l1_common_tensors[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 128 * 128 * sizeof(T1);
    l1_common_tensors[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 128 * 128 * sizeof(T1);
    // key
    l1_dy_tensors[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 512 * 128 * sizeof(T1);
    l1_dy_tensors[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 512 * 128 * sizeof(T1);
    // ds
    l1_ds_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);

    /*
     * Init L0 Tensor
     */
    aL0TensorPingPong[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(0);
    aL0TensorPingPong[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(128 * 128 * sizeof(T1));

    bL0TensorPingPong[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, T1>(0);
    bL0TensorPingPong[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, T1>(128 * 128 * sizeof(T1));

    cL0TensorPingPong[0] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    cL0TensorPingPong[1] = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(128 * 128 * sizeof(float));
}

#include "./cube_modules/cube1.h"
#include "./cube_modules/cube2.h"
#include "./cube_modules/cube3.h"
#include "./cube_modules/cube4.h"
#include "./cube_modules/cube5.h"
} // namespace SASG_BASIC