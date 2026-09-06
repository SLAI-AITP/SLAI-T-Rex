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
 * \file sfag_basic.h
 * \brief
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "basic_modules/cube_op.h"
#include "basic_modules/vec_op.h"
#include "basic_modules/common_header.h"
#include "sfag_post.h"

namespace SASG_BASIC {

template <typename SASGT>
class SelectedAttentionGradBasic {
    using TILING_CLASS = typename SASGT::tiling_class;
    using T1 = typename SASGT::t1;
    static constexpr bool IS_BSND = SASGT::is_bsnd;

public:
    __aicore__ inline SelectedAttentionGradBasic(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, 
                                   GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv,
                                   GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, GM_ADDR d_cmp_kv, GM_ADDR dsinks, 
                                   GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void CubeCompute(CubeOp<SASGT> &cubeOp);
    __aicore__ inline void VecCompute(VecOp<SASGT> &vecOp);
    __aicore__ inline void UpdateGmOffset(int64_t task);
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                        const GM_ADDR actual_seq_ori_kvlen_addr,
                                        const GM_ADDR actual_seq_cmp_kvlen_addr,
                                        const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount);

    uint32_t cubeBlockIdx;
    uint32_t subBlockIdx;
    uint32_t formerCoreNum;
    uint32_t processBS1ByCore;
    uint32_t usedCoreNum;

    // shape info
    int64_t dimG;
    int64_t dimN1;
    int64_t dimN2;
    int64_t dimDqk;
    int64_t dimDv;
    int64_t dimRope;
    int64_t t1Offset{0};
    int64_t t2Offset{0};
    int64_t t3Offset{0};
    int64_t curS1;
    int64_t curS2;
    int64_t curS3;
    int64_t curMaxS3;
    int64_t dimS1;
    int64_t cmpRatio;
    int64_t oriWinLeft;
    int64_t oriWinRight;
    int64_t oriWinStart{0};
    int64_t oriWinEnd{0};
    int64_t s1BasicSize;

    // gmoffset
    uint64_t queryGmOffset;
    uint64_t dyGmOffset;
    uint64_t indicesGmOffset;
    uint64_t lseGmOffset;
    uint64_t oriKeyGmOffset;
    uint64_t cmpKeyGmOffset;
    uint64_t valueGmOffset;
    uint64_t mm12GmOffset;
    uint64_t mm345GmOffset;
    uint64_t selectedKGmOffset;
    uint64_t selectedVGmOffset;

    // workspace
    uint32_t selectedKWorkspaceLen;
    uint32_t mm12WorkspaceLen;
    uint32_t mm345WorkspaceLen;

    // Index
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t loopCnt{0};
    int32_t blkCntOffset;
    int32_t lastblkCntOffset;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};
    uint32_t mmPingPongIdx{0};
    uint32_t selectdKPPPidx{0};
    int64_t scatterTaskId{0};

    // selectBlock相关
    int32_t selectedCountOffset{0};
    int32_t actualSelectedBlockCount{0};
    int32_t selectedBlockSize{0};
    int32_t selectedBlockCount{0};

    // flag
    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PING = 4;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PONG = 5;
    constexpr static uint32_t SCATTER_SYNC_FLAG = 6;
    bool changePingpong = false;
    bool isLastBlockSelected = false;

    RunInfo runInfo[2];
    RunInfo scatterRunInfo;

    // gm
    GlobalTensor<int32_t> topkIndicesGm;
};

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::Init(const TILING_CLASS *__restrict tilingData)
{
    dimS1 = tilingData->opInfo.S1;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimN1 = dimG * dimN2;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimDqk;
    dimRope = 0;
    cmpRatio = tilingData->opInfo.cmpRatio;
    oriWinLeft = tilingData->opInfo.oriWinLeft;
    oriWinRight = tilingData->opInfo.oriWinRight;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    mm345WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(T1);
    if constexpr (IS_BSND == true) {
        curS1 = tilingData->opInfo.S1;
        curS2 = tilingData->opInfo.S2;
        curS3 = tilingData->opInfo.S3;
    }

    if ASCEND_IS_AIC {
        cubeBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        cubeBlockIdx = GetBlockIdx() / 2;
        subBlockIdx = GetBlockIdx() % 2;
    }

    formerCoreNum = tilingData->opInfo.formerCoreNum;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    if (cubeBlockIdx < formerCoreNum) {
        processBS1ByCore = tilingData->opInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = tilingData->opInfo.remainCoreProcessNNum;
    }

    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedKWspOffset = selectedKWorkspaceLen / sizeof(T1) / 4;
    selectedCountOffset = tilingData->splitCoreParams.singleN;
    if (oriWinLeft + oriWinRight + tilingData->opInfo.selectedBlockCount <= tilingData->splitCoreParams.singleN) {
        selectedCountOffset = oriWinLeft + oriWinRight + tilingData->opInfo.selectedBlockCount;
    }
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    s1BasicSize = tilingData->splitCoreParams.s1BasicSize;
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::Process(
    GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR lse,
    GM_ADDR topk_indices, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv, GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, 
    GM_ADDR d_cmp_kv, GM_ADDR dsinks, GM_ADDR workspace, const TILING_CLASS *__restrict tilingData)
{
    Init(tilingData);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);

    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SASGT> cubeOp;
        cubeOp.Init(query, ori_kv, cmp_kv, attention_out_grad, workspace, tilingData, &pipeCube);
        AllocEventID();
        int64_t task = 0;
        bool changeS1 = false;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % 2;
            // SCFA backward batching: each task covers s1BasicSize (B_q) consecutive query
            // positions; they share one cmp gather (representative top-k) + one ori-union
            // window (approx, batch-shared mask). s1BasicSize==1 -> per-query == stable.
            int32_t t1Index = (cubeBlockIdx + usedCoreNum * i) * s1BasicSize;
            GetTndSeqLen(cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, t1Index, bIndex);
            changePingpong = false;
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset(task);
                    CubeCompute(cubeOp);
                    if (changeS1) {
                        CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
                        changeS1 = false;
                    }
                    task++;
                }
            }
            if (changePingpong) {
                changeS1 = actualSelectedBlockCount ? true : false;
            }
        }

        if (cubeBlockIdx < usedCoreNum && task > 0) {
            int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
        }
        FreeEventID();
    }

    // AIV Process
    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<SASGT> vecOp;
        vecOp.Init(ori_kv, cmp_kv, attention_out, attention_out_grad, lse, topk_indices, sinks, dsinks,
                    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, workspace, tilingData, &pipeVec);
        SyncAll();
        int64_t task = 0;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % 2;
            // SCFA backward batching: each task covers s1BasicSize (B_q) consecutive query
            // positions; they share one cmp gather (representative top-k) + one ori-union
            // window (approx, batch-shared mask). s1BasicSize==1 -> per-query == stable.
            int32_t t1Index = (cubeBlockIdx + usedCoreNum * i) * s1BasicSize;
            GetTndSeqLen(cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, t1Index, bIndex);
            changePingpong = false;
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset(task);
                    VecCompute(vecOp);
                    task++;
                }
            }
            if (changePingpong) {
                runInfo[1 - mmPingPongIdx].changeS1 = actualSelectedBlockCount ? true : false;
            }
        }
        if (cubeBlockIdx < usedCoreNum && task > 0) {
            if (scatterRunInfo.changeS1) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                vecOp.ScatterAdd(scatterRunInfo);
                scatterRunInfo.changeS1 = false;
            }

            int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
            if (subBlockIdx == 0) {
                vecOp.Process(runInfo[1 - mmPingPongIdx]);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);

            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            vecOp.ScatterAdd(runInfo[1 - mmPingPongIdx]);
        }
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        SparseFlashAttentionGradPost<T1, TILING_CLASS, true, IS_BSND ? 2 : 3, 0> opCast;
        opCast.Init(dq, d_ori_kv, d_cmp_kv, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::CubeCompute(CubeOp<SASGT> &cubeOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    // WaitVec for select & gather
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
    if (unlikely(loopCnt == 0)) {
        cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
        CrossCoreSetFlag<2, PIPE_FIX>(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
        SaveLastInfo();
        return;
    }
    cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
    CrossCoreSetFlag<2, PIPE_FIX>(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PONG : CUBE_WAIT_VEC_PING);
    cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
    SaveLastInfo();
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::VecCompute(VecOp<SASGT> &vecOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    if (cubeBlockIdx < usedCoreNum) {
        vecOp.GatherKV(n2Index, t1Offset, runInfo[mmPingPongIdx]);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);

    if (scatterRunInfo.changeS1) {
        CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
        vecOp.ScatterAdd(scatterRunInfo);
        scatterRunInfo.changeS1 = false;
    }

    if (runInfo[mmPingPongIdx].task > 0) {
        int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
        CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
        if (subBlockIdx == 0) {
            vecOp.Process(runInfo[1 - mmPingPongIdx]);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);

        if (runInfo[1 - mmPingPongIdx].changeS1) {
            scatterRunInfo = runInfo[1 - mmPingPongIdx];
            runInfo[1 - mmPingPongIdx].changeS1 = false;
        }
    }
    
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
    changePingpong = true;
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::UpdateGmOffset(int64_t task)
{
    /*
     *  query:    B S1 N2 G D
     *  dy/out:   B S1 N2 G D2
     *  indices:  B S1 N2 SELCTED_BLOCK_COUNT
     *  sum/max   B S1 N2 G 8 --> N2 T1 G or B N2 S1 G
     *  key:      B S2 N2 D
     *  value:    B S2 N2 D2
     */
    if constexpr (IS_BSND) {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = t1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        lseGmOffset = bIndex * dimN2 * dimS1 * dimG + n2Index * dimS1 * dimG + s1Index * dimG;
    } else {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + s1Index * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + s1Index * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset =
            t1Offset * (dimN2 * selectedBlockCount) + s1Index * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        lseGmOffset = n2Index * dimS1 * dimG + (t1Offset + s1Index) * dimG;
    }
    oriKeyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    cmpKeyGmOffset = t3Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm345WorkspaceLen;
    selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;
    selectedVGmOffset = selectedKGmOffset;

    runInfo[mmPingPongIdx].task = task;
    runInfo[mmPingPongIdx].lseGmOffset = lseGmOffset;
    runInfo[mmPingPongIdx].blkCntOffset = blkCntOffset;
    runInfo[mmPingPongIdx].queryGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].oriKeyGmOffset = oriKeyGmOffset;
    runInfo[mmPingPongIdx].cmpKeyGmOffset = cmpKeyGmOffset;
    runInfo[mmPingPongIdx].dyGmOffset = dyGmOffset;
    runInfo[mmPingPongIdx].indicesGmOffset = indicesGmOffset;
    runInfo[mmPingPongIdx].mm12GmOffset = mm12GmOffset;
    runInfo[mmPingPongIdx].mm345GmOffset = mm345GmOffset;
    runInfo[mmPingPongIdx].dqOutGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].actualSelCntOffset = blkCntOffset + selectedCountOffset <= actualSelectedBlockCount ? selectedCountOffset : actualSelectedBlockCount - blkCntOffset;
    runInfo[mmPingPongIdx].lastBlockSize = isLastBlockSelected && curMaxS3 % selectedBlockSize != 0 ? curMaxS3 % selectedBlockSize : selectedBlockSize;
    runInfo[mmPingPongIdx].isLastBasicBlock = (blkCntOffset + selectedCountOffset >= actualSelectedBlockCount);
    runInfo[mmPingPongIdx].scatterTaskId = scatterTaskId;
    runInfo[mmPingPongIdx].s1Index = s1Index;
    runInfo[mmPingPongIdx].actualSelectedBlockCount = actualSelectedBlockCount;
    runInfo[mmPingPongIdx].curS1 = curS1;
    runInfo[mmPingPongIdx].curS3 = curS3;
    runInfo[mmPingPongIdx].selectedKGmOffset = selectedKGmOffset;
    runInfo[mmPingPongIdx].selectedVGmOffset = selectedVGmOffset;
    runInfo[mmPingPongIdx].oriWinStart = oriWinStart;
    runInfo[mmPingPongIdx].oriWinEnd = oriWinEnd;
    runInfo[mmPingPongIdx].curS1g = s1BasicSize * dimG;
    runInfo[mmPingPongIdx].curS1Basic = s1BasicSize;
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::SaveLastInfo()
{
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
    changePingpong = true;
    loopCnt++;
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                                                       const GM_ADDR actual_seq_ori_kvlen_addr,
                                                                       const GM_ADDR actual_seq_cmp_kvlen_addr,
                                                                       const int64_t t1Idx, int64_t &bIdx)
{
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[++bIndex];
        }
        
        if (unlikely(bIndex == 0)) {
            t1Offset = 0;
            t2Offset = 0;
            t3Offset = 0;
            curS1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex + 1];
            curS2 = ((__gm__ int32_t *)actual_seq_ori_kvlen_addr)[bIndex + 1];
            curS3 = ((__gm__ int32_t *)actual_seq_cmp_kvlen_addr)[bIndex + 1];
        } else {
            t1Offset = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            t2Offset = ((__gm__ int32_t *)actual_seq_ori_kvlen_addr)[bIndex - 1];
            t3Offset = ((__gm__ int32_t *)actual_seq_cmp_kvlen_addr)[bIndex - 1];
            curS1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            curS2 = (((__gm__ int32_t *)actual_seq_ori_kvlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_ori_kvlen_addr)[bIndex - 1]);
            curS3 = (((__gm__ int32_t *)actual_seq_cmp_kvlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_cmp_kvlen_addr)[bIndex - 1]);
        }

        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        bIdx = t1Idx / curS1;
        s1Index = t1Idx % dimS1;
        t2Offset = bIdx * curS2;
        t3Offset = bIdx * curS3;
    }
}

template <typename SASGT>
__aicore__ inline void SelectedAttentionGradBasic<SASGT>::GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount)
{
    // batch spans query positions [s1Index, s1Last]; use the LAST for causal extent and
    // the ori-window union END, and the FIRST for the ori-window union START. cmp uses the
    // last query's causal count (representative top-k gathered via indicesGmOffset @ s1Index).
    int64_t s1Last = Min(s1Index + s1BasicSize - 1, curS1 - 1);
    // cmp causal count uses the FIRST query so we gather exactly its valid top-k blocks
    // (no padding region -> no duplicates / no -1 OOB). Later queries in the batch share
    // these blocks (the batch-shared approximation). ori union below still uses s1Last.
    int64_t maxS3 = Max((curS2 - curS1 + s1Index + 1) / cmpRatio, 0);
    curMaxS3 = (maxS3 + selectedBlockSize - 1) / selectedBlockSize;

    actualSelectedBlockCount = Min(selectedBlockCount, curMaxS3);

    int64_t oriWinDiagOffset = curS2 - curS1;
    oriWinEnd = Min(s1Last + oriWinRight + 1 + oriWinDiagOffset, curS2);   // union end (last query)
    oriWinEnd = Max(oriWinEnd, 0);
    oriWinStart = Max(s1Index - oriWinLeft + oriWinDiagOffset, 0);         // union start (first query)
    int64_t oriWinS2Len = oriWinEnd - oriWinStart;

    actualSelectedBlockCount += oriWinS2Len;
}

} // namespace SASG_BASIC
