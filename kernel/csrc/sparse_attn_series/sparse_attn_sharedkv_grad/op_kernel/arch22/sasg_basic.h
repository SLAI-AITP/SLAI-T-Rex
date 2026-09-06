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
 * \file sasg_basic.h
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
class SparseAttnSharedkvGrad {
    using TILING_CLASS = typename SASGT::tiling_class;
    using T1 = typename SASGT::t1;
    static constexpr bool IS_BSND = SASGT::is_bsnd;
    static constexpr uint32_t MODE = SASGT::mode;

public:
    __aicore__ inline SparseAttnSharedkvGrad(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, 
                                   GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv,
                                   GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, GM_ADDR d_cmp_kv, GM_ADDR dsinks, 
                                   GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS *__restrict tilingData, 
                                const GM_ADDR cu_seqlens_q, 
                                const GM_ADDR cu_seqlens_ori_kv, 
                                const GM_ADDR cu_seqlens_cmp_kv);
    __aicore__ inline void CubeCompute(CubeOp<SASGT> &cubeOp);
    __aicore__ inline void VecCompute(VecOp<SASGT> &vecOp);
    __aicore__ inline void UpdateGmOffset(int64_t task, int32_t loop);
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx, int32_t &s1Loop);
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &curS2Loop);
    __aicore__ inline void ChangeBatchUpdate();

    uint32_t cubeBlockIdx;
    uint32_t subBlockIdx;
    uint32_t formerCoreNum;
    uint32_t processBS1ByCore;
    uint32_t usedCoreNum;

    // shape info
    int64_t dimB;
    int64_t dimG;
    int64_t dimN1;
    int64_t dimN2;
    int64_t dimDqk;
    int64_t dimDv;
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
    int64_t curS1Basic;
    int64_t curLoopS1Basic;
    int64_t oriWinDiagOffset;

    // gmoffset
    uint64_t queryGmOffset;
    uint64_t dyGmOffset;
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

    // selectBlock相关
    int32_t selectedCountOffset{0};
    int32_t oriSelectedCount{0};
    int32_t cmpSelectedCount{0};
    int32_t oriS2Loop{0};
    int32_t cmpS2Loop{0};
    int32_t s1Loop{0};
    int32_t s2Loop{0};
    int32_t selectedBlockSize{0};
    int32_t selectedBlockCount{0};

    // flag
    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t POST_WAIT_CUBE = 4;

    // GM_ADDR
    const GM_ADDR cu_seqlens_q;
    const GM_ADDR cu_seqlens_ori_kv;
    const GM_ADDR cu_seqlens_cmp_kv;

    RunInfo runInfo[2];
};

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::Init(const TILING_CLASS *__restrict tilingData,
                                                           const GM_ADDR cu_seqlens_q, 
                                                           const GM_ADDR cu_seqlens_ori_kv, 
                                                           const GM_ADDR cu_seqlens_cmp_kv)
{
    dimB = tilingData->opInfo.B;
    dimS1 = tilingData->opInfo.S1;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimN1 = dimG * dimN2;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimDqk;
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

    selectedCountOffset = tilingData->splitCoreParams.singleN;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    s1BasicSize = tilingData->splitCoreParams.s1BasicSize;

    if constexpr (!IS_BSND) {
        this->cu_seqlens_q = cu_seqlens_q;
        this->cu_seqlens_ori_kv = cu_seqlens_ori_kv;
        this->cu_seqlens_cmp_kv = cu_seqlens_cmp_kv;
    }
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::Process(
    GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR lse,
    GM_ADDR topk_indices, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv, GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, 
    GM_ADDR d_cmp_kv, GM_ADDR dsinks, GM_ADDR workspace, const TILING_CLASS *__restrict tilingData)
{
    Init(tilingData, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv);

    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SASGT> cubeOp;
        cubeOp.Init(query, ori_kv, cmp_kv, attention_out_grad, workspace, tilingData, &pipeCube);
        AllocEventID();
        int64_t task = 0;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            int32_t t1Index = (cubeBlockIdx + usedCoreNum * i) * s1BasicSize;
            GetTndSeqLen(t1Index, bIndex, s1Loop);
            int32_t s1BasicAccum = 0;
            for (int32_t j = 0; j < s1Loop; j++) {
                bool changeB = s1Index + (curS1Basic - s1BasicAccum) > curS1;
                curLoopS1Basic = changeB ? curS1 - s1Index : (curS1Basic - s1BasicAccum);
                for (n2Index = 0; n2Index < dimN2; n2Index++) {
                    GetActualSelCount(t1Index, n2Index, s2Loop);
                    for (int32_t loop = 0; loop < s2Loop; loop++) {
                        UpdateGmOffset(task, loop);
                        CubeCompute(cubeOp);
                        task++;
                    }
                }
                s1BasicAccum += curLoopS1Basic;
                if (changeB) {
                    ChangeBatchUpdate();
                }
            }
        }

        if (cubeBlockIdx < usedCoreNum && task > 0) {
            int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(POST_WAIT_CUBE);
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
            int32_t t1Index = (cubeBlockIdx + usedCoreNum * i) * s1BasicSize;
            GetTndSeqLen(t1Index, bIndex, s1Loop);
            int32_t s1BasicAccum = 0;
            for (int32_t j = 0; j < s1Loop; j++) {
                bool changeB = s1Index + (curS1Basic - s1BasicAccum) > curS1;
                curLoopS1Basic = changeB ? curS1 - s1Index : (curS1Basic - s1BasicAccum);
                for (n2Index = 0; n2Index < dimN2; n2Index++) {
                    GetActualSelCount(t1Index, n2Index, s2Loop);
                    for (int32_t loop = 0; loop < s2Loop; loop++) {
                        UpdateGmOffset(task, loop);
                        VecCompute(vecOp);
                        task++;
                    }
                }
                s1BasicAccum += curLoopS1Basic;
                if (changeB) {
                    ChangeBatchUpdate();
                }
            }
        }
        if (cubeBlockIdx < usedCoreNum && task > 0) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(POST_WAIT_CUBE);
        }
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        SparseFlashAttentionGradPost<T1, TILING_CLASS, true, IS_BSND ? 2 : 3, 0, MODE> opCast;
        opCast.Init(dq, d_ori_kv, d_cmp_kv, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::CubeCompute(CubeOp<SASGT> &cubeOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
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
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::VecCompute(VecOp<SASGT> &vecOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    CrossCoreWaitFlag(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
    if (subBlockIdx == 0) {
        vecOp.Process(runInfo[mmPingPongIdx]);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
    mmPingPongIdx = 1 - mmPingPongIdx;
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::UpdateGmOffset(int64_t task, int32_t loop)
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
        lseGmOffset = bIndex * dimN2 * dimS1 * dimG + n2Index * dimS1 * dimG + s1Index * dimG;
    } else {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + s1Index * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + s1Index * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        lseGmOffset = n2Index * dimS1 * dimG + (t1Offset + s1Index) * dimG;
    }
    oriKeyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm345WorkspaceLen;

    if (loop == 0 || (MODE == SASG_CFA_MODE && loop == oriS2Loop)) {
        blkCntOffset = 0;
    } else {
        blkCntOffset += selectedCountOffset;
    }
    if constexpr (MODE == SASG_CFA_MODE) {
        cmpKeyGmOffset = t3Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
        runInfo[mmPingPongIdx].curS3 = curS3;
    }
    runInfo[mmPingPongIdx].isOri = loop < oriS2Loop;
    selectedKGmOffset = runInfo[mmPingPongIdx].isOri ? oriKeyGmOffset + (oriWinStart + blkCntOffset) * dimN2 * dimDqk : 
                                                       cmpKeyGmOffset + blkCntOffset * dimN2 * dimDqk;
    runInfo[mmPingPongIdx].task = task;
    runInfo[mmPingPongIdx].lseGmOffset = lseGmOffset;
    runInfo[mmPingPongIdx].blkCntOffset = blkCntOffset;
    runInfo[mmPingPongIdx].queryGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].dyGmOffset = dyGmOffset;
    runInfo[mmPingPongIdx].mm12GmOffset = mm12GmOffset;
    runInfo[mmPingPongIdx].mm345GmOffset = mm345GmOffset;
    runInfo[mmPingPongIdx].dqOutGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].actualSelCntOffset = runInfo[mmPingPongIdx].isOri ? 
                                                (blkCntOffset + selectedCountOffset <= oriSelectedCount ? selectedCountOffset : oriSelectedCount - blkCntOffset) :
                                                (blkCntOffset + selectedCountOffset <= cmpSelectedCount ? selectedCountOffset : cmpSelectedCount - blkCntOffset);
    runInfo[mmPingPongIdx].lastBlockSize = 1;
    runInfo[mmPingPongIdx].isLastBasicBlock = false;
    runInfo[mmPingPongIdx].s1Index = s1Index;
    runInfo[mmPingPongIdx].curS1 = curS1;
    runInfo[mmPingPongIdx].curS2 = curS2;
    runInfo[mmPingPongIdx].oriWinStart = oriWinStart;
    runInfo[mmPingPongIdx].oriWinEnd = oriWinEnd;
    runInfo[mmPingPongIdx].selectedKGmOffset = selectedKGmOffset;
    runInfo[mmPingPongIdx].selectedVGmOffset = selectedKGmOffset;
    runInfo[mmPingPongIdx].curS1g = curLoopS1Basic * dimG; // 尾块S1处理
    runInfo[mmPingPongIdx].curS1Basic = curLoopS1Basic;
    runInfo[mmPingPongIdx].oriWinDiagOffset = oriWinDiagOffset;
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::SaveLastInfo()
{
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    loopCnt++;
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::ChangeBatchUpdate()
{
    s1Index = 0;
    bIndex++;
    if constexpr (IS_BSND == false) {
        t1Offset = ((__gm__ int32_t *)cu_seqlens_q)[bIndex - 1];
        t2Offset = ((__gm__ int32_t *)cu_seqlens_ori_kv)[bIndex - 1];
        curS1 = ((__gm__ int32_t *)cu_seqlens_q)[bIndex] - ((__gm__ int32_t *)cu_seqlens_q)[bIndex - 1];
        curS2 = (((__gm__ int32_t *)cu_seqlens_ori_kv)[bIndex] - ((__gm__ int32_t *)cu_seqlens_ori_kv)[bIndex - 1]);
        if constexpr (MODE == SASG_CFA_MODE) {
            t3Offset = ((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIndex - 1];
            curS3 = (((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIndex] - ((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIndex - 1]);
        }
    } else {
        t1Offset = bIndex * curS1;
        t2Offset = bIndex * curS2;
        if constexpr (MODE == SASG_CFA_MODE) {
            t3Offset = bIndex * curS3;
        }
    }
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx, int32_t &s1Loop)
{
    s1Loop = 0;
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t *)cu_seqlens_q)[bIdx];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t *)cu_seqlens_q)[++bIdx];
        }
        
        t1Offset = ((__gm__ int32_t *)cu_seqlens_q)[bIdx - 1];
        t2Offset = ((__gm__ int32_t *)cu_seqlens_ori_kv)[bIdx - 1];
        curS1 = ((__gm__ int32_t *)cu_seqlens_q)[bIdx] - ((__gm__ int32_t *)cu_seqlens_q)[bIdx - 1];
        curS2 = (((__gm__ int32_t *)cu_seqlens_ori_kv)[bIdx] - ((__gm__ int32_t *)cu_seqlens_ori_kv)[bIdx - 1]);

        if constexpr (MODE == SASG_CFA_MODE) {
            t3Offset = ((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIdx - 1];
            curS3 = (((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIdx] - ((__gm__ int32_t *)cu_seqlens_cmp_kv)[bIdx - 1]);
        }

        s1Index = t1Idx - t1Offset;
        curS1Basic = t1Idx + s1BasicSize <= dimS1 ? s1BasicSize : dimS1 - t1Idx;

        int64_t bIndexTmp = bIdx;
        while (t1Idx + curS1Basic > curT1) {
            curT1 = ((__gm__ int32_t *)cu_seqlens_q)[++bIndexTmp];
            s1Loop++;
        }
        s1Loop++;
    } else {
        t1Offset = t1Idx;
        bIdx = t1Idx / curS1;
        s1Index = t1Idx % dimS1;
        t2Offset = bIdx * curS2;
        if constexpr (MODE == SASG_CFA_MODE) {
            t3Offset = bIdx * curS3;
        }
        curS1Basic = t1Idx + s1BasicSize <= dimB * dimS1 ? s1BasicSize : dimB * dimS1 - t1Idx;

        int64_t bIndexTmp = bIdx;
        while (t1Idx + curS1Basic > (bIndexTmp + 1) * dimS1) {
            bIndexTmp++;
            s1Loop++;
        }
        s1Loop++;
    }
}

template <typename SASGT>
__aicore__ inline void SparseAttnSharedkvGrad<SASGT>::GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &curS2Loop)
{
    oriWinDiagOffset = curS2 - curS1;
    oriWinEnd = Min(s1Index + (curLoopS1Basic - 1) + oriWinRight + 1 + oriWinDiagOffset, curS2);
    oriWinEnd = Max(oriWinEnd, 0);
    oriWinStart = Max(s1Index - oriWinLeft + oriWinDiagOffset, 0);
    oriSelectedCount = oriWinEnd - oriWinStart;

    if constexpr (MODE == SASG_CFA_MODE) {
        curMaxS3 = Max((oriWinDiagOffset + s1Index + (curLoopS1Basic - 1) + 1) / cmpRatio, 0);
        cmpSelectedCount = curMaxS3;
    }

    oriS2Loop = CeilDiv(oriSelectedCount, selectedCountOffset);
    cmpS2Loop = CeilDiv(cmpSelectedCount, selectedCountOffset);
    curS2Loop = oriS2Loop + cmpS2Loop;
}

} // namespace SASG_BASIC
