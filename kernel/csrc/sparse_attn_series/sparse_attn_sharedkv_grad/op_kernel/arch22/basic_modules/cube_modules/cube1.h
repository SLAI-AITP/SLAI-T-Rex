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
 * \file cube1.h
 * \brief Formula: s = q * k^T
 * l0_a： dimG * dimDqk * sizof(T1) (0-6k)
 * l0_b： selectedBlockSize * dimDqk * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 64k-68k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1Process(const int64_t queryGmOffset, const int64_t keyGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / K_SPLIT_SIZE;
    uint32_t perLoopDSize = K_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = N_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleN = MODE == SASG_SCFA_MODE ? selectedBlockSize * blockOffset : selectedCntOffset;
    mmParam.dstStride = singleN;

    // M-tiling: SCFA with B_q>2 makes curS1g (=B_q*gSize) up to 512, which overflows both
    // L0C (M*N*4B > 64KB) and l1_query (128x128). Tile M into <=128-row chunks. For
    // curS1g<=128 (CFA and SCFA B_q<=2) this runs once (mStart=0, mSize=curS1g), identical
    // to the original single-pass behaviour.
    const int32_t M_TILE = 128;
    for (int32_t mStart = 0; mStart < runInfo.curS1g; mStart += M_TILE) {
        int32_t mSize = (runInfo.curS1g - mStart < M_TILE) ? (int32_t)(runInfo.curS1g - mStart) : M_TILE;
        mmParam.singleM = mSize;
        const int64_t mQueryGmOffset = queryGmOffset + (int64_t)mStart * dimDqk;
        const int64_t mOutGmOffset = outGmOffset + (int64_t)mStart * singleN;
    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx+=blockOffset) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
        mmParam.isFixOut = false;
        mmParam.singleK = perLoopDSize;

        int64_t mm1WorkspaceGmOffset =  mOutGmOffset + (nIdx - blkCntOffset) * selectedBlockSize;
        LocalTensor<T1> l1_query_tensor, l1_key_tensor;
        int64_t currentQueryOffset, currentKeyOffset;
        // query node @ key node
        for (int32_t dIdx = 0; dIdx < dLoopTimes - 1; dIdx++) {
            mmParam.isOutKFisrt = dIdx == 0;
            l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
            l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];

            currentQueryOffset = mQueryGmOffset + dIdx * K_SPLIT_SIZE;
            currentKeyOffset = keyGmOffset + (nIdx - blkCntOffset) * selectedBlockSize * dimDTotal + dIdx * K_SPLIT_SIZE;

            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            CopyGmToL1(l1_query_tensor, queryGm[currentQueryOffset], mmParam.singleM, K_SPLIT_SIZE, dimDqk);
            if constexpr (MODE == SASG_SCFA_MODE) {
                CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            } else if (runInfo.isOri) {
                CopyGmToL1(l1_key_tensor, oriKvGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            } else {
                CopyGmToL1(l1_key_tensor, cmpKvGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            }

            MmadInnerWithSync<T1>(l0cTensor, l1_query_tensor, l1_key_tensor,
                        aL0TensorPingPong, bL0TensorPingPong,
                        mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm1WorkspaceGm[mm1WorkspaceGmOffset]);
            
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_query_);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }

        // query rope @ key rope
        mmParam.isFixOut = true;
        mmParam.isOutKFisrt = dLoopTimes == 1;
        mmParam.singleK = tailLoopDSize;
        l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
        l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
        
        currentQueryOffset = mQueryGmOffset + (dLoopTimes - 1) * K_SPLIT_SIZE;
        currentKeyOffset = keyGmOffset + (nIdx - blkCntOffset) * selectedBlockSize * dimDTotal + (dLoopTimes - 1) * K_SPLIT_SIZE;

        GlobalTensor<T1> srcGm = queryGm[currentQueryOffset];
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        int64_t srcDstride = dimDTotal;
        CopyGmToL1(l1_query_tensor, srcGm, mmParam.singleM, tailLoopDSize, srcDstride);
        if constexpr (MODE == SASG_SCFA_MODE) {
            CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleN, tailLoopDSize, dimDTotal);
        } else if (runInfo.isOri) {
            CopyGmToL1(l1_key_tensor, oriKvGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
        } else {
            CopyGmToL1(l1_key_tensor, cmpKvGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
        }

        MmadInnerWithSync<T1>(l0cTensor, l1_query_tensor, l1_key_tensor,
                aL0TensorPingPong, bL0TensorPingPong,
                mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm1WorkspaceGm[mm1WorkspaceGmOffset]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        
        UpdatePingPongFlag(ping_pong_flag_l1_query_);
        UpdatePingPongFlag(ping_pong_flag_l1_common_);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
    } // M-tile loop
}