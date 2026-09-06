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
 * \file cube4.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4Process(const int64_t dsGmOffset, const int64_t queryGmOffset,
                         const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    
    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = selectedBlockSize * blockOffset;
    mmParam.singleK = runInfo.curS1g;
    mmParam.isFixOut = true;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal * dimN2;

    int64_t mm4ResOutBaseOffset;
    if constexpr (MODE == SASG_SCFA_MODE) {
        mm4ResOutBaseOffset = runInfo.scatterTaskId * MAX_CORE_NUM * (selectedBlockCount + oriWinLen) * dimDTotal + cBlockIdx * (selectedBlockCount + oriWinLen) * dimDTotal;
    } else {
        mm4ResOutBaseOffset = runInfo.isOri ? 
                              runInfo.selectedKGmOffset - blkCntOffset * selectedBlockSize * dimDTotal :
                              runInfo.selectedKGmOffset - blkCntOffset * selectedBlockSize * dimDTotal + dOriKvSize;
    }
    // K_TILE gates the curS1g contraction tiling. curS1g<=K_TILE -> original single-K path
    // (B_q<=2, no change). Set K_TILE=64 to force the K-tiled path at B_q=2 for testing.
    const int32_t K_TILE = 128;
    if (runInfo.curS1g <= K_TILE) {
    CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[dsGmOffset], mmParam.singleK, selectedCntOffset * selectedBlockSize, singleN);
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx+=blockOffset) {
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * AlignTo<int64_t>(mmParam.singleK, SIZE_16);

        mmParam.singleN = perLoopDSize;
        mmParam.singleM = min(selectedBlockSize * blockOffset, selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);

        LocalTensor<T1> current_l1_ds_tensor, l1_query_tensor;
        current_l1_ds_tensor = l1_ds_tensor[l1Offset];
        int64_t currentQueryOffset;
        int64_t mm4ResOutOffset = mm4ResOutBaseOffset + mIdx * selectedBlockSize * dimDTotal;
        
        for (uint32_t dIdx = 0;dIdx < dLoopTimes - 1; dIdx++) {
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
            l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
            
            currentQueryOffset = queryGmOffset + dIdx * N_SPLIT_SIZE;
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            CopyGmToL1(l1_query_tensor, queryGm[currentQueryOffset], mmParam.singleK, N_SPLIT_SIZE, dimDqk);

            int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize;

            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            MmadInnerWithSync<T1, MODE!=SASG_SCFA_MODE>(l0cTensor, current_l1_ds_tensor, l1_query_tensor,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0, mm4ResWorkspaceGm[currentOutGmOffset]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            UpdatePingPongFlag(ping_pong_flag_l0c_);
            UpdatePingPongFlag(ping_pong_flag_l1_query_);
        }

        mmParam.singleN = tailLoopDSize;
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
        l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
        
        currentQueryOffset = queryGmOffset + (dLoopTimes - 1) * K_SPLIT_SIZE;
        GlobalTensor<T1> srcGm = queryGm[currentQueryOffset];
        int64_t destStride = dimDqk;
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        CopyGmToL1(l1_query_tensor, srcGm, mmParam.singleK, tailLoopDSize, destStride);

        int64_t currentOutGmOffset = mm4ResOutOffset + (dLoopTimes - 1) * perLoopDSize;
        
        MmadInnerWithSync<T1, MODE!=SASG_SCFA_MODE>(l0cTensor, current_l1_ds_tensor, l1_query_tensor,
                aL0TensorPingPong, bL0TensorPingPong,
                mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm4ResWorkspaceGm[currentOutGmOffset]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
        UpdatePingPongFlag(ping_pong_flag_l1_query_);
    }
    } else {
        // ===== K-tiled path: tile the curS1g contraction into <=K_TILE chunks, accumulating
        // dk in L0C (isOutKFisrt on first, isFixOut on last). ds reloaded per K-tile (all
        // selected width) with WAR via MM_L1_DS_EVENT; q reloaded per (dIdx,kStart) via the
        // MM_L1_QUERY_EVENTS ping-pong. The DS first/last flags keep the task-level
        // Wait(before cube4)/Set(after cube3) pairing balanced (cube4 enters & exits DS=0). =====
        bool firstDsLoad = true;
        for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx+=blockOffset) {
            mmParam.singleM = min(selectedBlockSize * blockOffset, selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);
            int64_t mm4ResOutOffset = mm4ResOutBaseOffset + mIdx * selectedBlockSize * dimDTotal;
            bool isLastM = (mIdx + blockOffset >= blkCntOffset + selectedCntOffset);
            for (uint32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
                mmParam.singleN = (dIdx == dLoopTimes - 1) ? tailLoopDSize : perLoopDSize;
                int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize;
                bool isLastD = (dIdx == dLoopTimes - 1);
                LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
                for (int32_t kStart = 0; kStart < runInfo.curS1g; kStart += K_TILE) {
                    int32_t kSize = (runInfo.curS1g - kStart < K_TILE) ? (int32_t)(runInfo.curS1g - kStart) : K_TILE;
                    bool isLastK = (kStart + K_TILE >= runInfo.curS1g);
                    mmParam.singleK = kSize;
                    mmParam.isOutKFisrt = (kStart == 0);
                    mmParam.isFixOut = isLastK;
                    int64_t l1Offset = (int64_t)(mIdx - blkCntOffset) * selectedBlockSize * AlignTo<int64_t>(kSize, SIZE_16);
                    // ds K-tile [kStart:+kSize, all selected]
                    if (!firstDsLoad) { WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT); }
                    CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[dsGmOffset + (int64_t)kStart * singleN], kSize, selectedCntOffset * selectedBlockSize, singleN);
                    // q K-tile [kStart:+kSize, this dIdx D-tile]
                    LocalTensor<T1> l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
                    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
                    CopyGmToL1(l1_query_tensor, queryGm[queryGmOffset + dIdx * N_SPLIT_SIZE + (int64_t)kStart * dimDqk], kSize, N_SPLIT_SIZE, dimDqk);
                    LocalTensor<T1> current_l1_ds_tensor = l1_ds_tensor[l1Offset];
                    uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
                    MmadInnerWithSync<T1, false>(l0cTensor, current_l1_ds_tensor, l1_query_tensor,
                            aL0TensorPingPong, bL0TensorPingPong,
                            mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm4ResWorkspaceGm[currentOutGmOffset]);
                    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
                    UpdatePingPongFlag(ping_pong_flag_l1_query_);
                    bool isVeryLast = isLastM && isLastD && isLastK;
                    if (!isVeryLast) { SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT); }
                    firstDsLoad = false;
                }
                UpdatePingPongFlag(ping_pong_flag_l0c_);
            }
        }
    }
}