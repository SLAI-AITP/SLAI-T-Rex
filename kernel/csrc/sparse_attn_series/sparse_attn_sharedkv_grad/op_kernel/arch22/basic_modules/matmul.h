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
 * \file matmul.h
 * \brief
 */
#pragma once
#include "kernel_operator.h"
#include "common_header.h"

using namespace AscendC;

namespace SASG_BASIC {

constexpr static uint64_t C0_SIZE = 16;

// mte2 <> mte1 EventID
// L1 3buf, 使用3个eventId
static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

static constexpr uint32_t MM_L1_QUERY_EVENTS[2] = {L1_EVENT0, L1_EVENT1};
static constexpr uint32_t MM_L1_COMMON_EVENTS[2] = {L1_EVENT2, L1_EVENT3};
static constexpr uint32_t MM_L1_DY_EVENTS[2] = {L1_EVENT4, L1_EVENT5};
static constexpr uint32_t MM_L1_DS_EVENT = L1_EVENT6;

// m <> mte1
static constexpr uint32_t L0A_EVENTS[2] = {EVENT_ID3, EVENT_ID4};
static constexpr uint32_t L0B_EVENTS[2] = {EVENT_ID5, EVENT_ID6};

// fix <> m
static constexpr uint32_t L0C_EVENTS[2] = {EVENT_ID1, EVENT_ID2};

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose = false;
    bool isRightTranspose = false;
    bool isOutKFisrt = true;
    bool isFixOut = true;
    int64_t dstStride = 0;
};

__aicore__ inline void UpdatePingPongFlag(uint32_t &flag) {
    flag = 1 - flag;
}

__aicore__ inline void AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID6);

    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    SetFlag<HardEvent::FIX_M>(EVENT_ID2);
}

__aicore__ inline void FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);

    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
}

template <typename T>
__aicore__ inline void CopyGmToL1(const LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmTensor, uint32_t srcN,
                                                                uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void LoadDataMm1A(LocalTensor<T> &aL0Tensor, LocalTensor<T> &aL1Tensor, struct MMParam &mmParam)
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

template <typename T>
__aicore__ inline void LoadDataMm1AWithTranspose(LocalTensor<T> &aL0Tensor, LocalTensor<T> &aL1Tensor, struct MMParam &mmParam)
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

template <typename T>
__aicore__ inline void LoadDataMm1B(LocalTensor<T> &l0Tensor,
                                                             LocalTensor<T> &l1Tensor,
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

// OUT_T (deduced from resGm) lets the fixpipe cast the fp32 L0C accumulator to a smaller
// store dtype (e.g. bf16) for the cube->vec score/dp handoff. Defaults to float, so all
// existing GlobalTensor<float> callers are unchanged.
template <typename T1, bool needAtomic = false, bool isScatterFixOut = false, typename OUT_T = float>
__aicore__ inline void MmadInnerWithSync(LocalTensor<float> &l0cTensor,
                                 LocalTensor<T1> &l1aTensor, LocalTensor<T1> &l1bTensor,
                                 LocalTensor<T1> (&aL0TensorPingPong)[2], LocalTensor<T1> (&bL0TensorPingPong)[2],
                                 struct MMParam &mmParam, uint32_t &l0aPingPongFlag, uint32_t &l0bPingPongFlag,
                                 uint32_t l0cPingPongFlag, bool needCopyL0a,
                                 const GlobalTensor<OUT_T> &resGm) {
    MmadParams mmadParams;
    mmadParams.m = (mmParam.singleM == 1) ? 2 : mmParam.singleM;
    mmadParams.n = mmParam.singleN;
    mmadParams.k = mmParam.singleK;
    mmadParams.cmatrixInitVal = mmParam.isOutKFisrt;

    LocalTensor<T1> l0aTensor = aL0TensorPingPong[l0aPingPongFlag & 1];
    LocalTensor<T1> l0bTensor = bL0TensorPingPong[l0bPingPongFlag & 1];

    uint32_t l0a_event = L0A_EVENTS[l0aPingPongFlag & 1];
    uint32_t l0b_event = L0B_EVENTS[l0bPingPongFlag & 1];
    uint32_t l0c_event = L0C_EVENTS[l0cPingPongFlag & 1];
    
    SetFlag<HardEvent::MTE2_MTE1>(l0b_event);
    WaitFlag<HardEvent::MTE2_MTE1>(l0b_event);

    // 反向同步
    WaitFlag<HardEvent::M_MTE1>(l0a_event);
    if (needCopyL0a) {
        if (mmParam.isLeftTranspose) {
            LoadDataMm1AWithTranspose<T1>(l0aTensor, l1aTensor, mmParam);
        } else {
            LoadDataMm1A<T1>(l0aTensor, l1aTensor, mmParam);
        }
    }

    // 反向同步
    WaitFlag<HardEvent::M_MTE1>(l0b_event);
    LoadDataMm1B<T1>(l0bTensor, l1bTensor, mmParam);

    SetFlag<HardEvent::MTE1_M>(l0b_event);
    WaitFlag<HardEvent::MTE1_M>(l0b_event);
    PIPE_BARRIER(PIPE_M);
    // 反向同步
    WaitFlag<HardEvent::FIX_M>(l0c_event);
    Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);

    SetFlag<HardEvent::M_MTE1>(l0a_event);
    SetFlag<HardEvent::M_MTE1>(l0b_event);
    
    if (mmParam.isFixOut) {
        SetFlag<HardEvent::M_FIX>(l0c_event);
        WaitFlag<HardEvent::M_FIX>(l0c_event);

        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = mmParam.singleN;
        fixpipeParams.mSize = mmParam.singleM;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = mmParam.dstStride;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (needAtomic) {
            AscendC::SetAtomicAdd<float>();
        }
        Fixpipe<OUT_T, float>(resGm, l0cTensor, fixpipeParams); // L0C(fp32) -> GM(OUT_T); casts if OUT_T!=float
        if constexpr (needAtomic) {
            AscendC::SetAtomicNone();
        }
    }
    // scatter out场景的反向同步在ScatterFixOut做
    if constexpr(!isScatterFixOut) {
        SetFlag<HardEvent::FIX_M>(l0c_event);
    }
    
    l0aPingPongFlag = 1 - l0aPingPongFlag;
    l0bPingPongFlag = 1 - l0bPingPongFlag;
}

} // namespace SASG_BASIC