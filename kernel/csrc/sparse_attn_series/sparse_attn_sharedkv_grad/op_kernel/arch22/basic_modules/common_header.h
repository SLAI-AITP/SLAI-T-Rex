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
 * \file common_header.h
 * \brief
 */
#pragma once
#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "kernel_tensor.h"
#include "kernel_macros.h"

namespace SASG_BASIC {

#define SASG_SWA_MODE 0
#define SASG_CFA_MODE 1
#define SASG_SCFA_MODE 2

constexpr static uint64_t MAX_CORE_NUM = 24;

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<pipe>()

/////////////////////////////////////////////////////
// common struct
/////////////////////////////////////////////////////

template <class TILING_CLASS, typename T1, const bool IS_BSND = false, const uint32_t MODE = SASG_SCFA_MODE, typename... Args>
struct SFAG_TYPE {
    using tiling_class = TILING_CLASS;
    using t1 = T1;
    static constexpr bool is_bsnd = IS_BSND;
    static constexpr uint32_t mode = MODE;
};

struct RunInfo {
    int64_t task;
    int64_t curS1;
    int64_t curS2;
    int64_t curS3;
    int64_t lseGmOffset;
    int64_t blkCntOffset;
    int64_t queryGmOffset;
    int64_t oriKeyGmOffset;
    int64_t cmpKeyGmOffset;
    int64_t dyGmOffset;
    int64_t indicesGmOffset;
    int64_t mm12GmOffset;
    int64_t mm345GmOffset;
    int64_t dqOutGmOffset;
    int64_t actualSelCntOffset;
    int64_t scatterTaskId;
    int64_t s1Index;
    int64_t actualSelectedBlockCount;
    int64_t changeS1 = false;
    int64_t selectedKGmOffset;
    int64_t selectedVGmOffset;
    int64_t lastBlockSize;
    int64_t oriWinStart;
    int64_t oriWinEnd;
    int64_t oriWinS2Len;
    int64_t curS1g;
    int64_t curS1Basic;
    int64_t oriWinDiagOffset;
    bool isLastBasicBlock;
    bool isOri;
};

/////////////////////////////////////////////////////
// hardware
/////////////////////////////////////////////////////
enum class ArchType {
    ASCEND_V220,
    ASCEND_V200,
    ASCEND_M200
};

template <ArchType ArchTag>
struct HardwareInfo {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 128 * 1024;
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 192 * 1024;
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 32;
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};

/////////////////////////////////////////////////////
// mem
/////////////////////////////////////////////////////
enum class BufferType {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

template <ArchType ArchTag>
struct AsdopsBuffer {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    __aicore__ AsdopsBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo<ArchTag>::ubSize, HardwareInfo<ArchTag>::l1Size, HardwareInfo<ArchTag>::l0ASize,
            HardwareInfo<ArchTag>::l0BSize, HardwareInfo<ArchTag>::l0CSize};
#ifdef __DAV_C220_VEC__
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
#elif __DAV_C220_CUBE__
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#else
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#endif
    };
#pragma GCC diagnostic pop

    template <BufferType BufferType_, typename DstDataType>
    __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
    }

public:
    AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};

/*
 * common function
 */
template <class T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

} // namespace SASG_BASIC