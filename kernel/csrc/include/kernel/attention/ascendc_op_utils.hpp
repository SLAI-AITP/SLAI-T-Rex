/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCENDC_OP_UTILS_H
#define ASCENDC_OP_UTILS_H
#include "kernel_operator.h"
#include "float.h"
#include "utils/op_math.h"
using namespace AscendC;
namespace Wind {
constexpr uint16_t FLOAT_BLOCK_LENS = 8; // 仅float使用
constexpr uint16_t FLOAT_REPEAT_LENS = 64; // 仅float使用
constexpr uint16_t HALF_BLOCK_LENS = 16; // 仅half使用
constexpr uint16_t HALF_REPEAT_LENS = 128; // 仅half使用
constexpr float FLOAT_NEG_INF = -FLT_MAX; // Negative infinity for Float
constexpr float HALF_NEG_INF = -65504; // Negative infinity for half

enum OpType : uint8_t {
    SUM = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MAX = 4
};

//     fullCol            fullCol    tailCol
// |----------------|----------------|----|  \
// |----------------|----------------|----|   | => reduceResultLen
// |----------------|----------------|----|  /
//                   <=====blockStride====
//  ===============>
//  fullIterateCnt = 2
// 默认前提，每行长度16的倍数，已经按Block对齐
struct ReduceParam {
    uint64_t fullCol;
    uint64_t fullIterateCnt;
    uint64_t tailCol;
    uint64_t blockStride;
    uint64_t reduceResultLen;
    uint64_t BLOCK_LENS;
    uint64_t REPEAT_LENS;
    __aicore__ ReduceParam() {}
    __aicore__ ReduceParam(uint64_t row, uint64_t col, uint64_t byteSize)
    {
        BLOCK_LENS = DIVFLOOR(32, byteSize); // 32:1个block为32bytes
        REPEAT_LENS = DIVFLOOR(256, byteSize); // 256:一个repeat为256bytes
        // maximum 256 Bytes while repeated in level-0 instruction
        fullCol = REPEAT_LENS;
        fullIterateCnt = DIVFLOOR(col, fullCol);
        tailCol = col - fullCol * fullIterateCnt;
        // 32 Bytes in one block
        blockStride = col / BLOCK_LENS; // 32:1个block为32bytes
        reduceResultLen = row;
    }
    __aicore__ ~ReduceParam() {}
};

template <Wind::OpType opType, typename T>
__aicore__ inline void OpXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                            const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::Max<T>(dstLocal, src0Local, src1Local, calCount);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::Add<T>(dstLocal, src0Local, src1Local, calCount);
    } else if constexpr (opType == Wind::OpType::SUB) {
        AscendC::Sub<T>(dstLocal, src0Local, src1Local, calCount);
    } else if constexpr (opType == Wind::OpType::MUL) {
        AscendC::Mul<T>(dstLocal, src0Local, src1Local, calCount);
    } else if constexpr (opType == Wind::OpType::DIV) {
        AscendC::Div<T>(dstLocal, src0Local, src1Local, calCount);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void OpXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                            const LocalTensor<T>& src1Local, uint64_t mask,
                            const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::Max<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::Add<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::SUB) {
        AscendC::Sub<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::MUL) {
        AscendC::Mul<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::DIV) {
        AscendC::Div<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void OpXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                            const LocalTensor<T>& src1Local, uint64_t mask[],
                            const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::Max<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::Add<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::SUB) {
        AscendC::Sub<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::MUL) {
        AscendC::Mul<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    } else if constexpr (opType == Wind::OpType::DIV) {
        AscendC::Div<T, isSetMask>(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                     const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
                                     const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::WholeReduceMax<T, isSetMask>(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, ReduceOrder::ORDER_ONLY_VALUE);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::WholeReduceSum<T, isSetMask>(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                     const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride,
                                     const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::WholeReduceMax<T, isSetMask>(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, ReduceOrder::ORDER_ONLY_VALUE);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::WholeReduceSum<T, isSetMask>(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                     const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride,
                                     const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::BlockReduceMax<T, isSetMask>(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::BlockReduceSum<T, isSetMask>(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <Wind::OpType opType, typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceXx(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                     const int32_t repeat, const int32_t maskCount, const int32_t dstRepStride,
                                     const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if constexpr (opType == Wind::OpType::MAX) {
        AscendC::BlockReduceMax<T, isSetMask>(dstLocal, srcLocal, repeat, maskCount, dstRepStride, srcBlkStride, srcRepStride);
    } else if constexpr (opType == Wind::OpType::SUM) {
        AscendC::BlockReduceSum<T, isSetMask>(dstLocal, srcLocal, repeat, maskCount, dstRepStride, srcBlkStride, srcRepStride);
    }
}
}

#endif // ASCENDC_OP_UTILS_H
