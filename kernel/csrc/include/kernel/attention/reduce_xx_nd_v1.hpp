/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_XX_ND_V1_HPP
#define REDUCE_XX_ND_V1_HPP
#include "ascendc_op_utils.hpp"
using namespace AscendC;
namespace Wind {
// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float, coleLen = 16
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_16(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                              const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    // 16->1
    uint32_t softmaxRepTimes = DIVCEIL(rowLen * colLen, HALF_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, inputLocal, softmaxRepTimes, HALF_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 按垂直方向折半Max操作
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxStep(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                     uint64_t rowLen, uint64_t colLen, uint64_t reduceLen)
{
    int32_t softmaxRepTimes = DIVCEIL(rowLen * reduceLen, HALF_REPEAT_LENS);
    Wind::OpXx<opType, T>(dstLocal, srcLocal, srcLocal[16],
        HALF_REPEAT_LENS, softmaxRepTimes, {1, 2, 2, 8, 16, 16});
    pipe_barrier(PIPE_V);
}

// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float, coleLen = 32
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_32(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                              const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    // 32 -> 16  max
    ReduceXxStep<opType, T>(reduceLocal, inputLocal, rowLen, colLen, 16);
    // 16->1
    int32_t softmaxRepTimes = DIVCEIL(rowLen * 16, HALF_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, reduceLocal, softmaxRepTimes, HALF_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float, coleLen = 128
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_128(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                               const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    // 128->64
    ReduceXxStep<opType, T>(reduceLocal, inputLocal, rowLen, colLen, 64);
    // 64->32
    ReduceXxStep<opType, T>(reduceLocal, reduceLocal, rowLen, colLen, 32);
    // 64->16
    ReduceXxStep<opType, T>(reduceLocal, reduceLocal, rowLen, colLen, 16);
    // 16-> 1
    int32_t softmaxRepTimes = DIVCEIL(rowLen * 16, HALF_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, reduceLocal, softmaxRepTimes, HALF_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 处理切片之后的tailColLen > 16部分
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_Slice_Tail(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                      uint64_t rowLen, uint64_t colLen,
                                                      uint64_t tailColLen, uint64_t &reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * HALF_BLOCK_LENS);
    uint64_t reduceOffset = (uint64_t)(reduceBlkNum * reduceOneBlkSize);
    uint64_t inputOffset = (uint64_t)(reduceBlkNum * HALF_REPEAT_LENS);
    int32_t srcRowStride = colLen / HALF_BLOCK_LENS;

    // 初始化为0
    if constexpr(opType == Wind::OpType::MAX) {
        Duplicate(reduceLocal[reduceOffset], (T)HALF_NEG_INF, reduceOneBlkSize);
    } else if (opType == Wind::OpType::SUM) {
        Duplicate(reduceLocal[reduceOffset], (T)0, reduceOneBlkSize);
    }
    pipe_barrier(PIPE_V);
    Wind::BlockReduceXx<opType, T>(reduceLocal[reduceOffset], inputLocal[inputOffset],
                   rowLen, tailColLen, 2, 1, srcRowStride);

    pipe_barrier(PIPE_V);
    reduceBlkNum++;
}

// 使用折半的方式reduce
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_HalfReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& reduceLocal,
                                                      uint64_t rowLen, uint64_t reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * HALF_BLOCK_LENS);
    // reduceLocal中有reduceBlkNum个blk，折半到1个blk
    while (reduceBlkNum > 1) {
        uint64_t offBlkNum = reduceBlkNum / 2;
        if (reduceBlkNum % 2 == 0) {
            reduceBlkNum = offBlkNum;
        } else {
            reduceBlkNum = (uint64_t)(offBlkNum + 1);
        }
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, reduceLocal[reduceBlkNum * reduceOneBlkSize], offBlkNum * reduceOneBlkSize);
        pipe_barrier(PIPE_V);
    }

    uint64_t mask[2];
    mask[0]=0x00FF00FF00FF00FF;
    mask[1]=0x00FF00FF00FF00FF;

    // 16->1
    int32_t softmaxRepTimes = DIVCEIL(reduceOneBlkSize, HALF_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, reduceLocal, softmaxRepTimes, mask, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float, coleLen > 32
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_Comm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                                const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    uint64_t iterCnt = colLen / HALF_REPEAT_LENS; // 先根据128切分成几段
    uint32_t tailColLen = colLen - iterCnt * HALF_REPEAT_LENS;
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * HALF_BLOCK_LENS);
    uint64_t reduceBlkNum = iterCnt; // reduce中有几个blk，开始不包含tail
    for (int64_t rowIdex = 0; rowIdex < rowLen; rowIdex++) {
        Wind::BlockReduceXx<opType, T>(reduceLocal[rowIdex * HALF_BLOCK_LENS], inputLocal[rowIdex * colLen],
                       iterCnt, HALF_REPEAT_LENS, rowLen * 2, 1, 8);
        pipe_barrier(PIPE_V);
    }

    if (tailColLen > 0) {
        Wind::ReduceXxHalfImpl_Nd_Slice_Tail<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    }

    Wind::ReduceXxHalfImpl_Nd_HalfReduce<opType, T>(dstLocal, reduceLocal, rowLen, reduceBlkNum);
}

// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float coleLen = 128
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxHalfImpl_Nd_V1(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                              const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    if (colLen == 128) {
        Wind::ReduceXxHalfImpl_Nd_128<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else if (colLen == 32) {
        Wind::ReduceXxHalfImpl_Nd_32<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else if (colLen == 16) {
        Wind::ReduceXxHalfImpl_Nd_16<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else {
        Wind::ReduceXxHalfImpl_Nd_Comm<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    }
}

template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_8(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                              const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    // 8->1
    int32_t softmaxRepTimes = DIVCEIL(rowLen * colLen, FLOAT_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, inputLocal, softmaxRepTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 使用折半的方式reduce
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_HalfReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& reduceLocal,
                                                    uint64_t rowLen, uint64_t reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    // reduceLocal中有reduceBlkNum个blk，折半到1个blk
    while (reduceBlkNum > 1) {
        uint64_t offBlkNum = reduceBlkNum / 2;
        if (reduceBlkNum % 2 == 0) {
            reduceBlkNum = offBlkNum;
        } else {
            reduceBlkNum = (uint64_t)(offBlkNum + 1);
        }
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, reduceLocal[reduceBlkNum * reduceOneBlkSize], offBlkNum * reduceOneBlkSize);
        pipe_barrier(PIPE_V);
    }

    // 8->1
    int32_t softmaxRepTimes = DIVCEIL(reduceOneBlkSize, FLOAT_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, reduceLocal, softmaxRepTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

// 处理切片之后的tailColLen = 16部分
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Slice_Tail_16(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                          uint64_t rowLen, uint64_t colLen,
                                                          uint64_t tailColLen, uint64_t &reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    uint64_t reduceOffset = (uint64_t)(reduceBlkNum * reduceOneBlkSize);
    uint64_t inputOffset = (uint64_t)(reduceBlkNum * FLOAT_REPEAT_LENS);
    int32_t srcRowStride = colLen / FLOAT_BLOCK_LENS;
    reduceBlkNum++;
    if ((srcRowStride * 8) > 255) {
        // srcRptStride超出最大范围，只能一行一行处理，每一行作为一个repeat
        for (int64_t rowIdex = 0; rowIdex < rowLen; rowIdex++) {
            Wind::OpXx<opType, T>(reduceLocal[reduceOffset], inputLocal[inputOffset], inputLocal[inputOffset + 8], 8);
            reduceOffset += FLOAT_BLOCK_LENS;
            inputOffset += colLen;
            pipe_barrier(PIPE_V);
        }
        return;
    }

    // 可以多行合并成一个repeat
    // 8和16特殊处理
    uint8_t srcRptStride = (uint8_t)(srcRowStride * 8);
    uint32_t fullRepTimes = reduceOneBlkSize / FLOAT_REPEAT_LENS;
    uint32_t tailLen = reduceOneBlkSize - fullRepTimes * FLOAT_REPEAT_LENS;
    if (fullRepTimes > 0) {
        Wind::OpXx<opType, T>(reduceLocal[reduceOffset], inputLocal[inputOffset], inputLocal[inputOffset + 8],
            FLOAT_REPEAT_LENS, fullRepTimes,
            {1, (uint8_t)srcRowStride, (uint8_t)srcRowStride, 8, srcRptStride, srcRptStride});
        pipe_barrier(PIPE_V);
    }
    if (tailLen > 0) {
        Wind::OpXx<opType, T>(reduceLocal[reduceOffset + fullRepTimes * FLOAT_REPEAT_LENS],
            inputLocal[inputOffset + fullRepTimes * 8 * colLen],
            inputLocal[inputOffset + 8 + fullRepTimes * 8 * colLen],
            tailLen, 1, {1, (uint8_t)srcRowStride, (uint8_t)srcRowStride, 8, srcRptStride, srcRptStride});
        pipe_barrier(PIPE_V);
    }
}

// 处理切片之后的tailColLen = 8部分
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Slice_Tail_8(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                         uint64_t rowLen, uint64_t colLen,
                                                         uint64_t tailColLen, uint64_t &reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    uint64_t reduceOffset = (uint64_t)(reduceBlkNum * reduceOneBlkSize);
    uint64_t inputOffset = (uint64_t)(reduceBlkNum * FLOAT_REPEAT_LENS);
    int32_t srcRowStride = colLen / FLOAT_BLOCK_LENS;
    if ((srcRowStride * 8) > 255) {
        // srcRptStride超出最大范围，只能一行一行处理，每一行作为一个repeat
        for (uint64_t rowIdex = 0; rowIdex < rowLen; rowIdex++) {
            Wind::OpXx<opType, T>(reduceLocal[rowIdex * FLOAT_BLOCK_LENS],
                                      reduceLocal[rowIdex * FLOAT_BLOCK_LENS],
                                      inputLocal[rowIdex * colLen + inputOffset],
                                      tailColLen, 1, {1, 1, 1, 8, 8, 8});
        }
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[inputOffset],
            tailColLen, rowLen, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        return;
    }

    // 可以多行合并成一个repeat
    // 8和16特殊处理
    uint8_t srcRptStride = (uint8_t)(srcRowStride * 8);
    uint32_t fullRepTimes = reduceOneBlkSize / FLOAT_REPEAT_LENS;
    uint32_t tailLen = reduceOneBlkSize - fullRepTimes * FLOAT_REPEAT_LENS;
    if (fullRepTimes > 0) {
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[inputOffset],
            FLOAT_REPEAT_LENS, fullRepTimes, {1, 1, (uint8_t)srcRowStride, 8, 8, srcRptStride});
        pipe_barrier(PIPE_V);
    }
    if (tailLen > 0) {
        Wind::OpXx<opType, T>(reduceLocal[fullRepTimes * FLOAT_REPEAT_LENS],
            reduceLocal[fullRepTimes * FLOAT_REPEAT_LENS],
            inputLocal[inputOffset + fullRepTimes * 8 * colLen],
            tailLen, 1, {1, 1, (uint8_t)srcRowStride, 8, 8, srcRptStride});
        pipe_barrier(PIPE_V);
    }
    return; // 8的时候直接叠加，不新增blk
}

// 处理切片之后的tailColLen > 16部分
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Slice_Tail_Lager16(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                               uint64_t rowLen, uint64_t colLen,
                                                               uint64_t tailColLen, uint64_t &reduceBlkNum)
{
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    uint64_t reduceOffset = (uint64_t)(reduceBlkNum * reduceOneBlkSize);
    uint64_t inputOffset = (uint64_t)(reduceBlkNum * FLOAT_REPEAT_LENS);
    int32_t srcRowStride = colLen / FLOAT_BLOCK_LENS;
    // 初始化为0
    if constexpr(opType == Wind::OpType::MAX)
    {
        Duplicate(reduceLocal[reduceOffset], (T)FLOAT_NEG_INF, reduceOneBlkSize);
    } else if constexpr(opType == Wind::OpType::SUM) {
        Duplicate(reduceLocal[reduceOffset], (T)0, reduceOneBlkSize);
    }
    pipe_barrier(PIPE_V);
    Wind::BlockReduceXx<opType, T>(reduceLocal[reduceOffset], inputLocal[inputOffset],
                   rowLen, tailColLen, 1, 1, srcRowStride);
    pipe_barrier(PIPE_V);
    reduceBlkNum++;
}

// 处理切片之后的tailColLen部分
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Slice_Tail(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                       uint64_t rowLen, uint64_t colLen,
                                                       uint64_t tailColLen, uint64_t &reduceBlkNum)
{
    if (tailColLen == 8) {
        Wind::ReduceXxFloatImpl_Nd_Slice_Tail_8<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    } else if (tailColLen == 16) {
        Wind::ReduceXxFloatImpl_Nd_Slice_Tail_16<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    } else {
        Wind::ReduceXxFloatImpl_Nd_Slice_Tail_Lager16<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    }
}

template <Wind::OpType opType, typename T>
__aicore__ inline uint64_t ReduceXxFloatImpl_Nd_Slice_ByRow(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                            uint64_t rowLen, uint64_t colLen)
{
    uint64_t iterCnt = colLen / FLOAT_REPEAT_LENS; // 先根据64切分成几段
    uint32_t tailColLen = colLen - iterCnt * FLOAT_REPEAT_LENS;
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    uint64_t reduceBlkNum = iterCnt; // reduce中有几个blk，开始不包含tail
    for (int64_t rowIdex = 0; rowIdex < rowLen; rowIdex++) {
        Wind::BlockReduceXx<opType, T>(reduceLocal[rowIdex * FLOAT_BLOCK_LENS], inputLocal[rowIdex * colLen],
                       iterCnt, FLOAT_REPEAT_LENS, rowLen, 1, 8);
        pipe_barrier(PIPE_V);
    }
    if (tailColLen > 0) {
        Wind::ReduceXxFloatImpl_Nd_Slice_Tail<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    }
    return reduceBlkNum;
}

template <Wind::OpType opType, typename T>
__aicore__ inline uint64_t ReduceXxFloatImpl_Nd_Slice(const LocalTensor<T>& inputLocal, const LocalTensor<T>& reduceLocal,
                                                      uint64_t rowLen, uint64_t colLen)
{
    uint64_t iterCnt = colLen / FLOAT_REPEAT_LENS; // 先根据64切分成几段
    uint32_t tailColLen = colLen - iterCnt * FLOAT_REPEAT_LENS;
    int32_t srcRepStride = colLen / FLOAT_BLOCK_LENS;
    uint64_t reduceOneBlkSize = (uint64_t)(rowLen * FLOAT_BLOCK_LENS);
    uint64_t reduceBlkNum = iterCnt; // reduce中有几个blk，开始不包含tail

    for (int64_t loop = 0; loop < iterCnt; loop++) {
        Wind::BlockReduceXx<opType, T>(reduceLocal[loop * reduceOneBlkSize], inputLocal[loop * FLOAT_REPEAT_LENS],
                       rowLen, FLOAT_REPEAT_LENS, 1, 1, srcRepStride);
        pipe_barrier(PIPE_V);
    }

    if (tailColLen > 0) {
        Wind::ReduceXxFloatImpl_Nd_Slice_Tail<opType, T>(inputLocal, reduceLocal, rowLen, colLen, tailColLen, reduceBlkNum);
    }
    return reduceBlkNum;
}

// 对colLen=8不是很友好，建议单独处理
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Comm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                                 const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    uint64_t iterCnt = colLen / FLOAT_REPEAT_LENS; // 先根据64切分成几段
    uint64_t reduceBlkNum = 0;
    if (rowLen >= iterCnt) { // tailColLen的指令数量一样，所以比较前面的
        reduceBlkNum = Wind::ReduceXxFloatImpl_Nd_Slice<opType, T>(inputLocal, reduceLocal, rowLen, colLen);
    } else {
        reduceBlkNum = Wind::ReduceXxFloatImpl_Nd_Slice_ByRow<opType, T>(inputLocal, reduceLocal, rowLen, colLen);
    }

    Wind::ReduceXxFloatImpl_HalfReduce<opType, T>(dstLocal, reduceLocal, rowLen, reduceBlkNum);
}

// 对colLen=64/128/256/512/1024的reduce，高性能
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_Hp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                               const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    uint64_t reduceBlkNum = colLen / FLOAT_REPEAT_LENS; // 进行过一次reduce之后，col上剩余的blk（32B）个数
    uint32_t softmaxRepTimes = DIVCEIL(rowLen * colLen, FLOAT_REPEAT_LENS);
    uint32_t repTimes = softmaxRepTimes > 255 ? 255 : softmaxRepTimes; // 255:repTimes最大值
    uint32_t tailRepTimes = softmaxRepTimes - repTimes;
    if (repTimes <= 255) { // 255:repTimes最大值
        Wind::BlockReduceXx<opType, T>(reduceLocal, inputLocal, repTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }
    if (tailRepTimes > 0) {
        Wind::BlockReduceXx<opType, T>(reduceLocal[repTimes * FLOAT_BLOCK_LENS], inputLocal[repTimes * FLOAT_REPEAT_LENS],
                       tailRepTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }

    if (reduceBlkNum % 8 == 0) { // 8:1个repeat有8个block
        uint64_t newColLen = (uint64_t)(reduceBlkNum * FLOAT_BLOCK_LENS);
        reduceBlkNum = newColLen / FLOAT_REPEAT_LENS; // 进行过一次reduce之后，col上剩余的blk（32B）个数
        // 再进行一次reduce， 这一次应该不会超过255
        softmaxRepTimes = DIVCEIL(rowLen * newColLen, FLOAT_REPEAT_LENS);
        Wind::BlockReduceXx<opType, T>(reduceLocal, reduceLocal, softmaxRepTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }

    // 折半，使reduceBlkNum变为1
    while (reduceBlkNum > 1) {
        reduceBlkNum = reduceBlkNum / 2; // 2:折半
        softmaxRepTimes = DIVCEIL(rowLen * reduceBlkNum * FLOAT_BLOCK_LENS, FLOAT_REPEAT_LENS);
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, reduceLocal[FLOAT_BLOCK_LENS],
            FLOAT_REPEAT_LENS, softmaxRepTimes, {1, 2, 2, 8, 16, 16});
        pipe_barrier(PIPE_V);
    }

    // 8->1
    softmaxRepTimes = DIVCEIL(rowLen * FLOAT_BLOCK_LENS, FLOAT_REPEAT_LENS);
    Wind::BlockReduceXx<opType, T>(dstLocal, reduceLocal, softmaxRepTimes, FLOAT_REPEAT_LENS, 1, 1, 8);
    pipe_barrier(PIPE_V);
}

template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxFloatImpl_Nd_V1(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                               const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    if (colLen == 64 || colLen == 128 || colLen == 256 || colLen == 512 || colLen == 1024) {
        Wind::ReduceXxFloatImpl_Nd_Hp<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else if (colLen == 8) {
        Wind::ReduceXxFloatImpl_Nd_8<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else {
        Wind::ReduceXxFloatImpl_Nd_Comm<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    }
}

// 将ND格式的矩阵inputLocal（rowLen * colLen）归约Max，归约方向：按行归约。目前只支持float coleLen = 128
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxImpl_Nd_V1(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                          const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen, uint64_t origColLen)
{
    if constexpr (IsSameType<T, float>::value) {
        Wind::ReduceXxFloatImpl_Nd_V1<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    } else {
        Wind::ReduceXxHalfImpl_Nd_V1<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen);
    }
}
}
#endif // REDUCE_XX_ND_V1_HPP
