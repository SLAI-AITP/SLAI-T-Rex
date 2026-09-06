/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_XX_ND_HPP
#define REDUCE_XX_ND_HPP
#include "reduce_xx_nd_v1.hpp"
#include "reduce_xx_nd_v2.hpp"
using namespace AscendC;
namespace Wind {
/* ND格式的ReduceMax/Sum，只适col为32B对齐场景，按列归约
 * */
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXx_vertically(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                           const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    if (rowLen == 1) {
        Adds(dstLocal, inputLocal, (T)0, colLen);
        pipe_barrier(PIPE_V);
        return;
    }
    Wind::OpXx<opType, T>(reduceLocal, inputLocal, inputLocal[rowLen / 2 * colLen], rowLen / 2 * colLen);
    pipe_barrier(PIPE_V);
    if (MOD(rowLen, 2) == 1) {
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[rowLen * colLen - colLen], colLen);
        pipe_barrier(PIPE_V);
    }
    uint64_t rowLenHalf = rowLen / 2;
    while (rowLenHalf > 1) {
        if (MOD(rowLenHalf, 2) == 1) {
            Wind::OpXx<opType, T>(reduceLocal, reduceLocal, reduceLocal[rowLenHalf * colLen - colLen], colLen);
            pipe_barrier(PIPE_V);
        }
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, reduceLocal[rowLenHalf / 2 * colLen], rowLenHalf / 2 * colLen);
        pipe_barrier(PIPE_V);
        rowLenHalf = rowLenHalf / 2;
    }
    Adds(dstLocal, reduceLocal, (T)0, colLen);
    pipe_barrier(PIPE_V);
}

/*
 * ND格式的ReduceMax/Sum，只适col为32B对齐场景，按行归约
 * inputLocal必须是ND格式， shape: rowLen * colLen，
 *     rowLen可以不用32B对齐，colLen必须是32B对齐
 * dstLocal存放Reduce结果，shape: rowLen * 1
 * reduceLocal需要的内存大小使用GetReduceXxNdTempBuffLen获取（op_tiling.h中，要在tiling中获取）
 * isPerfFirst为false时：
 *   reduceLocal是计算过程使用的临时缓存，最小缓存大小 rowLen * repeatLens（float为64，half为128）
 *   float类型colLen最大支持到255*64=2040； half类型的colLen最大支持到255*128=32640
 * isPerfFirst为true时：
 *   colLen可以比较大，没有限制
 *   reduceLocal需要的内存规则
 *   ┌───────┬────────────────────────────────┬────────────────────────────┬────────────────────────────┐
 *   │ 类型  │          colLen 条件           │   reduceLocal 最小元素数   │   reduceLocal 最小字节数   │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen == 8                    │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen ∈ {64,128,256,512,1024} │ ceil(rowLen×colLen/64)×8   │ ceil(rowLen×colLen/64)×32  │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ 其他                           │ reduceBlkNum×rowLen×8      │ reduceBlkNum×rowLen×32     │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 16                   │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 32                   │ ceil(rowLen/8)×128         │ ceil(rowLen/8)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 128                  │ ceil(rowLen/2)×128         │ ceil(rowLen/2)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ 其他                           │ ceil(colLen/128)×rowLen×16 │ ceil(colLen/128)×rowLen×32 │
 *   └───────┴────────────────────────────────┴────────────────────────────┴────────────────────────────┘
 *   其中 reduceBlkNum(浮点) = colLen/64 + isTail(tailColLen)
 *   tailColLen = colLen % 64
 *   isTail(t) = (t != 0 && t != 8) ? 1 : 0
 * */
template <Wind::OpType opType, typename T, bool isPerfFirst = true>
__aicore__ inline void ReduceXx_Nd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                   const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen)
{
    if constexpr (isPerfFirst) {
        Wind::ReduceXxImpl_Nd_V1<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen, colLen);
    } else {
        Wind::ReduceXxImpl_Nd_V2<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen, colLen);
    }
}

/*
 * ND格式的ReduceMax/Sum，支持col非32B对齐
 * inputLocal必须是ND格式， shape: rowLen * colLen，
 *     rowLen可以不用32B对齐，colLen必须是32B对齐，origColLen为实际参与Reduce计算的长度，可以不用32B对齐
 * dstLocal存放Reduce结果，shape: rowLen * 1
 * reduceLocal需要的内存大小使用GetReduceXxNdTempBuffLen获取（op_tiling.h中，要在tiling中获取）
 * isPerfFirst为false时：
 *   reduceLocal是计算过程使用的临时缓存，最小缓存大小 rowLen * repeatLens（float为64，half为128）
 *   float类型colLen最大支持到255*64=2040； half类型的colLen最大支持到255*128=32640
 * isPerfFirst为true时：
 *   colLen可以比较大，没有限制
 *   reduceLocal需要的内存规则
 *   ┌───────┬────────────────────────────────┬────────────────────────────┬────────────────────────────┐
 *   │ 类型  │          colLen 条件           │   reduceLocal 最小元素数   │   reduceLocal 最小字节数   │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen == 8                    │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen ∈ {64,128,256,512,1024} │ ceil(rowLen×colLen/64)×8   │ ceil(rowLen×colLen/64)×32  │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ 其他                           │ reduceBlkNum×rowLen×8      │ reduceBlkNum×rowLen×32     │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 16                   │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 32                   │ ceil(rowLen/8)×128         │ ceil(rowLen/8)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 128                  │ ceil(rowLen/2)×128         │ ceil(rowLen/2)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ 其他                           │ ceil(colLen/128)×rowLen×16 │ ceil(colLen/128)×rowLen×32 │
 *   └───────┴────────────────────────────────┴────────────────────────────┴────────────────────────────┘
 *   其中 reduceBlkNum(浮点) = colLen/64 + isTail(tailColLen)
 *   tailColLen = colLen % 64
 *   isTail(t) = (t != 0 && t != 8) ? 1 : 0
 * */
template <Wind::OpType opType, typename T, bool isPerfFirst = true>
__aicore__ inline void ReduceXx_Nd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                   const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen, uint64_t origColLen)
{
    if constexpr (isPerfFirst) {
        Wind::ReduceXxImpl_Nd_V1<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen, origColLen);
    } else {
        Wind::ReduceXxImpl_Nd_V2<opType, T>(dstLocal, inputLocal, reduceLocal, rowLen, colLen, origColLen);
    }
}
}
#endif // REDUCE_XX_ND_HPP
