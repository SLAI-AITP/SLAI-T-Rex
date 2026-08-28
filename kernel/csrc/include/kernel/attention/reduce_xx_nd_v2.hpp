/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_XX_ND_V2_HPP
#define REDUCE_XX_ND_V2_HPP
#include "ascendc_op_utils.hpp"
using namespace AscendC;
namespace Wind {
template <Wind::OpType opType, typename T>
__aicore__ inline void ReduceXxImpl_Nd_V2(const LocalTensor<T>& dstLocal, const LocalTensor<T>& inputLocal,
                                          const LocalTensor<T>& reduceLocal, uint64_t rowLen, uint64_t colLen, uint64_t origColLen)
{
    ReduceParam reduceParam(rowLen, colLen, sizeof(T));
    if (colLen <= reduceParam.REPEAT_LENS) {
        Wind::WholeReduceXx<opType, T>(dstLocal, inputLocal, origColLen, rowLen,
                                           1, 1, reduceParam.blockStride);
        pipe_barrier(PIPE_V);
        return;
    }

    if (colLen <= (reduceParam.REPEAT_LENS * 2)) {
        DataCopy(reduceLocal, inputLocal,
                 {static_cast<uint16_t>(rowLen),
                  static_cast<uint16_t>(reduceParam.REPEAT_LENS / reduceParam.BLOCK_LENS),
                  static_cast<uint16_t>((colLen - reduceParam.REPEAT_LENS) / reduceParam.BLOCK_LENS),
                  0});
        pipe_barrier(PIPE_V);

        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[reduceParam.REPEAT_LENS],
                                  origColLen - reduceParam.REPEAT_LENS, rowLen,
                                {1, 1, 1, 8, 8, static_cast<uint8_t>(reduceParam.blockStride)});
        pipe_barrier(PIPE_V);

        Wind::WholeReduceXx<opType, T>(dstLocal, reduceLocal, reduceParam.REPEAT_LENS,
                                               rowLen, 1, 1, 8);
        pipe_barrier(PIPE_V);
        return;
    }

    // 前面2个repeat加在一起
    Wind::OpXx<opType, T>(reduceLocal, inputLocal, inputLocal[reduceParam.REPEAT_LENS],
                            reduceParam.REPEAT_LENS, rowLen,
                            {1, 1, 1, 8, static_cast<uint8_t>(reduceParam.blockStride), static_cast<uint8_t>(reduceParam.blockStride)});
    pipe_barrier(PIPE_V);

    for (uint64_t i = 2; i < reduceParam.fullIterateCnt; i++) {
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[i * reduceParam.REPEAT_LENS],
                                reduceParam.REPEAT_LENS, rowLen,
                                {1, 1, 1, 8, 8, static_cast<uint8_t>(reduceParam.blockStride)});
        pipe_barrier(PIPE_V);
    }

    if (reduceParam.tailCol > 0) {
        uint64_t origTailCol = origColLen - reduceParam.fullIterateCnt * reduceParam.REPEAT_LENS;
        Wind::OpXx<opType, T>(reduceLocal, reduceLocal, inputLocal[reduceParam.fullIterateCnt * reduceParam.REPEAT_LENS],
                                  origTailCol, rowLen,
                                {1, 1, 1, 8, 8, static_cast<uint8_t>(reduceParam.blockStride)});
        pipe_barrier(PIPE_V);
    }

    Wind::WholeReduceXx<opType, T>(dstLocal, reduceLocal, reduceParam.REPEAT_LENS,
                                       rowLen, 1, 1, 8);
    pipe_barrier(PIPE_V);
}
}
#endif // REDUCE_XX_ND_V2_HPP
