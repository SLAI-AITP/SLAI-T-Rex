/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_SWI_GLU_LIMIT_FLOAT_HPP   // [Add by GTS-AILab] [feature GTSOps]
#define CANN_SWI_GLU_LIMIT_FLOAT_HPP   // [Add by GTS-AILab] [feature GTSOps]
#include "kernel_operator.h"
#include "kernel/entity/tqueue.h"

using namespace AscendC;

template<typename inType, typename outType, uint16_t bufferNum>
struct SwigluVector {
public:
    __aicore__ inline SwigluVector() {}
    __aicore__ inline ~SwigluVector() {}

protected:
    __aicore__ inline void InitUbBuffer(TPipe& pipe, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void Compute(uint64_t curTileLen, uint32_t curRowLen, uint32_t curColLen, uint64_t probsGmOffset);
    float beta = -1.0;
    float limit = 0.0f; // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪，由 GluSingle/GluSplit Init 透传

    TBuf<TPosition::VECCALC> tmpQueue;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<DTYPE_PROBS> probsGm;
    GlobalTensor<outType> cGm;
    GlobalTensor<outType> outprobsGm;

    LocalTensor<float> tempLocal;

    InQue<inType, bufferNum> inQueueA{EVENT_ID0};
    InQue<inType, bufferNum> inQueueB{EVENT_ID2};
    InQue<float, bufferNum> inQueueC{EVENT_ID4};
    OutQue<outType, bufferNum> outQueueC{EVENT_ID0};
    OutQue<outType, bufferNum> outQueueProbs{EVENT_ID1};
};

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::InitUbBuffer(TPipe& pipe, uint64_t tileLength) {
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    tempLocal = tmpQueue.Get<float>();
    Duplicate(tempLocal, (float)(1.0), tileLength);

    inQueueA.Init(pipe, tileLength);
    inQueueB.Init(pipe, tileLength);
    outQueueC.Init(pipe, tileLength);
}

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType>& aLocal = inQueueA.template DeQue<inType>();
    LocalTensor<outType>& cLocal = outQueueC.template AllocTensor<outType>();

    // [Add by GTS-AILab] [feature GTSOps] gate clamp max=limit 在 silu 之前（pre-silu），对齐线上 silu(clamp(gate))。
    // 之后 silu 全程使用 clamped gate；limit<=0 不裁剪。
    if (limit > 0.0f) {
        Mins(aLocal, aLocal, (inType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }

    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, (outType)(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    Div(cLocal, tempLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, aLocal, curTileLen); // cLocal = silu(clamped gate)
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal);

    LocalTensor<inType>& bLocal = inQueueB.template DeQue<inType>();
    // [Add by GTS-AILab] [feature GTSOps] up(=fp32 bLocal) clamp 到 [-limit, limit]，limit<=0 不裁剪
    if (limit > 0.0f) {
        Maxs(bLocal, bLocal, (inType)(-limit), curTileLen);
        pipe_barrier(PIPE_V);
        Mins(bLocal, bLocal, (inType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }
    Mul(cLocal, cLocal, bLocal, curTileLen);
    inQueueB.template FreeTensor(bLocal);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue(cLocal);
    // free input tensors for reuse
}

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::Compute(uint64_t curTileLen, uint32_t curRowLen, uint32_t curColLen, uint64_t probsGmOffset)
{
    LocalTensor<inType>& aLocal = inQueueA.template DeQue<inType>();
    LocalTensor<outType>& cLocal = outQueueC.template AllocTensor<outType>();

    // [Add by GTS-AILab] [feature GTSOps] gate clamp max=limit 在 silu 之前（pre-silu），对齐线上 silu(clamp(gate))。
    // 之后 silu 全程使用 clamped gate；limit<=0 不裁剪。
    if (limit > 0.0f) {
        Mins(aLocal, aLocal, (inType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }

    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, (outType)(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    Div(cLocal, tempLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, aLocal, curTileLen); // cLocal = silu(clamped gate)
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal);

    LocalTensor<inType>& bLocal = inQueueB.template DeQue<inType>();
    // [Add by GTS-AILab] [feature GTSOps] up(=fp32 bLocal) clamp 到 [-limit, limit]，limit<=0 不裁剪
    if (limit > 0.0f) {
        Maxs(bLocal, bLocal, (inType)(-limit), curTileLen);
        pipe_barrier(PIPE_V);
        Mins(bLocal, bLocal, (inType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }
    Mul(cLocal, cLocal, bLocal, curTileLen);

    LocalTensor<DTYPE_PROBS> probsIn = tempLocal.template ReinterpretCast<DTYPE_PROBS>();
    DataCopy(probsIn, probsGm[probsGmOffset], (curRowLen + 31U) / 32U * 32U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // 2. Cast 到 FP32（仍占用 outTempLocal 的前 curRowLen 个 float）
    if constexpr (!IsSameType<DTYPE_PROBS, outType>::value) {
        LocalTensor<outType> probsFP32 = tempLocal.template ReinterpretCast<outType>();
        Cast(probsFP32, probsIn, RoundMode::CAST_NONE, curRowLen);
        pipe_barrier(PIPE_V);

        // 3. 广播乘：cLocal_ 的每一行乘以对应的 probs
        for (uint32_t r = 0; r < curRowLen; ++r) {
            Muls(cLocal[r * curColLen], cLocal[r * curColLen], probsFP32.GetValue(r), curColLen);
        }
    } else {
        // 3. 广播乘：cLocal_ 的每一行乘以对应的 probs
        for (uint32_t r = 0; r < curRowLen; ++r) {
            Muls(cLocal[r * curColLen], cLocal[r * curColLen], probsIn.GetValue(r), curColLen);
        }
    }

    inQueueB.template FreeTensor(bLocal);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue(cLocal);
    // free input tensors for reuse
}


#endif // CANN_SWI_GLU_LIMIT_FLOAT_HPP   // [Add by GTS-AILab] [feature GTSOps]