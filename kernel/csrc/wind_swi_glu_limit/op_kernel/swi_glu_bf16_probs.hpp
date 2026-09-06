/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_SWI_GLU_LIMIT_BF16_PROBS_HPP   // [Add by GTS-AILab] [feature GTSOps]
#define CANN_SWI_GLU_LIMIT_BF16_PROBS_HPP   // [Add by GTS-AILab] [feature GTSOps]
#include "kernel_operator.h"
#include "kernel/entity/tqueue.h"

#define TEMPLATE_DECLARE \
template<typename inType, typename calcType, typename outType, uint16_t bufferNum, bool highPrecision>
#define TEMPLATE_ARGS inType, calcType, outType, bufferNum, highPrecision

TEMPLATE_DECLARE
struct SwigluVectorBF16Probs {
public:
    __aicore__ inline SwigluVectorBF16Probs() {}
    __aicore__ inline ~SwigluVectorBF16Probs() {}

protected:
    __aicore__ inline void InitUbBuffer(TPipe& pipe, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void Compute(uint64_t curTileLen, uint32_t curRowLen, uint32_t curColLen, uint64_t probsGmOffset);
    float beta = -1.0;
    float limit = 0.0f; // [Add by GTS-AILab] [feature GTSOps] gate/up clamp 阈值，0=不裁剪，由 GluSingle/GluSplit Init 透传

    TBuf<TPosition::VECCALC> inputTempBuffer; // a/b复用

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<DTYPE_PROBS> probsGm;
    GlobalTensor<outType> cGm;
    GlobalTensor<outType> outprobsGm;

    LocalTensor<calcType> inputTempLocal;
    LocalTensor<calcType> outTempLocal;

    InQue<inType, bufferNum> inQueueA{EVENT_ID0};
    InQue<inType, bufferNum> inQueueB{EVENT_ID2};
    InQue<float, bufferNum> inQueueC{EVENT_ID4};
    OutQue<outType, 1> outQueueC{EVENT_ID0};
    OutQue<outType, 1> outQueueProbs{EVENT_ID1};
};

TEMPLATE_DECLARE
__aicore__ inline void SwigluVectorBF16Probs<TEMPLATE_ARGS>::InitUbBuffer(
        TPipe& pipe, uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes

    pipe.InitBuffer(inputTempBuffer, tileLength * sizeof(calcType) * 2U);

    inQueueA.Init(pipe, tileLength);
    inQueueB.Init(pipe, tileLength);
    outQueueC.Init(pipe, tileLength);
    
    inQueueC.Init(pipe, tileLength);
    outQueueProbs.Init(pipe, tileLength);

    inputTempLocal = inputTempBuffer.Get<calcType>();
    outTempLocal = inputTempLocal[tileLength];
}

TEMPLATE_DECLARE
__aicore__ inline void SwigluVectorBF16Probs<TEMPLATE_ARGS>::Compute(
        uint64_t curTileLen, uint32_t curRowLen, uint32_t curColLen, uint64_t probsGmOffset)
{
    // 增加输入a：BF16->FP32的CAST
    LocalTensor<calcType>& aLocal = inputTempLocal;
    LocalTensor<calcType>& cLocal = outTempLocal;

    LocalTensor<inType>& aLocal_ = inQueueA.template DeQue<inType>();
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    // [Add by GTS-AILab] [feature GTSOps] gate clamp max=limit 在 silu 之前（pre-silu），对齐线上 silu(clamp(gate))。
    // 之后 silu 全程使用 clamped gate；limit<=0 不裁剪。
    if (limit > 0.0f) {
        Mins(aLocal, aLocal, (calcType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }

    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, calcType(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    Div(cLocal, aLocal, cLocal, curTileLen);  // cLocal = silu(clamped gate)
    pipe_barrier(PIPE_V);

    if constexpr (!highPrecision
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                && !std::is_same_v<inType, bfloat16_t>
#endif
        ) {
        LocalTensor<outType> cTempLocal = outTempLocal.template ReinterpretCast<outType>();

        // 增加输出FP32->BF16的CAST
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        Cast(cTempLocal, cLocal, RoundMode::CAST_RINT, curTileLen);
#else
        Cast(cTempLocal, cLocal, RoundMode::CAST_NONE, curTileLen); // PIPE_V: 写cLocal_ r-cLocal
#endif

        LocalTensor<inType>& bLocal_ = inQueueB.template DeQue<inType>();
        LocalTensor<outType>& cLocal_ = outQueueC.template AllocTensor<outType>();
        pipe_barrier(PIPE_V);

        // [Add by GTS-AILab] [feature GTSOps] up(=bf16 bLocal_) clamp 到 [-limit, limit]，limit<=0 不裁剪
        if (limit > 0.0f) {
            Maxs(bLocal_, bLocal_, (inType)(-limit), curTileLen);
            pipe_barrier(PIPE_V);
            Mins(bLocal_, bLocal_, (inType)limit, curTileLen);
            pipe_barrier(PIPE_V);
        }

        Mul(cLocal_, cTempLocal, bLocal_, curTileLen);

        // free input tensors for reuse
        inQueueB.template FreeTensor(bLocal_);

        // enque the output tensor to VECOUT queue
        outQueueC.template EnQue<outType>(cLocal_); // PIPE_V: set V->MTE3
        return;
    }

    LocalTensor<calcType>& bLocal = inputTempLocal;
    LocalTensor<inType>& bLocal_ = inQueueB.template DeQue<inType>();

    // 增加输入b：BF16->FP32的CAST
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
    // free input tensors for reuse
    inQueueB.template FreeTensor(bLocal_);
    pipe_barrier(PIPE_V);

    // [Add by GTS-AILab] [feature GTSOps] up(=fp32 bLocal) clamp 到 [-limit, limit]，limit<=0 不裁剪
    if (limit > 0.0f) {
        Maxs(bLocal, bLocal, (calcType)(-limit), curTileLen);
        pipe_barrier(PIPE_V);
        Mins(bLocal, bLocal, (calcType)limit, curTileLen);
        pipe_barrier(PIPE_V);
    }

    Mul(cLocal, cLocal, bLocal, curTileLen); // 写
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    LocalTensor<float> probsIn = this->inQueueC.template DeQue<float>();
    LocalTensor<calcType> probsOut = inputTempLocal.template ReinterpretCast<calcType>();
    // 3. 广播乘：cLocal_ 的每一行乘以对应的 probs
    for (uint32_t r = 0; r < curRowLen; ++r) {
        float probs = probsIn.GetValue(r);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Muls(probsOut[r * curColLen], cLocal[r * curColLen], probs, curColLen);
    }
    pipe_barrier(PIPE_V);
    this->inQueueC.FreeTensor(probsIn);

    // 增加输出FP32->BF16的CAST
    LocalTensor<outType>& cLocal1_ = outQueueC.template AllocTensor<outType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    Cast(cLocal1_, cLocal, RoundMode::CAST_RINT, curTileLen);
#else
    Cast(cLocal1_, cLocal, RoundMode::CAST_NONE, curTileLen); // PIPE_V: 写cLocal_ r-cLocal
#endif
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal1_); // PIPE_V: set V->MTE3

    LocalTensor<outType>& cLocal_ = outQueueProbs.template AllocTensor<outType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    Cast(cLocal_, probsOut, RoundMode::CAST_RINT, curTileLen);
#else
    Cast(cLocal_, probsOut, RoundMode::CAST_NONE, curTileLen); // PIPE_V: 写cLocal_ r-cLocal
#endif
    // enque the output tensor to VECOUT queue
    outQueueProbs.template EnQue<outType>(cLocal_); // PIPE_V: set V->MTE3
}
#endif  // CANN_SWI_GLU_LIMIT_BF16_PROBS_HPP   // [Add by GTS-AILab] [feature GTSOps]
