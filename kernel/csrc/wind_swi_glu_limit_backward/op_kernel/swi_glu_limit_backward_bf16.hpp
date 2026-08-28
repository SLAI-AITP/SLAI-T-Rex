/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_SWI_GLU_LIMIT_BACKWARD_BF16_HPP
#define CANN_SWI_GLU_LIMIT_BACKWARD_BF16_HPP

#include "kernel_operator.h"

using namespace AscendC;

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwiGluLimitBackwardBF16 {
public:
    __aicore__ inline SwiGluLimitBackwardBF16() {}

protected:
    __aicore__ inline void InitUbBuffer(TPipe& pipe, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void ToOutQue(uint64_t tileLength, LocalTensor<calcType>& mLocal, LocalTensor<calcType>& bLocal);
    __aicore__ inline void CalcSig(uint64_t tileLength, LocalTensor<calcType>& aLocal);

    calcType beta = -1.0;
    float limit = 0.0f; // swiglu_limit 反向 clamp 阈值，0=不裁剪，由 GluLimitBackwardSplit/Single Init 透传
    InQue<inType, bufferNum> inQueueA{EVENT_ID0};
    InQue<inType, bufferNum> inQueueB{EVENT_ID2};
    InQue<inType, bufferNum> inQueueL{EVENT_ID4};
    OutQue<outType, bufferNum> outQueueM{EVENT_ID0};
    OutQue<outType, bufferNum> outQueueN{EVENT_ID2};
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    TBuf<TPosition::VECCALC> maskQueue; // swiglu_limit: CompareScalar 输出的 uint8 选择掩码
    TBuf<TPosition::VECCALC> origAQueue; // pre-silu: 保存原始 a(fp32)，M 分支即时算 [a<=limit]，避免 mask 跨 N 分支被串扰
    TBuf<TPosition::VECCALC> bF32Queue; // swiglu_limit: up(b) fp32 暂存（一次加载供 N/M 共用）
    TBuf<TPosition::VECCALC> zeroQueue; // swiglu_limit: Select 的“否则取0”源
    LocalTensor<calcType> tempLocal;
    LocalTensor<calcType> sigLocal;
    LocalTensor<uint8_t>  maskLocal; // uint8 选择掩码
    LocalTensor<calcType> origALocal; // pre-silu 原始 a(fp32) 暂存
    LocalTensor<calcType> bF32Local; // up(b) fp32 暂存
    LocalTensor<calcType> zeroLocal; // Select 的0常量源

    TBuf<TPosition::VECCALC> aTempBuffer;
    TBuf<TPosition::VECCALC> lTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<inType> lGm;
    GlobalTensor<outType> mGm;
    GlobalTensor<outType> nGm;

    // probs(MoE per-token 权重) 支持：grad_h = gradout*probs。
    // 由 Single 驱动 Init/CopyIn 设置；probsVecLocal_ 已按 token 行广播好(含 MTE2/S 同步)，Compute 只做一次 Mul。
    GlobalTensor<float> probsGm_;
    TBuf<TPosition::VECCALC> probsBuf_;     // 本 tile 的 curRow 个 probs(fp32)
    TBuf<TPosition::VECCALC> probsVecBuf_;  // [curRow×curCol] 广播向量
    LocalTensor<float> probsLocal_;
    LocalTensor<float> probsVecLocal_;
    uint32_t pHasProbs_ = 0;     // 0=无 probs，全部 probs 逻辑跳过(等价现状)
    uint32_t pCurFirstRow_ = 0;  // 本 tile 首行(token)全局行号
    uint32_t pCurRow_ = 0;       // 本 tile 行数
    uint32_t pCurCol_ = 0;       // 本 tile 列数(=每行元素数)

    // grad_probs 进算子：grad_probs[t]=sum_H(gradout*h)，h=silu(ac)*clamp(up)。切列时各列片算部分和，CopyOut 原子累加汇总。
    // 不新开 UB buffer：hLocal_/gpRowLocal_ 别名复用 tempLocal/probsLocal_(见 InitUbBuffer)，避免撑大 singleDataSize。
    GlobalTensor<float> gradProbsGm_;
    LocalTensor<float> hLocal_;       // 别名 tempLocal
    LocalTensor<float> gpRowLocal_;   // 别名 probsLocal_（reduce 输出）
    uint32_t outputGradProbs_ = 0;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    RoundMode downcastRoundMode_ = RoundMode::CAST_RINT;
#else
    RoundMode downcastRoundMode_ = RoundMode::CAST_NONE;
#endif
};

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(
    TPipe& pipe, uint64_t tileLength) {
    // pipe alloc memory to queue, the unit is Bytes
    inQueueA.Init(pipe, tileLength);
    inQueueB.Init(pipe, tileLength);
    inQueueL.Init(pipe, tileLength);
    outQueueM.Init(pipe, tileLength); // The length must be an integer multiple of 32
    outQueueN.Init(pipe, tileLength); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(calcType));

    pipe.InitBuffer(aTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(lTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
    // uint8 掩码 buffer 的 UB footprint 必须按 fp32(4B) 槽分配：
    // CompareScalar/Select 的 uint8 mask 底层按 4B 槽写（参考 wind_moe_compute_expert_tokens 的 *sizeof(float)），
    // 按 sizeof(uint8_t) 申请会比实际写入小 4 倍。(tileLength+7)/8*8 => 8 个 fp32 = 32B 对齐。
    uint64_t maskAlignLen = (tileLength + 7U) / 8U * 8U;
    pipe.InitBuffer(maskQueue, maskAlignLen * sizeof(calcType)); // uint8 掩码（按 4B 槽分配）
    // pre-silu: 原始 a(fp32) 全长暂存，M 分支即时算 [a<=limit] 掩码（不跨分支存 mask，避免串扰）
    pipe.InitBuffer(origAQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(bF32Queue, tileLength * sizeof(calcType)); // up(b) fp32
    pipe.InitBuffer(zeroQueue, tileLength * sizeof(calcType)); // Select 0 源
    tempLocal = tmpQueue.Get<calcType>();
    sigLocal = sigQueue.Get<calcType>();
    maskLocal = maskQueue.Get<uint8_t>();
    origALocal = origAQueue.Get<calcType>(); // pre-silu 原始 a
    bF32Local = bF32Queue.Get<calcType>();
    zeroLocal = zeroQueue.Get<calcType>();
    Duplicate<calcType>(zeroLocal, (calcType)0.0, tileLength); // Select 的“否则取0”常量源
    pipe_barrier(PIPE_V);
    // probs buffer：probsBuf_ 存本 tile 的 curRow 个 probs(≤tileLength)，
    // probsVecBuf_ 广播成 [curRow×curCol]≤tileLength。footprint 与 origAQueue/bF32Queue 同量级。
    pipe.InitBuffer(probsBuf_, tileLength * sizeof(float));
    pipe.InitBuffer(probsVecBuf_, tileLength * sizeof(float));
    probsLocal_ = probsBuf_.Get<float>();
    probsVecLocal_ = probsVecBuf_.Get<float>();
    // grad_probs 进算子：不新开 buffer(新开会撑大 singleDataSize→maxTileLen 变小→切更多列片)。
    //   h 复用 tempLocal(tmpQueue，grad_probs 段在 N分支后/M分支前，tempLocal 此刻空闲，M分支会重写)；
    //   clamp(up) 直接原地写 bF32Local(M分支 line 后续会再 clamp 一次，clamp 幂等，无害)，省掉独立工作区；
    //   逐行 reduce 结果复用 probsLocal_(CopyIn 广播进 probsVecLocal_ 后 probsLocal_ 即空闲，长度≥baseRowLen)。
    hLocal_ = tempLocal;            // 别名：tmpQueue
    gpRowLocal_ = probsLocal_;      // 别名：probsBuf_（reduce 输出，≤baseRowLen 个 float）
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    // calc sigLocal
    LocalTensor<inType>& aLocal_ = inQueueA.template DeQue<inType>(); // input a

    // 增加输入a：BF16->FP32的CAST
    auto aLocal = aTempBuffer.Get<calcType>();
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    // pre-silu: 先把原始 a 备份到 origALocal（M 分支即时算 [a<=limit] 掩码用），
    // 再把 aLocal 原地 clamp 成 ac=min(a,limit)，之后 sigmoid/silu/silu' 全自动变 pre-silu(ac) 版。
    // 注意：不在此处算 mask 跨 N 分支保存（会被 N 分支的 maskLocal 串扰），改为 M 分支即时用 origALocal 重算。
    if (limit > 0.0f) {
        Adds(origALocal, aLocal, (calcType)0.0, tileLength); // 备份原始 a（用 V 算子，确保走 PIPE_V，barrier 能正确同步，避免备份未完成即被 Mins 覆盖）
        pipe_barrier(PIPE_V);
        Mins(aLocal, aLocal, (calcType)limit, tileLength); // aLocal = ac = min(a,limit)
        pipe_barrier(PIPE_V);
    }

    CalcSig(tileLength, aLocal); // sigLocal = sigmoid(ac)

    // swiglu_limit: 一次加载 up(b) 到 bF32Local，供 N(maskB)/M(clamp) 共用。
    // 必须在 N 分支前加载（v1 原本在 M 分支晚加载并复用 aLocal，会破坏 aLocal=gate），这里改为早加载到独立 buffer。
    LocalTensor<inType>& bLocal_ = inQueueB.template DeQue<inType>(); // input b
    Cast(bF32Local, bLocal_, RoundMode::CAST_NONE, tileLength);       // bF32Local = up(fp32, raw)
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal_);

    // ----------------N
    auto nLocal = outputTempBuffer.Get<calcType>();
    Mul(nLocal, sigLocal, aLocal, tileLength);      // nLocal = silu(ac)
    pipe_barrier(PIPE_V);

    // swiglu_limit N分支: grad_b = min(silu(ac),limit) * l * [-limit<=b<=limit]
    if (limit > 0.0f) {
        Mins(nLocal, nLocal, (calcType)limit, tileLength); // gc = min(silu(ac),limit)
        pipe_barrier(PIPE_V);
        // maskB = [-limit<=b<=limit]：两次 Select 把越界处的 grad_b 置 0
        CompareScalar(maskLocal, bF32Local, (calcType)(-limit), CMPMODE::GE, tileLength); // b >= -limit?
        pipe_barrier(PIPE_V);
        Select(nLocal, maskLocal, nLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);
        pipe_barrier(PIPE_V);
        CompareScalar(maskLocal, bF32Local, (calcType)limit, CMPMODE::LE, tileLength);    // b <= limit?
        pipe_barrier(PIPE_V);
        Select(nLocal, maskLocal, nLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);
        pipe_barrier(PIPE_V);
    }

    // 增加输出l：FP32->BF16的CAST
    LocalTensor<inType>& lLocal_ = inQueueL.template DeQue<inType>(); // input l
    auto lLocal = lTempBuffer.Get<calcType>();
    Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueL.template FreeTensor(lLocal_);

    // grad_probs 进算子: grad_probs[t]=sum_H(gradout[t,:]*h[t,:])，h=silu(ac)*clamp(up)。
    // 顺序铁律: 在 lLocal *= probs 之前用 raw gradout 算。复用 sigLocal(sigmoid)/aLocal(ac)/bF32Local(raw up)。
    // buffer 全复用(见 InitUbBuffer)：hLocal_=tempLocal，reduce 输出 gpRowLocal_=probsLocal_；clamp(up) 原地写 bF32Local。
    if (outputGradProbs_ != 0U) {
        Mul(hLocal_, sigLocal, aLocal, tileLength);              // silu(ac)
        pipe_barrier(PIPE_V);
        if (limit > 0.0f) {                                      // clamp(up) 原地写 bF32Local(M分支后面会再 clamp，幂等)
            Maxs(bF32Local, bF32Local, (calcType)(-limit), tileLength);
            pipe_barrier(PIPE_V);
            Mins(bF32Local, bF32Local, (calcType)limit, tileLength);
            pipe_barrier(PIPE_V);
        }
        Mul(hLocal_, hLocal_, bF32Local, tileLength);            // h = silu(ac)*clamp(up)（limit<=0 时 bF32Local=raw up）
        pipe_barrier(PIPE_V);
        // h 全程 fp32(不截 bf16)：UB 现场重算、从不落盘，天然保留高精度。
        Mul(hLocal_, hLocal_, lLocal, tileLength);               // gradout * h (lLocal 为 raw gradout，fp32)
        pipe_barrier(PIPE_V);
        // 逐行 ReduceSum：每行 pCurCol_ 个元素求和 -> gpRowLocal_[i]。src 自身当 work buffer。
        for (uint32_t i = 0; i < pCurRow_; ++i) {
            ReduceSum(gpRowLocal_[i], hLocal_[i * pCurCol_], hLocal_[i * pCurCol_], pCurCol_);
        }
        pipe_barrier(PIPE_V);
    }

    // probs: grad_h = gradout * probs。lLocal(=gradout, fp32) cast 后乘一次，
    // N/M 两分支自动用加权梯度。probsVecLocal_ 已由驱动 CopyIn 按 token 行广播好(含同步)。hasProbs=0 跳过=等价现状。
    if (pHasProbs_ != 0U) {
        Mul(lLocal, lLocal, probsVecLocal_, tileLength); // l *= probs (fp32)
        pipe_barrier(PIPE_V);
    }

    LocalTensor<outType>& nLocal_ = outQueueN.template AllocTensor<outType>(); // lb

    // 删除 bf16 round-trip：实测线上 torch_npu swiglu 反向=内部 fp32 末尾截断，
    // 原"小算子拼接精度对齐"的 round-trip 反而比线上精度低(error 0.26)。全程 fp32 算到底，末尾才 cast，对齐线上。
    Mul(nLocal, nLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    // 输出 n：FP32->BF16 的 CAST（仅此一次截断，对齐线上）
    Cast(nLocal_, nLocal, downcastRoundMode_, tileLength);
    outQueueN.template EnQue<outType>(nLocal_);

    // ----------------M
    Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
    pipe_barrier(PIPE_V);
    Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);

    auto& mLocal = nLocal;
    Mul(mLocal, sigLocal, tempLocal, tileLength);
    pipe_barrier(PIPE_V);
    Mul(mLocal, mLocal, aLocal, tileLength);
    pipe_barrier(PIPE_V);

    Muls(mLocal, mLocal, -beta, tileLength);
    pipe_barrier(PIPE_V);
    Add(mLocal, mLocal, sigLocal, tileLength);        // mLocal = silu'(ac)

    // swiglu_limit M分支: grad_a = clamp(b) * silu'(ac) * l * [a<=limit]
    // 注意：b 已在 N 分支前早加载到 bF32Local（不再复用 aLocal），aLocal 此处为 ac。
    if (limit > 0.0f) {
        // pre-silu: 即时用备份的原始 a 算 [a<=limit] 掩码（不跨分支保存，避免被 N 分支 maskLocal 串扰）
        CompareScalar(maskLocal, origALocal, (calcType)limit, CMPMODE::LE, tileLength); // [a<=limit]
        pipe_barrier(PIPE_V);
        Select(mLocal, maskLocal, mLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength); // silu'(ac)*[a<=limit]
        pipe_barrier(PIPE_V);
        // M 分支用 clamp(b) 替代 raw b
        Maxs(bF32Local, bF32Local, (calcType)(-limit), tileLength);
        pipe_barrier(PIPE_V);
        Mins(bF32Local, bF32Local, (calcType)limit, tileLength); // bF32Local = clamp(b)
        pipe_barrier(PIPE_V);
    }

    Mul(mLocal, mLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    // 最终乘 b 改用 bF32Local（limit>0 时为 clamp(b)，否则为 raw b）
    ToOutQue(tileLength, mLocal, bF32Local);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardBF16<inType, calcType, outType, bufferNum>::CalcSig(uint64_t tileLength,
                                                                                         LocalTensor<calcType>& aLocal)
{
    Muls(sigLocal, aLocal, beta, tileLength);
    pipe_barrier(PIPE_V);
    Exp(sigLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
    Adds(sigLocal, sigLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Duplicate<calcType>(tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardBF16<inType, calcType, outType, bufferNum>::ToOutQue(uint64_t tileLength,
    LocalTensor<calcType>& mLocal, LocalTensor<calcType>& bLocal)
{
    LocalTensor<outType>& mLocal_ = outQueueM.template AllocTensor<outType>(); // la
    // 删除 bf16 round-trip：全程 fp32 算到底，末尾才 cast，对齐线上精度。
    Mul(mLocal, mLocal, bLocal, tileLength);
    pipe_barrier(PIPE_V);

    // enque the output tensor to VECOUT queue（仅此一次 FP32->BF16 截断，对齐线上）
    Cast(mLocal_, mLocal, downcastRoundMode_, tileLength);
    outQueueM.template EnQue<outType>(mLocal_);
}
#endif  // CANN_SWI_GLU_LIMIT_BACKWARD_BF16_HPP
