/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_SWI_GLU_LIMIT_BACKWARD_FLOAT_HPP
#define CANN_SWI_GLU_LIMIT_BACKWARD_FLOAT_HPP
#include "kernel_operator.h"
#include "kernel/entity/tqueue.h"

using namespace AscendC;

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
struct SwiGluLimitBackwardVector {
public:
    __aicore__ inline SwiGluLimitBackwardVector() {}

protected:
    __aicore__ inline void InitUbBuffer(TPipe& pipe, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = -1.0;
    float limit = 0.0f; // swiglu_limit 反向 clamp 阈值，0=不裁剪，由 GluLimitBackwardSplit/Single Init 透传

    InQue<aType, bufferNum> inQueueA{EVENT_ID0};
    InQue<bType, bufferNum> inQueueB{EVENT_ID2};
    InQue<lType, bufferNum> inQueueL{EVENT_ID4};
    OutQue<mType, bufferNum> outQueueM{EVENT_ID0};
    OutQue<nType, bufferNum> outQueueN{EVENT_ID2};
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    TBuf<TPosition::VECCALC> maskQueue; // swiglu_limit: CompareScalar 输出的 uint8 选择掩码
    TBuf<TPosition::VECCALC> origAQueue; // pre-silu: 保存原始 a(fp32)，M 分支即时算 [a<=limit]，避免 mask 跨 N 分支被串扰
    TBuf<TPosition::VECCALC> zeroQueue; // swiglu_limit: Select 的“否则取0”源
    LocalTensor<float> tempLocal;
    LocalTensor<float> sigLocal;
    LocalTensor<uint8_t> maskLocal; // uint8 选择掩码
    LocalTensor<float> origALocal; // pre-silu 原始 a(fp32) 暂存
    LocalTensor<float> zeroLocal; // Select 的0常量源
    GlobalTensor<aType> aGm;
    GlobalTensor<bType> bGm;
    GlobalTensor<lType> lGm;
    GlobalTensor<mType> mGm;
    GlobalTensor<nType> nGm;

    // probs(MoE per-token 权重) 支持：grad_h = gradout*probs。
    // 由 Single 驱动 Init/CopyIn 设置：probsGm_、本 tile 的 probs 行(probsBuf_)、行列几何(pCurFirstRow_等)。
    // probsVecBuf_ 把每行标量广播成 [curRow×curCol] 的 fp32 向量，一次 Mul 乘到 lLocal(=gradout)。
    GlobalTensor<float> probsGm_;
    TBuf<TPosition::VECCALC> probsBuf_;     // 本 tile 的 curRow 个 probs(fp32)
    TBuf<TPosition::VECCALC> probsVecBuf_;  // [curRow×curCol] 广播向量
    LocalTensor<float> probsLocal_;
    LocalTensor<float> probsVecLocal_;
    uint32_t pHasProbs_ = 0;     // 0=无 probs，全部 probs 逻辑跳过(等价现状)
    uint32_t pCurFirstRow_ = 0;  // 本 tile 首行(token)全局行号
    uint32_t pCurRow_ = 0;       // 本 tile 行数
    uint32_t pCurCol_ = 0;       // 本 tile 列数(=每行元素数，probs 按此宽度广播)

    // grad_probs 进算子：grad_probs[t] = sum_H(gradout[t,:] * h[t,:])，h=swiglu_limit(x)。
    // outputGradProbs_=hasProbs；切列时各列片逐行 ReduceSum 出部分和，CopyOut 原子累加汇总。
    GlobalTensor<float> gradProbsGm_;        // [T,1] fp32 输出
    // grad_probs 进算子：不新开 buffer，hLocal_/gpRowLocal_ 别名复用 tempLocal/probsLocal_(见 InitUbBuffer)。
    LocalTensor<float> hLocal_;       // 别名 tempLocal
    LocalTensor<float> gpRowLocal_;   // 别名 probsLocal_（reduce 输出）
    uint32_t outputGradProbs_ = 0;           // 1=算并输出 grad_probs；0=无 probs 不算
};

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardVector<aType, bType, lType, mType, nType, bufferNum>::InitUbBuffer(
    TPipe& pipe, uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    inQueueA.Init(pipe, tileLength);
    inQueueB.Init(pipe, tileLength);
    inQueueL.Init(pipe, tileLength);
    outQueueM.Init(pipe, tileLength); // The length must be an integer multiple of 32
    outQueueN.Init(pipe, tileLength); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(float));
    // uint8 掩码 buffer 的 UB footprint 必须按 fp32(4B) 槽分配：
    // CompareScalar/Select 的 uint8 mask 底层按 4B 槽写（参考 wind_moe_compute_expert_tokens 的 *sizeof(float)），
    // 按 sizeof(uint8_t) 申请会比实际写入小 4 倍。(tileLength+7)/8*8 => 8 个 fp32 = 32B 对齐。
    uint64_t maskAlignLen = (tileLength + 7U) / 8U * 8U;
    pipe.InitBuffer(maskQueue, maskAlignLen * sizeof(float)); // uint8 掩码（按 4B 槽分配）
    // pre-silu: 原始 a(fp32) 全长暂存，M 分支即时算 [a<=limit] 掩码（不跨分支存 mask，避免串扰）
    pipe.InitBuffer(origAQueue, tileLength * sizeof(float));
    pipe.InitBuffer(zeroQueue, tileLength * sizeof(float)); // Select 0 源
    tempLocal = tmpQueue.Get<float>();
    sigLocal = sigQueue.Get<float>();
    maskLocal = maskQueue.Get<uint8_t>();
    origALocal = origAQueue.Get<float>(); // pre-silu 原始 a
    zeroLocal = zeroQueue.Get<float>();
    Duplicate<float>(zeroLocal, (float)0.0, tileLength); // Select 的“否则取0”常量源
    pipe_barrier(PIPE_V);
    // probs buffer：probsBuf_ 存本 tile 的 curRow 个 probs(≤tileLength)，
    // probsVecBuf_ 广播成 [curRow×curCol]≤tileLength。仅 hasProbs 时实际使用，footprint 与 origAQueue 同量级。
    pipe.InitBuffer(probsBuf_, tileLength * sizeof(float));
    pipe.InitBuffer(probsVecBuf_, tileLength * sizeof(float));
    probsLocal_ = probsBuf_.Get<float>();
    probsVecLocal_ = probsVecBuf_.Get<float>();
    // grad_probs 进算子：不新开 buffer(新开会撑大 singleDataSize→maxTileLen 变小→切更多列片)。
    //   h 复用 tempLocal(tmpQueue，grad_probs 段在 N分支后/M分支前空闲，M分支会重写)；
    //   clamp(up) 原地写 bLocal(M分支后续会再 clamp 一次，clamp 幂等)，省掉独立工作区；
    //   逐行 reduce 结果复用 probsLocal_(CopyIn 广播进 probsVecLocal_ 后即空闲，长度≥baseRowLen)。
    hLocal_ = tempLocal;            // 别名：tmpQueue
    gpRowLocal_ = probsLocal_;      // 别名：probsBuf_（reduce 输出，≤baseRowLen 个 float）
}

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluLimitBackwardVector<aType, bType, lType, mType, nType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    // calc sigLocal
    LocalTensor<aType>& aLocal = inQueueA.template DeQue<aType>(); // input a
    // pre-silu: 先把原始 a 备份到 origALocal（M 分支即时算 [a<=limit] 用），再原地 clamp aLocal=ac。
    // 不在此处算 mask 跨 N 分支保存（会被 N 分支 maskLocal 串扰），改为 M 分支即时用 origALocal 重算。
    if (limit > 0.0f) {
        Adds(origALocal, aLocal, (float)0.0, tileLength); // 备份原始 a（用 V 算子，确保走 PIPE_V，barrier 能正确同步，避免备份未完成即被 Mins 覆盖）
        pipe_barrier(PIPE_V);
        Mins(aLocal, aLocal, (float)limit, tileLength); // aLocal = ac = min(a,limit)
        pipe_barrier(PIPE_V);
    }
    Muls(sigLocal, aLocal, beta, tileLength);
    pipe_barrier(PIPE_V);
    Exp(sigLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
    Adds(sigLocal, sigLocal, (mType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);

    // swiglu_limit: 早加载 up(b)，供 N(maskB) 与 M(clamp) 共用。
    // b 为 fp32 输入，直接在 input 队列张量上 in-place clamp / 取掩码即可。
    LocalTensor<bType>& bLocal = inQueueB.template DeQue<bType>(); // input b

    // ----------------N: grad_b = min(silu(ac),limit) * l * [-limit<=b<=limit]
    LocalTensor<nType>& nLocal = outQueueN.template AllocTensor<nType>(); // lb
    Mul(nLocal, sigLocal, aLocal, tileLength); // nLocal = silu(ac)
    pipe_barrier(PIPE_V);
    if (limit > 0.0f) {
        Mins(nLocal, nLocal, (float)limit, tileLength); // gc = min(silu(ac),limit)
        pipe_barrier(PIPE_V);
        // maskB = [-limit<=b<=limit]：两次 Select 把越界处的 grad_b 置 0
        CompareScalar(maskLocal, bLocal, (float)(-limit), CMPMODE::GE, tileLength); // b >= -limit?
        pipe_barrier(PIPE_V);
        Select(nLocal, maskLocal, nLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);
        pipe_barrier(PIPE_V);
        CompareScalar(maskLocal, bLocal, (float)limit, CMPMODE::LE, tileLength);    // b <= limit?
        pipe_barrier(PIPE_V);
        Select(nLocal, maskLocal, nLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);
        pipe_barrier(PIPE_V);
    }
    LocalTensor<lType>& lLocal = inQueueL.template DeQue<lType>(); // input l (= gradout, fp32)

    // grad_probs 进算子: grad_probs[t] = sum_H(gradout[t,:] * h[t,:])，h=silu(ac)*clamp(up)。
    // 顺序铁律: 必须在 lLocal *= probs 之前用 raw gradout 算(grad_probs 用未乘 probs 的 grad_out)。
    // 复用现成中间量: sigLocal=sigmoid(ac) 未被改; aLocal=ac (M分支才改); bLocal=raw up (N分支只CompareScalar未改)。
    if (outputGradProbs_ != 0U) {
        Mul(hLocal_, sigLocal, aLocal, tileLength);              // silu(ac) = sigmoid(ac)*ac
        pipe_barrier(PIPE_V);
        if (limit > 0.0f) {                                      // clamp(up) 原地写 bLocal(M分支后面会再 clamp，幂等)
            Maxs(bLocal, bLocal, (bType)(-limit), tileLength);
            pipe_barrier(PIPE_V);
            Mins(bLocal, bLocal, (bType)limit, tileLength);
            pipe_barrier(PIPE_V);
        }
        Mul(hLocal_, hLocal_, bLocal, tileLength);               // h = silu(ac)*clamp(up)（limit<=0 时 bLocal=raw up）
        pipe_barrier(PIPE_V);
        Mul(hLocal_, hLocal_, lLocal, tileLength);               // gradout * h (lLocal 此时为 raw gradout)
        pipe_barrier(PIPE_V);
        // 逐行 ReduceSum：每行 pCurCol_ 个元素求和 -> gpRowLocal_[i]。src 自身当 work buffer。
        for (uint32_t i = 0; i < pCurRow_; ++i) {
            ReduceSum(gpRowLocal_[i], hLocal_[i * pCurCol_], hLocal_[i * pCurCol_], pCurCol_);
        }
        pipe_barrier(PIPE_V);
    }

    // probs: grad_h = gradout * probs。在 N/M 两分支用 l 之前乘一次，
    // 两分支自动用加权后的梯度（grad_a/grad_b 都正确）。probsVecLocal_ 已由驱动 CopyIn 按 token 行广播好
    // （第 i 行 curCol 列 = probs[firstRow+i]，并完成 MTE2→V 同步）。hasProbs=0 时跳过，与现状逐字节等价。
    if (pHasProbs_ != 0U) {
        Mul(lLocal, lLocal, probsVecLocal_, tileLength); // l *= probs (fp32)
        pipe_barrier(PIPE_V);
    }
    Mul(nLocal, nLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);
    outQueueN.template EnQue<nType>(nLocal);

    // ----------------M: grad_a = clamp(b) * silu'(ac) * l * [a<=limit]
    Muls(tempLocal, sigLocal, (mType)(-1.0), tileLength);
    pipe_barrier(PIPE_V);
    Adds(tempLocal, tempLocal, (mType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    LocalTensor<mType>& mLocal = outQueueM.template AllocTensor<mType>(); // la
    Mul(mLocal, sigLocal, tempLocal, tileLength);
    pipe_barrier(PIPE_V);
    Mul(mLocal, mLocal, aLocal, tileLength);
    pipe_barrier(PIPE_V);

    inQueueA.template FreeTensor(aLocal);

    Muls(mLocal, mLocal, -beta, tileLength);
    pipe_barrier(PIPE_V);
    Add(mLocal, mLocal, sigLocal, tileLength); // mLocal = silu'(a)
    pipe_barrier(PIPE_V);

    // swiglu_limit: 乘 maskG，并把 b clamp 到 [-limit,limit]
    if (limit > 0.0f) {
        // pre-silu: 即时用备份的原始 a 算 [a<=limit] 掩码（不跨分支保存，避免被 N 分支 maskLocal 串扰）
        CompareScalar(maskLocal, origALocal, (float)limit, CMPMODE::LE, tileLength); // [a<=limit]
        pipe_barrier(PIPE_V);
        Select(mLocal, maskLocal, mLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);
        pipe_barrier(PIPE_V);
        Maxs(bLocal, bLocal, (bType)(-limit), tileLength);
        pipe_barrier(PIPE_V);
        Mins(bLocal, bLocal, (bType)limit, tileLength); // bLocal = clamp(b)
        pipe_barrier(PIPE_V);
    }

    Mul(mLocal, mLocal, bLocal, tileLength);
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal);

    Mul(mLocal, mLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);
    // enque the output tensor to VECOUT queue
    outQueueM.template EnQue<mType>(mLocal);
    // free input tensors for reuse
    inQueueL.template FreeTensor(lLocal);
}
#endif // CANN_SWI_GLU_LIMIT_BACKWARD_FLOAT_HPP
