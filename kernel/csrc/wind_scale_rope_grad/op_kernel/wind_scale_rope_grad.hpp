/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * WindScaleRopeGrad kernel(WindScaleRope 反向)
 *   给 grad_out 求 grad_x, grad_scale, 和现场 q*rsqrt 反向链一致(grad_u 共享):
 *     grad_u[:passDim]  = grad_out[:passDim]               # 直通段
 *     grad_u[passDim:]  = inv_rope(grad_out[passDim:])     # inverse RoPE(调用方传 sin=-sinFwd)
 *     grad_x            = grad_u * scale                   # 一次乘(现场 Mul483)
 *     grad_scale        = ReduceSum(grad_u * x, -1)        # 一次乘+reduce, 单趟HBM(现场 Mul742+ReduceSum)
 *   旋转公式与前向同(用 sinSigned + rotate): out = u*cos + rotate(u)*sinSigned
 *     调用方传 inverse 的 sin(对 forward sinSigned 取反), 即得 inverse RoPE。
 * Create: 2026-06-23
 */
#ifndef WIND_SCALE_ROPE_GRAD_HPP
#define WIND_SCALE_ROPE_GRAD_HPP

#include <kernel_operator.h>
#include "utils/op_math.h"
using namespace AscendC;
#include "kernel/attention/reduce_xx_nd.hpp"

namespace AscendC {

constexpr int32_t WSRG_BUFFER_NUM = 2;
constexpr int32_t WSRG_NUM_TWO = 2;
constexpr int32_t WSRG_SIZE_OF_FLOAT = 4;
constexpr uint32_t WSRG_BATCH = 8;   // 一次处理同 seq 的 8 行
constexpr uint32_t WSRG_BLOCK_SIZE = 32;   // UB 搬运/对齐粒度(B)
// scale 的 dtype 独立于 x: BF16 或 FP32, 与前向 WindScaleRope 的两档一一对应。
//   sizeof 出的是 size_t, 显式收成 uint32_t 免得 AlignUp(uint32_t,uint32_t) 处走隐式窄化。
constexpr uint32_t WSRG_SCALE_PER_BLOCK =
    WSRG_BLOCK_SIZE / static_cast<uint32_t>(sizeof(DTYPE_SCALE));   // bf16:16, fp32:8

class WindScaleRopeGrad {
public:
    __aicore__ inline WindScaleRopeGrad(const WindScaleRopeGradTilingData& tiling, TPipe& pipe)
        : pipe_(pipe)
    {
        T_ = tiling.T;
        S_ = tiling.S;
        N_ = tiling.N;
        D_ = tiling.D;
        ropeDim_ = tiling.ropeDim;
        ropeHalf_ = ropeDim_ / WSRG_NUM_TWO;
        passDim_ = D_ - ropeDim_;
        rowsPerSeq_ = tiling.rowsPerSeq;
        usedCore_ = tiling.usedCore;
        rowsPerCore_ = tiling.rowsPerCore;

        blockIdx_ = GetBlockIdx();
        rowStart_ = blockIdx_ * rowsPerCore_;
        // rowStart_ 也要夹: usedCore*rowsPerCore 可能 > T, 尾核 rowStart_>T_ 会使 rowEnd_-rowStart_ 下溢 -> DataCopyPad 越界(507035)
        if (rowStart_ > T_) {
            rowStart_ = T_;
        }
        rowEnd_ = rowStart_ + rowsPerCore_;
        if (rowEnd_ > T_) {
            rowEnd_ = T_;
        }
    }

    __aicore__ inline void Process(GM_ADDR gradOutGm, GM_ADDR xGm, GM_ADDR scaleGm, GM_ADDR cosGm,
                                   GM_ADDR sinGm, GM_ADDR gradXGm, GM_ADDR gradScaleGm)
    {
        gradOutGm_.SetGlobalBuffer((__gm__ DTYPE_X*)gradOutGm);
        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)xGm);
        scaleGm_.SetGlobalBuffer((__gm__ DTYPE_SCALE*)scaleGm);   // scale dtype 独立于 x(bf16/fp32)
        cosGm_.SetGlobalBuffer((__gm__ float*)cosGm);
        sinGm_.SetGlobalBuffer((__gm__ float*)sinGm);
        gradXGm_.SetGlobalBuffer((__gm__ DTYPE_X*)gradXGm);
        gradScaleGm_.SetGlobalBuffer((__gm__ float*)gradScaleGm);

        // grad_out / x 各一块 batch buffer; grad_x 输出; grad_scale 输出
        pipe_.InitBuffer(inQueGrad_, WSRG_BUFFER_NUM, WSRG_BATCH * D_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(inQueX_, WSRG_BUFFER_NUM, WSRG_BATCH * D_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueGradX_, WSRG_BUFFER_NUM, WSRG_BATCH * D_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueGradS_, WSRG_BUFFER_NUM, WSRG_BATCH * sizeof(float));  // grad_scale[cnt]
        // 计算缓冲: gu[cnt×D] fp32(grad_u) + xf[cnt×D] fp32(x) + prod[cnt×D] fp32(gu*x)
        pipe_.InitBuffer(guBuf_, WSRG_BATCH * D_ * sizeof(float));
        pipe_.InitBuffer(xfBuf_, WSRG_BATCH * D_ * sizeof(float));
        pipe_.InitBuffer(scaleVecBuf_, WSRG_BATCH * D_ * sizeof(float));
        // ReduceXx_Nd 临时工作区(grad_scale 一次对 cnt 行各 reduce D 维)。
        //   GetReduceXxNdTempBuffLen 只在 host(op_tiling.h)可调, kernel 侧按其公式直接算:
        //   D(=colLen) 命中 {64,128,256,512,1024} 分支 -> ceil(rowLen*colLen/64)*8 元素;
        //   rowLen 用最大 WSRG_BATCH(尾块 cnt<8 也够)。
        reduceTmpElems_ = ((WSRG_BATCH * D_ + 63u) / 64u) * 8u;
        pipe_.InitBuffer(reduceTmpBuf_, reduceTmpElems_ * sizeof(float));
        // scale 整核搬入(元素数按 scale 自身 dtype 对齐到 32B: bf16 16个 / fp32 8个)
        uint32_t scaleAlign = AlignUp(rowsPerCore_, WSRG_SCALE_PER_BLOCK);
        pipe_.InitBuffer(scaleBuf_, scaleAlign * sizeof(DTYPE_SCALE));
        // cos/sin 用 TQue(double buffer, 框架自动管 MTE2↔V 同步), 每 batch 搬该 seq 的 cos/sin。
        //   之前用常驻 TBuf(csBuf_) + curSeq_ + 手动 SetFlag/WaitFlag(单 EVENT_ID0), 在 kv 等
        //   rowsPerSeq=1 场景每 batch 换 seq 的高频下, V_MTE2/MTE2_V 交替复用同一 event 配对错乱 ->
        //   少数行读到半搬运 cos/sin(grad_kv rel~0.53 race 指纹)。前向已改 TQue, 此处同步对齐。
        ropeDimAligned_ = AlignUp(ropeDim_, 8u);
        pipe_.InitBuffer(csQue_, WSRG_BUFFER_NUM, WSRG_NUM_TWO * ropeDimAligned_ * sizeof(float));  // [cos|sin] double buffer
        pipe_.InitBuffer(rotBuf_, ropeDimAligned_ * sizeof(float));
        pipe_.InitBuffer(maskBuf_, ropeDimAligned_ * sizeof(uint32_t));

        // 空核(rowStart_==rowEnd_==T_)跳过, 零长度 DataCopyPad 无意义且不保证合法。
        scaleLocal_ = scaleBuf_.Get<DTYPE_SCALE>();
        uint32_t n = rowEnd_ - rowStart_;
        if (n > 0) {
            DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(DTYPE_SCALE)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_SCALE> pad{false, 0, 0, 0};
            DataCopyPad(scaleLocal_, scaleGm_[rowStart_], p, pad);
        }

        mask_ = maskBuf_.Get<uint32_t>();
        for (uint32_t j = 0; j < ropeHalf_; ++j) {
            mask_.SetValue(WSRG_NUM_TWO * j,     (WSRG_NUM_TWO * j + 1) * WSRG_SIZE_OF_FLOAT);
            mask_.SetValue(WSRG_NUM_TWO * j + 1, (WSRG_NUM_TWO * j)     * WSRG_SIZE_OF_FLOAT);
        }

        uint32_t row = rowStart_;
        while (row < rowEnd_) {
            uint32_t seq = (rowsPerSeq_ == 0) ? 0 : (row / rowsPerSeq_);
            if (seq >= S_) { seq = S_ - 1; }
            uint32_t seqEnd = (rowsPerSeq_ == 0) ? rowEnd_ : ((seq + 1) * rowsPerSeq_);
            uint32_t end = row + WSRG_BATCH;
            if (end > seqEnd) { end = seqEnd; }
            if (end > rowEnd_) { end = rowEnd_; }
            ProcessBatch(row, end - row, seq);
            row = end;
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t rowBase, uint32_t cnt, uint32_t seq)
    {
        uint32_t total = cnt * D_;

        // ---- CopyIn grad_out[cnt×D], x[cnt×D] ----
        LocalTensor<DTYPE_X> gradLocal = inQueGrad_.AllocTensor<DTYPE_X>();
        DataCopy(gradLocal, gradOutGm_[static_cast<uint64_t>(rowBase) * D_], total);
        inQueGrad_.EnQue(gradLocal);
        LocalTensor<DTYPE_X> gradIn = inQueGrad_.DeQue<DTYPE_X>();

        LocalTensor<DTYPE_X> xLocal = inQueX_.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm_[static_cast<uint64_t>(rowBase) * D_], total);
        inQueX_.EnQue(xLocal);
        LocalTensor<DTYPE_X> xIn = inQueX_.DeQue<DTYPE_X>();

        // ---- grad_u = grad_out → fp32; 后rope维做 inverse RoPE(原地) ----
        LocalTensor<float> gu = guBuf_.Get<float>();            // [cnt×D] fp32 = grad_u
        Cast(gu, gradIn, RoundMode::CAST_NONE, total);
        inQueGrad_.FreeTensor(gradIn);
        PipeBarrier<PIPE_V>();

#ifndef WSRG_SKIP_ROPE
        // [cos/sin 加载] TQue double buffer, 框架自动管 MTE2↔V 同步; 每 batch 搬该 seq 的 cos/sin。
        //   (kv rowsPerSeq=1 每行换 seq 也安全, 不再有常驻 buffer 的 event 配对 race。)
        LocalTensor<float> csIn = csQue_.AllocTensor<float>();
        {
            DataCopyExtParams p{1, static_cast<uint32_t>(ropeDim_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(csIn, cosGm_[static_cast<uint64_t>(seq) * ropeDim_], p, pad);
            DataCopyPad(csIn[ropeDimAligned_], sinGm_[static_cast<uint64_t>(seq) * ropeDim_], p, pad);
        }
        csQue_.EnQue(csIn);
        LocalTensor<float> csDe = csQue_.DeQue<float>();        // 框架保证搬完才返回
        LocalTensor<float> cosL = csDe;                         // [ropeDim]
        LocalTensor<float> sinL = csDe[ropeDimAligned_];        // [ropeDim]
        // 逐行 inverse rope(gu 后rope维原地): gu_tail = gu_tail*cos + rotate(gu_tail)*sinSigned
        //   调用方传 sin=-sinFwd, 故此处即 inverse。
        LocalTensor<float> rot = rotBuf_.Get<float>();
        for (uint32_t i = 0; i < cnt; ++i) {
            LocalTensor<float> ropeIn = gu[i * D_ + passDim_];
            Gather(rot, ropeIn, mask_, 0, ropeDim_);
            PipeBarrier<PIPE_V>();
            Mul(ropeIn, ropeIn, cosL, ropeDim_);
            PipeBarrier<PIPE_V>();
            MulAddDst(ropeIn, rot, sinL, ropeDim_);
            PipeBarrier<PIPE_V>();
        }
        csQue_.FreeTensor(csDe);                               // 释放, 框架管 V→MTE2(下次搬入前等读完)
#endif

        // ---- grad_x = grad_u * scale (广播 scaleVec, 一次 Mul fp32, 整行) ----
        LocalTensor<float> scaleVec = scaleVecBuf_.Get<float>();
        for (uint32_t i = 0; i < cnt; ++i) {
            float sv = ScaleAt(rowBase - rowStart_ + i);       // bf16/fp32 都提到 fp32 再广播
            Duplicate(scaleVec[i * D_], sv, D_);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<DTYPE_X> gradXLocal = outQueGradX_.AllocTensor<DTYPE_X>();
        LocalTensor<float> xf = xfBuf_.Get<float>();           // [cnt×D] fp32 = x
        Cast(xf, xIn, RoundMode::CAST_NONE, total);
        inQueX_.FreeTensor(xIn);
        PipeBarrier<PIPE_V>();

        // grad_scale = sum(grad_u * x, -1): 先 prod = gu*x(复用 scaleVec 不行, 用 xf 原地);
        //   再逐行 ReduceSum 出 [cnt] 标量。注意要在 grad_x 计算前算, 因为 grad_x 会改 gu(乘scale)。
        LocalTensor<float> prod = xf;                          // prod 原地写回 xf
        Mul(prod, gu, xf, total);                             // prod = grad_u * x (整行 fp32)
        PipeBarrier<PIPE_V>();
        LocalTensor<float> gsLocal = outQueGradS_.AllocTensor<float>();
        // 一次对 cnt 行各 reduce D 维 -> gsLocal[cnt]; 替代逐行 ReduceSum+GetValue/SetValue 的标量循环。
        Wind::ReduceXx_Nd<Wind::OpType::SUM, float, true>(gsLocal, prod, reduceTmpBuf_.Get<float>(), cnt, D_);
        PipeBarrier<PIPE_V>();
        outQueGradS_.EnQue(gsLocal);
        LocalTensor<float> gsDe = outQueGradS_.DeQue<float>();
        {
            DataCopyExtParams p{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            DataCopyPad(gradScaleGm_[rowBase], gsDe, p);
        }
        outQueGradS_.FreeTensor(gsDe);

        // grad_x = grad_u * scale (gu 原地乘 scaleVec, 再 cast bf16)
        Mul(gu, gu, scaleVec, total);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<DTYPE_X, half>::value) {
            Cast(gradXLocal, gu, RoundMode::CAST_NONE, total);
        } else {
            Cast(gradXLocal, gu, RoundMode::CAST_RINT, total);
        }
        PipeBarrier<PIPE_V>();
        outQueGradX_.EnQue(gradXLocal);
        LocalTensor<DTYPE_X> gxDe = outQueGradX_.DeQue<DTYPE_X>();
        DataCopy(gradXGm_[static_cast<uint64_t>(rowBase) * D_], gxDe, total);
        outQueGradX_.FreeTensor(gxDe);
    }

    __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b)
    {
        return (b == 0) ? a : ((a + b - 1) / b * b);
    }

    // scale 值取到 fp32: fp32 档直接返回, bf16 档走 ToFloat。
    //   【务必保持 ToF32 是模板、且形参类型就是 S】: if constexpr 的弃置分支只有在
    //   依赖模板参数时才不实例化。若把 if constexpr 直接写进非模板的 ScaleAt(), 或写成模板
    //   但分支里用的是 scaleLocal_(类型固定为 LocalTensor<DTYPE_SCALE>, 不依赖模板参数),
    //   弃置分支都会照样在定义期被检查, 于是:
    //     fp32 档 -> 实例化 ToFloat<float>, 撞 kernel_scalar_convert.h:133 的 static_assert
    //               (ToFloat 只支持 bfloat16_t/hifloat8_t/fp8/fp4);
    //     bf16 档 -> `return v;` 报 no viable conversion bfloat16_t -> float。
    //   (两种失败均已用 clang -std=c++17 复刻 CANN 声明实测过。)
    template <typename S>
    __aicore__ inline float ToF32(const S& v)
    {
        if constexpr (IsSameType<S, float>::value) {
            return v;
        } else {
            return ToFloat(v);
        }
    }

    __aicore__ inline float ScaleAt(uint32_t idx) { return ToF32(scaleLocal_.GetValue(idx)); }

private:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, WSRG_BUFFER_NUM> inQueGrad_, inQueX_, csQue_;  // csQue_: cos/sin double buffer
    TQue<QuePosition::VECOUT, WSRG_BUFFER_NUM> outQueGradX_, outQueGradS_;
    TBuf<TPosition::VECCALC> guBuf_, xfBuf_, scaleVecBuf_, scaleBuf_, rotBuf_, maskBuf_, reduceTmpBuf_;
    LocalTensor<DTYPE_SCALE> scaleLocal_;
    LocalTensor<uint32_t> mask_;
    uint32_t ropeDimAligned_{0};
    uint32_t reduceTmpElems_{0};

    GlobalTensor<DTYPE_X> gradOutGm_, xGm_, gradXGm_;
    GlobalTensor<DTYPE_SCALE> scaleGm_;                  // scale 与 x 的 dtype 解耦
    GlobalTensor<float> cosGm_, sinGm_, gradScaleGm_;

    uint32_t T_, S_, N_, D_, ropeDim_, ropeHalf_, passDim_;
    uint32_t rowsPerSeq_{0};
    uint32_t usedCore_, rowsPerCore_;
    uint32_t blockIdx_, rowStart_, rowEnd_;
};

} // namespace AscendC

#endif // WIND_SCALE_ROPE_GRAD_HPP
