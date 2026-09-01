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
 *   每行 D 个元素:
 *     u = x * scale                         (全 D 维, fp32 计算)
 *     out[0 : D-rope) = u[0 : D-rope)       (直通)
 *     out[D-rope : D) = rope(u[D-rope : D)) (复数式 interleave RoPE, forward 乘 freqs)
 *   复数公式(对一对 (a,b)=(u[2j],u[2j+1]), freqs=(cos,sin)):
 *     out_re = a*cos - b*sin
 *     out_im = a*sin + b*cos
 *   实现(照 norm_rope_concat_base.h):sin 偶位取负后, out = u*cos + rotate(u)*sin_signed
 *     rotate: 相邻对交换 [a,b] -> [b,a];  sin_signed: 偶位 = -sin, 奇位 = +sin
 *       位 2j  : a*cos + b*(-sin) = a*cos - b*sin  ✓
 *       位 2j+1: b*cos + a*(+sin) = a*sin + b*cos  ✓
 * Create: 2026-06-22
 */
#ifndef WIND_SCALE_ROPE_HPP
#define WIND_SCALE_ROPE_HPP

#include "kernel_operator.h"

namespace AscendC {

constexpr int32_t WSR_BUFFER_NUM = 2;
constexpr int32_t WSR_NUM_TWO = 2;
constexpr int32_t WSR_SIZE_OF_FLOAT = 4;
constexpr uint32_t WSR_BLOCK_SIZE = 32;   // UB 搬运/对齐粒度(B)
// scale 的 dtype 独立于 x: BF16(rms0 出的 rstd) 或 FP32(npu_rms_norm/torch.rsqrt 出的 rstd)。
//   由 OpDef 的两档 dtype 组合决定, 编译期各出一份 kernel 二进制。
//   sizeof 出的是 size_t, 显式收成 uint32_t 免得 AlignUp(uint32_t,uint32_t) 处走隐式窄化。
constexpr uint32_t WSR_SCALE_PER_BLOCK =
    WSR_BLOCK_SIZE / static_cast<uint32_t>(sizeof(DTYPE_SCALE));   // bf16:16, fp32:8
// seq-block 批处理: 一次处理同 seq 的 WSR_BATCH 行(向量吃满)。
//   UB 估算(BATCH=8, D=512): inQueX 8×512×2×2(double)=16KB + outQue 16KB + calcBuf(fp32) 8×512×4=16KB
//   + scaleVec(fp32) 16KB ≈ 64KB, 远低于 UB(~192KB), 安全有余量。BATCH 可后续调大。
constexpr uint32_t WSR_BATCH = 8;

class WindScaleRope {
public:
    __aicore__ inline WindScaleRope(const WindScaleRopeTilingData& tiling, TPipe& pipe)
        : pipe_(pipe)
    {
        // 注意: device 侧 TilingData 是 POD 结构, 直接成员访问(点号), 不能用 host 侧的 get_XXX() getter
        T_ = tiling.T;
        S_ = tiling.S;
        N_ = tiling.N;
        D_ = tiling.D;
        ropeDim_ = tiling.ropeDim;
        ropeHalf_ = ropeDim_ / WSR_NUM_TWO;
        passDim_ = D_ - ropeDim_;
        rowsPerSeq_ = tiling.rowsPerSeq;   // = T/S = B*N
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

    __aicore__ inline void Process(GM_ADDR xGm, GM_ADDR scaleGm, GM_ADDR cosGm, GM_ADDR sinGm, GM_ADDR outGm)
    {
        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)xGm);
        scaleGm_.SetGlobalBuffer((__gm__ DTYPE_SCALE*)scaleGm);   // scale dtype 独立于 x(bf16/fp32)
        cosGm_.SetGlobalBuffer((__gm__ float*)cosGm);
        sinGm_.SetGlobalBuffer((__gm__ float*)sinGm);
        outGm_.SetGlobalBuffer((__gm__ DTYPE_X*)outGm);

        // [seq-block 批处理] 一次处理同一 seq 的 BATCH 行(向量吃满, 摊薄 launch/同步开销)。
        //   buffer 按 BATCH×D 分配。一个 batch 必须同 seq(cos/sin 一致)。
        pipe_.InitBuffer(inQueX_, WSR_BUFFER_NUM, WSR_BATCH * D_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQue_, WSR_BUFFER_NUM, WSR_BATCH * D_ * sizeof(DTYPE_X));
        // 计算缓冲(fp32): u[BATCH×D]
        pipe_.InitBuffer(calcBuf_, WSR_BATCH * D_ * sizeof(float));
        // scale 先搬进 UB; 一次搬该核所有行(pad 到 32B)。元素数按 scale 自身 dtype 算(bf16 16个/fp32 8个 = 32B)
        uint32_t scaleAlign = AlignUp(rowsPerCore_, WSR_SCALE_PER_BLOCK);
        pipe_.InitBuffer(scaleBuf_, scaleAlign * sizeof(DTYPE_SCALE));
        // scaleVec: BATCH×D 的 fp32 scale 广播向量, 供 fp32 Mul(每行 scale 不同, 不能用标量 Muls)
        pipe_.InitBuffer(scaleVecBuf_, WSR_BATCH * D_ * sizeof(float));
        // cos/sin: host 已预展开到 ropeDim 长度(每复数重复2份, sin偶位已取负)。
        //   用 TQue(double buffer, 框架自动管 MTE2↔V 同步)每 batch 搬该 seq 的 cos/sin。
        //   曾用常驻 TBuf"仅 seq 变化才搬"省 MTE2, 但 kv(rowsPerSeq=1 每 batch 换 seq)下手动同步
        //   配对错乱致 race(55/2048 行错)。TQue 框架同步最稳; 同 seq 多 batch 重复搬运由 double buffer 流水缓解。
        ropeDimAligned_ = AlignUp(ropeDim_, 8u);              // 8 fp32 = 32B
        pipe_.InitBuffer(csQue_, WSR_BUFFER_NUM, WSR_NUM_TWO * ropeDimAligned_ * sizeof(float));  // [cos|sin] double buffer
        // rot(gather 结果)临时: 旋转段逐行处理(rope 只 64 维, 批量化收益小且 mask 复杂)
        pipe_.InitBuffer(cosBuf2_, ropeDimAligned_ * sizeof(float));
        pipe_.InitBuffer(maskBuf_, ropeDimAligned_ * sizeof(uint32_t));

        // 把本核 [rowStart_, rowEnd_) 的 scale 一次性搬进 UB。空核(rowStart_==rowEnd_==T_)直接跳过,
        //   零长度 DataCopyPad 无意义且不保证合法。
        scaleLocal_ = scaleBuf_.Get<DTYPE_SCALE>();
        uint32_t n = rowEnd_ - rowStart_;
        if (n > 0) {
            DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(DTYPE_SCALE)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_SCALE> pad{false, 0, 0, 0};
            DataCopyPad(scaleLocal_, scaleGm_[rowStart_], p, pad);
        }

        // 构造 interleave rotate 的 gather mask(相邻对交换): 0,1,2,3 -> 1,0,3,2 (字节偏移)
        mask_ = maskBuf_.Get<uint32_t>();
        for (uint32_t j = 0; j < ropeHalf_; ++j) {
            mask_.SetValue(WSR_NUM_TWO * j,     (WSR_NUM_TWO * j + 1) * WSR_SIZE_OF_FLOAT);
            mask_.SetValue(WSR_NUM_TWO * j + 1, (WSR_NUM_TWO * j)     * WSR_SIZE_OF_FLOAT);
        }

        // [seq-block 批处理] 按 batch 推进: 每个 batch 取同 seq 的 cnt 行(cnt≤WSR_BATCH)。
        //   x*scale 与 cast 批量化(cnt×D 一次过, vector 吃满); 旋转段逐行(rope 64 维, 批量收益小)。
        uint32_t row = rowStart_;
        while (row < rowEnd_) {
            uint32_t seq = (rowsPerSeq_ == 0) ? 0 : (row / rowsPerSeq_);
            if (seq >= S_) { seq = S_ - 1; }
            // 本 batch 行数 = min(WSR_BATCH, 该 seq 在核内剩余行, 核剩余行)
            uint32_t seqEnd = (rowsPerSeq_ == 0) ? rowEnd_ : ((seq + 1) * rowsPerSeq_);
            uint32_t batchEnd = row + WSR_BATCH;
            uint32_t end = batchEnd;
            if (end > seqEnd) { end = seqEnd; }
            if (end > rowEnd_) { end = rowEnd_; }
            uint32_t cnt = end - row;
            ProcessBatch(row, cnt, seq);
            row = end;
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t rowBase, uint32_t cnt, uint32_t seq)
    {
        uint32_t total = cnt * D_;

        // ---- CopyIn x[cnt×D] (一次搬 cnt 行) ----
        LocalTensor<DTYPE_X> xLocal = inQueX_.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm_[static_cast<uint64_t>(rowBase) * D_], total);
        inQueX_.EnQue(xLocal);
        LocalTensor<DTYPE_X> xIn = inQueX_.DeQue<DTYPE_X>();

        // ---- x*scale: 批量 fp32。每行 scale 不同 => 广播成 cnt×D 的 fp32 scaleVec, 再一次 Mul ----
        LocalTensor<float> u = calcBuf_.Get<float>();            // [cnt×D] fp32
        Cast(u, xIn, RoundMode::CAST_NONE, total);              // x → fp32 (一次 cast cnt×D)
        inQueX_.FreeTensor(xIn);
        PipeBarrier<PIPE_V>();
        // 构造 scaleVec[cnt×D]: 第 i 行全 D 维 = scale[i]
        LocalTensor<float> scaleVec = scaleVecBuf_.Get<float>();
        for (uint32_t i = 0; i < cnt; ++i) {
            float sv = ScaleAt(rowBase - rowStart_ + i);       // bf16/fp32 都提到 fp32 再广播
            Duplicate(scaleVec[i * D_], sv, D_);               // 第 i 行 D 维填 scale[i]
        }
        PipeBarrier<PIPE_V>();
        Mul(u, u, scaleVec, total);                            // u = x*scale (一次 Mul cnt×D, fp32)
        PipeBarrier<PIPE_V>();

        LocalTensor<DTYPE_X> outLocal = outQue_.AllocTensor<DTYPE_X>();

#ifndef WSR_SKIP_ROPE
        // [cos/sin 加载] 用 TQue(double buffer, 框架自动管 MTE2↔V 同步), 每 batch 搬该 seq 的 cos/sin。
        //   之前用常驻 TBuf + 手动 SetFlag/WaitFlag(单 EVENT_ID0), 在 kv 等 rowsPerSeq=1 场景
        //   每 batch 换 seq 的高频下, V_MTE2/MTE2_V 交替复用同一 event 配对错乱 -> 少数行(实测55/2048)
        //   读到半搬运 cos/sin(out 既非正确seq也非seq0的脏值, race 指纹)。改回 TQue 由框架同步, 最稳。
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

        // 旋转段逐行: u 第 i 行的后 ropeDim 维(fp32)原地旋转。
        //   out = u_tail*cos + rotate(u_tail)*sinSigned
        LocalTensor<float> rot = cosBuf2_.Get<float>();         // [ropeDim] fp32
        for (uint32_t i = 0; i < cnt; ++i) {
            LocalTensor<float> ropeIn = u[i * D_ + passDim_];   // 第 i 行后 ropeDim 维
            Gather(rot, ropeIn, mask_, 0, ropeDim_);            // rot = rotate(ropeIn) = [b,a,..]
            PipeBarrier<PIPE_V>();
            Mul(ropeIn, ropeIn, cosL, ropeDim_);                // ropeIn *= cos
            PipeBarrier<PIPE_V>();
            MulAddDst(ropeIn, rot, sinL, ropeDim_);             // ropeIn += rot*sinSigned
            PipeBarrier<PIPE_V>();
        }
        csQue_.FreeTensor(csDe);                               // 释放, 框架管 V→MTE2(下次搬入前等读完)
#endif

        // ---- 整行批量 cast 回 bf16(cnt×D), CopyOut ----
        if constexpr (IsSameType<DTYPE_X, half>::value) {
            Cast(outLocal, u, RoundMode::CAST_NONE, total);
        } else {
            Cast(outLocal, u, RoundMode::CAST_RINT, total);
        }
        PipeBarrier<PIPE_V>();
        outQue_.EnQue(outLocal);
        LocalTensor<DTYPE_X> outDe = outQue_.DeQue<DTYPE_X>();
        DataCopy(outGm_[static_cast<uint64_t>(rowBase) * D_], outDe, total);
        outQue_.FreeTensor(outDe);
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
    TQue<QuePosition::VECIN, WSR_BUFFER_NUM> inQueX_;
    TQue<QuePosition::VECOUT, WSR_BUFFER_NUM> outQue_;
    TBuf<TPosition::VECCALC> calcBuf_;
    TBuf<TPosition::VECCALC> scaleBuf_, scaleVecBuf_, rotBuf_, cosBuf2_, maskBuf_;
    TQue<QuePosition::VECIN, WSR_BUFFER_NUM> csQue_;   // cos/sin 每 batch 搬运(double buffer, 框架自动同步)
    LocalTensor<DTYPE_SCALE> scaleLocal_;
    LocalTensor<uint32_t> mask_;
    uint32_t ropeDimAligned_{0};

    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_SCALE> scaleGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<DTYPE_X> outGm_;

    uint32_t T_, S_, N_, D_, ropeDim_, ropeHalf_, passDim_;
    uint32_t rowsPerSeq_{0};
    uint32_t usedCore_, rowsPerCore_;
    uint32_t blockIdx_, rowStart_, rowEnd_;
};

} // namespace AscendC

#endif // WIND_SCALE_ROPE_HPP
