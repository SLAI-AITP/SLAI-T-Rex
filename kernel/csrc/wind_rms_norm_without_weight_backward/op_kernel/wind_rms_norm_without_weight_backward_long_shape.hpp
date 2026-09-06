/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_LONG_SHAPE_HPP
#define CANN_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_LONG_SHAPE_HPP

#include "kernel_operator.h"
#include "wind_rms_norm_without_weight_backward_tiling_kernel.h"

using namespace AscendC;

// ========== 910B Multi Row ==========
template <uint8_t bufferNum = 2>
struct KernelRmsNormWithoutWeightBackwardLong {
public:
    __aicore__ inline KernelRmsNormWithoutWeightBackwardLong() {}
    __aicore__ inline ~KernelRmsNormWithoutWeightBackwardLong() {}

    __aicore__ inline void Init(GM_ADDR gradResGm, GM_ADDR xGm, GM_ADDR resGm, GM_ADDR gradXGm, GM_ADDR tiling, TPipe& pipe)
    {
        tiling_.GetTilingAndOffset(tiling, sizeof(DTYPE_X));
        this->gradResGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_RES*)gradResGm + tiling_.gmOffset / tiling_.colNum, tiling_.blockRowNumAliagn512);
        this->xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)xGm + tiling_.gmOffset, tiling_.blockRowNum * tiling_.colNum);
        this->resGm_.SetGlobalBuffer((__gm__ DTYPE_RES*)resGm + tiling_.gmOffset / tiling_.colNum, tiling_.blockRowNumAliagn512);
        this->gradXGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_X*)gradXGm + tiling_.gmOffset, tiling_.blockRowNum * tiling_.colNum);

        xBufSize_ = tiling_.rowEachCore * tiling_.colNumPerUBAlignDown32; // ub内部都按照float申请
        gradXBufSize_ = tiling_.rowEachCore * tiling_.colNumPerUBAlignDown32; // ub内部都按照float申请

        pipe.InitBuffer(inQueueX_, bufferNum, xBufSize_ * sizeof(float));
        pipe.InitBuffer(outQueueGradX_, bufferNum, gradXBufSize_ * sizeof(float));

        pipe.InitBuffer(const_fp32_buf, 2U * tiling_.blockRowNumAliagn32 * sizeof(float));

        index_grad_resfp32 = tiling_.blockRowNumAliagn32;

        Process();
    }

    __aicore__ inline void Process()
    {
        // 定制化fp32的输入
        LocalTensor<float> resfp32Local = const_fp32_buf.Get<float>();

        // 1. 先搬grad_res
        LocalTensor<DTYPE_GRAD_RES> gradResLocal = resfp32Local.ReinterpretCast<DTYPE_GRAD_RES>();
        if constexpr (!IsSameType<DTYPE_GRAD_RES, float>::value) {
            gradResLocal = gradResLocal[tiling_.blockRowNumAliagn32];
        }
        ::DataCopy(gradResLocal, this->gradResGm_, tiling_.blockRowNumAliagn32);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // 2. 计算grad_res * （-1/n)
        if constexpr (!IsSameType<DTYPE_GRAD_RES, float>::value) {
            // BF16/HALF -> float
            Cast(resfp32Local, gradResLocal, RoundMode::CAST_NONE, tiling_.blockRowNumAliagn32);
            PipeBarrier<PIPE_V>();
        }
        ::Muls(resfp32Local, resfp32Local, -tiling_.avg_factor, tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);

        // 搬运res
        LocalTensor<DTYPE_RES> resLocal = resfp32Local[index_grad_resfp32].ReinterpretCast<DTYPE_RES>();
        if constexpr (!IsSameType<DTYPE_RES, float>::value) {
            resLocal = resLocal[tiling_.blockRowNumAliagn32];
        }
        ::DataCopy(resLocal, this->resGm_, tiling_.blockRowNumAliagn32);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // 计算grad_res * （-1/n) * res^3
        if constexpr (!IsSameType<DTYPE_RES, float>::value) {
            // BF16/HALF -> float
            Cast(resfp32Local[index_grad_resfp32], resLocal, RoundMode::CAST_NONE, tiling_.blockRowNumAliagn32);
            PipeBarrier<PIPE_V>();
        }
        ::Mul(resfp32Local, resfp32Local, resfp32Local[index_grad_resfp32], tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);
        ::Mul(resfp32Local, resfp32Local, resfp32Local[index_grad_resfp32], tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);
        ::Mul(resfp32Local, resfp32Local, resfp32Local[index_grad_resfp32], tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);

        for (uint64_t i = 0; i < tiling_.tileLoopNum; i++) {
            uint64_t offsetRow = i * tiling_.rowEachCore * tiling_.colNum;
            float factor = resfp32Local.GetValue(i * tiling_.rowEachCore);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            for (uint64_t j = 0; j < tiling_.colLoopNum; j++) {
                uint64_t srcOffset = offsetRow + j * tiling_.colNumPerUBAlignDown32;
                CopyInX(srcOffset, tiling_.colNumPerUBAlignDown32);
                this->ComputeLoop(tiling_.rowEachCore, tiling_.colNumPerUBAlignDown32, factor);
                CopyOut(srcOffset, tiling_.colNumPerUBAlignDown32);
            }
            if (tiling_.colTileNumAlignUp32 > 0) {
                uint64_t srcOffset = offsetRow + tiling_.colLoopNum * tiling_.colNumPerUBAlignDown32;
                CopyInX(srcOffset, tiling_.colTileNumAlignUp32);
                this->ComputeLoop(tiling_.rowEachCore, tiling_.colTileNumAlignUp32, factor);
                CopyOut(srcOffset, tiling_.colTileNumAlignUp32);
            }
        }
    }

private:
    __aicore__ inline void CopyInX(uint64_t offset, uint64_t curTileLen)
    {
        auto xLocal = this->inQueueX_.template AllocTensor<float>();
        LocalTensor<DTYPE_X> xLocal_orig = xLocal.template ReinterpretCast<DTYPE_X>();
        if constexpr (!IsSameType<DTYPE_X, float>::value) {
            xLocal_orig = xLocal_orig[xBufSize_];
        }
        ::DataCopy(xLocal_orig, this->xGm_[offset], curTileLen);
        this->inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t curTileLen)
    {
        LocalTensor<float> gradXLocal = this->outQueueGradX_.template DeQue<float>();
        LocalTensor<DTYPE_GRAD_X> gradXLocal_orig = gradXLocal.template ReinterpretCast<DTYPE_GRAD_X>();
        ::DataCopy(this->gradXGm_[offset], gradXLocal_orig, curTileLen);
        this->outQueueGradX_.FreeTensor(gradXLocal);
    }

    __aicore__ inline void ComputeLoop(uint64_t rowNum, uint64_t colNum, float factor)
    {
        auto xLocal = inQueueX_.template DeQue<float>();
        if constexpr (!IsSameType<DTYPE_X, float>::value) {
            LocalTensor<DTYPE_X> xLocal_orig = xLocal.template ReinterpretCast<DTYPE_X>();
            xLocal_orig = xLocal_orig[xBufSize_];
            // BF16/HALF -> float
            Cast(xLocal, xLocal_orig, RoundMode::CAST_NONE, colNum);
            PipeBarrier<PIPE_V>();
        }
        auto outLocal = outQueueGradX_.template AllocTensor<float>();
        for (uint64_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            ::Muls(outLocal, xLocal, factor, colNum);
            pipe_barrier(PIPE_V);
        }
        inQueueX_.FreeTensor(xLocal);

        if constexpr (!IsSameType<DTYPE_GRAD_X, float>::value) {
            LocalTensor<DTYPE_GRAD_X> gradXLocal_orig = outLocal.template ReinterpretCast<DTYPE_GRAD_X>();
            // float -> BF16/HALF
            Cast(gradXLocal_orig, outLocal, RoundMode::CAST_RINT, colNum);
            PipeBarrier<PIPE_V>();
        }
        outQueueGradX_.EnQue(outLocal);
    }

private:
    TQue<QuePosition::VECIN, bufferNum> inQueueX_;
    TQue<QuePosition::VECOUT, bufferNum> outQueueGradX_;
    TBuf<TPosition::VECCALC> const_fp32_buf;

    GlobalTensor<DTYPE_GRAD_RES> gradResGm_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_RES> resGm_;
    GlobalTensor<DTYPE_GRAD_X> gradXGm_;
    RmsNormWithoutWeightBackwardTilingKernel tiling_;

    uint64_t index_grad_resfp32 = 0;
    uint32_t xBufSize_;
    uint32_t gradXBufSize_;
};

#endif
