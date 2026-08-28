/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_HPP
#define CANN_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_HPP

#include "kernel_operator.h"
#include "wind_rms_norm_without_weight_backward_tiling_kernel.h"

using namespace AscendC;

constexpr const bool GRAD_RES_IS_FLOAT = IsSameType<DTYPE_GRAD_RES, float>::value;
constexpr const bool RES_IS_FLOAT = IsSameType<DTYPE_RES, float>::value;

// ========== 910B Multi Row ==========
template <uint8_t bufferNum = 2>
struct KernelRmsNormWithoutWeightBackward {
public:
    __aicore__ inline KernelRmsNormWithoutWeightBackward() {}
    __aicore__ inline ~KernelRmsNormWithoutWeightBackward() {}

    __aicore__ inline void Init(GM_ADDR gradResGm, GM_ADDR xGm, GM_ADDR resGm, GM_ADDR gradXGm, GM_ADDR tiling, TPipe& pipe)
    {
        tiling_.GetTilingAndOffset(tiling, sizeof(DTYPE_X));
        this->gradResGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_RES*)gradResGm + tiling_.gmOffset / tiling_.colNum, tiling_.blockRowNumAliagn512);
        this->xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)xGm + tiling_.gmOffset, tiling_.blockRowNum * tiling_.colNum); // colNum为512的倍数，Tiling保证
        this->resGm_.SetGlobalBuffer((__gm__ DTYPE_RES*)resGm + tiling_.gmOffset / tiling_.colNum, tiling_.blockRowNumAliagn512);
        this->gradXGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_X*)gradXGm + tiling_.gmOffset, tiling_.blockRowNum * tiling_.colNum);

        uint64_t xBufSize = tiling_.rowEachCore * tiling_.colNum * sizeof(DTYPE_X);
        uint64_t gradXBufSize = tiling_.rowEachCore * tiling_.colNum * sizeof(DTYPE_GRAD_X);

        pipe.InitBuffer(inQueueX_, bufferNum, xBufSize);
        pipe.InitBuffer(outQueueGradX_, bufferNum, gradXBufSize);

        pipe.InitBuffer(x_fp32_buf, tiling_.rowEachCore * tiling_.colNum * sizeof(float));
        pipe.InitBuffer(const_fp32_buf, tiling_.blockRowNumAliagn32 * sizeof(float));
        pipe.InitBuffer(res_fp32_buf, tiling_.blockRowNumAliagn32 * sizeof(float));

        index_grad_res = tiling_.blockRowNumAliagn32;
        index_res = tiling_.blockRowNumAliagn32;

        Process();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> gradResfp32Local = const_fp32_buf.Get<float>();

        const_buf = gradResfp32Local.ReinterpretCast<DTYPE_GRAD_RES>();
        if constexpr (!GRAD_RES_IS_FLOAT) {
            const_buf = const_buf[index_grad_res]; // 低精度，放在后面一半内存，方便后面Cast half->float
        }

        // 1. 搬运gradres
        ::DataCopy(const_buf, this->gradResGm_, tiling_.blockRowNumAliagn32);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        if constexpr (!GRAD_RES_IS_FLOAT) {
            ::Cast(gradResfp32Local, const_buf, RoundMode::CAST_NONE, tiling_.blockRowNumAliagn32);
        }

        pipe_barrier(PIPE_V);
        // 2. 计算gradres * （-1/D)
        ::Muls(gradResfp32Local, gradResfp32Local, -tiling_.avg_factor, tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);

        // 3. 搬运res
        LocalTensor<float> resfp32Local = res_fp32_buf.Get<float>();

        res_buf = resfp32Local.ReinterpretCast<DTYPE_RES>();
        if constexpr (!RES_IS_FLOAT) {
            res_buf = res_buf[index_res]; // 低精度，放在后面一半内存，方便后面Cast half->float
        }
        ::DataCopy(res_buf, this->resGm_, tiling_.blockRowNumAliagn32);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (!RES_IS_FLOAT) {
            ::Cast(resfp32Local, res_buf, RoundMode::CAST_NONE, tiling_.blockRowNumAliagn32);
            pipe_barrier(PIPE_V);
        }

        // 计算gradres * （-1/D) *res^3
        ::Mul(gradResfp32Local, gradResfp32Local, resfp32Local, tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);
        ::Mul(gradResfp32Local, gradResfp32Local, resfp32Local, tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);
        // 计算-res^3/n
        ::Mul(gradResfp32Local, gradResfp32Local, resfp32Local, tiling_.blockRowNumAliagn32);
        pipe_barrier(PIPE_V);

        if (tiling_.tailTileRow > 0) {
            CopyInX(tiling_.tileLoopNum * tiling_.rowEachCore * tiling_.colNum, tiling_.tailTileRow * tiling_.colNum);
            this->ComputeLoop(tiling_.tailTileRow, tiling_.colNum, tiling_.tileLoopNum * tiling_.rowEachCore, gradResfp32Local);
            CopyOut(tiling_.tileLoopNum * tiling_.rowEachCore * tiling_.colNum, tiling_.tailTileRow * tiling_.colNum);
        }
        for (uint64_t i = 0; i < tiling_.tileLoopNum; i++) {
            CopyInX(i * tiling_.rowEachCore * tiling_.colNum, tiling_.rowEachCore * tiling_.colNum);
            this->ComputeLoop(tiling_.rowEachCore, tiling_.colNum, i * tiling_.rowEachCore, gradResfp32Local);
            CopyOut(i * tiling_.rowEachCore * tiling_.colNum, tiling_.rowEachCore * tiling_.colNum);
        }
    }

private:
    __aicore__ inline void CopyInX(uint64_t offset, uint64_t curTileLen)
    {
        auto xLocal = this->inQueueX_.template AllocTensor<DTYPE_X>();
        ::DataCopy(xLocal, this->xGm_[offset], curTileLen);
        this->inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t curTileLen)
    {
        LocalTensor<DTYPE_GRAD_X> outLocal = this->outQueueGradX_.template DeQue<DTYPE_GRAD_X>();
        ::DataCopy(this->gradXGm_[offset], outLocal, curTileLen);
        this->outQueueGradX_.FreeTensor(outLocal);
    }

    __aicore__ inline void ComputeLoop(uint64_t rowNum, uint64_t colNum, uint64_t startRow, LocalTensor<float>& gradResfp32Local)
    {
        auto xLocal = inQueueX_.template DeQue<DTYPE_X>();
        LocalTensor<float> xTemp = x_fp32_buf.Get<float>();
        ::Cast(xTemp, xLocal, RoundMode::CAST_NONE, rowNum * colNum);
        pipe_barrier(PIPE_V);
        inQueueX_.FreeTensor(xLocal);

        for (uint64_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            float factor = gradResfp32Local.GetValue(startRow + rowIndex);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            ::Muls(xTemp[rowIndex * colNum], xTemp[rowIndex * colNum], factor, colNum);

            /***
             * // 可以考虑方案2 后期一次次拷贝这个值
            // float gradRes, res;
            // ::DataCopy(&gradRes, this->gradResGm_[startRow + rowIndex], 1);
            // ::DataCopy(&res, this->resGm_[startRow + rowIndex], 1);
            // set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            //
            // float factor = (-1.0f) * gradRes * res * res * res / (float)colNum;
            // ::Muls(gradXTemp[rowIndex * colNum], xTemp[rowIndex * colNum], factor, colNum);
            // pipe_barrier(PIPE_V);
             **/
        }
        pipe_barrier(PIPE_V);

        auto outLocal = outQueueGradX_.template AllocTensor<DTYPE_GRAD_X>();
        if constexpr (::IsSameType<DTYPE_GRAD_X, half>::value) {
            ::Cast(outLocal, xTemp, RoundMode::CAST_NONE, rowNum * colNum);
        } else {
            ::Cast(outLocal, xTemp, RoundMode::CAST_RINT, rowNum * colNum);
        }
        outQueueGradX_.EnQue(outLocal);
    }

private:
    TQue<QuePosition::VECIN, bufferNum> inQueueX_;
    TQue<QuePosition::VECOUT, bufferNum> outQueueGradX_;
    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> const_fp32_buf;
    TBuf<TPosition::VECCALC> res_fp32_buf;
    LocalTensor<DTYPE_GRAD_RES> const_buf;
    LocalTensor<DTYPE_RES> res_buf;

    GlobalTensor<DTYPE_GRAD_RES> gradResGm_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_RES> resGm_;
    GlobalTensor<DTYPE_GRAD_X> gradXGm_;
    RmsNormWithoutWeightBackwardTilingKernel tiling_;

    uint64_t index_grad_res = 0;
    uint64_t index_grad_resfp32 = 0;
    uint64_t index_res = 0;
    uint64_t index_resfp32 = 0;
};

#endif
