/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_RMS_NORM_WITHOUT_WEIGHT_HPP
#define CANN_RMS_NORM_WITHOUT_WEIGHT_HPP
#include <kernel_operator.h>
#include "utils/op_math.h"
#include "kernel/attention/reduce_xx_nd.hpp"
using namespace Wind;
using namespace AscendC;

namespace AscendC {
constexpr const uint64_t REPEAT_LEN_FLOAT = 64; // 64:float类型的一个repeat
constexpr const uint8_t MIN_BLK_LEN_FLOAT = 8; // 8:float类型的32B
constexpr const uint8_t MIN_BLK_LEN = DIVCEIL(32U, sizeof(DTYPE_X));

    class WindRmsNormWithoutWeight {
    public:
        __aicore__ inline WindRmsNormWithoutWeight(const RmsNormWithoutWeightTilingData& tiling, TPipe& pipe)
            : tiling_(tiling), pipe_(pipe)
        {
        }
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR res)
        {
            // cacl kernel tiling param
            tLenPerCore_ = DIVFLOOR(tiling_.T, GetBlockNum());
            uint32_t preTLen = tLenPerCore_ * GetBlockIdx(); // 本core前面其他core的长度
            uint32_t tLenRemainder = MOD(tiling_.T, GetBlockNum());
            if (tLenRemainder > 0 && GetBlockIdx() < tLenRemainder) {
                tLenPerCore_ += 1; // 不能均分，前面的核多处理一行
                preTLen += GetBlockIdx(); // 前面的核，需要增加对应行数
            } else if (tLenRemainder > 0) {
                preTLen += tLenRemainder; // 后面的核，需要增加余数部分
            }
            tTailLen_ = MOD(tLenPerCore_, tiling_.tBaseLen);
            tTileNum_ = DIVCEIL(tLenPerCore_, tiling_.tBaseLen); // 包含尾块
            dTailLen_ = MOD(tiling_.D, tiling_.dBaseLen);
            dTileNum_ = DIVCEIL(tiling_.D, tiling_.dBaseLen); // 包含尾块

            xBlkSize32B_ = tiling_.tBaseLen * tiling_.dBaseLen; // dBaseLen一定是32B对齐
            resBlkSize32B_ = ALIGNUP(tLenPerCore_, MIN_BLK_LEN_FLOAT);
            lsmBlkSize32B_ = ALIGNUP(tiling_.tBaseLen, MIN_BLK_LEN_FLOAT);

            // x:[T,D]
            xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + preTLen * tiling_.D, tLenPerCore_ * tiling_.D);
            // out:[T,1]
            resGm.SetGlobalBuffer((__gm__ DTYPE_RES*)res + preTLen, tLenPerCore_);

            pipe_.InitBuffer(xBuf_, xBlkSize32B_ * sizeof(float) * 2U);
            pipe_.InitBuffer(outQueueRes, 1, resBlkSize32B_ * sizeof(float));
            pipe_.InitBuffer(sLsumTbuf, lsmBlkSize32B_ * sizeof(float));
            pipe_.InitBuffer(tempBuf, tiling_.reduceBuffLen * sizeof(float)); // 64: ReduceSum 临时工作区， 申请一个repeat的大小

            sLsumTensor_ = sLsumTbuf.Get<float>();
            tempTensor_ = tempBuf.Get<float>();
            resTensor_ = outQueueRes.AllocTensor<float>();
        }

        __aicore__ inline void Process()
        {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            // reducesum
            uint32_t tTileLen = tiling_.tBaseLen;
            uint32_t iter = 0; // 切换 Ping-Pong
            for (uint32_t tTileIdx = 0; tTileIdx < tTileNum_; tTileIdx++) {
                if (tTailLen_ > 0 && (tTileIdx + 1) == tTileNum_) {
                    tTileLen = tTailLen_;
                }
                for (uint32_t dTileIdx = 0; dTileIdx < dTileNum_; dTileIdx++) {
                    uint32_t dTileLen = (dTailLen_ > 0 && (dTileIdx + 1) == dTileNum_) ? dTailLen_ : tiling_.dBaseLen;
                    uint32_t pingPongIdx = iter % 2U; // 0 or 1
                    CopyInX(tTileIdx, dTileIdx, tTileLen, dTileLen, pingPongIdx);
                    ComputeReduceSum(tTileIdx, dTileIdx, tTileLen, dTileLen, pingPongIdx);
                    iter++;
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            // mean,adds,rqst
            ComputeNorm();
        }

        __aicore__ inline void ComputeReduceSum(uint32_t tTileIdx, uint32_t dTileIdx, uint32_t tTileLen, uint32_t dTileLen, uint32_t pingPongIdx)
        {
            if (pingPongIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            }

            uint32_t xBlkSize = tTileLen * dTileLen;
            LocalTensor<float> xLocal = xBuf_.Get<float>();
            xLocal = xLocal[pingPongIdx * xBlkSize32B_];
            // Cast(根据需要 bf16->float)
            if constexpr (!IsSameType<DTYPE_X, float>::value) {
                LocalTensor<DTYPE_X> xLocal_orig = xLocal.ReinterpretCast<DTYPE_X>();
                xLocal_orig = xLocal_orig[xBlkSize32B_]; // 原始数据放在后面一半， cast的结果可以复用内存
                // BF16/HALF -> float
                Cast(xLocal, xLocal_orig, RoundMode::CAST_NONE, xBlkSize);
                PipeBarrier<PIPE_V>();
            }
            // x = x*x
            Mul(xLocal, xLocal, xLocal, xBlkSize);
            PipeBarrier<PIPE_V>();
            // ReduceSum
            uint32_t resOffset = tTileIdx * tiling_.tBaseLen;
            if (dTileIdx == 0) {
                Wind::ReduceXx_Nd<Wind::OpType::SUM, float, true>(resTensor_[resOffset], xLocal, tempTensor_, tTileLen, dTileLen);
            } else {
                Wind::ReduceXx_Nd<Wind::OpType::SUM, float, true>(sLsumTensor_, xLocal, tempTensor_, tTileLen, dTileLen);
                // sLsumTensor_累加到到resTensor_
                Add(resTensor_[resOffset], resTensor_[resOffset], sLsumTensor_, tTileLen);
                PipeBarrier<PIPE_V>();
            }
            if (pingPongIdx == 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
        }

        __aicore__ inline void ComputeNorm()
        {
            // resTensor_ mean,adds,rqst
            Muls(resTensor_, resTensor_, (float)tiling_.avgFactor, tLenPerCore_);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(resTensor_, resTensor_, (float)tiling_.epsilon, tLenPerCore_);
            AscendC::PipeBarrier<PIPE_V>();
            // Rsqrt精度有问题，不能用
            WindRsqrt(resTensor_, resTensor_, tLenPerCore_);
            // Cast(根据需要 float->bf16)
            if constexpr (!IsSameType<DTYPE_RES, float>::value) {
                LocalTensor<DTYPE_RES> resLocal_orig = resTensor_.ReinterpretCast<DTYPE_RES>();
                // float -> BF16/HALF
                Cast(resLocal_orig, resTensor_, RoundMode::CAST_RINT, tLenPerCore_);
                PipeBarrier<PIPE_V>();
            }
            outQueueRes.EnQue(resTensor_);
            // CopyOut
            CopyOutRes();
        }

        // Rsqrt精度有问题，不能用
        // 支持dst和src同地址
        __aicore__ inline void WindRsqrt(const LocalTensor<float>& dst, const LocalTensor<float>& src, const int32_t& count)
        {
            Sqrt(dst, dst, count);
            AscendC::PipeBarrier<PIPE_V>();
            // 求倒数
            Duplicate(tempTensor_, static_cast<float>(1.0), MIN_BLK_LEN_FLOAT);
            PipeBarrier<PIPE_V>();
            uint8_t fullRepeatTime = DIVFLOOR(count, REPEAT_LEN_FLOAT);
            uint64_t tailLen = MOD(count, REPEAT_LEN_FLOAT);
            /* dst和src是同一个对象(resTensor_)，当执行原先的Sqrt(dst, dst) 后，由于地址相同，src内的数据在硬件底层的物理空间上
             也变成了开方后的值，随后的Div(dst, tempTensor_, src)实际上就是用1.0除以了开方后的src */
            if (fullRepeatTime > 0) {
                Div(dst, tempTensor_, src, REPEAT_LEN_FLOAT, fullRepeatTime, {1, 0, 1, 8, 0, 8});
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tailLen) {
                uint64_t offset = fullRepeatTime * REPEAT_LEN_FLOAT;
                Div(dst[offset], tempTensor_, src[offset], tailLen, 1, {1, 0, 1, 8, 0, 8});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        __aicore__ inline void CopyInX(uint32_t tTileIdx, uint32_t dTileIdx, uint32_t tTileLen, uint32_t dTileLen, uint32_t pingPongIdx)
        {
            if (pingPongIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            LocalTensor<float> xLocal = xBuf_.Get<float>();
            xLocal = xLocal[pingPongIdx * xBlkSize32B_]; // 根据 Index 偏移定位物理空间

            LocalTensor<DTYPE_X> xLocal_orig = xLocal.ReinterpretCast<DTYPE_X>();
            if constexpr (!IsSameType<DTYPE_X, float>::value) {
                xLocal_orig = xLocal_orig[xBlkSize32B_]; // 原始数据放在后面一半， cast的结果可以复用内存
            }
            uint64_t gmOffset = tTileIdx * tiling_.tBaseLen * tiling_.D + dTileIdx * tiling_.dBaseLen;
            uint16_t blockLen = DIVCEIL(dTileLen, MIN_BLK_LEN);
            uint16_t srcStride = DIVCEIL((tiling_.D - dTileLen), MIN_BLK_LEN);
            DataCopy(xLocal_orig, xGm[gmOffset], {(uint16_t)tTileLen, blockLen, srcStride, 0});
            if (pingPongIdx == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            }
        }

        __aicore__ inline void CopyOutRes()
        {
            LocalTensor<float> resLocal = outQueueRes.DeQue<float>();
            LocalTensor<DTYPE_RES> resLocal_orig = resLocal.ReinterpretCast<DTYPE_RES>();
            if (MOD(tLenPerCore_, MIN_BLK_LEN) == 0) {
                DataCopy(resGm, resLocal_orig, tLenPerCore_);
            } else {
                DataCopyExtParams dataCopyExtParams;
                dataCopyExtParams.blockCount = 1;
                dataCopyExtParams.blockLen = tLenPerCore_ * sizeof(DTYPE_RES); // 单位：byte
                DataCopyPad(resGm, resLocal_orig, dataCopyExtParams);
            }
            PipeBarrier<PIPE_MTE2>();
            outQueueRes.FreeTensor(resLocal_orig);
        }

    private:
        TBuf<AscendC::TPosition::VECIN> xBuf_;
        TQue<QuePosition::VECOUT, 1> outQueueRes;
        TBuf<AscendC::TPosition::VECCALC> sLsumTbuf;
        TBuf<AscendC::TPosition::VECCALC> tempBuf;

        GlobalTensor<DTYPE_X> xGm;
        GlobalTensor<DTYPE_RES> resGm;
        LocalTensor<float> resTensor_;
        LocalTensor<float> sLsumTensor_;
        LocalTensor<float> tempTensor_;

        // 分核和切分信息
        uint32_t tLenPerCore_;
        uint32_t tTileNum_;
        uint32_t dTileNum_;
        uint32_t tTailLen_;
        uint32_t dTailLen_;

        uint32_t xBlkSize32B_ = 0; // [tBaseLen * dBaseLen]
        uint32_t resBlkSize32B_ = 0;  // [tLenPerCore_ 向32B对齐]
        uint32_t lsmBlkSize32B_ = 0;  // [tBaseLen 向32B对齐]

        TPipe& pipe_;
        const RmsNormWithoutWeightTilingData& tiling_;
    };
} // namespace AscendC
#endif // CANN_RMS_NORM_WITHOUT_WEIGHT_HPP