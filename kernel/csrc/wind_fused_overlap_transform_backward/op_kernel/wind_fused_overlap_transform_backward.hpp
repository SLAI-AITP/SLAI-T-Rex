/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_WindFusedOverlapTransformBackward_HPP
#define GTSOPS_WindFusedOverlapTransformBackward_HPP
#include <kernel_operator.h>
#include "utils/op_math.h"

using namespace AscendC;
namespace AscendC {
#define ONE_BLOCK_LEN_FLOAT 8 // float类型一个block（32B）的长度
#define ONE_REPEAT_LEN_FLOAT 64 // float类型一个repeat（256B）的长度
#define ONE_BLOCK_SIZE 32 // 一个block（32B）的长度
#define IN_TBUF_NUM 2
#define OUT_TBUF_NUM 2

constexpr const uint16_t MIN_BLK_LEN = DIVCEIL(ONE_BLOCK_SIZE, sizeof(DTYPE_KV_OUT_GRAD));

template<bool isKvScoreComb = false>
struct WindFusedOverlapTransformBackwardKernel {
public:
    __aicore__ inline WindFusedOverlapTransformBackwardKernel(const WindFusedOverlapTransformBackwardTilingData& tiling, TPipe& pipe)
        : tiling_(tiling),
        pipe_(pipe)
    {
    }

    __aicore__ inline void Init(GM_ADDR kv_out_gradGm, GM_ADDR score_out_gradGm, GM_ADDR grad_kvGm, GM_ADDR grad_scoreGm)
    {
        uint16_t blockNum_half = DIVFLOOR(GetBlockNum(), 2U); // 一半和核处理kv，一半的和处理score
        bool isKv = GetBlockIdx() < blockNum_half; // 前面一半核处理kv，后面一半核处理score
        lastBlockIdx_ = blockNum_half - 1;
        curBlockIdx_ = (GetBlockIdx() < blockNum_half) ? GetBlockIdx() : (GetBlockIdx() - blockNum_half);
        uint32_t S = tiling_.S; // 如果kv和score合并，这里还要取一半

        // cacl kernel tiling param
        tLenPerCore_ = DIVFLOOR(S, blockNum_half);
        preTLenPerCore_ = tLenPerCore_ * curBlockIdx_; // 本core前面其他core的长度
        uint32_t tLenRemainder = MOD(S, blockNum_half);
        if (tLenRemainder > 0 && curBlockIdx_ < tLenRemainder) {
            tLenPerCore_ += 1; // 不能均分，前面的核多处理一行
            preTLenPerCore_ += curBlockIdx_; // 前面的核，需要增加对应行数
        } else if (tLenRemainder > 0) {
            preTLenPerCore_ += tLenRemainder; // 后面的核，需要增加余数部分
        }

        tTailLen_ = MOD(tLenPerCore_, tiling_.tBaseLen);
        tTileNum_ = DIVCEIL(tLenPerCore_, tiling_.tBaseLen); // 包含尾块

        // ub内存操作参数，32B对齐
        colLen_per_seq_ = 2U * tiling_.R * tiling_.D;
        inBlkSize32B_ = tiling_.tBaseLen * tiling_.R * tiling_.D; // D一定是16对齐的，tiling保证
        outBlkSize32B_ = tiling_.tBaseLen * colLen_per_seq_; // D一定是16对齐的，tiling保证

        // init gm buffer 从最原始开始
        GM_ADDR inGm = isKv ? kv_out_gradGm : score_out_gradGm;
        inGm_.SetGlobalBuffer((__gm__ DTYPE_KV_OUT_GRAD*)inGm, tiling_.S * colLen_per_seq_);
        if constexpr (isKvScoreComb) {
            // 如果kv_score合并，前一半是kv_grad，后面一半是score_grad
            if (isKv) {
                outGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_KV*)grad_kvGm, tiling_.S * colLen_per_seq_);
            } else {
                outGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_KV*)grad_kvGm + tiling_.S * colLen_per_seq_, tiling_.S * colLen_per_seq_);
            }
        } else {
            GM_ADDR outGm = isKv ? grad_kvGm : grad_scoreGm;
            outGm_.SetGlobalBuffer((__gm__ DTYPE_GRAD_KV*)outGm, tiling_.S * colLen_per_seq_);
        }

        // init kernel buffer, 精度要求，都按照float来申请
        pipe_.InitBuffer(inTbuf_, IN_TBUF_NUM * inBlkSize32B_ * sizeof(DTYPE_KV_OUT_GRAD));
        pipe_.InitBuffer(outTbuf_, OUT_TBUF_NUM * outBlkSize32B_ * sizeof(DTYPE_GRAD_KV));

        inTensor_ = inTbuf_.Get<DTYPE_KV_OUT_GRAD>();
        outTensor_ = outTbuf_.Get<DTYPE_GRAD_KV>();
    }

    __aicore__ inline void Process(GM_ADDR kv_out_gradGm, GM_ADDR score_out_gradGm, GM_ADDR grad_kvGm, GM_ADDR grad_scoreGm)
    {
        Init(kv_out_gradGm, score_out_gradGm, grad_kvGm, grad_scoreGm);
        SyncBegin();

        in_pp_flag_ = 0;
        out_pp_flag_ = 0;

        uint32_t tTileLen = tiling_.tBaseLen;
        for (uint32_t tTileIdx = 0; tTileIdx < tTileNum_; tTileIdx++) {
            // T尾块判决
            if ((tTailLen_ > 0) && ((tTileIdx + 1) == tTileNum_)) {
                tTileLen = tTailLen_;
            }

            bool isLastTile = (curBlockIdx_ == lastBlockIdx_ && (tTileIdx + 1) == tTileNum_); // kv/score的最后一个分块

            ProcessTile(tTileIdx, tTileLen, isLastTile);

            if constexpr (OUT_TBUF_NUM > 1) {
                out_pp_flag_ = 1 - out_pp_flag_;
            }
        }

        SyncEnd();
    }

private:
    __aicore__ inline void SyncBegin()
    {
        set_flag(PIPE_V, PIPE_MTE2, (event_t)0); // VEC通知MTE2, inTensor已经空闲
        if constexpr (IN_TBUF_NUM > 1) {
            set_flag(PIPE_V, PIPE_MTE2, (event_t)1); // VEC通知MTE2, inTensor已经空闲
        }
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0); // MTE3通知VEC, outTensor_已经空闲
        if constexpr (OUT_TBUF_NUM > 1) {
            set_flag(PIPE_MTE3, PIPE_V, (event_t)1); // MTE3通知VEC, outTensor_已经空闲
        }
    }

    __aicore__ inline void SyncEnd()
    {
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)0); // VEC通知MTE2, inTensor已经空闲
        if constexpr (IN_TBUF_NUM > 1) {
            wait_flag(PIPE_V, PIPE_MTE2, (event_t)1); // VEC通知MTE2, inTensor已经空闲
        }
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0); // 等待MTE3通知VEC, outTensor_已经空闲
        if constexpr (OUT_TBUF_NUM > 1) {
            wait_flag(PIPE_MTE3, PIPE_V, (event_t)1); // 等待MTE3通知VEC, outTensor_已经空闲
        }
    }

    // 处理一个分块 矩形分块
    __aicore__ inline void ProcessTile(uint32_t tTileIdx, uint32_t tTileLen, bool isLastTile)
    {
        LocalTensor<DTYPE_KV_OUT_GRAD> inLocal_frontD = inTensor_[inBlkSize32B_ * in_pp_flag_];
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)in_pp_flag_); // VEC通知MTE2, inTensor已经空闲
        CopyInFrontD(inLocal_frontD, tTileIdx, tTileLen, isLastTile);
        set_flag(PIPE_MTE2, PIPE_V, (event_t)in_pp_flag_); // MTE2通知VEC, inTensor已经copy好
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)in_pp_flag_); // 等待MTE2通知VEC, inTensor已经copy好
        if (isLastTile) {
            InitValueFrontD(inLocal_frontD, tTileLen);
        }

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)out_pp_flag_); // 等待MTE3通知VEC, outTensor_已经空闲
        LocalTensor<DTYPE_GRAD_KV> outLocal = out_pp_flag_ ? outTensor_[outBlkSize32B_] : outTensor_;
        TransFrontD2Out(outLocal, inLocal_frontD, tTileIdx, tTileLen); // 将front D内存转移到out中
        set_flag(PIPE_V, PIPE_MTE2, (event_t)in_pp_flag_); // VEC通知MTE2, inTensor已经空闲
        in_pp_flag_++;
        in_pp_flag_ = ((in_pp_flag_ == IN_TBUF_NUM) ? 0 : in_pp_flag_);

        LocalTensor<DTYPE_KV_OUT_GRAD> inLocal_backD = inTensor_[inBlkSize32B_ * in_pp_flag_];
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)in_pp_flag_); // VEC通知MTE2, inTensor已经空闲
        CopyInBackD(inLocal_backD, tTileIdx, tTileLen);
        set_flag(PIPE_MTE2, PIPE_V, (event_t)in_pp_flag_); // MTE2通知VEC, inTensor已经copy好
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)in_pp_flag_); // 等待MTE2通知VEC, inTensor已经copy好
        TransBackD2Out(outLocal, inLocal_backD, tTileIdx, tTileLen); // 将back D内存转移到out中
        set_flag(PIPE_V, PIPE_MTE2, (event_t)in_pp_flag_); // VEC通知MTE2, inTensor已经空闲
        in_pp_flag_++;
        in_pp_flag_ = ((in_pp_flag_ == IN_TBUF_NUM) ? 0 : in_pp_flag_);

        set_flag(PIPE_V, PIPE_MTE3, (event_t)out_pp_flag_); // VEC通知MTE3, outTensor_已经计算好
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)out_pp_flag_); // 等待VEC通知MTE3, outTensor_已经计算好
        CopyOut(outLocal, tTileIdx, tTileLen);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)out_pp_flag_); // MTE3通知VEC, outTensor_已经空闲
    }

    __aicore__ inline void InitValueFrontD(LocalTensor<DTYPE_KV_OUT_GRAD>& inLocal_frontD, uint32_t tTileLen)
    {
        // 将最后行的前面R*D个元素初始化为0， 有间隔D
        Duplicate(inLocal_frontD[(tTileLen - 1) * tiling_.R * tiling_.D], static_cast<DTYPE_GRAD_KV>(0.0), tiling_.R * tiling_.D);
        pipe_barrier(PIPE_V);
    }

    // tTileIdx和tTileLen是out的，对应in需要进行转换
    __aicore__ inline void CopyInFrontD(LocalTensor<DTYPE_KV_OUT_GRAD>& inLocal_frontD, uint32_t tTileIdx, uint32_t tTileLen, bool isLastTile)
    {
        // 先将其他核和其他分块的跳过
        uint64_t gmOffset = (preTLenPerCore_ + tTileIdx * tiling_.tBaseLen) * colLen_per_seq_;
        // 往后看一行
        gmOffset += colLen_per_seq_;
        uint16_t blockCnt;
        uint16_t blockLen = tiling_.R * tiling_.D / MIN_BLK_LEN ; // element -> 32B;
        uint16_t srcGap = blockLen; // 计算结果与blockLen一样，直接复用
        if (isLastTile) {
            if (tTileLen == 1) {
                return; // 只有一行，需不要copyin front D
            }
            // 少copy一个seq
            blockCnt = tTileLen - 1;
        } else {
            blockCnt = tTileLen;
        }

        ::DataCopy(inLocal_frontD, inGm_[gmOffset], {blockCnt, blockLen, srcGap, 0});
        pipe_barrier(PIPE_MTE2);
    }

    __aicore__ inline void CopyInBackD(LocalTensor<DTYPE_KV_OUT_GRAD>& inLocal_backD, uint32_t tTileIdx, uint32_t tTileLen)
    {
        // 先将其他核和其他分块的跳过
        uint64_t gmOffset = (preTLenPerCore_ + tTileIdx * tiling_.tBaseLen) * colLen_per_seq_;
        // 跳过front D
        gmOffset += tiling_.R * tiling_.D;
        uint16_t blockCnt = tTileLen;
        uint16_t blockLen = tiling_.R * tiling_.D / MIN_BLK_LEN ; // element -> 32B;
        uint16_t srcGap = blockLen; // 计算结果与blockLen一样，直接复用
        ::DataCopy(inLocal_backD, inGm_[gmOffset], {blockCnt, blockLen, srcGap, 0});
        pipe_barrier(PIPE_MTE2);
    }

    // inLocal_frontD中已经补齐了最后一个seq的0，这里直接映射到out
    __aicore__ inline void TransFrontD2Out(LocalTensor<DTYPE_GRAD_KV>& outLocal, LocalTensor<DTYPE_KV_OUT_GRAD>& inLocal_frontD, uint32_t tTileIdx, uint32_t tTileLen)
    {
        uint16_t blockCnt = tTileLen * tiling_.R;
        uint16_t blockLen = tiling_.D / MIN_BLK_LEN; // element -> 32B;
        uint16_t srcGap = blockLen; // 计算结果与blockLen一样，直接复用

        ::DataCopy(outLocal, inLocal_frontD, {blockCnt, blockLen, 0, srcGap});
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void TransBackD2Out(LocalTensor<DTYPE_GRAD_KV>& outLocal, LocalTensor<DTYPE_KV_OUT_GRAD>& inLocal_backD, uint32_t tTileIdx, uint32_t tTileLen)
    {
        uint16_t blockCnt = tTileLen * tiling_.R;
        uint16_t blockLen = tiling_.D / MIN_BLK_LEN ; // element -> 32B;
        uint16_t srcGap = blockLen; // 计算结果与blockLen一样，直接复用
        ::DataCopy(outLocal[tiling_.D], inLocal_backD, {blockCnt, blockLen, 0, srcGap});
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void CopyOut(LocalTensor<DTYPE_GRAD_KV>& outLocal, uint32_t tTileIdx, uint32_t tTileLen)
    {
        // 先将其他核和其他分块的跳过
        uint64_t gmOffset = (preTLenPerCore_ + tTileIdx * tiling_.tBaseLen) * colLen_per_seq_;
        ::DataCopy(outGm_[gmOffset], outLocal, tTileLen * colLen_per_seq_);
        pipe_barrier(PIPE_MTE3);
    }

private:
    // tiling切分参数
    uint32_t tLenPerCore_ = 0; // 每个核需要处理的tLenPerCore
    uint32_t preTLenPerCore_ = 0; // 每个核需要处理的行，前面有多少行是被其他核处理的。 out的角度看
    uint32_t tTailLen_ = 0; // T维度切分后的尾块
    uint32_t tTileNum_ = 0; // T维度切分的基块数量，包含尾块
    // // ub内存操作参数，32B对齐
    uint32_t inBlkSize32B_ = 0; // ub上kv/score分块的大小（单位：element）
    uint32_t outBlkSize32B_ = 0; // ub上kv_out/socre_out分块的大小（单位：element）
    uint8_t in_pp_flag_ = 0; // 0:ping, 1:pong
    uint8_t out_pp_flag_ = 0; // 0:ping, 1:pong

    uint16_t curBlockIdx_; // 经过转换后的blockidx，后面一半核处理socre，blockIdx映射从0开始（折半之后）
    uint16_t lastBlockIdx_; // 处理kv或者score的最后一个block idx(折半之后)
    uint32_t colLen_per_seq_; // tiling_R * 2 * tiling_.D

    TBuf<TPosition::VECCALC> inTbuf_; // pingpong
    TBuf<TPosition::VECCALC> outTbuf_; // pingpong

    LocalTensor<DTYPE_KV_OUT_GRAD> inTensor_;
    LocalTensor<DTYPE_GRAD_KV> outTensor_;

    GlobalTensor<DTYPE_KV_OUT_GRAD> inGm_;
    GlobalTensor<DTYPE_GRAD_KV> outGm_;

    const WindFusedOverlapTransformBackwardTilingData& tiling_;
    TPipe& pipe_;
};
} // namespace AscendC
#endif // GTSOPS_WindFusedOverlapTransformBackward_HPP