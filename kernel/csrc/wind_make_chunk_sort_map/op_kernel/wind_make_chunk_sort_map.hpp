/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef WIND_MAKE_CHUNK_SORT_MAP_HPP
#define WIND_MAKE_CHUNK_SORT_MAP_HPP

#include "kernel_operator.h"

namespace AscendC {

template <typename SplitT, typename SortedT>
class WindMakeChunkSortMap {
public:
    __aicore__ inline WindMakeChunkSortMap(const WindMakeChunkSortMapTilingData& tiling, TPipe& pipe)
        : pipe_(pipe)
    {
        numTokens_ = tiling.numTokens;
        numSplits_ = tiling.numSplits;
        usedCore_ = tiling.usedCore;
        tokensPerCore_ = tiling.tokensPerCore;
        outTileLen_ = tiling.outTileLen;
        blockIdx_ = GetBlockIdx();

        tokenStart_ = blockIdx_ * tokensPerCore_;
        tokenEnd_ = tokenStart_ + tokensPerCore_;
        if (tokenEnd_ > numTokens_) {
            tokenEnd_ = numTokens_;
        }
    }

    __aicore__ inline void Process(GM_ADDR splitSizesGm, GM_ADDR sortedIndicesGm, GM_ADDR rowIdMapGm)
    {
        if (blockIdx_ >= usedCore_ || tokenStart_ >= tokenEnd_) {
            return;
        }

        splitSizesGm_.SetGlobalBuffer((__gm__ SplitT*)splitSizesGm);
        sortedIndicesGm_.SetGlobalBuffer((__gm__ SortedT*)sortedIndicesGm);
        rowIdMapGm_.SetGlobalBuffer((__gm__ int32_t*)rowIdMapGm);

        numSplitsAlign_ = AlignUp(numSplits_ + 1U, 8U);
        outTileAlign_ = AlignUp(outTileLen_, 8U);

        pipe_.InitBuffer(splitRawBuf_, numSplitsAlign_ * sizeof(SplitT));
        pipe_.InitBuffer(sortedRawBuf_, numSplitsAlign_ * sizeof(SortedT));
        pipe_.InitBuffer(splitBuf_, numSplitsAlign_ * sizeof(int32_t));
        pipe_.InitBuffer(prefixBuf_, numSplitsAlign_ * sizeof(int32_t));
        pipe_.InitBuffer(inverseBuf_, numSplitsAlign_ * sizeof(int32_t));
        pipe_.InitBuffer(outPrefixBuf_, numSplitsAlign_ * sizeof(int32_t));
        pipe_.InitBuffer(outBuf_, outTileAlign_ * sizeof(int32_t));

        LoadAndBuildMeta();
        EmitRows();
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b)
    {
        return (b == 0U) ? a : ((a + b - 1U) / b * b);
    }

    template <typename T>
    __aicore__ inline int32_t ToInt32(LocalTensor<T> tensor, uint32_t idx)
    {
        return static_cast<int32_t>(tensor.GetValue(idx));
    }

    __aicore__ inline void LoadAndBuildMeta()
    {
        LocalTensor<SplitT> splitRaw = splitRawBuf_.Get<SplitT>();
        LocalTensor<SortedT> sortedRaw = sortedRawBuf_.Get<SortedT>();
        LocalTensor<int32_t> split = splitBuf_.Get<int32_t>();
        LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
        LocalTensor<int32_t> inverse = inverseBuf_.Get<int32_t>();
        LocalTensor<int32_t> outPrefix = outPrefixBuf_.Get<int32_t>();

        DataCopyExtParams copySplit{1, static_cast<uint32_t>(numSplits_ * sizeof(SplitT)), 0, 0, 0};
        DataCopyPadExtParams<SplitT> padSplit{false, 0, 0, 0};
        DataCopyPad(splitRaw, splitSizesGm_, copySplit, padSplit);

        DataCopyExtParams copySorted{1, static_cast<uint32_t>(numSplits_ * sizeof(SortedT)), 0, 0, 0};
        DataCopyPadExtParams<SortedT> padSorted{false, 0, 0, 0};
        DataCopyPad(sortedRaw, sortedIndicesGm_, copySorted, padSorted);
        PipeBarrier<PIPE_ALL>();

        int32_t running = 0;
        for (uint32_t i = 0; i < numSplits_; ++i) {
            int32_t size = ToInt32(splitRaw, i);
            split.SetValue(i, size);
            prefix.SetValue(i, running);
            running += size;
        }
        prefix.SetValue(numSplits_, running);

        int32_t outRunning = 0;
        for (uint32_t r = 0; r < numSplits_; ++r) {
            int32_t srcChunk = ToInt32(sortedRaw, r);
            inverse.SetValue(static_cast<uint32_t>(srcChunk), static_cast<int32_t>(r));
            outPrefix.SetValue(r, outRunning);
            outRunning += split.GetValue(static_cast<uint32_t>(srcChunk));
        }
    }

    __aicore__ inline uint32_t FindChunk(uint32_t token)
    {
        LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
        uint32_t chunk = 0U;
        while ((chunk + 1U) < numSplits_ &&
               static_cast<int32_t>(token) >= prefix.GetValue(chunk + 1U)) {
            ++chunk;
        }
        return chunk;
    }

    __aicore__ inline void EmitRows()
    {
        LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
        LocalTensor<int32_t> inverse = inverseBuf_.Get<int32_t>();
        LocalTensor<int32_t> outPrefix = outPrefixBuf_.Get<int32_t>();
        LocalTensor<int32_t> outLocal = outBuf_.Get<int32_t>();

        uint32_t base = tokenStart_;
        uint32_t chunk = FindChunk(base);
        while (base < tokenEnd_) {
            uint32_t cnt = outTileLen_;
            if (cnt > tokenEnd_ - base) {
                cnt = tokenEnd_ - base;
            }

            for (uint32_t i = 0; i < cnt; ++i) {
                uint32_t token = base + i;
                while ((chunk + 1U) < numSplits_ &&
                       static_cast<int32_t>(token) >= prefix.GetValue(chunk + 1U)) {
                    ++chunk;
                }
                int32_t srcStart = prefix.GetValue(chunk);
                int32_t dstChunk = inverse.GetValue(chunk);
                int32_t dstStart = outPrefix.GetValue(static_cast<uint32_t>(dstChunk));
                outLocal.SetValue(i, dstStart + static_cast<int32_t>(token) - srcStart);
            }

            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(rowIdMapGm_[base], outLocal, copyOut);
            base += cnt;
        }
    }

private:
    TPipe& pipe_;
    TBuf<TPosition::VECCALC> splitRawBuf_, sortedRawBuf_;
    TBuf<TPosition::VECCALC> splitBuf_, prefixBuf_, inverseBuf_, outPrefixBuf_, outBuf_;

    GlobalTensor<SplitT> splitSizesGm_;
    GlobalTensor<SortedT> sortedIndicesGm_;
    GlobalTensor<int32_t> rowIdMapGm_;

    uint32_t numTokens_{0};
    uint32_t numSplits_{0};
    uint32_t usedCore_{1};
    uint32_t tokensPerCore_{0};
    uint32_t outTileLen_{1024};
    uint32_t blockIdx_{0};
    uint32_t tokenStart_{0};
    uint32_t tokenEnd_{0};
    uint32_t numSplitsAlign_{0};
    uint32_t outTileAlign_{0};
};

} // namespace AscendC

#endif // WIND_MAKE_CHUNK_SORT_MAP_HPP
