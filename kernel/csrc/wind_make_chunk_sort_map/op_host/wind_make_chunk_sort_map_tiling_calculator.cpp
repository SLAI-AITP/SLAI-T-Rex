/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <register/op_def_registry.h>
#include "tiling/common_tiling_base.h"
#include "utils/op_math.h"
#include "wind_make_chunk_sort_map_tiling.hpp"

namespace optiling {
namespace {
constexpr uint32_t INPUT_SPLIT_SIZES = 0U;
constexpr uint32_t ATTR_NUM_TOKENS = 0U;
constexpr uint32_t OUT_TILE_LEN = 1024U;
} // namespace

class WindMakeChunkSortMapTilingCalculator final : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;

private:
    uint32_t usedCoreNum_ = 1U;
    WindMakeChunkSortMapTilingData tilingData_;

    bool ParseInputParam()
    {
        auto splitTensor = GetInputTensor(INPUT_SPLIT_SIZES);
        auto splitShape = splitTensor->GetOriginShape();
        OP_TILING_CHECK((splitShape.GetDimNum() != 1U),
            OP_LOGE("wind_make_chunk_sort_map", "[ERROR] split_sizes must be 1D"), return false);

        auto attrs = context_->GetAttrs();
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        const int64_t* numTokensPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOKENS);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, numTokensPtr);

        int64_t numTokens = *numTokensPtr;
        OP_TILING_CHECK((numTokens < 0),
            OP_LOGE("wind_make_chunk_sort_map", "[ERROR] num_tokens must be non-negative"), return false);

        tilingData_.set_numTokens(static_cast<uint32_t>(numTokens));
        tilingData_.set_numSplits(static_cast<uint32_t>(splitShape.GetDim(0)));
        tilingData_.set_outTileLen(OUT_TILE_LEN);
        return true;
    }

    bool CalcOptTiling()
    {
        uint32_t numTokens = tilingData_.get_numTokens();
        usedCoreNum_ = (numTokens == 0U) ? 1U : MIN(numTokens, limitCoreNum_);
        tilingData_.set_usedCore(usedCoreNum_);
        tilingData_.set_tokensPerCore(DIVCEIL(numTokens, usedCoreNum_));
        return true;
    }

    ge::graphStatus CalcTiling()
    {
        if (!ParseInputParam()) {
            return ge::GRAPH_FAILED;
        }
        if (!CalcOptTiling()) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckTiling() final { return ge::GRAPH_SUCCESS; }
    [[nodiscard]] uint64_t GetTilingKey() const final { return 0U; }
    size_t GetUserWorkspaceSize() final { return 0U; }
    uint32_t GetBlockDim() final { return usedCoreNum_; }

    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tilingData_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
};

inline ge::graphStatus Tiling4WindMakeChunkSortMap(gert::TilingContext* context)
{
    WindMakeChunkSortMapTilingCalculator calculator(context);
    return calculator.DoTiling();
}

IMPL_OP(WindMakeChunkSortMap).Tiling(Tiling4WindMakeChunkSortMap);
} // namespace optiling
