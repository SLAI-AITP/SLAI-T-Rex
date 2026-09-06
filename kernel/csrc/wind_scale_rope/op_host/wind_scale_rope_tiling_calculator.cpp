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
#include "utils/op_tiling.h"
#include "utils/op_math.h"
#include "wind_scale_rope_tiling.hpp"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t WSR_ALIGN_SIZE = 32;

#define WSR_INPUT_X_INDEX 0
#define WSR_INPUT_SCALE_INDEX 1
#define WSR_INPUT_COS_INDEX 2
#define WSR_INPUT_SIN_INDEX 3
#define WSR_OUT_INDEX 0

class WindScaleRopeTilingCalculator final : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;

private:
    uint32_t usedCoreNum_ = 1;
    uint8_t xDTypeLen_ = 2;        // bf16

    bool IsCapable()
    {
        if (context_ == nullptr) {
            return false;
        }
        auto inTensor = GetInputTensor(WSR_INPUT_X_INDEX);
        if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
            return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
        }
        return true;
    }

    bool ParseInputParam()
    {
        // x: [T, D] 或 [B,S,N,D] / [S,B,N,D] 摊平成 [T, D]
        auto xTensor = GetInputTensor(WSR_INPUT_X_INDEX);
        xDTypeLen_ = ge::GetSizeByDataType(xTensor->GetDataType());
        auto xShape = xTensor->GetOriginShape();
        uint32_t dimNum = xShape.GetDimNum();
        uint32_t T = 1;
        uint32_t D = xShape.GetDim(dimNum - 1U);
        uint32_t N = 1;
        for (uint32_t i = 0; i < dimNum - 1U; ++i) {
            T *= xShape.GetDim(i);
        }
        // N = 倒数第二维(head 数);若只有2维则 N=1。seq 反算用 seq=(row/N)%S。
        if (dimNum >= 2U) {
            N = xShape.GetDim(dimNum - 2U);
        }
        tilingData_.set_T(T);
        tilingData_.set_D(D);
        tilingData_.set_N(N);

        // cos: [S, ropeDim] (host 已预展开, 每复数 cos 重复2份; sin 同理且偶位取负)
        auto cosTensor = GetInputTensor(WSR_INPUT_COS_INDEX);
        auto cosShape = cosTensor->GetOriginShape();
        uint32_t cosDim = cosShape.GetDimNum();
        uint32_t S = cosShape.GetDim(0);
        uint32_t ropeDim = cosShape.GetDim(cosDim - 1U);
        tilingData_.set_S(S);
        tilingData_.set_ropeDim(ropeDim);
        // seq = row / rowsPerSeq, rowsPerSeq = T/S = B*N。不依赖 x 的维度布局(flatten 与否都对)
        uint32_t rowsPerSeq = (S == 0) ? T : (T / S);
        tilingData_.set_rowsPerSeq(rowsPerSeq);

        // D 必须 16 对齐, ropeDim 必须偶数且 <= D
        OP_TILING_CHECK((MOD(D, 16U) != 0),
            OP_LOGE("wind_scale_rope", "[ERROR] D:%d must be aligned to 16", D), return false);
        OP_TILING_CHECK((tilingData_.get_ropeDim() > D),
            OP_LOGE("wind_scale_rope", "[ERROR] ropeDim:%d > D:%d", tilingData_.get_ropeDim(), D), return false);
        return true;
    }

    // UB 占用估算(每行 D 个元素):
    //  x bf16(in, 2 buf) + xf32(fp32 计算) + scale(brcd) + cos/sin(ropeHalf fp32) + rotate tmp + out bf16(2 buf)
    uint64_t GetUsedUbSizePerRow()
    {
        uint32_t D = tilingData_.get_D();
        uint32_t rope = tilingData_.get_ropeDim();
        uint64_t bytes = 0;
        bytes += 2U * D * sizeof(uint16_t);     // x in, double buffer (bf16)
        bytes += D * sizeof(float);             // xf32 计算缓冲
        bytes += rope * sizeof(float);          // rotate 临时(后64维交换)
        bytes += 2U * rope * sizeof(float);     // cos/sin (fp32)
        bytes += 2U * D * sizeof(uint16_t);     // out, double buffer (bf16)
        return bytes + WSR_ALIGN_SIZE;
    }

    bool CalcOptTiling()
    {
        uint32_t T = tilingData_.get_T();
        usedCoreNum_ = MIN(T, limitCoreNum_);
        uint32_t rowsPerCore = DIVCEIL(T, usedCoreNum_);
        tilingData_.set_usedCore(usedCoreNum_);
        tilingData_.set_rowsPerCore(rowsPerCore);

        // 核内每次 tile 多少行: 用 UB 容量限制
        uint64_t perRow = GetUsedUbSizePerRow();
        uint32_t rowTile = (perRow == 0) ? 1U : static_cast<uint32_t>(limitUbMemSize_ / perRow);
        if (rowTile == 0U) {
            rowTile = 1U;
        }
        if (rowTile > rowsPerCore) {
            rowTile = rowsPerCore;
        }
        tilingData_.set_rowTileLen(rowTile);
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
    [[nodiscard]] uint64_t GetTilingKey() const final { return 0; }
    size_t GetUserWorkspaceSize() final { return 0; }
    uint32_t GetBlockDim() final { return usedCoreNum_; }
    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tilingData_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

private:
    WindScaleRopeTilingData tilingData_;
};

inline ge::graphStatus Tiling4WindScaleRope(gert::TilingContext* context)
{
    WindScaleRopeTilingCalculator tilingCalculator(context);
    return tilingCalculator.DoTiling();
}

IMPL_OP(WindScaleRope).Tiling(Tiling4WindScaleRope);
}
