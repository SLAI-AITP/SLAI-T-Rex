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
#include "wind_scale_rope_grad_tiling.hpp"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t WSRG_ALIGN_SIZE = 32;

#define WSRG_INPUT_GRADOUT_INDEX 0
#define WSRG_INPUT_X_INDEX 1
#define WSRG_INPUT_SCALE_INDEX 2
#define WSRG_INPUT_COS_INDEX 3
#define WSRG_INPUT_SIN_INDEX 4

class WindScaleRopeGradTilingCalculator final : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;

private:
    uint32_t usedCoreNum_ = 1;

    bool IsCapable()
    {
        if (context_ == nullptr) {
            return false;
        }
        auto inTensor = GetInputTensor(WSRG_INPUT_GRADOUT_INDEX);
        if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
            return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
        }
        return true;
    }

    bool ParseInputParam()
    {
        // grad_out: [T, D] 或 [B,S,N,D] 摊平成 [T, D]
        auto gradTensor = GetInputTensor(WSRG_INPUT_GRADOUT_INDEX);
        auto gradShape = gradTensor->GetOriginShape();
        uint32_t dimNum = gradShape.GetDimNum();
        uint32_t T = 1;
        uint32_t D = gradShape.GetDim(dimNum - 1U);
        uint32_t N = 1;
        for (uint32_t i = 0; i < dimNum - 1U; ++i) {
            T *= gradShape.GetDim(i);
        }
        if (dimNum >= 2U) {
            N = gradShape.GetDim(dimNum - 2U);
        }
        tilingData_.set_T(T);
        tilingData_.set_D(D);
        tilingData_.set_N(N);

        auto cosTensor = GetInputTensor(WSRG_INPUT_COS_INDEX);
        auto cosShape = cosTensor->GetOriginShape();
        uint32_t cosDim = cosShape.GetDimNum();
        uint32_t S = cosShape.GetDim(0);
        uint32_t ropeDim = cosShape.GetDim(cosDim - 1U);
        tilingData_.set_S(S);
        tilingData_.set_ropeDim(ropeDim);
        // seq = row / rowsPerSeq, rowsPerSeq = T/S = B*N (shape-agnostic)
        uint32_t rowsPerSeq = (S == 0) ? T : (T / S);
        tilingData_.set_rowsPerSeq(rowsPerSeq);

        OP_TILING_CHECK((MOD(D, 16U) != 0),
            OP_LOGE("wind_scale_rope_grad", "[ERROR] D:%d must be aligned to 16", D), return false);
        OP_TILING_CHECK((tilingData_.get_ropeDim() > D),
            OP_LOGE("wind_scale_rope_grad", "[ERROR] ropeDim:%d > D:%d", tilingData_.get_ropeDim(), D), return false);
        return true;
    }

    bool CalcOptTiling()
    {
        uint32_t T = tilingData_.get_T();
        usedCoreNum_ = MIN(T, limitCoreNum_);
        uint32_t rowsPerCore = DIVCEIL(T, usedCoreNum_);
        tilingData_.set_usedCore(usedCoreNum_);
        tilingData_.set_rowsPerCore(rowsPerCore);
        tilingData_.set_rowTileLen(1U);  // kernel 内按 WSRG_BATCH 固定批, 此字段保留
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
    WindScaleRopeGradTilingData tilingData_;
};

inline ge::graphStatus Tiling4WindScaleRopeGrad(gert::TilingContext* context)
{
    WindScaleRopeGradTilingCalculator tilingCalculator(context);
    return tilingCalculator.DoTiling();
}

IMPL_OP(WindScaleRopeGrad).Tiling(Tiling4WindScaleRopeGrad);
}
