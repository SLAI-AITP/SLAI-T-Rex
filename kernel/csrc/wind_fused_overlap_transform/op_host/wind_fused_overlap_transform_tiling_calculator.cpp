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
#include "utils/op_math.h"
#include "tiling/common_tiling_base.h"
#include "wind_fused_overlap_transform_tiling.hpp"

namespace optiling {
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B

class WindFusedOverlapTransformTilingCalculator final : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;

private:
    uint16_t usedCoreNum_ = 1; // 最终实际使用的核数
    uint8_t inDTypeLen_ = 1; // x输入数据类型的长度
    uint32_t blkAlignLen_;

    uint8_t inTbuf_num_ = 2;
    uint8_t outTbuf_num_ = 2;

    uint32_t tLenPerCore_;
    bool isKvScoreComb_ = false; // kv和comb是否合并在一起，前面1个batch是kv，后面1个batch是score

private:
    // 0 是否支持
    bool IsCapable() { return true; }

    void Print()
    {
#ifdef ENABLE_LOG_ON
        std::cout << "======== wind_fused_overlap_transform TilingPrint ========" << std::endl;
        std::cout << "usedCoreNum:" << usedCoreNum_ << std::endl;
        std::cout << "S:" << tilingData_.get_S() << std::endl;
        std::cout << "R:" << tilingData_.get_R() << std::endl;
        std::cout << "D:" << tilingData_.get_D() << std::endl;
        std::cout << "kvValue:" << tilingData_.get_kvValue() << std::endl;
        std::cout << "scoreValue:" << tilingData_.get_scoreValue() << std::endl;
        std::cout << "tBaseLen:" << tilingData_.get_tBaseLen() << std::endl;
        std::cout << "isKvScoreComb:" << isKvScoreComb_ << std::endl;
        std::cout << "--------------------" << std::endl;
        std::cout << "tTailLen:" << MOD(tilingData_.get_S(), tilingData_.get_tBaseLen()) << " per kv/score" << std::endl;
        std::cout << "tTileNum:" << DIVCEIL(tilingData_.get_S(), tilingData_.get_tBaseLen()) << " per kv/score" << std::endl;
        std::cout << "tLenPerCore:" << tLenPerCore_ << std::endl;
        uint64_t usedUbBytes = GetUsedUbSize(tilingData_.get_tBaseLen(), tilingData_.get_R(), tilingData_.get_D());
        std::cout << "usedUb(byte):" << usedUbBytes << " usedUb(kb):" << (usedUbBytes / 1024U) <<
            " ratio(%):" << (((float)usedUbBytes / limitUbMemSize_) * 100U) << std::endl;
#endif
    }

    bool ParseInputParam()
    {
        tilingData_.set_R(GetAttr<int64_t>(0));
        tilingData_.set_D(GetAttr<int64_t>(1));
        tilingData_.set_kvValue(GetAttr<float>(2U));
        tilingData_.set_scoreValue(GetAttr<float>(3U));

        // kv和score合并为一个tensor，B=2，前面一个B为kv，后面一个B为score
        auto scoreTensor = GetInputTensor(1);
        if (scoreTensor == nullptr) {
            isKvScoreComb_ = true;
        } else {
            isKvScoreComb_ = scoreTensor->GetOriginShape().GetDimNum() == 0;
        }

        // kv
        auto kvTensor = GetInputTensor(0);
        inDTypeLen_ = ge::GetSizeByDataType(kvTensor->GetDataType());
        blkAlignLen_ = DIVCEIL(DEFAULT_ALIGN_SIZE, inDTypeLen_);

        // kv
        auto kvShape = kvTensor->GetOriginShape();
        // 暂时只支持[B, S, R, 2D]，[S, R, 2D]
        if (kvShape.GetDimNum() == 3U) {
            if (isKvScoreComb_) {
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv shape must be [B, S,R,2D], when kv_score_comb is true");
                return false;
            }
            tilingData_.set_S(kvShape.GetDim(0));
            OP_TILING_CHECK((kvShape.GetDim(1) != tilingData_.get_R() || (kvShape.GetDim(2U) != 2U * tilingData_.get_D())),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv shape is invalid, [S,R,2D], dim[0]:%d, dim[1]:%d, dim[2]:%d, R:%d, D:%d",
                kvShape.GetDim(0), kvShape.GetDim(1), kvShape.GetDim(2),
                tilingData_.get_R(), tilingData_.get_D()), return false);
        } else if (kvShape.GetDimNum() == 4U) {
            OP_TILING_CHECK(((isKvScoreComb_ && kvShape.GetDim(0) != 2U) || (!isKvScoreComb_ && kvShape.GetDim(0) != 1)),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv shape is invalid, [B,S,R,2D], dim[0]:%d, isKvScoreComb_:%d. "
                "B must be 2 when kv_score_comb is true, or B must be 1 when kv_score_comb is false.",
                kvShape.GetDim(0), isKvScoreComb_), return false);
            tilingData_.set_S(kvShape.GetDim(1));
            OP_TILING_CHECK((kvShape.GetDim(2) != tilingData_.get_R() || (kvShape.GetDim(3U) != 2U * tilingData_.get_D())),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv shape is invalid, [B,S,R,2D], dim[0]:%d, dim[1]:%d, dim[2]:%d, dim[3]:%d, R:%d, D:%d",
                kvShape.GetDim(0), kvShape.GetDim(1), kvShape.GetDim(2), kvShape.GetDim(3),
                tilingData_.get_R(), tilingData_.get_D()), return false);
        } else {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] kv shape is invalid, dimnum:%d, only support [B, S, R, 2D] or [S, R, 2D]", kvShape.GetDimNum());
            return false;
        }

        if (!isKvScoreComb_) {
            // score
            auto scoreShape = scoreTensor->GetOriginShape();
            // socre的shape必须与kv一样
            if (scoreShape.GetDimNum() != kvShape.GetDimNum()) {
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] score dims:%d must be same with kv dims:%d.",
                        scoreShape.GetDimNum(), kvShape.GetDimNum());
                return false;
            }

            for (uint32_t i = 0; i < scoreShape.GetDimNum(); i++) {
                if (scoreShape.GetDim(i) != kvShape.GetDim(i)) {
                    OP_LOGE("wind_fused_overlap_transform", "[ERROR] score dim[%d]:%d must be same with kv dim[%d]:%d.",
                            i, scoreShape.GetDim(i), i, scoreShape.GetDim(i));
                    return false;
                }
            }
        }

        OP_TILING_CHECK((tilingData_.get_R() != 4U),
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] compress_ratio:%d is invalid, only support 4.",
            tilingData_.get_R()), return false);
        OP_TILING_CHECK((MOD(tilingData_.get_D(), blkAlignLen_) != 0),
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] head_dim:%d is invalid, must be aligned to 32B.",
            tilingData_.get_D()), return false);
        return true;
    }

    uint64_t GetUsedUbSize(uint32_t tBaseLen, uint32_t compress_ratio, uint32_t head_dim)
    {
        // inBuf   tBaseLen * R * D
        uint64_t usedUbSize = inTbuf_num_ * tBaseLen * compress_ratio * head_dim * inDTypeLen_;
        // outBuf  tBaseLen * 2R * D
        usedUbSize += outTbuf_num_ * tBaseLen* 2U * compress_ratio * head_dim * inDTypeLen_;
        return usedUbSize;
    }

    uint64_t GetMaxTBaseLenPerUB(uint32_t compress_ratio, uint32_t head_dim)
    {
        // inBuf   tBaseLen * R * D
        uint64_t factor = inTbuf_num_ * compress_ratio * head_dim * inDTypeLen_;
        // outBuf  tBaseLen * 2R * D
        factor += outTbuf_num_ * 2U * compress_ratio * head_dim * inDTypeLen_;
        return DIVFLOOR(limitUbMemSize_, factor);
    }

    bool CaclOptTiling()
    {
        // kv和score各用一半的核，先分核，前面一半核处理kv，后面一半核处理score
        uint16_t limitCoreNum_half = limitCoreNum_ / 2U;
        uint16_t usedCoreNum_half = MIN(tilingData_.get_S(), limitCoreNum_half);
        usedCoreNum_ = usedCoreNum_half * 2U;
        tLenPerCore_ = DIVCEIL(tilingData_.get_S(), usedCoreNum_half);

        // 核内切分, colLen保持固定
        uint32_t colLen = 2U * tilingData_.get_R() * tilingData_.get_D();
        uint64_t tBaseLen = GetMaxTBaseLenPerUB(tilingData_.get_R(), tilingData_.get_D());
        tBaseLen = MIN(tBaseLen, tLenPerCore_);
        tilingData_.set_tBaseLen(tBaseLen);
        return true;
    }

    // 3.1 计算Tiling
    ge::graphStatus CalcTiling()
    {
        if (!ParseInputParam()) {
            return ge::GRAPH_FAILED;
        }
        if (!CaclOptTiling()) {
            return ge::GRAPH_FAILED;
        }

        Print();
        return ge::GRAPH_SUCCESS;
    }
    // 3.2 校验Tiling结果
    ge::graphStatus CheckTiling() { return ge::GRAPH_SUCCESS; }
    // 5 计算TilingKey
    [[nodiscard]] uint64_t GetTilingKey() const final { return isKvScoreComb_ ? 1 : 0; }
    // 6.1 获取用户workspace大小
    size_t GetUserWorkspaceSize() final { return 0; }
    // 7.1 获取计算核数
    uint32_t GetBlockDim() final { return usedCoreNum_; }
    // 7.2 设置TilingData
    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tilingData_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

private:
    WindFusedOverlapTransformTilingData tilingData_;
};

inline ge::graphStatus Tiling4WindFusedOverlapTransform(gert::TilingContext* context)
{
    WindFusedOverlapTransformTilingCalculator tilingCalculator(context);
    return tilingCalculator.DoTiling();
}

IMPL_OP(WindFusedOverlapTransform).Tiling(Tiling4WindFusedOverlapTransform);
}
