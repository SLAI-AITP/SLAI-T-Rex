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
#include "wind_fused_overlap_transform_backward_tiling.hpp"

namespace optiling {
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B

class WindFusedOverlapTransformBackwardTilingCalculator final : public CommonTilingBase {
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
        std::cout << "======== wind_fused_overlap_transform_backward TilingPrint ========" << std::endl;
        std::cout << "usedCoreNum:" << usedCoreNum_ << std::endl;
        std::cout << "S:" << tilingData_.get_S() << std::endl;
        std::cout << "R:" << tilingData_.get_R() << std::endl;
        std::cout << "D:" << tilingData_.get_D() << std::endl;
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

        // kv和score合并为一个tensor，B=2，前面一个B为kv，后面一个B为score
        isKvScoreComb_ = GetAttr<bool>(2U); // kv_score_comb

        // grad_kv_out
        auto gradKvOutTensor = GetInputTensor(0);
        inDTypeLen_ = ge::GetSizeByDataType(gradKvOutTensor->GetDataType());
        blkAlignLen_ = DIVCEIL(DEFAULT_ALIGN_SIZE, inDTypeLen_);

        // grad_kv_out
        auto gradKvOutShape = gradKvOutTensor->GetOriginShape();
        // 暂时只支持[B, S, 2R, D]，[S, 2R, D]
        if (gradKvOutShape.GetDimNum() == 3U) {
            if (isKvScoreComb_) {
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_kv_out shape must be [B,S,2R,D], when kv_score_comb is true");
                return false;
            }
            tilingData_.set_S(gradKvOutShape.GetDim(0));
            OP_TILING_CHECK((gradKvOutShape.GetDim(1) != 2U * tilingData_.get_R() || (gradKvOutShape.GetDim(2U) != tilingData_.get_D())),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_kv_out shape is invalid, [S,2R,D], dim[0]:%d, dim[1]:%d, dim[2]:%d, R:%d, D:%d",
                gradKvOutShape.GetDim(0), gradKvOutShape.GetDim(1), gradKvOutShape.GetDim(2),
                tilingData_.get_R(), tilingData_.get_D()), return false);
        } else if (gradKvOutShape.GetDimNum() == 4U) {
            OP_TILING_CHECK((gradKvOutShape.GetDim(0) != 1),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_kv_out shape is invalid, [B,S,2R,D], B:%d, B must be 1.",
                        gradKvOutShape.GetDim(0)), return false);
            tilingData_.set_S(gradKvOutShape.GetDim(1));
            OP_TILING_CHECK((gradKvOutShape.GetDim(2) != 2U * tilingData_.get_R() || (gradKvOutShape.GetDim(3U) != tilingData_.get_D())),
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_kv_out shape is invalid, [B,S,2R,D], dim[0]:%d, dim[1]:%d, dim[2]:%d, dim[3]:%d, R:%d, D:%d",
                gradKvOutShape.GetDim(0), gradKvOutShape.GetDim(1), gradKvOutShape.GetDim(2), gradKvOutShape.GetDim(3),
                tilingData_.get_R(), tilingData_.get_D()), return false);
        } else {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_kv_out shape is invalid, dimnum:%d, only support [B, S, 2R, D] or [S, 2R, D]", gradKvOutShape.GetDimNum());
            return false;
        }

        // grad_score_out
        auto gradScoreOutTensor = GetInputTensor(1);
        auto gradScoreOutShape = gradScoreOutTensor->GetOriginShape();
        if (gradScoreOutShape.GetDimNum() == 0) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_score_out dims=0, when isKvScoreComb_ is false.");
            return false;
        }
        // socre的shape必须与kv一样
        if (gradScoreOutShape.GetDimNum() != gradKvOutShape.GetDimNum()) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_score_out dims:%d must be same with grad_kv_out dims:%d.",
                    gradScoreOutShape.GetDimNum(), gradKvOutShape.GetDimNum());
            return false;
        }

        for (uint32_t i = 0; i < gradScoreOutShape.GetDimNum(); i++) {
            if (gradScoreOutShape.GetDim(i) != gradKvOutShape.GetDim(i)) {
                OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_score_out dim[%d]:%d must be same with grad_kv_out dim[%d]:%d.",
                        i, gradScoreOutShape.GetDim(i), i, gradScoreOutShape.GetDim(i));
                return false;
            }
        }

        // grad_score
        auto gradScoreShape = GetOptionalOutputShape(1); // grad_score
        if (!isKvScoreComb_ && gradScoreShape == nullptr) {
            OP_LOGE("wind_fused_overlap_transform", "[ERROR] grad_score is null, when isKvScoreComb_ is false.");
            return false;
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
    WindFusedOverlapTransformBackwardTilingData tilingData_;
};

inline ge::graphStatus Tiling4WindFusedOverlapTransformBackward(gert::TilingContext* context)
{
    WindFusedOverlapTransformBackwardTilingCalculator tilingCalculator(context);
    return tilingCalculator.DoTiling();
}

IMPL_OP(WindFusedOverlapTransformBackward).Tiling(Tiling4WindFusedOverlapTransformBackward);
}
