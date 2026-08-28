/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_RMS_NORM_WITHOUT_WEIGHT_TILING_H
#define OP_RMS_NORM_WITHOUT_WEIGHT_TILING_H
#include <register/tilingdata_base.h>
#include <exe_graph/runtime/tiling_context.h>
#include "tiling/common_tiling_base.h"
#include "../../include/log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/op_math.h"
#include "utils/op_tiling.h"

namespace optiling {
const uint32_t RNW_DEFAULT_BUFFER_NUM = 2;
const uint32_t RNW_MAX_CORE_NUMBER = 64;
const uint32_t RNW_T_BUFFER_NUM = 1;
const uint32_t RNW_ALIGN_NUM = 16;
const uint32_t RNW_LONG_SHAPE_COLNUM_310 = 8192;

uint32_t RNW_LONG_SHAPE_COLNUM = 5120;

const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B
const uint32_t MIN_INPUT_DTYPELEN = 2;
const uint32_t MIN_BLK_LEN_FLOAT = 8; // float32字节对齐
const uint32_t MIN_TBASELEN = 8; // t维度基块切分最小8个

    BEGIN_TILING_DATA_DEF(RmsNormWithoutWeightTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, T);          // 合并后的总行数 (B * S)
    TILING_DATA_FIELD_DEF(uint32_t, D);          // 总列数
    TILING_DATA_FIELD_DEF(uint32_t, tBaseLen);   // 行基块切分长度
    TILING_DATA_FIELD_DEF(uint32_t, dBaseLen);   // 列基块切分长度
    TILING_DATA_FIELD_DEF(uint32_t, reduceBuffLen);   // reduceSum需要的buff長度，元素個數
    TILING_DATA_FIELD_DEF(float, epsilon);      // Adds 标量值
    TILING_DATA_FIELD_DEF(float, avgFactor);
    END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindRmsNormWithoutWeight, RmsNormWithoutWeightTilingData);

struct RmsNormWithoutWeightTilingCalculator : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;
    [[nodiscard]] uint64_t GetTilingKey() const final { return 0; }
    size_t GetUserWorkspaceSize() final { return 0; }
    uint32_t GetBlockDim() final { return usedCoreNum_; }
    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tilingData_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    inline bool IsCapable()
    {
        if (context_ == nullptr) {
            return false;
        }
        auto inTensor = GetInputTensor(0);
        if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
            return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
        }
        return true;
    }

    inline void PrintTiling()
    {
        std::cout << "======RmsNormWithoutWeight tiling======" << std::endl;
        std::cout << " T=" << tilingData_.get_T() << std::endl;
        std::cout << " D=" << tilingData_.get_D() << std::endl;
        std::cout << " tBaseLen=" << tilingData_.get_tBaseLen() << std::endl;
        std::cout << " dBaseLen=" << tilingData_.get_dBaseLen() << std::endl;
        std::cout << " reduceBuffLen=" << tilingData_.get_reduceBuffLen() << std::endl;
        std::cout << " avgFactor=" << tilingData_.get_avgFactor() << std::endl;
        std::cout << " epsilon=" << tilingData_.get_epsilon() << std::endl;
        std::cout << "-------------------------" << std::endl;
        std::cout << " xDTypeLen_=" << xDTypeLen_ << std::endl;
        std::cout << " l2CacheLineLen_=" << l2CacheLineLen_ << std::endl;
        std::cout << " blkAlignLen_=" << blkAlignLen_ << std::endl;
        std::cout << " usedCoreNum_=" << usedCoreNum_ << std::endl;
        std::cout << " tLenPerCore_=" << tLenPerCore_ << std::endl;
        uint64_t usedUbBytes = CaclUbSize(tilingData_.get_tBaseLen(), tilingData_.get_dBaseLen());
        std::cout << "usedUb(byte):" << usedUbBytes << " usedUb(kb):" << (usedUbBytes / 1024U) <<
                  " ratio(%):" << (((float)usedUbBytes / limitUbMemSize_) * 100U) << std::endl;
    }

    ge::graphStatus ParseParam()
    {
        auto xTensor = GetInputTensor(0);
        xDTypeLen_ = ge::GetSizeByDataType(xTensor->GetDataType());
        blkAlignLen_ = DIVCEIL(DEFAULT_ALIGN_SIZE, xDTypeLen_);
        l2CacheLineLen_ = DIVCEIL(PACK_SIZE, xDTypeLen_);

        auto xShape = xTensor->GetOriginShape();
        if (xShape.GetDimNum() < 2U) {
            OP_LOGE("CheckTiling error", "input dim num = %d of x is invalid, must be >= 2.", xShape.GetDimNum());
            return ge::GRAPH_PARAM_INVALID;
        }
        tilingData_.set_D(xShape.GetDim(xShape.GetDimNum() - 1));
        uint32_t T = 1;
        for (int i = 0; i < xShape.GetDimNum() - 1; ++i) {
            T *= xShape.GetDim(i);
        }
        tilingData_.set_T(T);

        if (tilingData_.get_D() == 0 || MOD(tilingData_.get_D(), blkAlignLen_) != 0) {
            OP_LOGE("CheckTiling",
                    "The input tensor last dimension (D=%ld) is invalid. "
                    "Requirement: must be 32-byte aligned. ",
                    tilingData_.get_D());
            return ge::GRAPH_PARAM_INVALID;
        }
        float avg_factor =  1.0 / tilingData_.get_D();
        tilingData_.set_avgFactor(avg_factor);
        float epsilon = *(GetAttrs()->GetAttrPointer<float>(0));
        tilingData_.set_epsilon(epsilon);
        return ge::GRAPH_SUCCESS;
    }

    // 将tBaseLen=1,根据ub内存使用情况，计算最大的dBaseLen
    uint64_t GetMaxDBaseLenPerUB(uint32_t tBaseLen_min)
    {
        uint64_t leftUbSize = limitUbMemSize_ - CalcUbSize_without_x(tBaseLen_min);
        // x和tempBuffer與dBaseLen有关，tempBuffer先简化处理（按照1行dBaseLen来计算）
        uint32_t factor = (2U * tBaseLen_min + 1) * sizeof(float); // 64: float一個repeat的長度
        uint32_t dBaseLen = DIVFLOOR(leftUbSize, factor);
        return Align(dBaseLen, blkAlignLen_);
    }

    uint64_t CalcUbSize_without_x(uint32_t tBaseLen)
    {
        // res
        uint64_t ubUsedSize = ALIGNUP(tLenPerCore_, MIN_BLK_LEN_FLOAT) * sizeof(float);
        // sLsm
        ubUsedSize += ALIGNUP(tBaseLen, MIN_BLK_LEN_FLOAT) * sizeof(float);
        return ubUsedSize;
    }

    uint64_t CaclUbSize(uint32_t tBaseLen, uint32_t dBaseLen)
    {
        // x
        uint64_t ubUsedSize = 2U * tBaseLen * dBaseLen * sizeof(float); // ub内存大小按照float开辟
        // tempBuffer
        ubUsedSize += GetReduceXxNdTempBuffLen<float, true>(tBaseLen, dBaseLen) * sizeof(float);
        // others
        ubUsedSize += CalcUbSize_without_x(tBaseLen);
        return ubUsedSize;
    }

    bool CheckUbSize(uint32_t tBaseLen, uint32_t dBaseLen)
    {
        return CaclUbSize(tBaseLen, dBaseLen) <= limitUbMemSize_;
    }

    void CaclOptTiling()
    {
        // 先分核
        usedCoreNum_ = MIN(tilingData_.get_T(), limitCoreNum_);
        tLenPerCore_ = DIVCEIL(tilingData_.get_T(), usedCoreNum_);
        // 核内分配最优基块计算
        uint64_t dBaseLenMax = GetMaxDBaseLenPerUB(MIN_TBASELEN);
        // 第一版tiling简化处理
        if (dBaseLenMax < tilingData_.get_D()) {
            // ub放不下整个D，TtBaseLen维度就1行，D需要切分，dBaseLenMax调整到32B对齐或者512B对齐
            tilingData_.set_tBaseLen(MIN(MIN_TBASELEN, tLenPerCore_));
            if (MOD(dBaseLenMax, l2CacheLineLen_) == 0 || dBaseLenMax <= blkAlignLen_) {
                tilingData_.set_dBaseLen(dBaseLenMax);
            } else if (dBaseLenMax > l2CacheLineLen_) {
                // 大于512B，单不是512B对齐，512B向下取整
                tilingData_.set_dBaseLen(ALIGNDOWN(dBaseLenMax, l2CacheLineLen_));
            } else {
                // 大于32B，小于512B， 32B向下取整
                tilingData_.set_dBaseLen(ALIGNDOWN(dBaseLenMax, blkAlignLen_));
            }
            return;
        }

        // ub能放下整个D，逐步扩大tBaseLen，直到放不下
        tilingData_.set_dBaseLen(tilingData_.get_D());
        uint32_t tBaseLen_opt = MIN(MIN_TBASELEN, tLenPerCore_);
        uint32_t tBaseLen = tBaseLen_opt;

        while (tBaseLen <= tLenPerCore_) {
            if (!CheckUbSize(tBaseLen, tilingData_.get_dBaseLen())) {
                break;
            }
            // 保存较优切分结果
            tBaseLen_opt = tBaseLen;
            if (tBaseLen >= tLenPerCore_) {
                break;
            }
            tBaseLen = MIN(tBaseLen + MIN_TBASELEN, tLenPerCore_);
        }
        tilingData_.set_tBaseLen(tBaseLen_opt);
    }

    ge::graphStatus CalcTiling()
    {
        if (ParseParam() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        CaclOptTiling();
        tilingData_.set_reduceBuffLen(GetReduceXxNdTempBuffLen<float, true>(tilingData_.get_tBaseLen(), tilingData_.get_dBaseLen()));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckTiling()
    {
#ifdef ENABLE_LOG_ON
        PrintTiling();
#endif
        return ge::GRAPH_SUCCESS;
    }

private:
    mutable RmsNormWithoutWeightTilingData tilingData_;
    uint32_t xDTypeLen_ = 1; // x输入数据类型的长度
    uint32_t l2CacheLineLen_; // 512B / sizeof(dtype_x)
    uint32_t blkAlignLen_; // 32B / sizeof(dtype_x)
    uint32_t usedCoreNum_ = 1;
    uint32_t tLenPerCore_ = 0; // 每个核计算的tLen(最大值)
};

inline ge::graphStatus Tiling4RmsNormWithoutWeight(gert::TilingContext* context)
{
    RmsNormWithoutWeightTilingCalculator tilingCalc(context);
    return tilingCalc.DoTiling();
}
} // end optiling
#endif
