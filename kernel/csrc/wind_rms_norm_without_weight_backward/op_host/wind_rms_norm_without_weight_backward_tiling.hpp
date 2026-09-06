/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_TILING_H
#define OP_RMS_NORM_WITHOUT_WEIGHT_BACKWARD_TILING_H
#include <register/tilingdata_base.h>
#include <exe_graph/runtime/tiling_context.h>
#include "tiling/common_tiling_base.h"
#include "../../include/log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/op_math.h"

namespace optiling {
const uint32_t RNWB_DEFAULT_BUFFER_NUM = 2;
const uint32_t RNWB_ALIGN_NUM = 16;
const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B
const uint32_t MIN_INPUT_DTYPELEN = 2;

const uint32_t RNWB_LONG_SHAPE_COLNUM_310 = 8192;
uint32_t RNWB_LONG_SHAPE_COLNUM = 5120;

BEGIN_TILING_DATA_DEF(RmsNormWithoutWeightBackwardTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
    TILING_DATA_FIELD_DEF(uint64_t, maxTileRow);
    TILING_DATA_FIELD_DEF(uint64_t, rowNum);
    TILING_DATA_FIELD_DEF(uint64_t, colNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailBlockRowNums);
    TILING_DATA_FIELD_DEF(uint64_t, baseBlockRowIndex);
    TILING_DATA_FIELD_DEF(uint64_t, alignNum);
    TILING_DATA_FIELD_DEF(uint64_t, packLen);
    TILING_DATA_FIELD_DEF(uint64_t, colNumPerUB);
    TILING_DATA_FIELD_DEF(uint64_t, colNumPerUBAlignDown32);
    TILING_DATA_FIELD_DEF(uint64_t, colLoopNum);
    TILING_DATA_FIELD_DEF(uint64_t, colTileNum);
    TILING_DATA_FIELD_DEF(uint64_t, colTileNumAlignUp32);
    TILING_DATA_FIELD_DEF(uint64_t, longMode);
    TILING_DATA_FIELD_DEF(float, avgFactor);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindRmsNormWithoutWeightBackward, RmsNormWithoutWeightBackwardTilingData);

struct RmsNormWithoutWeightBackwardTilingCalculator : public CommonTilingBase {
public:
    using CommonTilingBase::CommonTilingBase;

    bool CalcVectorTiling(uint32_t totalCore, uint64_t ubSize, uint64_t rowNum, uint64_t colNum, uint32_t inputDTypeLen);
    bool CalcVectorTiling4Long(uint32_t totalCore, uint64_t ubSize, uint64_t rowNum, uint64_t colNum, uint32_t inputDTypeLen);

    bool IsCapable() final;
    ge::graphStatus CalcTiling() final;
    ge::graphStatus CheckTiling() final;
    [[nodiscard]] uint64_t GetTilingKey() const final { return 0; }
    size_t GetUserWorkspaceSize() final { return 0; }
    uint32_t GetBlockDim() final { return usedCoreNum; }
    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tiling_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tiling_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint32_t usedCoreNum = 1;

private:
    inline void PrintTiling();

    mutable RmsNormWithoutWeightBackwardTilingData tiling_;
};

bool RmsNormWithoutWeightBackwardTilingCalculator::CalcVectorTiling(uint32_t totalCore, uint64_t ubSize, uint64_t rowNum, uint64_t colNum,
                                                                    uint32_t inputDTypeLen)
{
    if (rowNum <= totalCore) {
        usedCoreNum = rowNum;
        tiling_.set_isDoubleBuffer(0);
    } else {
        usedCoreNum = totalCore;
        tiling_.set_isDoubleBuffer(1);
    }

    uint64_t tailBlockRowNums = OpDiv(rowNum, usedCoreNum);
    uint64_t baseBlockRowIndex = OpMod(rowNum, usedCoreNum);

    uint32_t bufferNum = RNWB_DEFAULT_BUFFER_NUM;

    uint64_t alignNum = OpDiv(DEFAULT_ALIGN_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);
    uint32_t packLen = OpDiv(PACK_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);

    uint64_t tailBlockRowNumsAlign = (tailBlockRowNums + 1 + alignNum - 1) / alignNum * alignNum;

    // backward: x(in) + grad_x(out) + x_fp32 + grad_fp32 = 4 * inputDTypeLen per element
    uint32_t gradResBufferSize = 4U * tailBlockRowNumsAlign;
    uint32_t ResBufferSize = 4U * tailBlockRowNumsAlign;

    uint32_t singleRowXDataSize = (bufferNum * 1U + 4U) * inputDTypeLen * colNum;
    uint32_t singleRowGradResDataSize = (bufferNum * 1U * inputDTypeLen) * colNum;

    uint64_t rowNumPerUB = OpDiv(ubSize - gradResBufferSize - ResBufferSize, singleRowXDataSize + singleRowGradResDataSize);
    if (rowNumPerUB == 0) {
        rowNumPerUB = 1;
    }

    tiling_.set_usedCoreNum(usedCoreNum);
    tiling_.set_maxTileRow(rowNumPerUB);
    tiling_.set_colNum(colNum);
    tiling_.set_rowNum(rowNum);
    tiling_.set_tailBlockRowNums(tailBlockRowNums);
    tiling_.set_baseBlockRowIndex(baseBlockRowIndex);

    tiling_.set_alignNum(alignNum);
    tiling_.set_packLen(packLen);

    float avg_factor = (colNum == 0) ? 0 : 1.0 / colNum;
    tiling_.set_avgFactor(avg_factor);

    return true;
}

bool RmsNormWithoutWeightBackwardTilingCalculator::CalcVectorTiling4Long(uint32_t totalCore, uint64_t ubSize, uint64_t rowNum, uint64_t colNum,
                                                                         uint32_t inputDTypeLen)
{
    if (rowNum <= totalCore) {
        usedCoreNum = rowNum;
    } else {
        usedCoreNum = totalCore;
    }
    tiling_.set_isDoubleBuffer(1);

    uint64_t tailBlockRowNums = OpDiv(rowNum, usedCoreNum);
    uint64_t baseBlockRowIndex = OpMod(rowNum, usedCoreNum);

    uint32_t bufferNum = RNWB_DEFAULT_BUFFER_NUM;

    uint64_t alignNum = OpDiv(DEFAULT_ALIGN_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);
    uint32_t packLen = OpDiv(PACK_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);

    uint64_t tailBlockRowNumsAlign = (tailBlockRowNums + 1 + alignNum - 1) / alignNum * alignNum;

    // backward: x(in) + grad_x(out) + x_fp32 + grad_fp32 = 4 * inputDTypeLen per element
    uint32_t TempBufferSize = 2 * sizeof(float) * tailBlockRowNumsAlign; // ub内部都用float申请内存

    uint32_t singleColDataSize = (bufferNum * 2U) * sizeof(float); // ub内部都用float申请内存

    uint64_t colNumPerUB = OpDiv(ubSize - TempBufferSize, singleColDataSize); // 按列切分
    uint64_t colNumPerUBAlignDown32 = colNumPerUB  / alignNum * alignNum; // 按列切分
    uint64_t colLoopNum = OpDiv(colNum, colNumPerUBAlignDown32); // 列搬运循环次数
    uint64_t colTileNum = colNum - colLoopNum * colNumPerUBAlignDown32; // 列搬运剩余列数需要向上32B对齐
    uint64_t colTileNumAlignUp32 = (colTileNum + alignNum - 1) / alignNum * alignNum; // 列搬运剩余列数需要向上32B对齐

    uint32_t rowNumPerUB = 1;

    tiling_.set_usedCoreNum(usedCoreNum);
    tiling_.set_maxTileRow(rowNumPerUB);
    tiling_.set_colNum(colNum);
    tiling_.set_rowNum(rowNum);
    tiling_.set_tailBlockRowNums(tailBlockRowNums);
    tiling_.set_baseBlockRowIndex(baseBlockRowIndex);

    tiling_.set_alignNum(alignNum);
    tiling_.set_packLen(packLen);

    // 设置和列相关的变量
    tiling_.set_colNumPerUB(colNumPerUB);
    tiling_.set_colNumPerUBAlignDown32(colNumPerUBAlignDown32);
    tiling_.set_colLoopNum(colLoopNum);
    tiling_.set_colTileNum(colTileNum);
    tiling_.set_colTileNumAlignUp32(colTileNumAlignUp32);
    tiling_.set_longMode(1);

    float avg_factor = (colNum == 0) ? 0 : 1.0 / colNum;
    tiling_.set_avgFactor(avg_factor);

    return true;
}

inline bool RmsNormWithoutWeightBackwardTilingCalculator::IsCapable()
{
    if (context_ == nullptr) {
        return false;
    }

    auto inTensor = GetInputTensor(1); // x
    if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
        return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
    }
    return inTensor->GetDataType() == ge::DataType::DT_FLOAT || inTensor->GetDataType() == ge::DataType::DT_FLOAT16;
}

ge::graphStatus RmsNormWithoutWeightBackwardTilingCalculator::CalcTiling()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        RNWB_LONG_SHAPE_COLNUM = RNWB_LONG_SHAPE_COLNUM_310;
    }

    auto inTensor_X = GetInputTensor(1);
    auto xShape = inTensor_X->GetStorageShape();

    uint64_t tmpColNum = 1;
    if (xShape.GetDimNum() > 0) {
        tmpColNum = xShape.GetDim(xShape.GetDimNum() - 1);
    }

    uint64_t tmpRowNum = 1;
    for (uint32_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        auto tempShapeDim = xShape.GetDim(i);
        tmpRowNum *= tempShapeDim;
    }

    if (xShape.GetDimNum() == 0) {
        OP_LOGE("Unsupported input data shape");
        return ge::GRAPH_FAILED;
    }
    if (tmpColNum % RNWB_ALIGN_NUM != 0) {
        OP_LOGE("Unsupported unaligned input data");
        return ge::GRAPH_FAILED;
    }

    auto inputDTypeLen = ge::GetSizeByDataType(inTensor_X->GetDataType());
    if (inputDTypeLen < 0) {
        OP_LOGE(context->GetNodeName(), "Unsupported input data type %d", inTensor_X->GetDataType());
        return ge::GRAPH_FAILED;
    }

    if (tmpColNum > RNWB_LONG_SHAPE_COLNUM) {
        if (!RmsNormWithoutWeightBackwardTilingCalculator::CalcVectorTiling4Long(limitCoreNum_, limitUbMemSize_, tmpRowNum, tmpColNum, inputDTypeLen)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!RmsNormWithoutWeightBackwardTilingCalculator::CalcVectorTiling(limitCoreNum_, limitUbMemSize_, tmpRowNum, tmpColNum, inputDTypeLen)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline void RmsNormWithoutWeightBackwardTilingCalculator::PrintTiling()
{
    std::cout << "======RmsNormWithoutWeightBackward tiling======" << std::endl;
    std::cout << " usedCoreNum=" << tiling_.get_usedCoreNum() << std::endl;
    std::cout << " isDoubleBuffer=" << tiling_.get_isDoubleBuffer() << std::endl;
    std::cout << " maxTileRow=" << tiling_.get_maxTileRow() << std::endl;
    std::cout << " rowNum=" << tiling_.get_rowNum() << std::endl;
    std::cout << " colNum=" << tiling_.get_colNum() << std::endl;
    std::cout << " tailBlockRowNums=" << tiling_.get_tailBlockRowNums() << std::endl;
    std::cout << " baseBlockRowIndex=" << tiling_.get_baseBlockRowIndex() << std::endl;
    std::cout << " alignNum=" << tiling_.get_alignNum() << std::endl;
    std::cout << " packLen=" << tiling_.get_packLen() << std::endl;
    std::cout << " colNumPerUB=" << tiling_.get_colNumPerUB() << std::endl;
    std::cout << " colNumPerUBAlignDown32=" << tiling_.get_colNumPerUBAlignDown32() << std::endl;
    std::cout << " colLoopNum=" << tiling_.get_colLoopNum() << std::endl;
    std::cout << " colTileNum=" << tiling_.get_colTileNum() << std::endl;
    std::cout << " colTileNumAlignUp32=" << tiling_.get_colTileNumAlignUp32() << std::endl;
    std::cout << " longMode=" << tiling_.get_longMode() << std::endl;
    std::cout << " avgFactor=" << tiling_.get_avgFactor() << std::endl;
}

ge::graphStatus RmsNormWithoutWeightBackwardTilingCalculator::CheckTiling()
{
#ifdef ENABLE_LOG_ON
    PrintTiling();
#endif
    OP_TILING_CHECK((tiling_.get_usedCoreNum() == 0 || tiling_.get_usedCoreNum() > this->limitCoreNum_),
                    OP_LOGE("[ERROR] invalid corenum"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tiling_.get_baseBlockRowIndex() >= tiling_.get_usedCoreNum()),
                    OP_LOGE("[ERROR] invalid baseBlockRowIndex"), return ge::GRAPH_FAILED);

    size_t totalCalcRows = tiling_.get_tailBlockRowNums() * tiling_.get_usedCoreNum() + tiling_.get_baseBlockRowIndex();
    OP_TILING_CHECK((totalCalcRows != tiling_.get_rowNum()),
                    OP_LOGE("[ERROR] invalid totalCalcRows != rowNum"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus Tiling4RmsNormWithoutWeightBackward(gert::TilingContext* context)
{
    RmsNormWithoutWeightBackwardTilingCalculator tilingCalc(context);
    return tilingCalc.DoTiling();
}
} // end optiling
#endif
