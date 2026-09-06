/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_XXGLU_LIMIT_TILING_HOST_H
#define OP_XXGLU_LIMIT_TILING_HOST_H
#include "xxglu_limit_tiling_data.h"
#include "tiling/common_tiling_base.h"
#include "log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optilingLimitBackward { // 独立 namespace，避免与老 swi_glu_backward 的 optiling::GluTilingCalculator/Tiling4Glu ODR 冲突
const uint32_t UB_RESERVED_BUFF = 0; // reserve 0k
const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t DEFAULT_ALIGN_SIZE = 32; // align unit in cache 32B
const uint32_t DEFAULT_BUFFER_NUM = 2;

// 每次新增算子增加算子标志
enum GLU_FLAG {
    REGLU,
    GEGLU,
    SWIGLU,
    REGLU_BACKWARD,
    GEGLU_BACKWARD,
    SWIGLU_BACKWARD,
    YULU,
    FASTGELU,
    SIGMOID,
};

// 此处的代码表示逻辑计算所需的搬入队列的数量、临时存储单元的数量
const uint32_t XXGLU_TQUE_NUM = 3;
const uint32_t GEGLU_TBUF_NUM_HALF = 3;
const uint32_t GEGLU_TBUF_NUM_FLOAT = 1;
const uint32_t SWIGLU_TBUF_NUM_HALF = 2;
const uint32_t SWIGLU_TBUF_NUM_FLOAT = 1;
const uint32_t XXGLU_BW_TQUE_NUM = 5;
const uint32_t REGLU_BW_TBUF_NUM_FP = 1;
// swiglu_limit 反向新增 UB buffer，必须同步增大预算常量，否则大 shape UB 越界：
//   bf16/fp16: 新增 maskQueue(uint8)+maskGQueue(uint8)+bF32Queue(fp32)+zeroQueue(fp32) => +4
//   fp32:      新增 maskQueue(uint8)+maskGQueue(uint8)+zeroQueue(fp32)                 => +3
// probs 融合再加 2 个 fp32 TBuf(probsBuf_ + probsVecBuf_)，同步 +2，否则大 shape UB 越界。
const uint32_t SWIGLU_BW_TBUF_NUM_FLOAT = 2 + 3 + 2; // pre-silu +maskG 2->5; +probs ->7
const uint32_t SWIGLU_BW_TBUF_NUM_BF16 = 5 + 4 + 2;  // pre-silu +maskG 5->9; +probs ->11
const uint32_t SWIGLU_BW_TBUF_NUM_HALF = 4 + 4 + 2;  // pre-silu +maskG 4->8; +probs ->10
const uint32_t GEGLU_BW_TBUF_NUM_HALF = 3;
const uint32_t GEGLU_BW_TBUF_NUM_FLOAT = 2;
const uint32_t MIN_INPUT_DTYPELEN = 2;
const uint32_t FASTGELU_TQUE_NUM = 2;
const uint32_t FASTGELU_TBUF_NUM = 2;
const uint32_t SIGMOID_TQUE_NUM = 2;

template <GLU_FLAG Glu_Flag>
struct GluTilingCalculator : public optiling::CommonTilingBase {
public:
    static constexpr uint32_t INPUT_NUMBER = ((Glu_Flag == GLU_FLAG::YULU) ? 3U : 2U);
    using optiling::CommonTilingBase::CommonTilingBase;

    bool CalcVectorTiling(uint32_t totalCore, uint64_t ubSize, uint64_t dataLen, uint32_t inputDTypeLen,
                          int32_t dtype);
    bool CalcVectorTiling512Bsort(uint32_t totalCore, uint64_t ubSize, uint64_t dataLen, uint32_t inputDTypeLen,
                                  int32_t dtype);

    bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, uint32_t inputDTypeLen, int32_t dtype,
                                     uint32_t& bufferNum, uint64_t& dataLenPerUB);

    // 0 是否支持
    bool IsCapable() final;
    // 3.1 计算Tiling
    ge::graphStatus CalcTiling() final;
    // 3.2 校验Tiling结果
    ge::graphStatus CheckTiling() final { return ge::GRAPH_SUCCESS;}
    // 5、计算TilingKey
    [[nodiscard]] uint64_t GetTilingKey() const final
    {
        if (Glu_Flag == GLU_FLAG::SIGMOID && inputDataLen == 1) {
            return 1; // 1个元素，走特殊的kernel
        } else {
            return 0;
        }
    }
    // 6.1 获取用户workspace大小
    size_t GetUserWorkspaceSize() final { return 0; }
    // 7.1 获取计算核数
    uint32_t GetBlockDim() final { return usedCoreNum; }
    // 7.2 设置TilingData
    ge::graphStatus SaveTilingData(gert::TilingData& tilingData) final
    {
        tiling_.SaveToBuffer(tilingData.GetData(), tilingData.GetCapacity());
        tilingData.SetDataSize(tiling_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint32_t usedCoreNum = 0; // 最终实际使用的核数
    uint64_t inputDataLen; // 输入的长度

private:
    void Print() const;
    bool Check(uint32_t totalCore, uint64_t dataLen, uint32_t inputDTypeLen) const;
    uint32_t CalculateDataSizeForREGLU(uint32_t bufferNum, uint32_t inputDTypeLen) const;
    uint32_t CalculateDataSizeForGEGLU(uint32_t bufferNum, uint32_t inputDTypeLen) const;
    uint32_t CalculateDataSizeForSWIGLU(uint32_t bufferNum, uint32_t inputDTypeLen) const;
    uint32_t CalculateDataSizeForREGLU_BACKWARD(uint32_t bufferNum, uint32_t inputDTypeLen, int32_t dtype) const;
    uint32_t CalculateDataSizeForSWIGLU_BACKWARD(uint32_t bufferNum, uint32_t inputDTypeLen, int32_t dtype) const;
    uint32_t CalculateDataSizeForGEGLU_BACKWARD(uint32_t bufferNum, uint32_t inputDTypeLen) const;
    uint32_t CalculateDataSizeForFASTGELU(uint32_t bufferNum, uint32_t inputDTypeLen) const;

private:
    mutable optiling::XxGluLimitBackwardTilingData tiling_;
};

template <GLU_FLAG Glu_Flag>
bool GluTilingCalculator<Glu_Flag>::CalcVectorTiling(uint32_t totalCore, uint64_t ubSize, uint64_t dataLen,
                                                     uint32_t inputDTypeLen, int32_t dtype)
{
    OP_LOGI("GluTilingCalc", "Input totalCore:%u, ubSize:%u, dataLen:%u, inputDTypeLen:%u",
            totalCore, ubSize, dataLen, inputDTypeLen);
    // dataLen:待分配的数据总长度、inputDTypeLen：输入数据类型长度
    uint32_t alignNum = OpDiv(DEFAULT_ALIGN_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);
    uint32_t packLen = OpDiv(PACK_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);

    // (dataLen + alignNum - 1) / alignNum：向上取整，dataLen:待分配的数据总长度、alignNum：单个计算单元最大计算数，例如alignNum=64，则加上63，可以向上取整，
    // 再乘以单个计算单元大小，即64*4=256，即256个字节，即64个float32
    // dataLenAlign:理论上要计算的数据长度
    uint64_t dataLenAlign = (dataLen + alignNum - 1) / alignNum * alignNum;

    // 是否double_buffer
    uint32_t bufferNum = DEFAULT_BUFFER_NUM;
    uint64_t dataLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB(ubSize, inputDTypeLen, dtype, bufferNum, dataLenPerUB)) {
        OP_LOGE("GluTilingCalc", "Get bufferNum %u and dataLenPerUB %lu failed", bufferNum, dataLenPerUB);
        return false;
    }
    // 每个UB上计算的数据数，packLen表示每次能打包的数据数
    // 向下取整，计算每个UB可以计算的实际数据数；dataLenPerUB：每个UB可以计算的理论数据数
    uint64_t availableUB = dataLenPerUB / packLen * packLen;
    OP_LOGI("GluTilingCalc", "bufferNum:%u, dataLenPerUB:%lu", bufferNum, dataLenPerUB);

    // 计算所需核数
    // compute core num used
    // 当数据总长度小于一个UB上可以计算的实际数据数时，只需使用一个核
    if (dataLenAlign <= availableUB) {
        usedCoreNum = 1;
    } else if (dataLenAlign > availableUB && dataLenAlign <= availableUB * totalCore) {
        // 当数据总长度大于一个UB上可以计算的实际数据数，且小于一个UB上可以计算的实际数据数乘以核数时，使用多个核
        // 计算理论数据总长度除以一个UB上可以计算的实际数据数，向上取整，得到使用的核数
        usedCoreNum = (dataLenAlign + availableUB - 1) / availableUB;
    } else {
        usedCoreNum = totalCore;
    }

    // compute tiling
    uint64_t index = 0;
    // 每个核可以处理的相同最大数据量
    uint64_t baseBlockDims = dataLenAlign / (usedCoreNum * availableUB) * availableUB;
    // singleCoreBlockLens：存储每个核计算的数据量
    uint64_t singleCoreBlockLens[MAX_CORE_NUMBER] = {0};
    std::fill(singleCoreBlockLens, singleCoreBlockLens + totalCore, baseBlockDims);
    // 计算剩余数据长度
    uint64_t resDataLenAlign = dataLenAlign - baseBlockDims * usedCoreNum;

    for (uint64_t i = availableUB; i <= resDataLenAlign; i += availableUB) {
        singleCoreBlockLens[index % usedCoreNum] += availableUB;
        index++;
    }
    singleCoreBlockLens[usedCoreNum - 1] += dataLenAlign % availableUB;
    tiling_.set_usedCoreNum(usedCoreNum);
    tiling_.set_isDoubleBuffer(bufferNum > 1 ? 1 : 0);
    tiling_.set_maxTileLen(std::min(availableUB, dataLenAlign));
    tiling_.set_singleCoreBlockLens(singleCoreBlockLens);

    return Check(totalCore, dataLen, inputDTypeLen);
}

template <GLU_FLAG Glu_Flag>
bool GluTilingCalculator<Glu_Flag>::CalcVectorTiling512Bsort(uint32_t totalCore, uint64_t ubSize, uint64_t dataLen,
                                                             uint32_t inputDTypeLen, int32_t dtype)
{
    OP_LOGI("GluTilingCalcBy512B", "Input totalCore:%u, ubSize:%u, dataLen:%u, inputDTypeLen:%u",
            totalCore, ubSize, dataLen, inputDTypeLen);
    uint32_t alignNum = OpDiv(DEFAULT_ALIGN_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);
    uint32_t packLen = OpDiv(PACK_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);

    uint64_t dataLenAlign = (dataLen + alignNum - 1) / alignNum * alignNum;

    uint32_t bufferNum = DEFAULT_BUFFER_NUM;
    uint64_t dataLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB(ubSize, inputDTypeLen, dtype, bufferNum, dataLenPerUB)) {
        OP_LOGE("GluTilingCalcBy512B", "Get bufferNum %u and dataLenPerUB %lu failed", bufferNum, dataLenPerUB);
        return false;
    }
    uint64_t availableUB = dataLenPerUB / packLen * packLen;
    OP_LOGI("GluTilingCalcBy512B", "bufferNum:%u, dataLenPerUB:%lu", bufferNum, dataLenPerUB);

    uint32_t maxTileLen = std::min(availableUB, dataLenAlign);
    // compute core num used
    if (dataLenAlign <= availableUB) {
        usedCoreNum = 1;
    } else if (dataLenAlign > availableUB && dataLenAlign <= availableUB * totalCore) {
        usedCoreNum = (dataLenAlign + availableUB - 1) / availableUB;
    } else {
        usedCoreNum = totalCore;
    }

    // sigmoid 重新算，优化
    if constexpr (Glu_Flag == GLU_FLAG::SIGMOID) {
        // compute core num used， availableUB < packLen暂不考虑，不可能
        // 一个和最少处理packLen * 4
        uint64_t minLen = packLen * 4; // 4: 1个核最少处理4个512B
        if (dataLenAlign <= minLen) {
            usedCoreNum = 1;
        } else {
            usedCoreNum = MIN(DIVCEIL(dataLenAlign, MIN(availableUB, minLen)), totalCore);
        }
        maxTileLen = MIN(availableUB, DIVFLOOR(dataLenAlign, usedCoreNum));
    }

    // compute tiling
    uint64_t index = 0;
    uint64_t baseBlockDims = dataLenAlign / (usedCoreNum * packLen) * packLen;
    uint64_t singleCoreBlockLens[MAX_CORE_NUMBER] = {0};
    std::fill(singleCoreBlockLens, singleCoreBlockLens + totalCore, baseBlockDims);
    uint64_t resDataLenAlign = dataLenAlign - baseBlockDims * usedCoreNum;
    for (uint64_t i = packLen; i <= resDataLenAlign; i += packLen) {
        singleCoreBlockLens[index % usedCoreNum] += packLen;
        index++;
    }
    singleCoreBlockLens[usedCoreNum - 1] += dataLenAlign % packLen;
    tiling_.set_usedCoreNum(usedCoreNum);
    tiling_.set_isDoubleBuffer(bufferNum > 1 ? 1 : 0);
    tiling_.set_maxTileLen(maxTileLen);
    tiling_.set_singleCoreBlockLens(singleCoreBlockLens);

    return Check(totalCore, dataLen, inputDTypeLen);
}

template <GLU_FLAG Glu_Flag>
inline void GluTilingCalculator<Glu_Flag>::Print() const
{
    OP_LOGI("GluTilingPrint", "usedCoreNum:%u isDoubleBuffer:%u maxTileLen:%u",
            tiling_.get_usedCoreNum(), tiling_.get_isDoubleBuffer(), tiling_.get_maxTileLen());
    auto singleCoreBlockLens = tiling_.get_singleCoreBlockLens();
    uint64_t tmp = singleCoreBlockLens[0];
    uint32_t cnt = 1;
    for (uint32_t i = 1; i < tiling_.get_usedCoreNum(); i++) {
        if (singleCoreBlockLens[i] == tmp) {
            cnt++;
        } else {
            OP_LOGI("GluTilingPrint", "singleCoreBlockLens:%u cnt:%u ", tmp, cnt);
            tmp = singleCoreBlockLens[i];
            cnt = 1;
        }
    }
    OP_LOGI("GluTilingPrint", "singleCoreBlockLens:%u cnt:%u ", tmp, cnt);
}

template <GLU_FLAG Glu_Flag>
inline bool GluTilingCalculator<Glu_Flag>::Check(uint32_t totalCore, uint64_t dataLen, uint32_t inputDTypeLen) const
{
    Print();
    uint32_t packNum = OpDiv(PACK_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);
    uint32_t alignNum = OpDiv(DEFAULT_ALIGN_SIZE, inputDTypeLen, MIN_INPUT_DTYPELEN);

    if (tiling_.get_usedCoreNum() <= 0 || tiling_.get_usedCoreNum() > totalCore) {
        OP_LOGI("GluTilingCheck", "usedCoreNum(%u) wrong", tiling_.get_usedCoreNum());
        return false;
    }

    if (tiling_.get_maxTileLen() % packNum != 0 && (tiling_.get_usedCoreNum() == 1 &&
        tiling_.get_maxTileLen() % alignNum != 0)) {
        OP_LOGI("GluTilingCheck", "maxTileLen(%u) wrong", tiling_.get_maxTileLen());
        return false;
    }

    uint64_t dataLenAlign = (dataLen + alignNum - 1) / alignNum * alignNum;
    uint64_t coreBlockDims = 0;
    bool packNumFlag = true;
    bool atonicAlignFlag = true;
    uint32_t cntTail = 0;
    auto singleCoreBlockLens = tiling_.get_singleCoreBlockLens();
    for (uint32_t i = 0; i < tiling_.get_usedCoreNum(); i++) {
        coreBlockDims += singleCoreBlockLens[i];
        if (singleCoreBlockLens[i] % packNum != 0) {
            if (singleCoreBlockLens[i] % alignNum != 0) {
                atonicAlignFlag = false;
            } else {
                cntTail++;
            }
            if (cntTail > 1) {
                packNumFlag = false;
            }
        }
    }
    if (!atonicAlignFlag || !packNumFlag) {
        OP_LOGI("GluTilingCheck", "packNumFlag(%d) or atonicAlignFlag(%d) wrong", packNumFlag, atonicAlignFlag);
        return false;
    }
    if (coreBlockDims != dataLenAlign) {
        OP_LOGI("GluTilingCheck", "totalCoreBlockDims(%u) != dataLenAlign(%u), totalCoreBlockDims wrong",
                coreBlockDims, dataLenAlign);
        return false;
    }
    OP_LOGI("GluTilingCheck", "check ok !");
    return true;
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForREGLU(
    uint32_t bufferNum, uint32_t inputDTypeLen) const
{
    return (bufferNum * XXGLU_TQUE_NUM * inputDTypeLen);
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForGEGLU(
    uint32_t bufferNum, uint32_t inputDTypeLen) const
{
    if (inputDTypeLen == sizeof(int16_t)) {
        return (bufferNum * XXGLU_TQUE_NUM * sizeof(int16_t) + GEGLU_TBUF_NUM_HALF * sizeof(int32_t));
    } else {
        return (bufferNum * XXGLU_TQUE_NUM * sizeof(int32_t) + GEGLU_TBUF_NUM_FLOAT * sizeof(int32_t));
    }
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForSWIGLU(
    uint32_t bufferNum, uint32_t inputDTypeLen) const
{
    if (inputDTypeLen == sizeof(int16_t)) {
        // 2 inputs, 1 output, 2 temp tensors
        return (bufferNum * 2 * sizeof(int16_t) + 1 * sizeof(int16_t) + SWIGLU_TBUF_NUM_HALF * sizeof(int32_t));
    } else {
        return (bufferNum * XXGLU_TQUE_NUM * sizeof(int32_t) + SWIGLU_TBUF_NUM_FLOAT * sizeof(int32_t));
    }
}

template <GLU_FLAG Glu_Flag>
inline bool GluTilingCalculator<Glu_Flag>::GetBufferNumAndDataLenPerUB(
    uint64_t ubSize, uint32_t inputDTypeLen, int32_t dtype, uint32_t& bufferNum, uint64_t& dataLenPerUB)
{
    uint32_t singleDataSize = 0;
    bufferNum = DEFAULT_BUFFER_NUM;
    switch (Glu_Flag) {
        case GLU_FLAG::REGLU:
            singleDataSize = CalculateDataSizeForREGLU(bufferNum, inputDTypeLen);
            break;
        case GLU_FLAG::GEGLU:
            singleDataSize = CalculateDataSizeForGEGLU(bufferNum, inputDTypeLen);
            break;
        case GLU_FLAG::SWIGLU:
            singleDataSize = CalculateDataSizeForSWIGLU(bufferNum, inputDTypeLen);
            break;
        case GLU_FLAG::REGLU_BACKWARD:
            singleDataSize = CalculateDataSizeForREGLU_BACKWARD(bufferNum, inputDTypeLen, dtype);
            break;
        case GLU_FLAG::SWIGLU_BACKWARD:
            singleDataSize = CalculateDataSizeForSWIGLU_BACKWARD(bufferNum, inputDTypeLen, dtype);
            break;
        case GLU_FLAG::GEGLU_BACKWARD:
            singleDataSize = CalculateDataSizeForGEGLU_BACKWARD(bufferNum, inputDTypeLen);
            break;
        case GLU_FLAG::YULU:
            singleDataSize = (bufferNum * 4U * sizeof(int16_t) + 2U * sizeof(int32_t));
            break;
        case GLU_FLAG::FASTGELU:
            singleDataSize = CalculateDataSizeForFASTGELU(bufferNum, inputDTypeLen);
            break;
        case GLU_FLAG::SIGMOID:
            singleDataSize = (bufferNum * SIGMOID_TQUE_NUM +1) * sizeof(float); // 输入输出+tempbuffer，都用float
            break;
        default:
            return false;
    }
    dataLenPerUB = OpDiv(ubSize, singleDataSize, ubSize);
    return true;
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForREGLU_BACKWARD(
    uint32_t bufferNum, uint32_t inputDTypeLen, int32_t dtype) const
{
    if (dtype == ge::DT_INT32) {
        return (bufferNum * XXGLU_BW_TQUE_NUM * inputDTypeLen);
    } else if (dtype == ge::DT_FLOAT) {
        return (bufferNum * XXGLU_BW_TQUE_NUM * inputDTypeLen + REGLU_BW_TBUF_NUM_FP * inputDTypeLen);
    } else {
        return (bufferNum * XXGLU_BW_TQUE_NUM * inputDTypeLen + REGLU_BW_TBUF_NUM_FP * sizeof(int32_t) +
                INPUT_NUMBER * sizeof(int32_t));
    }
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForSWIGLU_BACKWARD(
    uint32_t bufferNum, uint32_t inputDTypeLen, int32_t dtype) const
{
    if (dtype == ge::DT_BF16) {
        return (bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) + SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t));
    } else if (inputDTypeLen == sizeof(int16_t)) {
        return (bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) + SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t));
    } else {
        return (bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int32_t) + SWIGLU_BW_TBUF_NUM_FLOAT * sizeof(int32_t));
    }
}

template <GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForGEGLU_BACKWARD(
    uint32_t bufferNum, uint32_t inputDTypeLen) const
{
    if (inputDTypeLen == sizeof(int16_t)) {
        return (bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) + GEGLU_BW_TBUF_NUM_HALF * sizeof(int32_t));
    } else {
        return (bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int32_t) + GEGLU_BW_TBUF_NUM_FLOAT * sizeof(int32_t));
    }
}

template<GLU_FLAG Glu_Flag> inline uint32_t GluTilingCalculator<Glu_Flag>::CalculateDataSizeForFASTGELU(
    uint32_t bufferNum, uint32_t inputDTypeLen) const
{
    if (inputDTypeLen == sizeof(int16_t)) {
        return (bufferNum * FASTGELU_TQUE_NUM * sizeof(int16_t) + FASTGELU_TBUF_NUM * sizeof(int16_t));
    } else {
        return (bufferNum * FASTGELU_TQUE_NUM * sizeof(int32_t) + FASTGELU_TBUF_NUM * sizeof(int32_t));
    }
}

template <GLU_FLAG Glu_Flag> bool GluTilingCalculator<Glu_Flag>::IsCapable()
{
    if (context_ == nullptr) {
        return false;
    }

    if (CheckSameInputs(INPUT_NUMBER)) {
        return false;
    }

    auto inTensor = GetInputTensor(0);
    if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
        // 310P不支持BF16
        return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
    }
    return inTensor->GetDataType() == ge::DataType::DT_FLOAT || inTensor->GetDataType() == ge::DataType::DT_FLOAT16;
}

template <GLU_FLAG Glu_Flag> ge::graphStatus GluTilingCalculator<Glu_Flag>::CalcTiling()
{
    auto inTensor = GetInputTensor(0);
    auto inShape = inTensor->GetOriginShape();
    inputDataLen = uint64_t((inShape.GetDimNum() == 0) ? 1 : inShape.GetShapeSize());
    auto inputDTypeLen = ge::GetSizeByDataType(inTensor->GetDataType());
    if (inputDTypeLen < 0) {
        OP_LOGE(context_->GetNodeName(), "Unsupported input data type %d", inTensor->GetDataType());
    }
    if (!CalcVectorTiling512Bsort(limitCoreNum_, limitUbMemSize_, inputDataLen, inputDTypeLen, inTensor->GetDataType())) {
        return ge::GRAPH_FAILED;
    }
    // 读 limit attr(index1) 写入 tiling，供 kernel clamp 使用
    tiling_.set_limit(GetAttrWithDefault<float>(1, 0.0f));
    // probs(input index3) 是否存在 → kernel 决定是否做 grad_h = gradout*probs
    tiling_.set_hasProbs((context_ != nullptr && context_->GetOptionalInputShape(3) != nullptr) ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

template <GLU_FLAG Glu_Flag>
inline ge::graphStatus Tiling4Glu(gert::TilingContext* context)
{
    GluTilingCalculator<Glu_Flag> gluTiling(context);
    return gluTiling.DoTiling();
}
} // end optiling
#endif  // OP_XXGLU_LIMIT_TILING_HOST_H
