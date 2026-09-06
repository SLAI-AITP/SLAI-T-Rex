/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_XXGLU_LIMIT_TILING_HOST_SINGLE_H   // [Add by GTS-AILab] [feature GTSOps]
#define OP_XXGLU_LIMIT_TILING_HOST_SINGLE_H   // [Add by GTS-AILab] [feature GTSOps]
#include "xxglu_limit_tiling_data.h"          // [Add by GTS-AILab] [feature GTSOps]
#include "tiling/common_tiling_base.h"
#include "log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"
#include <array>

namespace optilingLimitSingle { // [Add by GTS-AILab] [feature GTSOps] 独立 namespace，避免与老 swi_glu_v2/v3 的 optilingSingle::GluSingleTilingCalculator/Tiling4Glu ODR 冲突
const uint32_t UB_RESERVED_BUFF = 0; // reserve 0k
const uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B
const uint32_t UB_MIN_BLOCK_SIZE = 32; // align unit in cache 32B
const uint32_t DEFAULT_BUFFER_NUM = 2;
const uint32_t MAX_BLOCK_COUNT = 4095; // datacopy指令包含的连续传输数据块的最大个数
const uint32_t MAX_BLOCK_LEN = 65535 * 32; // datacopy指令每个连续传输数据块的最长长度为65535，单位为32bytes
const uint32_t MAX_UINT32 = 4294967295;
const uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095; // 非连续拷贝，blockCount最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535; // 非连续拷贝，blockLen最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535; // 非连续拷贝，srcStride/dstStride最大值,AscendC接口限制

enum GLU_FLAG {
    REGLU,
    GEGLU,
    SWIGLU,
    REGLU_BACKWARD,
    GEGLU_BACKWARD,
    SWIGLU_BACKWARD,
    PANGLU
};

const uint32_t XXGLU_TQUE_NUM = 3;
const uint32_t SWIGLU_TBUF_NUM_HALF = 2;
const uint32_t SWIGLU_TBUF_NUM_BF16 = 2;
const uint32_t SWIGLU_TBUF_NUM_FLOAT = 1;
const uint32_t XXGLU_BW_TQUE_NUM = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_FLOAT = 2;
const uint32_t SWIGLU_BW_TBUF_NUM_BF16 = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_HALF = 4;
const uint32_t PANGLU_TW_TQUE_NUM_FLOAT = 4;
const uint32_t PANGLU_TW_TQUE_NUM_HALF = 4;
const uint32_t PANGLU_TW_TBUF_NUM_FLOAT = 1;
const uint32_t PANGLU_TW_TBUF_NUM_HALF = 2;

// Tiling优选参数
struct GluSingleTilingOptParam {
    // Maximum amount of data that can be transferred by an operator UB at a time. Unit:element
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0; // 最优的BaseRowLen
    uint32_t optBaseColLen = 0; // 最优的BaseColLen
    uint64_t optTotalTileNum = 0; // 最优的分割后的数据块数量
    uint64_t optBaseSize = 0; // 最优的分割后的base shape数据块的大小， optBaseRowLen*optBaseColLen, Unit:element
    uint64_t optBaseTileNum = 0; // 最优的分割后的base shape数据块数量，不包含尾块

    uint32_t totalUsedCoreNum = 0; // 最终实际使用的核数
    uint64_t tileNumPerCore = 0; // 每个核需要处理的TileNum，如果不均匀，按照多的计算

    inline void Print()
    {
        OP_LOGI("GluSingleTilingOptParamPrint", "GluSingleTilingOptParam");
        OP_LOGI("GluSingleTilingOptParamPrint", "maxTileLen:%u optBaseRowLen:%u optBaseColLen:%u",
                maxTileLen, optBaseRowLen, optBaseColLen);
        OP_LOGI("GluSingleTilingOptParamPrint", "optTotalTileNum:%u", optTotalTileNum);
        OP_LOGI("GluSingleTilingOptParamPrint", "optBaseSize:%u", optBaseSize);
        OP_LOGI("GluSingleTilingOptParamPrint", "optBaseTileNum:%u", optBaseTileNum);
        OP_LOGI("GluSingleTilingOptParamPrint", "optTotalTileNum:%u", optTotalTileNum);
        OP_LOGI("GluSingleTilingOptParamPrint", "totalUsedCoreNum:%u", totalUsedCoreNum);
        OP_LOGI("GluSingleTilingOptParamPrint", "tileNumPerCore:%u", tileNumPerCore);
    }
};

template <GLU_FLAG Glu_Flag>
struct GluSingleTilingCalculator : public optiling::CommonTilingBase {
public:
    static constexpr uint32_t REAL_INPUT_NUM = ((Glu_Flag == GLU_FLAG::PANGLU) ? 3U : 2U);
    using optiling::CommonTilingBase::CommonTilingBase;

    ~GluSingleTilingCalculator() final = default;

    bool IsCapable() final;
    // 3.1 计算Tiling
    ge::graphStatus CalcTiling() final;
    // 3.2 校验Tiling结果
    ge::graphStatus CheckTiling() final { return ge::GRAPH_SUCCESS;}
    // 5、计算TilingKey
    [[nodiscard]] uint64_t GetTilingKey() const final { return 1; }
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

private:
    bool CalcTiling(uint32_t totalCore, uint64_t ubSize, ge::DataType& dtype);
    bool SetTotalShape(const gert::Shape &inShape, int64_t inDim);
    void Print() const;
    template <uint16_t bufferNum>
    inline bool CalcUbMaxTileLen(uint64_t ubSize, ge::DataType& dtype, GluSingleTilingOptParam &optTiling);
    inline void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam &optTiling);
    inline uint32_t getBaseColLenUpBound(GluSingleTilingOptParam &optTiling);
    inline uint32_t getBaseRowLenUpBound();

    uint32_t usedCoreNum = 0; // 最终实际使用的核数

    uint32_t inputDTypeLen = 2;
    // Indicates the minimum processing data unit of the UB. Unit:element.
    // Formula: 32B/sizeof(DType). For example, if Dtype is BF16, ubMinBlockLen = 32/2 = 16
    uint32_t ubMinBlockLen = 0;
    // Length of the L2 cache line. Unit:element.
    // Formula: 512B/sizeof(DType). For example, if the Dtype is BF16, cacheLineLen = 512/2 = 256
    uint32_t cacheLineLen = 0;
    // baseColLen aligned package Len. elelment:Unit. 512-byte alignment or 32-byte alignment
    uint32_t alignPackLen = 0;
    uint32_t totalAvailableCore = 0; // total avaliable core in device

    mutable optiling::XxGluLimitSingleTilingData tiling_;

private:
    bool GetOptTiling(ge::DataType& dtype, GluSingleTilingOptParam &optTiling);
    template <uint16_t bufferNum>
    bool CalcOptTiling(uint64_t ubSize, ge::DataType& dtype, GluSingleTilingOptParam &optTiling);
    template <uint16_t bufferNum>
    bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, ge::DataType& dtype, uint64_t &dataLenPerUB);
    inline bool MustBeSingleBaseRowLen(uint32_t baseColLen_);
    inline bool isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_);
    inline bool CalcOptBaseShape(GluSingleTilingOptParam &optTiling);
    inline uint32_t GetSelfIdx() { return Glu_Flag == GLU_FLAG::SWIGLU_BACKWARD ? 1 : 0; }
};

template <GLU_FLAG Glu_Flag> inline bool GluSingleTilingCalculator<Glu_Flag>::SetTotalShape(
    const gert::Shape& inShape, int64_t inDim)
{
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int64_t dimNum = inShape.GetDimNum();
    if (inDim < -dimNum || inDim >= dimNum) {
        OP_LOGE("SetTotalShape", "Unsupported inDim %ld, dimNum %ld", inDim, dimNum);
        return false;
    }
    int64_t splitDim = inDim < 0 ? dimNum + inDim : inDim; // inDim default -1
    for (int64_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    for (int64_t j = splitDim; j < dimNum; j++) {
        shapeAfter *= inShape.GetDim(j);
    }

    if (shapeAfter % REAL_INPUT_NUM != 0) {
        OP_LOGE("SetTotalShape", "Unsupported inDim %ld, shapeAfter %ld mod %d != 0", inDim, shapeAfter, REAL_INPUT_NUM);
        return false;
    }

    tiling_.set_rowLen(shapeBefore);
    tiling_.set_colLen(shapeAfter / REAL_INPUT_NUM);
    return true;
}

template <GLU_FLAG Glu_Flag> inline void GluSingleTilingCalculator<Glu_Flag>::Print() const
{
    OP_LOGI("GluTilingPrint", "is32BAligned:%u isDoubleBuffer:%u rowLen:%u colLen:%u",
            tiling_.get_is32BAligned(), tiling_.get_isDoubleBuffer(), tiling_.get_rowLen(), tiling_.get_colLen());
    OP_LOGI("GluTilingPrint", "baseRowLen:%u baseColLen:%u", tiling_.get_baseRowLen(), tiling_.get_baseColLen());
    OP_LOGI("GluTilingPrint", "totalUsedCoreNum:%u", usedCoreNum);
}

template <GLU_FLAG Glu_Flag>
template <uint16_t bufferNum>
bool GluSingleTilingCalculator<Glu_Flag>::GetBufferNumAndDataLenPerUB(
    uint64_t ubSize, ge::DataType& dtype, uint64_t& dataLenPerUB)
{
    uint32_t singleDataSize = 0;
    switch (Glu_Flag) {
        case GLU_FLAG::SWIGLU:
            if (dtype == ge::DT_FLOAT16) {
                // 2 input * sizeof(half) + 1 output * sizeof(half) + 2 * sizeof(float)
                singleDataSize = bufferNum * 2 * sizeof(int16_t) + 1 * sizeof(int16_t) +
                        SWIGLU_TBUF_NUM_HALF * sizeof(int32_t);
                if (tiling_.get_hasProbs()) {
                    singleDataSize += bufferNum * 1 * sizeof(float) + 1 * sizeof(int16_t);
                }
            } else if (dtype == ge::DT_BF16) {
                // 2 input * sizeof(half) + 1 output * sizeof(half) + 2 * sizeof(float)
                singleDataSize = bufferNum * 2 * sizeof(int16_t) + 1 * sizeof(int16_t) +
                        SWIGLU_TBUF_NUM_BF16 * sizeof(int32_t);
                if (tiling_.get_hasProbs()) {
                    singleDataSize += bufferNum * 1 * sizeof(float) + 1 * sizeof(int16_t);
                }
            } else {
                // 2*3*sizeof(float)+1*sizeof(float)
                singleDataSize = bufferNum * XXGLU_TQUE_NUM * sizeof(int32_t) + SWIGLU_TBUF_NUM_FLOAT * sizeof(int32_t);
                if (tiling_.get_hasProbs()) {
                    singleDataSize += bufferNum * 1 * sizeof(float) + 1 * sizeof(int32_t);
                }
            }
            break;
        case GLU_FLAG::SWIGLU_BACKWARD:
            if (dtype == ge::DT_FLOAT16) {
                singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                    SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t);
            } else if (dtype == ge::DT_BF16) {
                // 2*3*sizeof(half)+2*sizeof(float)
                singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                    SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t);
            } else {
                singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int32_t) +
                    SWIGLU_BW_TBUF_NUM_FLOAT * sizeof(int32_t);
            }
            break;
        case GLU_FLAG::PANGLU:
            if (dtype == ge::DT_FLOAT16) {
                singleDataSize = bufferNum * PANGLU_TW_TQUE_NUM_HALF * sizeof(int16_t) +
                    PANGLU_TW_TBUF_NUM_HALF * sizeof(int32_t);
            } else {
                singleDataSize = bufferNum * PANGLU_TW_TQUE_NUM_FLOAT * sizeof(int32_t) +
                    PANGLU_TW_TBUF_NUM_FLOAT * sizeof(int32_t);
            }
            break;
        default:
            return false;
    }
    dataLenPerUB = OpDiv(ubSize, singleDataSize, ubSize);
    return true;
}

template <GLU_FLAG Glu_Flag>
template <uint16_t bufferNum>
inline bool GluSingleTilingCalculator<Glu_Flag>::CalcUbMaxTileLen(
    uint64_t ubSize, ge::DataType& dtype, GluSingleTilingOptParam& optTiling)
{
    // get buffernum and maxTileLen
    uint64_t maxTileLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB<bufferNum>(ubSize, dtype, maxTileLenPerUB)) {
        OP_LOGE("CalcTiling", "Get bufferNum %u and maxTileLenPerUB %lu failed", bufferNum, maxTileLenPerUB);
        return false;
    }
    optTiling.maxTileLen = ALIGNDOWN(maxTileLenPerUB, ubMinBlockLen);
    OP_LOGI("CalcTiling", "ubSize:%u, bufferNum:%u, maxTileLenPerUB:%lu", ubSize, bufferNum, optTiling.maxTileLen);
    return true;
}

template <GLU_FLAG Glu_Flag>
inline void GluSingleTilingCalculator<Glu_Flag>::SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_,
    GluSingleTilingOptParam& optTiling)
{
    uint64_t totalTileNum = DIVCEIL(tiling_.get_rowLen(), (baseRowLen_)) * DIVCEIL(tiling_.get_colLen(), (baseColLen_));
    uint64_t baseSize = baseRowLen_ * baseColLen_;
    if (baseRowLen_ == 0 || baseColLen_ == 0) {
        OP_LOGI("SaveOptBaseShape", "baseRowLen_:%u or baseColLen:%u is zero.", baseRowLen_, baseColLen_);
        return;
    }
    uint64_t baseTileNum = (tiling_.get_rowLen() / baseRowLen_) * (tiling_.get_colLen() / baseColLen_);
    uint16_t totalUsedCoreNum = std::min(totalTileNum, (uint64_t)totalAvailableCore);
    if ((optTiling.optTotalTileNum == 0) ||
        (totalUsedCoreNum > optTiling.totalUsedCoreNum) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum < optTiling.optTotalTileNum)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) &&
        (baseSize > optTiling.optBaseSize)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) &&
        (baseSize == optTiling.optBaseSize) && (baseTileNum > optTiling.optBaseTileNum))) {
        optTiling.optBaseRowLen = baseRowLen_;
        optTiling.optBaseColLen = baseColLen_;
        optTiling.optTotalTileNum = totalTileNum;
        optTiling.optBaseSize = baseSize;
        optTiling.optBaseTileNum = baseTileNum;
        optTiling.totalUsedCoreNum = totalUsedCoreNum;
        optTiling.tileNumPerCore = DIVCEIL(totalTileNum, totalUsedCoreNum);
    }
}

template <GLU_FLAG Glu_Flag>
inline uint32_t GluSingleTilingCalculator<Glu_Flag>::getBaseColLenUpBound(GluSingleTilingOptParam& optTiling)
{
    uint32_t upBound = std::min(tiling_.get_colLen(), (uint64_t)optTiling.maxTileLen);
    const uint32_t minInputDTypeLen = 2;
    if (tiling_.get_is32BAligned() == 1) {
        upBound = std::min(upBound, (uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN);
    } else {
        upBound = std::min(upBound, OpDiv((uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN, inputDTypeLen, minInputDTypeLen));
    }

    if (upBound < tiling_.get_colLen() && upBound > cacheLineLen) {
        // 该种场景，每一个colLen至少被切割成2块，需要保证baseColLen为512B整数倍才高效
        return ALIGNDOWN(upBound, cacheLineLen);
    } else {
        return upBound;
    }
}

template <GLU_FLAG Glu_Flag>
inline uint32_t GluSingleTilingCalculator<Glu_Flag>::getBaseRowLenUpBound()
{
    return std::min(tiling_.get_rowLen(), (uint64_t)DISCONTINE_COPY_MAX_BLOCKCNT);
}

/**
 * 为了避免stride长度大于上限
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535 * ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
template <GLU_FLAG Glu_Flag>
inline bool GluSingleTilingCalculator<Glu_Flag>::MustBeSingleBaseRowLen(uint32_t baseColLen_)
{
    if (tiling_.get_is32BAligned() == 1) {
        return ((tiling_.get_colLen() * REAL_INPUT_NUM - baseColLen_) > (DISCONTINE_COPY_MAX_STRIDE * ubMinBlockLen));
    } else {
        return (((tiling_.get_colLen() * REAL_INPUT_NUM - baseColLen_) * inputDTypeLen) > DISCONTINE_COPY_MAX_STRIDE);
    }
}

/**
 * 若则baseRowLen大于1，且约束判决baseRowLen必须等于1时，则是不合法的
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
template <GLU_FLAG Glu_Flag>
inline bool GluSingleTilingCalculator<Glu_Flag>::isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_)
{
    return ((baseRowlen_ < 1) || (baseRowlen_ > 1 && MustBeSingleBaseRowLen(baseColLen_)));
}

template <GLU_FLAG Glu_Flag>
inline bool GluSingleTilingCalculator<Glu_Flag>::CalcOptBaseShape(GluSingleTilingOptParam& optTiling)
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    OP_LOGI("CalcOptBaseShape", "init baseColLen : %u", baseColLen_);
    if (MustBeSingleBaseRowLen(baseColLen_)) {
        SaveOptBaseShape(1, baseColLen_, optTiling);
        return true;
    }

    while (true) {
        // colLen非32B对齐时，数据copy到ub时，每一行的尾部会补齐32B
        uint32_t baseRowlen_ = std::min(optTiling.maxTileLen / ALIGNUP(baseColLen_, ubMinBlockLen),
                                        getBaseRowLenUpBound());
        if (isInvalidBaseShape(baseRowlen_, baseColLen_)) {
            OP_LOGI("CalcOptBaseShape", "baseRowLen:%u or baseColLen:%u is invalid. optTotalTileNum:%u end",
                    baseRowlen_, baseColLen_, optTiling.optTotalTileNum);
            // optTotalTileNum有效，则前面有最优解，返回true;否则返回false
            return (optTiling.optTotalTileNum > 0);
        }
        // 保存较优的base shape
        SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);

        // baseColLen已经到达下限 或者 baseRowlen已经达到上限，无法继续调整，结束
        if (baseColLen_ <= alignPackLen || (baseRowlen_ >= getBaseRowLenUpBound())) {
            return true; // baseColLen无法继续调整了，结束
        }

        // 继续调整baseColLen
        // baseColLen为若alignPackLen的整数倍，则baseColLen减少1个alignPackLen的长度
        // 否则baseColLen减少到alignPackLen的整数倍（最接近的）
        if (baseColLen_ % alignPackLen == 0) {
            baseColLen_ -= alignPackLen;
        } else {
            baseColLen_ = ALIGNDOWN(baseColLen_, alignPackLen);
        }
    }
}

using RowColParamPair = std::pair<std::pair<size_t, size_t>, std::pair<GluSingleTilingOptParam, bool>>;

constexpr RowColParamPair createEntry(size_t first, size_t second, GluSingleTilingOptParam param, bool doubleBuffer)
{
    return { {first, second}, {param, doubleBuffer} };
}

constexpr std::array<RowColParamPair, 14U> ParamMap310P = {
    // row, col
    // {maxTileLen, BaseRowLen, BaseColLen, TotalTileNum, BaseSize, BaseTileNum, totalUsedCoreNum, tileNumPerCore}
    createEntry(8U, 9504U, {14560U, 1U, 9504U, 8U, 9504U, 8U, 8U, 1U}, true),
    createEntry(8U, 6912U, {18720U, 2U, 3584U, 8U, 7168U, 4U, 8U, 1U}, false),
    createEntry(8192U, 9504U, {18720U, 36U, 512U, 4332U, 18432U, 4086U, 8U, 542U}, false),
    createEntry(8192U, 6912U, {18720U, 73U, 256U, 3051U, 18688U, 3024U, 8U, 382U}, false),

    createEntry(16U, 9504U, {14560U, 1U, 9504U, 16U, 9504U, 16U, 8U, 2U}, true),
    createEntry(16U, 6912U, {14560U, 2U, 6912U, 8U, 13824U, 8U, 8U, 1U}, true),
    createEntry(16384U, 9504U, {18720U, 36U, 512U, 8664U, 18432U, 8190U, 8U, 1083U}, false),
    createEntry(16384U, 6912U, {18720U, 8U, 2304U, 6144U, 18432U, 6144U, 8U, 768U}, false),

    createEntry(4U, 9504U, {18720U, 1U, 4864U, 8U, 4864U, 4U, 8U, 1U}, false),
    createEntry(4U, 6912U, {18720U, 2U, 1792U, 8U, 3584U, 6U, 8U, 1U}, false),
    createEntry(4096U, 9504U, {18720U, 36U, 512U, 2166U, 18432U, 2034U, 8U, 271U}, false),
    createEntry(4096U, 6912U, {18720U, 73U, 256U, 1539U, 18688U, 1512U, 8U, 193U}, false),
};

constexpr std::array<RowColParamPair, 14U> ParamMap910B = {
    // {maxTileLen, BaseRowLen, BaseColLen, TotalTileNum, BaseSize, BaseTileNum, totalUsedCoreNum, tileNumPerCore}
    createEntry(8U, 9504U, {14016U, 1U, 4864U, 16U, 4864U, 8U, 16U, 1U}, false),
    createEntry(8U, 6912U, {14016U, 2U, 2304U, 12U, 4608U, 12U, 12U, 1U}, false),
    createEntry(8192U, 9504U, {14016U, 1U, 9504U, 8192U, 9504U, 8192U, 40U, 205U}, false),
    createEntry(8192U, 6912U, {14016U, 2U, 6912U, 4096U, 13824U, 4096U, 40U, 103U}, false),

    createEntry(16U, 9504U, {14016U, 1U, 9504U, 16U, 9504U, 16U, 16U, 1U}, false),
    createEntry(16U, 6912U, {14016U, 9U, 1536U, 10U, 13824U, 4U, 10U, 1U}, false),
    createEntry(16384U, 9504U, {14016U, 1U, 9504U, 16384U, 9504U, 16384U, 40U, 410U}, false),
    createEntry(16384U, 6912U, {14016U, 2U, 6912U, 8192U, 13824U, 8192U, 40U, 205U}, false),

    createEntry(4U, 9504U, {14016U, 1U, 6656U, 8U, 6656U, 4U, 8U, 1U}, false),
    createEntry(4U, 6912U, {14016U, 2U, 3840U, 4U, 7680U, 2U, 4U, 1U}, false),
    createEntry(4096U, 9504U, {14016U, 1U, 9504U, 4096U, 9504U, 4096U, 40U, 103U}, false),
    createEntry(4096U, 6912U, {14016U, 2U, 6912U, 2048U, 13824U, 2048U, 40U, 52U}, false),

    createEntry(96U, 4864U, {14016U, 54U, 256U, 38U, 13824U, 19U, 38U, 1U}, false),  // 10.79
    createEntry(57600U, 4864U, {10896U, 2U, 4864U, 28800U, 9728U, 28800U, 48U, 600U}, true), // 1423.79
};

template <GLU_FLAG Glu_Flag>
bool GluSingleTilingCalculator<Glu_Flag>::GetOptTiling(ge::DataType& dtype, GluSingleTilingOptParam& optTiling)
{
    if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16) {
        return false;
    }
    uint32_t enableDoubleBuffer = (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) ? 0 : 1;
    auto& paramMap = (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) ? ParamMap310P : ParamMap910B;
    for (const auto& entry : paramMap) {
        if (entry.first.first == tiling_.get_rowLen() && entry.first.second == tiling_.get_colLen()) {
            optTiling = entry.second.first;
            tiling_.set_baseRowLen(optTiling.optBaseRowLen);
            tiling_.set_baseColLen(optTiling.optBaseColLen);
            usedCoreNum = optTiling.totalUsedCoreNum;
            tiling_.set_isDoubleBuffer(entry.second.second);
            return true;
        }
    }
    return false;
}

// 如果开启double buffer，则bufferNum为2，否则为1
template <GLU_FLAG Glu_Flag>
template <uint16_t bufferNum>
bool GluSingleTilingCalculator<Glu_Flag>::CalcOptTiling(uint64_t ubSize, ge::DataType& dtype, GluSingleTilingOptParam& optTiling)
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen<bufferNum>(ubSize, dtype, optTiling)) {
        return false;
    }

    // 计算最优的base块形状
    if (!CalcOptBaseShape(optTiling)) {
        return false;
    }

    optTiling.Print();
    return true;
}

template <GLU_FLAG Glu_Flag>
bool GluSingleTilingCalculator<Glu_Flag>::CalcTiling(uint32_t totalCore, uint64_t ubSize, ge::DataType& dtype)
{
    totalAvailableCore = totalCore;
    inputDTypeLen = ge::GetSizeByDataType(dtype);
    if (inputDTypeLen < 0) {
        OP_LOGI("CalcTiling", "Unsupported input data type %d", dtype);
        return false;
    }
    if (inputDTypeLen == 0) {
        OP_LOGI("CalcTiling", "inputDTypeLen is invalid: %d", inputDTypeLen);
        return false;
    }
    ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen; // min block size
    cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen; // bandwidth max efficiency
    alignPackLen = cacheLineLen; // 默认512对齐，策略可调整
    OP_LOGI("CalcTiling", "inputDTypeLen:%u ubMinBlockLen:%u cacheLineLen:%u alignPackLen:%u",
        inputDTypeLen, ubMinBlockLen, cacheLineLen, alignPackLen);

    // Is 32-byte aligned for split colLen?
    tiling_.set_is32BAligned(tiling_.get_colLen() % ubMinBlockLen == 0);
    if (!tiling_.get_is32BAligned() && socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        OP_LOGE("CalcTiling", "XxGLU single doesn't support unaligned(32B) input");
        return false;
    }

    GluSingleTilingOptParam optTilingDb;
    if (GetOptTiling(dtype, optTilingDb)) {
        return true;
    }

    // 先计算开启double buffer的tiling参数
    tiling_.set_isDoubleBuffer(1);
    if (!CalcOptTiling<SWIGLU_TBUF_NUM_HALF>(ubSize, dtype, optTilingDb)) {
        return false;
    }
    GluSingleTilingOptParam *optTiling = &optTilingDb;

    // 如果double buffer开启的tiling参数中，每个核需要处理的tileNum等于2，尝试关闭double buffer;
    // 若关闭double buffer后只需要搬运1次数据，且使用的核没有减少, 则使用关闭double buffer的tiling
    if (optTilingDb.tileNumPerCore <= REAL_INPUT_NUM) {
        GluSingleTilingOptParam optTilingNoDb;
        if (CalcOptTiling<1>(ubSize, dtype, optTilingNoDb) &&
            (optTilingNoDb.tileNumPerCore == 1) && (optTilingNoDb.totalUsedCoreNum >= optTilingDb.totalUsedCoreNum)) {
            optTiling = &optTilingNoDb;
            tiling_.set_isDoubleBuffer(0);
        }
    }
    optTiling->Print();

    // 记录最优的结果
    tiling_.set_baseRowLen(optTiling->optBaseRowLen);
    tiling_.set_baseColLen(optTiling->optBaseColLen);
    usedCoreNum = optTiling->totalUsedCoreNum;
    Print();

    return true;
}

template <GLU_FLAG Glu_Flag> bool GluSingleTilingCalculator<Glu_Flag>::IsCapable()
{
    if (context_ == nullptr) {
        return false;
    }

    auto inTensor = GetInputTensor(0);
    if (inTensor->GetDataType() == ge::DataType::DT_BF16) {
        // 310P不支持BF16
        return (socVersion_ != platform_ascendc::SocVersion::ASCEND310P);
    }
    return inTensor->GetDataType() == ge::DataType::DT_FLOAT || inTensor->GetDataType() == ge::DataType::DT_FLOAT16;
}

template <GLU_FLAG Glu_Flag> ge::graphStatus GluSingleTilingCalculator<Glu_Flag>::CalcTiling()
{
    auto inTensor = GetInputTensor(GetSelfIdx());
    auto inShape = inTensor->GetOriginShape();
    int64_t inDim = GetInputFormat(0) == ge::FORMAT_FRACTAL_NZ ? 0 : -1;
    auto dataType = inTensor->GetDataType();
    tiling_.set_hasProbs(GetAttrWithDefault<bool>(3U, false));
    if (!SetTotalShape(inShape, inDim) || !CalcTiling(limitCoreNum_, limitUbMemSize_, dataType)) {
        return ge::GRAPH_FAILED;
    }

    if constexpr (Glu_Flag == SWIGLU) {
        // [Add by GTS-AILab] [feature GTSOps] limit 占用 attr index1，highPrecision 顺移到 index2
        tiling_.set_highPrecision(GetAttrWithDefault<bool>(2, true));
    }

    // [Add by GTS-AILab] [feature GTSOps] 读 limit attr(index1) 写入 tiling，供 kernel clamp 使用
    tiling_.set_limit(GetAttrWithDefault<float>(1, 0.0f));

    return ge::GRAPH_SUCCESS;
}

template <GLU_FLAG Glu_Flag>
inline ge::graphStatus Tiling4Glu(gert::TilingContext* context)
{
    GluSingleTilingCalculator<Glu_Flag> tilingCalculator(context);
    return tilingCalculator.DoTiling();
}

} // end optilingSingle
#endif  // OP_XXGLU_LIMIT_TILING_HOST_SINGLE_H   // [Add by GTS-AILab] [feature GTSOps]
