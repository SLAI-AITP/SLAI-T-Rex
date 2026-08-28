/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_H

#include <numeric>
#include <algorithm>
#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_template_tiling_key.h"
#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "err/ops_err.h"

// tiling_base 需要
#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

using namespace ge;
using namespace AscendC;

namespace optiling {

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t blockDim = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

class TilingBaseClass {
public:
    explicit TilingBaseClass(gert::TilingContext* context) : context_(context)
    {}

    virtual ~TilingBaseClass() = default;

    // Tiling执行框架
    //     1、GRAPH_SUCCESS: 成功，并且不需要继续执行后续Tiling类的实现
    //     2、GRAPH_FAILED: 失败，中止整个Tiling流程
    //     3、GRAPH_PARAM_INVALID: 本类不支持，需要继续往下执行其他Tiling类的实现
    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context_->SetTilingKey(GetTilingKey());
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }

    // 更新 context
    virtual void Reset(gert::TilingContext* context)
    {
        context_ = context;
    }

protected:
    virtual bool IsCapable() = 0;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    virtual ge::graphStatus GetPlatformInfo() = 0;
    // 2、获取INPUT/OUTPUT/ATTR信息
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    // 3、计算数据切分TilingData
    virtual ge::graphStatus DoOpTiling() = 0;
    // 4、计算TilingKey
    [[nodiscard]] virtual uint64_t GetTilingKey() const = 0;
    // 5、计算Workspace 大小
    virtual ge::graphStatus GetWorkspaceSize() = 0;
    // 6、Dump Tiling数据
    virtual void DumpTilingInfo()
    {
        int32_t enable = 1;
        if (enable != 1) {
            return;
        }
        auto buf = (uint32_t*)context_->GetRawTilingData()->GetData();
        auto bufLen = context_->GetRawTilingData()->GetDataSize();
        std::ostringstream oss;
        oss << "Start to dump tiling info. tilingkey:" << context_->GetTilingKey() << ", tiling data size:" << bufLen
            << ", content:";
        for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
            oss << *(buf + i) << ",";
            if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
                OP_LOGD(context_, "%s", oss.str().c_str());
                oss.str("");
            }
        }
        OP_LOGD(context_, "%s", oss.str().c_str());
    }

    static uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
    {
        uint32_t ration;
        if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
            return sliceNum;
        }
        ration = aivCoreNum / aicCoreNum;
        return (sliceNum + (ration - 1)) / ration;
    }

    template <typename T>
    [[nodiscard]] std::string GetShapeDebugStr(const T& shape) const
    {
        std::ostringstream oss;
        oss << "[";
        if (shape.GetDimNum() > 0) {
            for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
                oss << shape.GetDim(i) << ", ";
            }
            oss << shape.GetDim(shape.GetDimNum() - 1);
        }
        oss << "]";
        return oss.str();
    }

    [[nodiscard]] std::string GetTensorDebugStr(
        const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
    {
        if (shape == nullptr || tensor == nullptr) {
            return "nil ";
        }
        std::ostringstream oss;
        oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
        oss << "(shape:" << GetShapeDebugStr(shape->GetStorageShape()) << "),";
        oss << "(ori_shape:" << GetShapeDebugStr(shape->GetOriginShape()) << "),";
        oss << "(format: "
            << ge::TypeUtils::FormatToSerialString(
                   static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
            << "),";
        oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";
        return oss.str();
    }

    [[nodiscard]] std::string GetTilingContextDebugStr()
    {
        std::ostringstream oss;
        for (size_t i = 0; i < context_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
            oss << "input" << i << ": ";
            oss << GetTensorDebugStr(context_->GetInputShape(i), context_->GetInputDesc(i));
        }

        for (size_t i = 0; i < context_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
            oss << "output" << i << ": ";
            oss << GetTensorDebugStr(context_->GetOutputShape(i), context_->GetOutputDesc(i));
        }
        return oss.str();
    }

    [[nodiscard]] std::string GetTilingDataDebugStr() const
    {
        auto rawTilingData = context_->GetRawTilingData();
        auto rawTilingDataSize = rawTilingData->GetDataSize();
        auto data = reinterpret_cast<const int32_t*>(rawTilingData->GetData());
        size_t len = rawTilingDataSize / sizeof(int32_t);
        std::ostringstream oss;
        for (size_t i = 0; i < len; i++) {
            oss << data[i] << ", ";
        }
        return oss.str();
    }

protected:
    gert::TilingContext* context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_;
};

// SLIGKLLOSS tiling类定义 继承TilingBaseClass
struct SparseLightningIndexerGradKLLossCompileInfo {
    int64_t core_num;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
};

class SparseLightningIndexerGradKLLossTilingBase : public TilingBaseClass {
public:
    explicit SparseLightningIndexerGradKLLossTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~SparseLightningIndexerGradKLLossTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset() {
        bSize = 0;
        gSizeQuery = 0;
        gSizeQueryIndex = 0;
        dSizeQuery = 0;
        dSizeQueryIndex = 0;
        kSize = 2048;
        n2Size = 0;
        s1Size = 0;
        s2Size = 0;
        accumS1 = 0;
        accumS2 = 0;
        sparseMode = 3;
        scaleValue = 1.0f;
        dQueryRopeSize = 0;
        dKeyRopeSize = 0;
        cmpRatio = 0;
        hasSoftmaxInput = 0;
        deterministic = false;

        opName = nullptr;
        inputLayout = nullptr;
    }

    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable()
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    void GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;
    ge::graphStatus CheckContext();
    bool AnalyzeAttrs();
    bool CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShap);
    bool AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape,
                         const gert::Shape &topKShape, size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    int64_t GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size, int32_t s1Idx);
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray);
    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                            const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    bool Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray, 
                    const int64_t balanceNum);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                       int64_t maxCoreNum);
    void SetSparseParamsRegbase(int64_t maxCoreNum);
    int64_t CalcTotalSize();
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    void InitOutputSplit();

    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t gSizeQuery;
    int32_t gSizeQueryIndex;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSizeQuery;
    int32_t dSizeQueryIndex;
    int32_t kSize;
    int32_t sparseMode;
    int32_t rsvd;
    int64_t cmpRatio;
    int64_t hasSoftmaxInput;
    float scaleValue;

    const char *templateName = "slikbase";

    LayoutType tilingKeyLayout;
    TopKRange topkSize;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize;

    int64_t realT1Size;
    int64_t dQueryRopeSize;
    int64_t dKeyRopeSize;
    int64_t accumS1;
    int64_t accumS2;
    int64_t maxS1Val;
    int64_t maxS2Val;

    platform_ascendc::SocVersion socVersion;
    bool deterministic;
    bool hasRope = false;

    const char *opName;
    const char *inputLayout;

    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKData;

    SparseLightningIndexerGradKLLossTilingData *tilingData = context_->GetTilingData<SparseLightningIndexerGradKLLossTilingData>();
    SLIGradKLLossBaseParams *sliGradkllossBaseParams_ = &tilingData->baseParams;
    SLIGradKLLossMultiCoreParams *sliGradkllossMultiCoreParams_ = &tilingData->multiCoreParams;
};

} // optiling
#endif