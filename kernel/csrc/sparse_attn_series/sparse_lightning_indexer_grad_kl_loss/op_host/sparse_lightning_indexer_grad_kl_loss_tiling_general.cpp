/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp
 * \brief
 */
#include "sparse_lightning_indexer_grad_kl_loss_tiling_general.h"
#include <tiling/tiling_api.h>
using namespace ge;
using namespace AscendC;

namespace optiling {

static const int64_t PING_PONG_VALUE = 2L;
static const int64_t GM_ALIGN = 512;
static const int32_t FRACTAL_NUM = 16L;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;

static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_33K = 33 * 1024;

static constexpr uint32_t NQUERY_SIZE_8   = 8;
static constexpr uint32_t NQUERY_SIZE_16  = 16;
static constexpr uint32_t NQUERY_SIZE_32  = 32;
static constexpr uint32_t NQUERY_SIZE_64  = 64;
static constexpr uint32_t NQUERY_SIZE_128 = 128;

static constexpr uint32_t NQUERYINDEX_SIZE_8  = 8;
static constexpr uint32_t NQUERYINDEX_SIZE_16 = 16;
static constexpr uint32_t NQUERYINDEX_SIZE_32 = 32;
static constexpr uint32_t NQUERYINDEX_SIZE_64 = 64;

static constexpr uint32_t N2_SIZE_1 = 1;
static constexpr uint32_t D_SIZE_512 = 512;
static constexpr uint32_t DINDEX_SIZE_128 = 128;
static constexpr uint32_t DROPE_SIZE_64 = 64;
static constexpr uint32_t TOPK_SIZE_2048 = 2048;
static constexpr int64_t  SPARSE_MODE_SIZE_3 = 3;
static constexpr uint64_t CMP_RATIO_INDEX = 6;
static constexpr int64_t CMP_RATIO_SUPPORT_4 = 4;

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    // 输入属性参数验证
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    // idx++跳过pretoken和nexttoken
    idx++;
    idx++;
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    // 输入shape验证
    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX);
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX);
    auto queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX);
    auto keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX);
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX);
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    // softmaxMax和softmaxSum为可选输入
    auto softmaxMaxShape = context_->GetOptionalInputShape(SOFTMAX_MAX_INPUT_INDEX);
    auto softmaxSumShape = context_->GetOptionalInputShape(SOFTMAX_SUM_INPUT_INDEX);
    // 输出shape验证
    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_INDEX_OUTPUT_INDEX);
    auto dkeyIndexShape = context_->GetOutputShape(D_KEY_INDEX_OUTPUT_INDEX);
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX);
    auto lossShape = context_->GetOutputShape(LOSS_OUTPUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dkeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lossShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    
    return ge::GRAPH_SUCCESS;
}

bool SparseLightningIndexerGradKLLossTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto cmpRatioPtr = attrs->GetAttrPointer<int64_t>(CMP_RATIO_INDEX);

    // idx++跳过pretoken和nexttoken
    idx++;
    idx++;
    scaleValue = *scaleValuePtr;
    inputLayout = inputLayoutPtr;
    sparseMode = *sparseModePtr;
    cmpRatio = *cmpRatioPtr;

    deterministic = (context_->GetDeterministic() == 1);

    OP_CHECK_IF((cmpRatio != 0) && (cmpRatio != CMP_RATIO_SUPPORT_4),
                OP_LOGE(opName, " The value of cmp_ratio only support 0 or 4 but got [%d]", cmpRatio),
                return false);

    OP_CHECK_IF(sparseMode != SPARSE_MODE_SIZE_3,
                OP_LOGE(opName, " The value of sparse_mode is [%d], but currently only supports mode [3].", sparseMode),
                return false);
    OP_LOGD(context_, "attrs: scaleValue[%f] input_layout[%s] sparse_mode[%ld].",
            scaleValue, inputLayout, sparseMode);
    return true;
}

void SparseLightningIndexerGradKLLossTilingBase::GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    int64_t seqLen = actualSeqLenShape.GetDim(0);
    try{
        res.reserve(seqLen);
    } catch (...) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "Init actual_seq_len failed, array is too long.");
        return;
    }
    res.emplace_back(value[0]);
    actualLen++;
    for (int64_t i = 1; i < seqLen; ++i) {
        auto qLen = value[i] - value[i - 1]; // value[i]代表偏移位置的索引
        qLen = qLen < 0 ? 0 : qLen;
        res.emplace_back(qLen);
        actualLen++;
    }
}

bool SparseLightningIndexerGradKLLossTilingBase::AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape, const gert::Shape &topKShape,
                                                                    size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape)
{
    // dRopeSize的确定，有queryRopeShape 和 keyRopeShape
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            OP_CHECK_IF(true,
                OP_LOGE(opName, "op current not support tnd layout"), return false);
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKLen = 0;
            int64_t t1Size = queryShape.GetDim(0); // TND只有三个数值 T:0 N:1 D:2
            int64_t t2Size = keyShape.GetDim(0);
            realT1Size = t1Size;
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKData.begin(), actualSeqLenKData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_QUERY_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_KEY_INPUT_INDEX, actualSeqLenKData, actualSeqKLen);
            OP_CHECK_IF(actualSeqQLen != actualSeqKLen,
                OP_LOGE(opName, "VarLen scene, q[%ld] is not equal k[%ld].", actualSeqQLen, actualSeqKLen), return false);
            // 校验actualQ 对应每一个元素是否大于 actualK ，大于则拦截
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.begin() + actualSeqQLen, 0LL);
            accumS2 = std::accumulate(actualSeqLenKData.begin(), actualSeqLenKData.begin() + actualSeqKLen, 0LL);
            OP_CHECK_IF(
                t1Size != accumS1 || t2Size != accumS2,
                OP_LOGE(
                    opName,
                    "Query Tsize(%ld) and key Tsize(%ld) must be equal to sum of seqQLen(%ld) and seqkLen(%ld), respectively.",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKData.begin(), actualSeqLenKData.end());
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            n2Size = keyShape.GetDim(1);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSizeQuery = queryShape.GetDim(1) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(1) / n2Size;
            dSizeQuery = queryShape.GetDim(2);
            dSizeQueryIndex = queryIndexShape.GetDim(2);
            kSize = topKShape.GetDim(2);
            if (cmpRatio != 4) {
                OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || kSize % BUFFER_SIZE_BYTE_1K > 0,
                    OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024.", kSize),
                    return false);
            } else {
                OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || (kSize % BUFFER_SIZE_BYTE_1K > 0 && kSize != 512),
                    OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024 or 512.", kSize),
                    return false);
                OP_CHECK_IF(kSize == 512 && hasRope,
                    OP_LOGE(opName, "topK(512) is only supported when hasRope is false."),
                    return false);
            }
            topkSize = (TopKRange) kSize;
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(2);
                dKeyRopeSize = keyRopeShape.GetDim(2);
            }
            tilingData->baseParams.set_layoutType(LAYOUT_TND);
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        }
    } else if (layoutLen == 4UL){
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2);
            if (cmpRatio != 4) {
                OP_CHECK_IF(
                    s1Size > s2Size,
                    OP_LOGE(
                        opName,
                        "Query s1Size(%ld) must be small than Key s2Size(%ld).",
                        s1Size, s2Size),
                    return false);
            } else {
                OP_CHECK_IF(
                    s1Size / cmpRatio != s2Size,
                    OP_LOGE(
                        opName,
                        "current op only support ori_kv = q_s1"),
                    return false);
            }
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSizeQuery = queryShape.GetDim(2) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(2) / n2Size;
            dSizeQuery = queryShape.GetDim(3);
            dSizeQueryIndex = queryIndexShape.GetDim(3);
            kSize = topKShape.GetDim(3);
            if (cmpRatio != 4) {
                OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || kSize % BUFFER_SIZE_BYTE_1K > 0,
                    OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024.", kSize),
                    return false);
            } else {
                OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || (kSize % BUFFER_SIZE_BYTE_1K > 0 && kSize != 512),
                    OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024 or 512.", kSize),
                    return false);
                OP_CHECK_IF(kSize == 512 && hasRope,
                    OP_LOGE(opName, "topK(512) is only supported when hasRope is false."),
                    return false);
            }

            topkSize = (TopKRange) kSize;
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(3);
                dKeyRopeSize = keyRopeShape.GetDim(3);
            }
            tilingData->baseParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        }        
    } else {
        return false;
    }

    return true;
}

// 对每一个输入参数的变量类型进行对比校验
bool SparseLightningIndexerGradKLLossTilingBase::AnalyzeDtype()
{
    // 对8个必须输入的参数进行参数类型判断
    // 输入空指针校验
    auto queryDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto keyDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto queryIndexDesc = context_->GetInputDesc(QUERY_INDEX_INPUT_INDEX);
    auto keyIndexDesc = context_->GetInputDesc(KEY_INDEX_INPUT_INDEX);
    auto weightsDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto sparseIndicesDesc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    auto softmaxMaxDesc = context_->GetOptionalInputDesc(SOFTMAX_MAX_INPUT_INDEX);
    auto softmaxSumDesc = context_->GetOptionalInputDesc(SOFTMAX_SUM_INPUT_INDEX); 
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesDesc);

    // 以下5个为16类型保持一致
    auto queryDtype = queryDesc->GetDataType();
    auto keyDtype = keyDesc->GetDataType();
    auto queryIndexDtype = queryIndexDesc->GetDataType();
    auto keyIndexDtype = keyIndexDesc->GetDataType();
    auto weightsDtype = weightsDesc->GetDataType();

    // 以下1个为32类型，softmaxMax和softmaxSum为可选输入
    auto sparseIndicesDtype = sparseIndicesDesc->GetDataType();
    bool hasSoftmaxInput = (softmaxMaxDesc != nullptr && softmaxSumDesc != nullptr);
    bool same16 = false; // 判断输入为fp16或者部分16的是否一致
    bool same32 = false; // 判断输入为int32或者fp32的类型是否正确
    if (queryDtype == ge::DT_FLOAT16 && keyDtype == ge::DT_FLOAT16 && queryIndexDtype == ge::DT_FLOAT16 && 
                            keyIndexDtype == ge::DT_FLOAT16 && weightsDtype == ge::DT_FLOAT16) {
        same16 = true;
    } else if (queryDtype == ge::DT_BF16 && keyDtype == ge::DT_BF16 && queryIndexDtype == ge::DT_BF16 && 
                            keyIndexDtype == ge::DT_BF16 && weightsDtype == ge::DT_BF16) {
        same16 = true;
    } else {
        OP_LOGW(context_, "InputDtype is not same.: queryDtype[%s], keyDtype[%s], queryIndexDtype[%s], keyIndexDtype[%s], weightsDtype[%s]",
            ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(queryIndexDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyIndexDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str());
        same16 = false;
    } 
    if (sparseIndicesDtype == ge::DT_INT32) {
        if (hasSoftmaxInput) {
            auto softmaxMaxDtype = softmaxMaxDesc->GetDataType();
            auto softmaxSumDtype = softmaxSumDesc->GetDataType();
            if (softmaxMaxDtype == ge::DT_FLOAT && softmaxSumDtype == ge::DT_FLOAT) {
                same32 = true;
            } else {
                OP_LOGW(context_, "InputDtype is fault: sparseIndicesDtype must be int32, but[%s]; softmaxMaxDtype must be float32, but[%s]; softmaxSumDtype must be float32, but[%s].",
                    ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(softmaxMaxDtype).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(softmaxSumDtype).c_str());
                same32 = false;
            }
        } else {
            same32 = true;
        }
    } else {
        OP_LOGW(context_, "InputDtype is fault: sparseIndicesDtype must be int32, but[%s].",
            ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str());
        same32 = false;
    }
    // 所有类型不满足返回false
    if(same16 == false || same32 == false) {
        return false;
    }
    OP_LOGW(context_, "InputDtype: queryDtype[%s], keyDtype[%s], queryIndexDtype[%s], keyIndexDtype[%s], weightsDtype[%s],",
        ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(queryIndexDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyIndexDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str());
    if (hasSoftmaxInput) {
        auto softmaxMaxDtype = softmaxMaxDesc->GetDataType();
        auto softmaxSumDtype = softmaxSumDesc->GetDataType();
        OP_LOGW(context_, "sparseIndicesDtype[%s], softmaxMaxDtype[%s], softmaxSumDtype[%s].",
            ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(softmaxMaxDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(softmaxSumDtype).c_str());
    } else {
        OP_LOGW(context_, "sparseIndicesDtype[%s], softmaxMax/softmaxSum: not provided.",
            ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str());
    }
    return true;
}

// 输入shape进行交叉验证，防止数据错误输入
bool SparseLightningIndexerGradKLLossTilingBase::CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape)
{
    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    
    // 检查softmax_max和softmax_sum输入状态
    auto softmaxMax = context_->GetOptionalInputShape(SOFTMAX_MAX_INPUT_INDEX);
    bool hasSoftmaxMax = softmaxMax != nullptr && softmaxMax->GetStorageShape().GetDimNum() != 0;
    auto softmaxSum = context_->GetOptionalInputShape(SOFTMAX_SUM_INPUT_INDEX);
    bool hasSoftmaxSum = softmaxSum != nullptr && softmaxSum->GetStorageShape().GetDimNum() != 0;
    if (hasSoftmaxMax ^ hasSoftmaxSum) {
        OP_LOGE(opName, "softmax_max and softmax_sum should be present or absent at the same time, check this.");
        return false;
    }
    hasSoftmaxInput = (hasSoftmaxMax && hasSoftmaxSum) ? 1 : 0;
    
    // 仅当cmpRatio ==4 时支持同时不传softmax_max softmax_sum
    if (hasSoftmaxInput == 0 && cmpRatio != 4) {
        OP_LOGE(opName, "Only when cmp_ratio is 4, softmax_max and softmax_sum can be absent at the same time. current cmp_ratio is %ld", cmpRatio);
        return false;
    }
    
    // 获取softmax_max和softmax_sum的形状（如果存在）
    gert::Shape defaultShape;
    auto softmaxMaxShape = (hasSoftmaxMax) ? softmaxMax->GetStorageShape() : defaultShape;
    auto softmaxSumShape = (hasSoftmaxSum) ? softmaxSum->GetStorageShape() : defaultShape;
    // 下面数字对应shape输入位置
    if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
        int64_t t1Len = queryShape[0];
        int64_t n1Len = queryShape[1];
        int64_t n1indexLen = queryIndexShape[1];
        int64_t t2Len = keyShape[0];
        int64_t n2Len = keyShape[1];
        // 验证T1
        OP_CHECK_IF(queryIndexShape[0] != t1Len || weightsShape[0] != t1Len || (hasSoftmaxInput && (softmaxMaxShape[1] != t1Len || softmaxSumShape[1] != t1Len)),
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify T1 is failed, the value of query[0], query_index[0], weights[0], softmax_max[1] \
                 and softmax_sum[1] are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.", queryShape[0], queryIndexShape[0], weightsShape[0], softmaxMaxShape[1], softmaxSumShape[1]), return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryShape[1] != NQUERY_SIZE_64 && queryShape[1] != NQUERY_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query must be one of the {64, 128}, but the value of query[1] is (%ld)", queryShape[1]), return false);        
        // 验证N Index数字是否正确
        OP_CHECK_IF(queryIndexShape[1] != NQUERYINDEX_SIZE_8 && queryIndexShape[1] != NQUERYINDEX_SIZE_16 && queryIndexShape[1] != NQUERYINDEX_SIZE_32 && queryIndexShape[1] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query_index must be one of the {8, 16, 32, 64}, but the value of query_index[1] is (%ld).", queryIndexShape[1]), return false);
        OP_CHECK_IF(weightsShape[1] != NQUERYINDEX_SIZE_8 && weightsShape[1] != NQUERYINDEX_SIZE_16 && weightsShape[1] != NQUERYINDEX_SIZE_32 && weightsShape[1] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of weights must be one of the {8, 16, 32, 64}, but the value of weights[1] is (%ld).", weightsShape[1]), return false);
        // 验证N Index
        OP_CHECK_IF(queryIndexShape[1] != weightsShape[1],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N index is failed, the value of query_index[1] and weights[1] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[1], weightsShape[1]), return false);
        // 验证T2
        OP_CHECK_IF(keyIndexShape[0] != keyShape[0],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify T2 is failed, the value of key[0] and key_index[0] are respectively (%ld), (%ld). Their values should be equal.", keyShape[0], keyIndexShape[0]), return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key[1] is (%ld).", keyShape[1]), return false);
        OP_CHECK_IF(keyIndexShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[1] is (%ld).", keyIndexShape[1]), return false);
        OP_CHECK_IF(hasSoftmaxInput && softmaxMaxShape[0] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_max[0] is (%ld).", softmaxMaxShape[0]), return false);
        OP_CHECK_IF(hasSoftmaxInput && softmaxSumShape[0] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_sum[0] is (%ld).", softmaxSumShape[0]), return false);
        // 验证N2
        OP_CHECK_IF(keyIndexShape[1] != keyShape[1] || (hasSoftmaxInput && (softmaxMaxShape[0] != keyShape[1] || softmaxSumShape[0] != keyShape[1])),
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N2 is failed, the value of key[1], key_index[1], softmax_max[0] and softmax_sum[0] are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                 keyShape[1], keyIndexShape[1], softmaxMaxShape[0], softmaxSumShape[0]), return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(queryShape[2] != D_SIZE_512,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query must be 512, but the value of query[2] is (%ld)", queryShape[2]), return false);
        OP_CHECK_IF(keyShape[2] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key must be 512, but the value of key[2] is (%ld)", keyShape[2]), return false);
        OP_CHECK_IF(queryIndexShape[2] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[2] is (%ld)", queryIndexShape[2]), return false);
        OP_CHECK_IF(keyIndexShape[2] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[2] is (%ld)", keyIndexShape[2]), return false);        
        // 验证D
        OP_CHECK_IF(keyShape[2] != queryShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key shape D is failed, the value of query[2] and key[2] are respectively (%ld), (%ld). Their values should be equal.", queryShape[2], keyShape[2]), return false);
        OP_CHECK_IF(queryIndexShape[2] != keyIndexShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key_index shape D is failed, the value of query_index[2] and key_index[2] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[2], keyIndexShape[2]), return false);
        // 验证ROPE是否使能
        OP_CHECK_IF(hasRope == 0 && cmpRatio != 4,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query or key rope is failed, rope can't be null"), return false);
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(queryRopeShape[0] != t1Len || queryRopeShape[1] != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to tlSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to nQuerySize(%ld).", queryRopeShape[0], t1Len, queryRopeShape[1], n1Len), return false);
            // 验证keyrope
            OP_CHECK_IF(keyRopeShape[0] != t2Len || keyRopeShape[1] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to t2Size(%ld). \
                    the value of key_rope[1] is (%ld), it should be equal to n2Size(%ld)", keyRopeShape[0], t2Len, keyRopeShape[1], n2Len), return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[2] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_rope must be 64, but the value of query_rope[2] is (%ld).", queryRopeShape[2]), return false);
            OP_CHECK_IF(keyRopeShape[2] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_rope must be 64, but the value of key_rope[2] is (%ld).", keyRopeShape[2]), return false);
            // 验证rope D
            OP_CHECK_IF(queryRopeShape[2] != keyRopeShape[2],
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope D is not equal to key_rope D, the value of query_rope[2] and key_rope[2] is (%ld), (%ld).", queryRopeShape[2], keyRopeShape[2]), return false);
        }           
    } else if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
        int64_t bLen = queryShape[0];
        int64_t s1Len = queryShape[1];
        int64_t n1Len = queryShape[2];
        int64_t n1indexLen = queryIndexShape[2];
        int64_t s2Len = keyShape[1];
        int64_t n2Len = keyShape[2];
        
        // 验证B
        OP_CHECK_IF(queryIndexShape[0] != bLen || weightsShape[0] != bLen || (hasSoftmaxInput && (softmaxMaxShape[0] != bLen || softmaxSumShape[0] != bLen)) ||
                    keyShape[0] != bLen || keyIndexShape[0] != bLen || sparseIndicesShape[0] != bLen,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify shape B is failed, the value of query[0], key[0], query_index[0], key_index[0], weights[0], softmax_max[0], softmax_sum[0] and sparse_indices[0] are respectively (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld).\
                 Their values should be equal.", queryShape[0], keyShape[0], queryIndexShape[0], keyIndexShape[0], weightsShape[0], softmaxMaxShape[0], softmaxSumShape[0], sparseIndicesShape[0]), return false);
        // 验证s1
        OP_CHECK_IF(queryIndexShape[1] != s1Len || weightsShape[1] != s1Len || (hasSoftmaxInput && (softmaxMaxShape[2] != s1Len || softmaxSumShape[2] != s1Len)),
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify S1 is failed, the value of query[1], query_index[1], weights[1], softmax_max[2] \
                 and softmax_sum[2] are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.", queryShape[1], queryIndexShape[1], weightsShape[1], softmaxMaxShape[2], softmaxSumShape[2]), return false);
        // 验证s2
        OP_CHECK_IF(keyIndexShape[1] != s2Len,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify shape S2 is failed, the value of key[1] and key_index[1] are respectively (%ld), (%ld). Their values should be equal.", keyShape[1], keyIndexShape[1]), return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryShape[2] != NQUERY_SIZE_64 && queryShape[2] != NQUERY_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query must be one of the {64, 128}, but the value of query[2] is (%ld)", queryShape[2]), return false);        
        // 验证N Index数字是否正确
        OP_CHECK_IF(queryIndexShape[2] != NQUERYINDEX_SIZE_8 && queryIndexShape[2] != NQUERYINDEX_SIZE_16 && queryIndexShape[2] != NQUERYINDEX_SIZE_32 && queryIndexShape[2] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query_index must be one of the {8, 16, 32, 64}, but the value of query_index[2] is (%ld).", queryIndexShape[2]), return false);        
        OP_CHECK_IF(weightsShape[2] != NQUERYINDEX_SIZE_8 && weightsShape[2] != NQUERYINDEX_SIZE_16 && weightsShape[2] != NQUERYINDEX_SIZE_32 && weightsShape[2] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of weights must be one of the {8, 16, 32, 64}, but the value of weights[2] is (%ld).", weightsShape[2]), return false);         
        // 验证N Index
        OP_CHECK_IF(queryIndexShape[2] != weightsShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N index is failed, the value of query_index[2] and weights[2] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[2], weightsShape[2]), return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyShape[2] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key[2] is (%ld).", keyShape[2]), return false);        
        OP_CHECK_IF(keyIndexShape[2] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[2] is (%ld).", keyIndexShape[2]), return false);
        OP_CHECK_IF(hasSoftmaxInput && softmaxMaxShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_max[1] is (%ld).", softmaxMaxShape[1]), return false);        
        OP_CHECK_IF(hasSoftmaxInput && softmaxSumShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_sum[1] is (%ld).", softmaxSumShape[1]), return false);
        // 验证N2
        OP_CHECK_IF(keyIndexShape[2] != n2Len || (hasSoftmaxInput && (softmaxMaxShape[1] != n2Len || softmaxSumShape[1] != n2Len)),
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N2 is failed, the value of key[2], key_index[2], softmax_max[1] and softmax_sum[1] are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                 keyShape[2], keyIndexShape[2], softmaxMaxShape[1], softmaxSumShape[1]), return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(queryShape[3] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query must be 512, but the value of query[3] is (%ld)", queryShape[3]), return false);
        OP_CHECK_IF(keyShape[3] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key must be 512, but the value of key[3] is (%ld)", keyShape[3]), return false);
        OP_CHECK_IF(queryIndexShape[3] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[3] is (%ld)", queryIndexShape[3]), return false);        
        OP_CHECK_IF(keyIndexShape[3] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[3] is (%ld)", keyIndexShape[3]), return false);       
        // 验证D
        OP_CHECK_IF(keyShape[3] != queryShape[3],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key shape D is failed, the value of query[3] and key[3] are respectively (%ld), (%ld). Their values should be equal.", queryShape[3], keyShape[3]), return false);
        OP_CHECK_IF(queryIndexShape[3] != keyIndexShape[3],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key_index shape D is Failed, the value of query_index[3] and key_index[3] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[3], keyIndexShape[3]), return false);
        // 验证ROPE是否使能
        OP_CHECK_IF(hasRope == 0 && cmpRatio != 4,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query or key rope is failed, rope can't be null"), return false);
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(queryRopeShape[0] != bLen || queryRopeShape[1] != s1Len || queryRopeShape[2] != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to bSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to s1Size(%ld), the value of query_rope[2] is (%ld), it should be equal to nQuerySize(%ld).", queryRopeShape[0], bLen, queryRopeShape[1], s1Len, queryRopeShape[2], n1Len), return false);
            // 验证keyrope
            OP_CHECK_IF(keyRopeShape[0] != bLen || keyRopeShape[1] != s2Len || keyRopeShape[2] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to bSize(%ld), the value of key_rope[1] is (%ld), it should be equal to s2Size(%ld). \
                    the value of key_rope[2] is (%ld), it should be equal to n2Size(%ld)", keyRopeShape[0], bLen, keyRopeShape[1], s2Len, keyRopeShape[2], n2Len), return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[3] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_rope  must be 64, but the value of query_rope[3] is (%ld).", queryRopeShape[3]), return false);
            OP_CHECK_IF(keyRopeShape[3] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_rope must be 64, but the value of key_rope[3] is (%ld).", keyRopeShape[3]), return false);           
            // 验证rope D
            OP_CHECK_IF(queryRopeShape[3] != keyRopeShape[3],
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope D is not equal to key_rope D, the value of queryRope[3] and keyRope[3] is (%ld), (%ld).", queryRopeShape[3], keyRopeShape[3]), return false);
        }
    }
    return true;
}

bool SparseLightningIndexerGradKLLossTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto &keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto &topKShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();

    auto queryRope = context_->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX); 
    bool hasQueryRope = queryRope != nullptr && queryRope->GetStorageShape().GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX);
    bool hasKeyRope = keyRope != nullptr && keyRope->GetStorageShape().GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
        return false;
    }
    // TND和BSND的dim位置不同来赋值
    hasRope = hasQueryRope && hasKeyRope;
    
    auto &queryRopeShape = queryRope->GetStorageShape();
    auto &keyRopeShape = keyRope->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
        queryIndexShape.GetDimNum() != layoutLen || keyIndexShape.GetDimNum() != layoutLen, OP_LOGE(opName, "Invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!CrossShapeVerify(queryRopeShape, keyRopeShape), OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify Failed"), return false);    
    OP_CHECK_IF(!AnalyzeDimLayout(queryShape, keyShape, queryIndexShape, topKShape, layoutLen, queryRopeShape, keyRopeShape),
               OP_LOGE(opName, "Layout: %s, Run Failed", inputLayout), return false);
    OP_CHECK_IF(gSizeQuery == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSizeQuery is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSizeQuery <= 0,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "dSizeQuery  is not support <= 0"), return false);
    return true;
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SparseLightningIndexerGradKLLossCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
               OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    // 将基本输入参数传给tilingdata
    sliGradkllossBaseParams_->set_bSize(bSize);
    sliGradkllossBaseParams_->set_n2Size(n2Size);
    sliGradkllossBaseParams_->set_gSizeQuery(gSizeQuery);
    sliGradkllossBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    sliGradkllossBaseParams_->set_s1Size(s1Size);
    sliGradkllossBaseParams_->set_s2Size(s2Size);
    sliGradkllossBaseParams_->set_dSizeQuery(dSizeQuery);
    sliGradkllossBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    sliGradkllossBaseParams_->set_kSize(kSize);
    sliGradkllossBaseParams_->set_sparseMode(sparseMode);
    sliGradkllossBaseParams_->set_scaleValue(scaleValue);
    sliGradkllossBaseParams_->set_cmpRatio(cmpRatio);
    sliGradkllossBaseParams_->set_hasSoftmaxInput(hasSoftmaxInput);

    OP_LOGW(context_, "INPUTPARAM bsize:[%d], n2Size:[%d], gSizeQuery:[%d], dSizeQueryIndex:[%d], s1Size:[%d], s2Size:[%d], dSizeQuery:[%d], dSizeQueryIndex:[%d], kSize:[%d], sparseMode:[%d], scaleValue:[%f] .", 
            bSize, n2Size, gSizeQuery, gSizeQueryIndex, s1Size, s2Size, dSizeQuery, dSizeQueryIndex, kSize, sparseMode, scaleValue);
    return ge::GRAPH_SUCCESS;
}

int64_t SparseLightningIndexerGradKLLossTilingBase::GetS2RealSize(int32_t sparseMode, int32_t s1Size, 
                                                                        int32_t s2Size, int32_t s1Idx) 
{
    int64_t s2RealSize = 0;
    // sparsemode = 3 的情形
    if (sparseMode == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = static_cast<int64_t>((s2Size - s1Size) + s1Idx + 1);
        if (s2RealSize <= 0) {
            s2RealSize = static_cast<int64_t>(s2Size);
        }
    }
    s2RealSize = AlignUp(s2RealSize, 512L);
    return std::min(s2RealSize, (int64_t)sliGradkllossBaseParams_->get_kSize());
}

bool SparseLightningIndexerGradKLLossTilingBase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."),
                return false);

    uint32_t sparseMode = sliGradkllossBaseParams_->get_sparseMode();
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t totalSize = 0;
        int64_t accumS1 = 0;
        for (int32_t i = 0; i < bSize; i++) {
            int64_t s1Size = actualSeqLenData[i];
            int64_t s2Size = actualSeqLenKData[i];
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accumS1] = s2RealSize;
                accumS1++;
            }
        }        
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t accum = 0;
        for (int32_t i = 0; i < bSize; i++) {
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accum] = s2RealSize;
                accum++;
            }
        } 
    }

    return true;
}

inline bool SparseLightningIndexerGradKLLossTilingBase::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                                                                      const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
{
    for (int64_t idx = 0; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAicNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] =
                std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        } else {
            break;
        }
    }
    return true;
}

bool SparseLightningIndexerGradKLLossTilingBase::BalanceLoad(const std::vector<int64_t> &sparseValidArray, std::vector<int64_t> &localValue,
                                                                std::vector<int64_t> &sparseStartIdx)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAicNum = std::min(static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum()),
                                    static_cast<int64_t>(aicNum));
    int64_t totalSize = sliGradkllossMultiCoreParams_->get_totalSize();
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAicNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

bool SparseLightningIndexerGradKLLossTilingBase::Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray, 
                                                                const int64_t balanceNum)
{
    int64_t tmpIndex = 0;
    tmpSparseValue[tmpIndex] = 0;
    int64_t sumTmpArray = 0;
    int64_t sumTmpArrayLast = 0; // 记录上次的总和值
    for (int64_t idx = 0; idx < sparseValidArray.size(); ++idx) {
        // 第一次分到最后一块后就没必须计算，剩余的直接分给最后一个核即可
        if (tmpIndex == static_cast<int64_t>(tmpSparseValue.size()) - 1) {
            break; 
        }

        sumTmpArrayLast = sumTmpArray;
        sumTmpArray += sparseValidArray[idx];
        if (sumTmpArray == balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            tmpSparseValue[tmpIndex] = idx + 1; //刚好等于均值时核的末尾为idx
            sumTmpArray = 0; // 重新计算总和
        } else if (sumTmpArray > balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            // 第一次总和大于均值，判断前面一次和当前哪个更接近均值
            if (balanceNum - sumTmpArrayLast >= sumTmpArray - balanceNum) {
                // 当前更接近，取当前值
                tmpSparseValue[tmpIndex] = idx + 1; //刚好等于均值时核的末尾为idx                
            }else {
                // 上一次更接近， 取上一次值
                tmpSparseValue[tmpIndex] = idx; //刚好等于均值时核的末尾为idx                
                idx--; // 记录的上一个值，重新计算时也需要从上一个值开始计算                 
            }
            sumTmpArray = 0; // 重新计算总和
        }
    }
    return true;
}

// 负载均衡
bool SparseLightningIndexerGradKLLossTilingBase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t maxCoreNum)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAicNum = static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum());
    int64_t totalSize = sliGradkllossMultiCoreParams_->get_totalSize();
    int64_t *sparseStartIdx = sliGradkllossMultiCoreParams_->get_bS1Ptr();
    int64_t splitFactorSize = sliGradkllossMultiCoreParams_->get_splitFactorSize();

    OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                   return false);
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        // initLoad: 使用均分策略, 保证后续不会比均分差
        std::vector<int64_t> localSparseStartIdx(aicNum, totalSize);
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAicNum, 0);
        //得到了每个核需要计算的S2数量，在T方向平均分配，但是有于有Sparse的存在，localValue中的数值，每个核之间应该差距很大
        InitLoadValue(sparseValidArray, validAicNum, totalSize, localSparseStartIdx, localValue); 

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAicNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(aicNum, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有T上的S2的总数量
        int64_t avgVal = CeilDivision(sparseArraySum, validAicNum); // 平均每个核需要处理的S2的总数量

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < static_cast<int64_t>(aicNum); ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                            (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                    singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAicNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAicNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有BS上的S2的总数量
        int64_t balanceNum = CeilDivision(sparseArraySum, validAicNum);
        std::vector<int64_t> tmpSparseValue(validAicNum, 0);
        Balance4DLoad(tmpSparseValue, sparseValidArray, balanceNum);
        for (int64_t idx = 0; idx < static_cast<int64_t>(validAicNum); ++idx) {
            sparseStartIdx[idx] = tmpSparseValue[idx];
        }
    }

    for (int64_t idx = 1; idx < static_cast<int64_t>(MAX_CORE_NUM); ++idx) {
        if (sparseStartIdx[idx] == 0) { 
            sparseStartIdx[idx] = static_cast<int64_t>(sparseValidArray.size()); // 赋值为s1最大值
        }
    }    
    return true;
}

void SparseLightningIndexerGradKLLossTilingBase::SetSparseParamsRegbase(int64_t maxCoreNum)
{
    std::vector<int64_t> sparseValidArray(sliGradkllossMultiCoreParams_->get_totalSize(), 0);
    InitSparseValidArray(sparseValidArray);
    SetSparseStartIdx(sparseValidArray, maxCoreNum);
}

// 计算S1的总个数
int64_t SparseLightningIndexerGradKLLossTilingBase::CalcTotalSize() {
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return realT1Size;
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) { 
        return bSize * s1Size;
    }
    return 0; //什么也不走就返回0
}

void SparseLightningIndexerGradKLLossTilingBase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    sliGradkllossMultiCoreParams_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    sliGradkllossMultiCoreParams_->set_totalSize(totalSize);
    sliGradkllossMultiCoreParams_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
}

void SparseLightningIndexerGradKLLossTilingBase::InitOutputSplit()
{
    SLIGradKLLossInitOutputParams *initoutput =  &tilingData->initOutputParams;
    auto &dKeyIndexShape = context_->GetOutputShape(1)->GetStorageShape();
    int64_t totalsize = 0;
    uint32_t singlecoresize = 0;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(2); // TND场景下D为第二个维度
        int64_t totalT2Size = dKeyIndexShape.GetDim(0); // T为第一个维度
        totalsize = totalT2Size * static_cast<int64_t>(dsizeDkeyindex);
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(3); // BSND场景下D为第三个维度
        int64_t totals2Size = dKeyIndexShape.GetDim(1); // s为第二个维度
        totalsize = static_cast<int64_t>(bSize) * totals2Size * dsizeDkeyindex;
    }
    // 单个核均分元素数量
    singlecoresize = static_cast<uint32_t>(CeilDivision(totalsize, static_cast<int64_t>(aivNum))); // 输出k-index总大小TD或者BSD 除以 总的aiv核数 向上取整
    initoutput->set_singleCoreSize(singlecoresize);
    initoutput->set_totalOutputSize(totalsize);
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBase::DoOpTiling()
{
    OP_LOGD(context_, "try template[%s]", templateName);
    // 无多余操作，分核，目前只实现TND场景分核
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    context_->SetBlockDim(sliGradkllossMultiCoreParams_->get_coreNum()); // 使用的核数确定

    std::vector<int64_t> shapeVec = {1, kSize};
    ge::Shape srcShape(shapeVec);
    int64_t softmaxTmpBufferSize = BUFFER_SIZE_BYTE_32K; // 需要32KB
    if (kSize > BUFFER_SIZE_BYTE_2K) {
        softmaxTmpBufferSize = BUFFER_SIZE_BYTE_33K; //kSize >2048 需要33kB
    }
    SoftMaxTilingFunc(srcShape, 4, softmaxTmpBufferSize, tilingData->vectorParams.softmaxYTilingData);

    SetSparseParamsRegbase(static_cast<int64_t>(aicNum));
    InitOutputSplit(); // output分核
    OP_LOGD(context_, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseLightningIndexerGradKLLossTilingBase::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(hasRope), static_cast<uint32_t>(topkSize), static_cast<uint8_t>(tilingKeyLayout), 
        static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(sparseMode), static_cast<uint8_t>(deterministic));
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t pSize = kSize * (dSizeQuery + dQueryRopeSize) * 2; // 2代表half大小
    int64_t sySize = kSize * dSizeQueryIndex * 2; // 2使用DB
    int64_t bmm1Size = gSizeQuery * kSize * sizeof(float);
    int64_t bmm2Size = gSizeQueryIndex * kSize * sizeof(float);
    int64_t reluGradSize = gSizeQueryIndex * kSize * sizeof(float);
    int64_t psySyncSize = kSize * 2 * sizeof(float); // 2使用DB
    int64_t bmm3Size = kSize * dSizeQueryIndex * sizeof(float);
    int64_t scatterAddOutSize;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        scatterAddOutSize = accumS2 * dSizeQueryIndex * sizeof(float); //batch
    } else {
        scatterAddOutSize = bSize * s2Size * dSizeQueryIndex * sizeof(float); //batch
    }

    int64_t singlecoreTotalSize = PING_PONG_VALUE * (pSize + bmm1Size + bmm2Size + reluGradSize + sySize + psySyncSize + bmm3Size);
    int64_t multicoreTotalsize = singlecoreTotalSize * static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum()) + scatterAddOutSize;
    workspaces[0] = static_cast<size_t>(multicoreTotalsize) + WORK_SPACE_RESERVE_SIZE; // 预留16M空间必须加;
    OP_LOGW(context_, "workspace size:[%ld], multicoreTotalsize:[%ld]", workspaces[0], multicoreTotalsize);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
