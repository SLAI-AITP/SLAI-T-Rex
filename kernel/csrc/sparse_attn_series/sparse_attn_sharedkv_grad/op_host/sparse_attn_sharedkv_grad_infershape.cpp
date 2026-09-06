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
 * \file sparse_attn_sharedkv_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace sasg {

enum class InputIndex : uint32_t {
    QUERY = 0,
    ORI_KV,
    CMP_KV,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    LSE,
    ORI_SPARSE_INDICES,
    CMP_SPARSE_INDICES,
    CU_SEQLENS_Q,
    CU_SEQLENS_OPI_KV,
    CU_SEQLENS_CMP_KV,
    SINKS
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DORI_KV,
    DCMP_KV,
    DSINKS
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    CMP_RATIO,
    INPUT_LAYOUT,
    ORI_MASK_MODE,
    CMP_MASK_MODE,
    ORI_WIN_LEFT,
    ORI_WIN_RIGHT,
    LAYOUT
};

ge::graphStatus InferShape4SparseAttnSharedkvGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseAttnSharedkvGrad", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4SparseAttnSharedkvGrad.");

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    const gert::Shape *oriKvShape = context->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    const gert::Shape *cmpKvShape = context->GetInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    const gert::Shape *sinksShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINKS));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, oriKvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKvShape);

    auto attrs = context->GetAttrs();
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    auto cmpRatio = attrs->GetInt(static_cast<size_t>(AttrIndex::CMP_RATIO));
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpRatio);

    gert::Shape *dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    gert::Shape *dOriKvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DORI_KV));
    gert::Shape *dCmpKvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DCMP_KV));
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dOriKvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dCmpKvShape);
    *dqShape = *queryShape;
    *dOriKvShape = *oriKvShape;
    *dCmpKvShape = *cmpKvShape;

    if (sinksShape != nullptr) {
        gert::Shape *dSinksShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DSINKS));
        OP_CHECK_NULL_WITH_CONTEXT(context, dSinksShape);
        *dSinksShape = *sinksShape;
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseAttnSharedkvGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseAttnSharedkvGrad", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4SparseAttnSharedkvGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    auto sinksDtype = context->GetInputDataType(static_cast<size_t>(InputIndex::SINKS));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DORI_KV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DCMP_KV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DSINKS), sinksDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseAttnSharedkvGrad)
    .InferShape(InferShape4SparseAttnSharedkvGrad)
    .InferDataType(InferDataType4SparseAttnSharedkvGrad);
} // namespace sasg
} // namespace ops
