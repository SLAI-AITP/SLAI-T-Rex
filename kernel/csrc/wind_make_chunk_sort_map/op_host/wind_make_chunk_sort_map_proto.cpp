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
#include "log/ops_log.h"

namespace ge {
namespace {
constexpr size_t ATTR_NUM_TOKENS = 0U;
constexpr size_t OUTPUT_ROW_ID_MAP = 0U;
} // namespace

inline ge::graphStatus InferShape4WindMakeChunkSortMap(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numTokensPtr = attrs->GetInt(ATTR_NUM_TOKENS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTokensPtr);

    gert::Shape* outShape = context->GetOutputShape(OUTPUT_ROW_ID_MAP);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(1);
    outShape->SetDim(0, *numTokensPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4WindMakeChunkSortMap(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_ROW_ID_MAP, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(WindMakeChunkSortMap)
    .InferShape(InferShape4WindMakeChunkSortMap)
    .InferDataType(InferDataType4WindMakeChunkSortMap);
} // namespace ge
