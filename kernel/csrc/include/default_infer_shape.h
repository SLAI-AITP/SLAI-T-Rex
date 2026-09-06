/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_DEFAULT_INFER_SHAPE_HPP
#define OP_DEFAULT_INFER_SHAPE_HPP
namespace ge {
template <bool IsTwoOutput = false>
static ge::graphStatus DefaultInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    if constexpr (IsTwoOutput) {
        gert::Shape* y2_shape = context->GetOutputShape(1);
        *y2_shape = *x1_shape;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    y_shape->operator [](y_shape->GetDimNum() - 1) /= 2;    // 2 最后一维数据减半
    return GRAPH_SUCCESS;
}

static ge::graphStatus Split3InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    y_shape->operator [](y_shape->GetDimNum() - 1) /= 3; // 3表示交织参数为3
    return GRAPH_SUCCESS;
}

template <bool IsGrad = false>
static ge::graphStatus XxGluInferShape(gert::InferShapeContext* context)
{
    auto x2_shape = context->GetOptionalInputShape(IsGrad ? 2 : 1); // 反向第2个参数，正向第1个参数
    if (x2_shape == nullptr) {
        return SplitInferShape(context);
    } else {
        return DefaultInferShape<IsGrad>(context);
    }
}

static ge::graphStatus XxGluInferDataType(gert::InferDataTypeContext* context)
{
    auto x_dtype = context->GetInputDataType(0);
    if (x_dtype == ge::DT_UNDEFINED) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, x_dtype);
}
}
#endif  // OP_DEFAULT_INFER_SHAPE_HPP
