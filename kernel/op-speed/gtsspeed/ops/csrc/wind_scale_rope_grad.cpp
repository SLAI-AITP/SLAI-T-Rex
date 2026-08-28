/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inc/aclnn_common.h"
#include "pybind.h"
#include <iostream>

std::tuple<at::Tensor, at::Tensor> npu_wind_scale_rope_grad_meta(
    const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &scale,
    const at::Tensor &cos, const at::Tensor &sin)
{
    at::Tensor grad_x = at::empty(grad_out.sizes().vec(), grad_out.options());      // [T,D] bf16
    at::Tensor grad_scale = at::empty(scale.sizes().vec(), scale.options().dtype(at::kFloat));  // [T,1] fp32
    return std::make_tuple(grad_x, grad_scale);
}

std::tuple<at::Tensor, at::Tensor> npu_wind_scale_rope_grad(
    const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &scale,
    const at::Tensor &cos, const at::Tensor &sin)
{
    auto outs = npu_wind_scale_rope_grad_meta(grad_out, x, scale, cos, sin);
    at::Tensor grad_x = std::get<0>(outs);
    at::Tensor grad_scale = std::get<1>(outs);
    ACLNN_CMD(aclnnWindScaleRopeGrad, grad_out, x, scale, cos, sin, grad_x, grad_scale);
    return std::make_tuple(grad_x, grad_scale);
}
