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

std::tuple<at::Tensor, at::Tensor> graph_wind_hc_pre_bmm_backward_meta(const at::Tensor& h_pre, const at::Tensor& x, const at::Tensor& grad_y)
{
    return {
        at::empty(h_pre.sizes(), h_pre.options()),
        at::empty(x.sizes(), x.options()),
    };
}

std::tuple<at::Tensor, at::Tensor> npu_wind_hc_pre_bmm_backward(const at::Tensor& h_pre, const at::Tensor& x, const at::Tensor& grad_y)
{
    auto [grad_h_pre, grad_x] = graph_wind_hc_pre_bmm_backward_meta(h_pre, x, grad_y);
    if (IS_ACLNN_MODE()) {
        // aclnn只调

        ACLNN_CMD(aclnnWindHcPreBmmBackward, h_pre, x, grad_y, grad_h_pre, grad_x);
    } else {
        // 图模式
        at_npu::native::OpCommand cmd;
        cmd.Name("WindHcPreBmmBackward")
            .Input(h_pre)
            .Input(x)
            .Input(grad_y)
            .Output(grad_h_pre)
            .Output(grad_x)
            .Run();
    }
    return {grad_h_pre, grad_x};
}
