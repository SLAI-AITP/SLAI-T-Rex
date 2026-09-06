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

at::Tensor graph_wind_hc_pre_bmm_forward_meta(const at::Tensor& h_pre, const at::Tensor& x, int64_t out_dtype)
{
    // h_pre [T,N], x [T,N,D] -> y [T,D]
    auto xSizes = x.sizes();
    std::vector<int64_t> ySizes;
    if (x.dim() == 4) {
        ySizes = {xSizes[0], xSizes[1], xSizes[3]};
    } else {
        ySizes = {xSizes[0], xSizes[2]};
    }
    at::ScalarType outDtype = static_cast<at::ScalarType>(out_dtype);
    return at::empty(ySizes, x.options().dtype(outDtype));
}

at::Tensor npu_wind_hc_pre_bmm_forward(const at::Tensor& h_pre, const at::Tensor& x, int64_t out_dtype)
{
    auto y = graph_wind_hc_pre_bmm_forward_meta(h_pre, x, out_dtype);
    if (IS_ACLNN_MODE()) {
        ACLNN_CMD(aclnnWindHcPreBmmForward, h_pre, x, out_dtype, y);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("WindHcPreBmmForward")
            .Input(h_pre)
            .Input(x)
            .Output(y)
            .Run();
    }
    return y;
}
