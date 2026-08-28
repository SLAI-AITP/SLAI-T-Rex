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

at::Tensor npu_rms_norm_without_weight(const at::Tensor &x, double epsilon, int64_t out_dtype)
{
    // output shape: same as x but last dim = 1
    std::vector<int64_t> resSizes(x.sizes().begin(), x.sizes().end());
    resSizes.back() = 1;
    at::ScalarType outDtype = static_cast<at::ScalarType>(out_dtype);
    at::Tensor res = at::empty(resSizes, x.options().dtype(outDtype));
    ACLNN_CMD(aclnnWindRmsNormWithoutWeight, x, epsilon, outDtype, res);
    return res;
}

at::Tensor npu_rms_norm_without_weight_backward(const at::Tensor &grad_res, const at::Tensor &x, const at::Tensor &res)
{
    at::Tensor grad_x = at::empty(x.sizes(), x.options());
    ACLNN_CMD(aclnnWindRmsNormWithoutWeightBackward, grad_res, x, res, grad_x);
    return grad_x;
}