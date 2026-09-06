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

at::Tensor npu_wind_scale_rope_meta(const at::Tensor &x, const at::Tensor &scale,
                                    const at::Tensor &cos, const at::Tensor &sin)
{
    // out 形状与 x 完全一致
    return at::empty(x.sizes().vec(), x.options());
}

at::Tensor npu_wind_scale_rope(const at::Tensor &x, const at::Tensor &scale,
                               const at::Tensor &cos, const at::Tensor &sin)
{
    at::Tensor out = npu_wind_scale_rope_meta(x, scale, cos, sin);
    ACLNN_CMD(aclnnWindScaleRope, x, scale, cos, sin, out);
    return out;
}
