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

at::Tensor npu_wind_mhc_post_part_meta(const at::Tensor &x, const at::Tensor &h_post, const at::Tensor &bmm2)
{
    // 从 h_post 的最后一维获取 N
    if (h_post.dim() < 1) {
        throw std::runtime_error("wind_mhc_post_part: h_post must have at least 1 dimension");
    }
    int64_t N = h_post.size(-1);

    auto x_sizes = x.sizes().vec();
    // 在倒数第二个位置（最后一维之前）插入 N
    // 例如 [B, S, D] -> [B, S, N, D]； [T, D] -> [T, N, D]
    int64_t insert_pos = x_sizes.size() - 1;  // 最后一维之前的位置
    x_sizes.insert(x_sizes.begin() + insert_pos, N);

    // 创建空 tensor，其他属性与 x 一致
    return at::empty(x_sizes, x.options());
}

at::Tensor npu_wind_mhc_post_part(const at::Tensor &x, const at::Tensor &h_post, const at::Tensor &bmm2)
{
    at::Tensor out = npu_wind_mhc_post_part_meta(x, h_post, bmm2);
    ACLNN_CMD(aclnnWindMhcPostPart, x, h_post, bmm2, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_wind_mhc_post_part_backward_meta(const at::Tensor &grad_output,
    const at::Tensor &x, const at::Tensor &residual, const at::Tensor &h_post, const at::Tensor &h_res)
{
    auto x_sizes = x.sizes().vec();
    auto h_post_sizes = h_post.sizes().vec();
    auto h_res_sizes = h_res.sizes().vec();
    return {
            at::empty(x_sizes, x.options()),
            at::empty(h_post_sizes, h_post.options()),
            at::empty(h_res_sizes, h_res.options())
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_wind_mhc_post_part_backward(const at::Tensor &grad_output,
    const at::Tensor &x, const at::Tensor &residual, const at::Tensor &h_post, const at::Tensor &h_res)
{
    auto [grad_x, grad_h_post, grad_h_res] = npu_wind_mhc_post_part_backward_meta(grad_output, x, residual, h_post, h_res);
    ACLNN_CMD(aclnnWindMhcPostPartBackward, grad_output, x, residual, h_post, h_res, grad_x, grad_h_post, grad_h_res);
    return {grad_x, grad_h_post, grad_h_res};
}