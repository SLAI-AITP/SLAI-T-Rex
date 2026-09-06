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

std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform_meta(const at::Tensor& kv, const at::Tensor& score, int64_t compress_ratio, int64_t head_dim,
                                                                         double kv_value, double score_value)
{
    bool isKvScoreComb = (!score.defined()) || (score.dim() == 0);
    auto kv_sizes = kv.sizes().vec();

    uint32_t B;
    uint32_t R;
    uint32_t two_D;
    // [B,S,R,2D]
    if (kv.dim() == 4U) {
        if ((isKvScoreComb && kv_sizes[0] != 2U) || (!isKvScoreComb && kv_sizes[0] != 1)) {
            throw std::runtime_error("wind_fused_overlap_transform: [B,S,R,2D], B must be 2 when kv_score_comb is true, or B must be 1 when kv_score_comb is false.");
        }
        R = kv_sizes[2U];
        two_D = kv_sizes[3U];
    } else if (kv.dim() == 3U) { // [S,R,2D]
        R = kv_sizes[1];
        two_D = kv_sizes[2U];
    } else {
        throw std::runtime_error("wind_fused_overlap_transform_backward: kv only support [B,S,R,2D] or [S,R,2D].");
    }

    if (R != compress_ratio) {
        throw std::runtime_error("wind_fused_overlap_transform: R must be same with compress_ratio.");
    }
    if (two_D != 2U * head_dim) {
        throw std::runtime_error("wind_fused_overlap_transform: D must be same with head_dim.");
    }

    if (!isKvScoreComb) {
        // score和kv的shape必须一样
        if (score.dim() != kv.dim()) {
            throw std::runtime_error("wind_fused_overlap_transform: score shape must be same with kv.");
        }
        auto score_sizes = score.sizes().vec();
        for (uint32_t i = 0; i < score.dim(); i++) {
            if (score_sizes[i] != kv_sizes[i]) {
                throw std::runtime_error("wind_fused_overlap_transform: score shape must be same with kv.");
            }
        }
    }

    // [B, S, R, 2D] -> [B, S, 2R, D]
    int lastDimIdx = kv_sizes.size() - 1;
    kv_sizes[0] = 1; // B只支持1，前面有校验
    kv_sizes[lastDimIdx - 1] = compress_ratio * 2U;
    kv_sizes[lastDimIdx] = head_dim;

    // kv_out和score_out的shape一样
    return {
        at::empty(kv_sizes, kv.options()),
        at::empty(kv_sizes, score.options())
    };
}

std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform(const at::Tensor& kv, const at::Tensor& score, int64_t compress_ratio, int64_t head_dim,
                                                                    double kv_value, double score_value)
{
    auto [kv_out, score_out] = npu_wind_fused_overlap_transform_meta(kv, score, compress_ratio, head_dim, kv_value, score_value);
    ACLNN_CMD(aclnnWindFusedOverlapTransform, kv, score, compress_ratio, head_dim, kv_value, score_value, kv_out, score_out);
    return {kv_out, score_out};
}

std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform_backward_meta(const at::Tensor& grad_kv_out, const at::Tensor& grad_score_out,
                                                                                  int64_t compress_ratio, int64_t head_dim, bool kv_score_comb)
{
    auto grad_kv_out_sizes = grad_kv_out.sizes().vec();
    uint32_t B;
    uint32_t two_R;
    uint32_t D;
    // [B,S,2R,D]
    if (grad_kv_out.dim() == 4U) {
        if (grad_kv_out_sizes[0] != 1U) {
            throw std::runtime_error("wind_fused_overlap_transform_backward: [B,S,2R,D], B must be 1.");
        }
        two_R = grad_kv_out_sizes[2U];
        D = grad_kv_out_sizes[3U];
    } else if (grad_kv_out.dim() == 3U) { // [S,2R,D]
        two_R = grad_kv_out_sizes[1];
        D = grad_kv_out_sizes[2U];
    } else {
        throw std::runtime_error("wind_fused_overlap_transform_backward: grad_kv_out only support [B,S,2R,D] or [S,2R,D].");
    }

    if (two_R != 2U * compress_ratio) {
        throw std::runtime_error("wind_fused_overlap_transform_backward: R must be same with compress_ratio.");
    }
    if (D != head_dim) {
        throw std::runtime_error("wind_fused_overlap_transform_backward: D must be same with head_dim.");
    }

    // score和kv的shape必须一样
    if (grad_score_out.dim() != grad_kv_out.dim()) {
        throw std::runtime_error("wind_fused_overlap_transform_backward: grad_score_out shape must be same with grad_kv_out.");
    }
    auto grad_score_out_sizes = grad_score_out.sizes().vec();
    for (uint32_t i = 0; i < grad_score_out.dim(); i++) {
        if (grad_score_out_sizes[i] != grad_kv_out_sizes[i]) {
            throw std::runtime_error("wind_fused_overlap_transform_backward: grad_score_out shape must be same with grad_kv_out.");
        }
    }

    // [B, S, 2R, D] -> [B, S, R, 2D]
    grad_kv_out_sizes[0] = kv_score_comb ? 2U : 1U;
    int lastDimIdx = grad_kv_out_sizes.size() - 1;
    grad_kv_out_sizes[lastDimIdx - 1] = compress_ratio;
    grad_kv_out_sizes[lastDimIdx] = head_dim * 2U;

    auto grad_kv = at::empty(grad_kv_out_sizes, grad_kv_out.options());
    at::Tensor grad_score;
    if (kv_score_comb) {
        grad_score = at::empty({0}, grad_kv_out.options());
    } else {
        grad_score = at::empty(grad_kv_out_sizes, grad_kv_out.options());
    }
    return {grad_kv, grad_score};
}

std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform_backward(const at::Tensor& grad_kv_out, const at::Tensor& grad_score_out,
                                                                             int64_t compress_ratio, int64_t head_dim, bool kv_score_comb)
{
    auto [grad_kv, grad_score] = npu_wind_fused_overlap_transform_backward_meta(grad_kv_out, grad_score_out, compress_ratio, head_dim, kv_score_comb);
    ACLNN_CMD(aclnnWindFusedOverlapTransformBackward, grad_kv_out, grad_score_out, compress_ratio, head_dim, kv_score_comb, grad_kv, grad_score);
    return {grad_kv, grad_score};
}
