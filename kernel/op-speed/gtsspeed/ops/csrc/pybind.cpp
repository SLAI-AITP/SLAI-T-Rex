/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pybind.h"

// ------------------------------------begin release declare------------------------------------
std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform(const at::Tensor& kv, const at::Tensor& score, int64_t compress_ratio, int64_t head_dim,
                                                                    double kv_value, double score_value);
std::tuple<at::Tensor, at::Tensor> npu_wind_fused_overlap_transform_backward(const at::Tensor& grad_kv_out, const at::Tensor& grad_score_out,
                                                                             int64_t compress_ratio, int64_t head_dim, bool kv_score_comb);

at::Tensor npu_wind_make_chunk_sort_map(const at::Tensor &splitSizes, const at::Tensor &sortedIndices, int64_t numTokens);


at::Tensor npu_rms_norm_without_weight(const at::Tensor &x, double epsilon, int64_t out_dtype);
at::Tensor npu_rms_norm_without_weight_backward(const at::Tensor &grad_res, const at::Tensor &x, const at::Tensor &res);

at::Tensor npu_wind_scale_rope(const at::Tensor &x, const at::Tensor &scale, const at::Tensor &cos, const at::Tensor &sin);
std::tuple<at::Tensor, at::Tensor> npu_wind_scale_rope_grad(const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &scale, const at::Tensor &cos, const at::Tensor &sin);

at::Tensor npu_swiglu_limit(const at::Tensor &x, const c10::optional<at::Tensor> &y, int64_t dim, double limit, std::string backend = "v2");
at::Tensor npu_swiglu_limit_backward(const at::Tensor &grad_output, const at::Tensor &x, int64_t dim, double limit, const c10::optional<at::Tensor> &probs = c10::nullopt, std::string backend = "v2");
std::tuple<at::Tensor, at::Tensor> npu_swiglu_limit_backward_probs(const at::Tensor &grad_output, const at::Tensor &x, int64_t dim, double limit, const at::Tensor &probs, std::string backend = "v2");
// swi_glu_limit_v3：v2 + probs向量乘法融合（probs[float, Mx1] 广播乘 output[bf16, MxH]）
std::tuple<at::Tensor, at::Tensor> npu_swiglu_limit_with_probs(const at::Tensor &x, const c10::optional<at::Tensor> &y, const c10::optional<at::Tensor> &probs, int64_t dim, double limit, bool highPrecision, bool hasProbs,
                                                               std::string backend);
// ------------------------------------end release declare------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    Pybind::GetInstance().Init(m);

    // ------------------------------------begin release pybind------------------------------------
    m.def("npu_wind_fused_overlap_transform", &npu_wind_fused_overlap_transform, "npu_wind_fused_overlap_transform");
    m.def("npu_wind_fused_overlap_transform_backward", &npu_wind_fused_overlap_transform_backward, "npu_wind_fused_overlap_transform_backward");

    m.def("npu_wind_make_chunk_sort_map", &npu_wind_make_chunk_sort_map, "make chunk sort row id map");

    m.def("npu_rms_norm_without_weight", &npu_rms_norm_without_weight,  "rmsnorm without weight forward");
    m.def("npu_rms_norm_without_weight_backward", &npu_rms_norm_without_weight_backward,  "rmsnorm without weight backward");

    m.def("npu_wind_scale_rope", &npu_wind_scale_rope,  "wind scale rope forward");
    m.def("npu_wind_scale_rope_grad", &npu_wind_scale_rope_grad,  "wind scale rope backward (grad_x, grad_scale)");

    m.def("npu_swiglu_limit", &npu_swiglu_limit, "swiglu limit forward (gate clamp max=limit, up clamp[-limit,limit])");
    m.def("npu_swiglu_limit_backward", &npu_swiglu_limit_backward, "swiglu limit backward (with clamp passthrough masks; optional probs=MoE per-token weight)",
          pybind11::arg("grad_output"), pybind11::arg("x"), pybind11::arg("dim"), pybind11::arg("limit"),
          pybind11::arg("probs") = c10::nullopt, pybind11::arg("backend") = "v2");  // probs 可选
    m.def("npu_swiglu_limit_backward_probs", &npu_swiglu_limit_backward_probs, "swiglu limit backward, return (grad_x, grad_probs); grad_probs 由算子内逐行 reduce 算",
          pybind11::arg("grad_output"), pybind11::arg("x"), pybind11::arg("dim"), pybind11::arg("limit"),
          pybind11::arg("probs"), pybind11::arg("backend") = "v2");
    m.def("npu_swiglu_limit_with_probs", &npu_swiglu_limit_with_probs, "swiglu limit v3 (v2 + probs vector mul fusion)");
    // ------------------------------------end release pybind------------------------------------
}
