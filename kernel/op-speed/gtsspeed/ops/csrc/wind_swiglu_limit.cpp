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

inline std::vector<int64_t> swiglu_grad_infershape(const at::Tensor &x, const c10::optional<at::Tensor> &y, int64_t dim)
{
    if (dim < 0) {
        dim += x.sizes().size();
    }
    TORCH_CHECK(dim < x.sizes().size(), "dim out of range", dim);

    auto input_sizes = x.sizes();
    std::vector<int64_t> output_sizes;
    output_sizes.reserve(input_sizes.size());
    for (size_t i = 0; i < input_sizes.size(); ++i) {
        output_sizes.push_back(input_sizes[i]);
    }
    if (!y.has_value()) {
        output_sizes[dim] /= 2U;
    }
    return output_sizes;
}

// [Add by GTS-AILab] [feature GTSOps] swiglu_limit 前向：x[..2H] -> y[..H]，gate(post-silu)clamp max=limit，up clamp[-limit,limit]
// 调 aclnnSwiGluLimitV2(self, other, dim, limit, highPrecision, out)（v2 op def attr 序：dim,limit,highPrecision）。
// limit<=0 等价普通 swiglu。仅支持合并输入（y 为空）。backend 参数保留以兼容签名，当前仅 v2。
at::Tensor npu_swiglu_limit(const at::Tensor &x, const c10::optional<at::Tensor> &y, int64_t dim, double limit,
                            std::string backend)
{
    TORCH_CHECK(!y.has_value(), "npu_swiglu_limit only supports merged input (y must be None)");
    auto output_sizes = swiglu_grad_infershape(x, y, dim);
    at::Tensor result = at::empty(output_sizes, x.options());
    double limitVal = limit;   // aclnn Float attr ABI = double，必须传具名 double 左值
    bool highPrecision = true;  // 与 v2 op def 默认一致
    bool hasProbs = false;  // 与 v2 op def 默认一致
    at::Tensor out_probs;
    at::Tensor probs;
    ACLNN_CMD(aclnnWindSwiGluLimit, x, y, probs, dim, limitVal, highPrecision, hasProbs, result, out_probs);
    return result;
}

std::tuple<at::Tensor, at::Tensor> npu_swiglu_limit_with_probs(const at::Tensor &x, const c10::optional<at::Tensor> &y, const c10::optional<at::Tensor> &probs, int64_t dim,
                                                               double limit, bool highPrecision, bool hasProbs, std::string backend)
{
    TORCH_CHECK(!y.has_value(), "npu_swiglu_limit_v3 only supports merged input (y must be None)");
    auto output_sizes = swiglu_grad_infershape(x, y, dim);
    at::Tensor result1 = at::empty(output_sizes, x.options());
    at::Tensor result2 = at::empty(output_sizes, x.options());
    double limitVal = limit;   // aclnn Float attr ABI = double，必须传具名 double 左值
    ACLNN_CMD(aclnnWindSwiGluLimit, x, y, probs, dim, limitVal, highPrecision, hasProbs, result1, result2);
    return std::make_tuple(result1, result2);
}

// [Add by GTS-AILab] [feature GTSOps] swiglu_limit 反向：grad_out[..H], x[..2H] -> x_grad[..2H]（与 x 同形）。limit<=0 等价普通 swiglu 反向。
// SwiGluLimitBackwardV2 op def=输入(gradout,self,other OPTIONAL,probs OPTIONAL)+attr(dim,limit)+输出(out1,out2 OPTIONAL)，
//   合并输入下 other 空、out2 空：aclnnSwiGluLimitBackwardV2(gradout, self, other, probs, dim, limit, out1, out2)。
// probs(MoE per-token 权重 [T,1] fp32) 可选：传入则算子内做 grad_h=gradout*probs，
//   省掉外部独立 Mul([*,1])+Cast；probs=None 时与原行为逐字节等价。grad_probs(=sum(grad_u*x)) 仍由调用方外部 reduce。
// backend 参数保留以兼容签名，当前仅 v2。
at::Tensor npu_swiglu_limit_backward(const at::Tensor &grad_output, const at::Tensor &x, int64_t dim, double limit,
                                     const c10::optional<at::Tensor> &probs, std::string backend)
{
    at::Tensor result = at::empty(x.sizes(), x.options());   // x_grad / out1 与 x 同形（全 2H）
    double limitVal = limit;
    c10::optional<at::Tensor> other = c10::nullopt;      // 合并输入：other 为空
    at::Tensor result2 = at::empty({0}, x.options());    // out2 空（无 grad_probs 时）
    ACLNN_CMD(aclnnWindSwiGluLimitBackward, grad_output, x, other, probs, dim, limitVal, result, result2);
    return result;
}

// 重型 probs 版：算子一次出 (grad_x, grad_probs)。grad_probs fp32 = sum_H(grad_out*h)，
// 由算子内逐行 reduce 算（消除外部重算 h + Mul + ReduceSum）。probs 须非空。
std::tuple<at::Tensor, at::Tensor> npu_swiglu_limit_backward_probs(
    const at::Tensor &grad_output, const at::Tensor &x, int64_t dim, double limit,
    const at::Tensor &probs, std::string backend)
{
    at::Tensor result = at::empty(x.sizes(), x.options());   // grad_x 与 x 同形 [T,2H]
    // grad_probs 形状跟随 probs 输入（[T] 或 [T,1] 皆可），autograd 才不会报 grad 与 probs 形状不一致；
    //   dtype 强制 fp32（算子内 fp32 reduce）。必须 at::zeros：切列时原子累加(SetAtomicAdd)要求初值 0。
    //   kernel 按连续 numel 个 float 写 grad_probs，[T]/[T,1] 内存布局相同，故按 probs.sizes() 分配 kernel 照写不误。
    at::Tensor result2 = at::zeros(probs.sizes(), x.options().dtype(at::kFloat));
    double limitVal = limit;
    c10::optional<at::Tensor> other = c10::nullopt;
    c10::optional<at::Tensor> probsOpt = probs;
    ACLNN_CMD(aclnnWindSwiGluLimitBackward, grad_output, x, other, probsOpt, dim, limitVal, result, result2);
    return std::make_tuple(result, result2);
}
