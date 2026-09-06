 # ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from gts_custom_op import *
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F
from utils.compare import compare

class TestGts(TestCase):
    # [Add by GTS-AILab] [feature GTSOps] swiglu_limit 接入测试
    def _golden_swiglu_limit(self, x, limit, dtype):
        # last-dim, pre-silu（对齐线上）：silu(clamp(gate,max=limit)) * clamp(up,-limit,limit)
        gate, up = x.float().chunk(2, dim=-1)
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        gate = F.silu(gate)
        return (gate * up).to(dtype)

    def test_gts_swiglu_limit_fwd(self):
        # 前向数值对拍（limit>0 触发 clamp）
        x = (torch.rand((1024, 6144)) * 8 - 4).to(torch.bfloat16).npu()
        limit = 10.0
        golden = self._golden_swiglu_limit(x.cpu(), limit, torch.bfloat16)
        is_enable_op("swiglu1", True)
        gts = gts_swiglu_limit(x, limit)
        is_enable_op("swiglu1", False)
        self.assertTrue(compare(golden, gts.cpu(), torch.float16, "swiglu_limit_fwd:",
                                show_detail_count=30, sort_mask=0, show_all=False, new_standard=False))

    def test_gts_swiglu_limit_fallback(self):
        # 回退：开关关 / limit<=0 应返回 None（由 fused_swiglu_with_limit 走老路）
        x = torch.rand((16, 32)).to(torch.bfloat16).npu()
        is_enable_op("swiglu1", False)
        self.assertIsNone(gts_swiglu_limit(x, 10.0))
        is_enable_op("swiglu1", True)
        self.assertIsNone(gts_swiglu_limit(x, 0.0))
        is_enable_op("swiglu1", False)

    def _golden_swiglu_limit_grad(self, input, gradout, limit, dim=-1):
        # 反向 golden：纯 fp32 autograd（v2 反向已删 round-trip = fp32 末尾截断 = 线上）。
        # pre-silu（对齐线上）：gate 先 clamp(max=limit) 再 silu。
        x = input.type(torch.float32).npu()
        x.requires_grad_(True); x.retain_grad()
        a, b = torch.chunk(x, 2, dim=dim)
        gate = a; up = b
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        gate = F.silu(gate)
        (gate * up).backward(gradout.type(torch.float32).npu())
        return x.grad.cpu().type(input.dtype)

    def test_gts_swiglu_limit_bwd(self):
        # 反向走 autograd：gts_swiglu_limit -> NpuSwiGluLimit.apply，.backward() 自动调 v2 反向算子。
        dim = -1
        x = (torch.rand((1024, 6144)) * 8 - 4).to(torch.bfloat16)
        gradout = (torch.rand((1024, 3072)) * 2 - 1).to(torch.bfloat16)
        limit = 10.0
        golden = self._golden_swiglu_limit_grad(x, gradout, limit, dim)

        is_enable_op("swiglu1", True)
        inp = x.npu(); inp.requires_grad_(True); inp.retain_grad()
        out = gts_swiglu_limit(inp, limit)           # = NpuSwiGluLimit.apply(...,"v2")
        out.backward(gradout.npu())
        grad = inp.grad.cpu()
        is_enable_op("swiglu1", False)
        self.assertTrue(compare(golden, grad, torch.float16, "swiglu_limit_bwd:",
                                show_detail_count=30, sort_mask=0, show_all=False, new_standard=False))

    def test_rms_norm_without_weight_forward_backward(self):
        """完整测试 RMSNormWithoutWeight 的前向和反向"""
        def run_forward_backward(x_shape, x_dtype, epsilon = 1e-6, out_dtype = None):
            torch.manual_seed(42)
            if out_dtype is None:
                out_dtype = x_dtype

            total_elements = 1
            for dim in x_shape:
                total_elements *= dim
            D = x_shape[-1]                 # 最后一维大小
            T = total_elements // D         # 批量大小

            # ---------- 1. 前向对比 ----------
            x = torch.randn(T, D, dtype=x_dtype, requires_grad=True, device="npu")
            # 算子计算
            r = gts_rmsnorm_without_weight(x, epsilon, out_dtype)

            # =============== golden ===============
            x_calc = x.type(torch.float32)
            x_square_mean = (x_calc ** 2).mean(dim=-1, keepdim=True)
            r_manual = torch.rsqrt(x_square_mean + epsilon).type(out_dtype)
            self.assertTrue(compare(r_manual.cpu().type(out_dtype), r.cpu().type(out_dtype), out_dtype))
            print("✓ Forward pass matches manual implementation.")

            # ---------- 2. 反向梯度对比 ----------
            # 构造随机的上游梯度 grad_res（形状与 r 相同）
            grad_res = torch.randn_like(r)

            # 手动计算梯度：-grad_res * r^3 / D * x
            r_manual_detach = r_manual.detach()   # 避免干扰计算图
            grad_x_manual = -grad_res * (r_manual_detach ** 3) / D * x.to(torch.float32)

            # 重新前向以保留计算图（因为之前 r 已脱离计算图？实际未脱离，但为了安全重新计算）
            x.grad = None
            r = gts_rmsnorm_without_weight(x, epsilon, out_dtype)
            # 反向传播，传入上游梯度
            r.backward(grad_res)
            grad_x_auto = x.grad

            grad_x_auto_f32 = grad_x_auto.to(torch.float32)  # 转换为 float32
            assert torch.allclose(grad_x_auto_f32, grad_x_manual, atol=1e-3), "Backward mismatch!"  # 适当放宽容差
            print("✓ Backward gradient matches analytical derivation.")

        # 测试参数
        run_forward_backward([1024, 28672], torch.bfloat16, 1e-6, torch.float32)
        run_forward_backward([1024, 28672], torch.float32, 1e-6, torch.float32)
        run_forward_backward([4096,1,64,512], torch.bfloat16, 1e-6, torch.bfloat16)

    def test_rms_norm_without_weight_forward(self):
        def run_forward(shape_x, x_dtype, epsilon, out_dtype):
            x = torch.randn(shape_x, dtype=x_dtype, requires_grad=True, device="npu")
            # ================ test ================
            npu_result = gts_rmsnorm_without_weight(x, epsilon, out_dtype)
            # =============== golden ===============
            x_calc = x.type(torch.float32)
            x_square_mean = (x_calc ** 2).mean(dim=-1, keepdim=True)
            golden = torch.rsqrt(x_square_mean + epsilon).type(out_dtype)
            self.assertTrue(compare(golden.type(out_dtype), npu_result.type(out_dtype), out_dtype))

        # 测试参数
        run_forward([1024, 1, 28672], torch.bfloat16, 1e-12, torch.float32)
        run_forward([1024, 1, 28672], torch.float32, 1e-12, torch.float32)
        run_forward([4096, 1, 64, 512], torch.bfloat16, 1e-12, torch.bfloat16)

    def _golden_swiglu_limit_probs(self, input, gradout, limit, probs, dim=-1):
        # Golden for npu_swiglu_limit_with_probs (v3)：
        #   out[1]=h_fp32×probs→bf16（kernel A220 高精度路径，不截断 h）；grad_probs 用 fp32-h reduce。
        x = input.type(torch.float32)
        x.requires_grad_(True); x.retain_grad()
        p = probs.type(torch.float32)
        p.requires_grad_(True); p.retain_grad()
        a, b = torch.chunk(x, 2, dim=dim)
        gate = a; up = b
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        gate = F.silu(gate)
        h_fp32 = gate * up                                      # h 全精度 fp32（不截断）
        # 前向 kernel（A220 高精度路径）：h_fp32 × probs → cast bf16（outQueueProbs）。
        # out[1] golden 与 kernel 对齐：不对 h 做 bf16 截断，直接 h_fp32 × probs → bf16。
        out = (h_fp32 * p).type(input.dtype).type(torch.float32)
        out_fwd = out.detach().cpu().type(input.dtype)
        gd = gradout.type(torch.float32)
        out.backward(gd)
        # grad_probs 用 fp32-h 单独算：grad_probs[t] = sum_H(gradout * h_fp32)。
        grad_probs_fp32h = (gd * h_fp32).sum(dim=dim, keepdim=True)
        return out_fwd, x.grad.cpu().type(input.dtype), grad_probs_fp32h.detach().cpu().type(probs.dtype)

    def test_gts_swiglu_limit_probs_bwd(self):
        # 有 probs（MoE overlap 路径）：adapter 返回单输出 h*probs，.backward() 触发 backward_probs。
        import gtsops
        self.assertTrue(hasattr(gtsops, "npu_swiglu_limit_with_probs"),
                        "gtsops 缺少正向 v3 pybind: npu_swiglu_limit_with_probs")

        dim = -1
        for shape, dtype, limit in [((16, 32), torch.bfloat16, 1.0),
                                    ((1024, 6144), torch.bfloat16, 10.0)]:
            print(f"====== adapter swiglu_limit_probs case: shape={shape} dtype={dtype} limit={limit}")
            x = (torch.rand(shape) * 8 - 4).to(dtype)
            grad_shape = list(shape)
            grad_shape[dim] = grad_shape[dim] // 2
            gradout = (torch.rand(grad_shape) * 2 - 1).to(dtype)
            T = gradout.numel() // grad_shape[dim]
            probs = (torch.rand((T, 1)) * 2.0 - 0.5).to(torch.float32)
            golden_fwd, golden_grad, golden_probs_grad = self._golden_swiglu_limit_probs(
                x, gradout, limit, probs, dim)

            is_enable_op("swiglu1_probs", True)
            try:
                inp = x.npu(); inp.requires_grad_(True); inp.retain_grad()
                prob = probs.npu(); prob.requires_grad_(True); prob.retain_grad()
                out = gts_swiglu_limit_probs(inp, limit, prob)
                self.assertIsNotNone(out, "gts_swiglu_limit_probs unexpectedly returned None")
                out_fwd = out[1].detach().cpu()
                out[1].backward(gradout.npu().contiguous())
                grad = inp.grad.cpu()
                probs_grad = prob.grad.cpu()
            finally:
                is_enable_op("swiglu1_probs", False)

            self.assertTrue(compare(golden_fwd, out_fwd, torch.float16, "swiglu_limit_probs_fwd:",
                                    show_detail_count=30, sort_mask=0, show_all=False, new_standard=False))
            self.assertTrue(compare(golden_grad, grad, torch.float16, "swiglu_limit_probs_bwd_x:",
                                    show_detail_count=30, sort_mask=0, show_all=False, new_standard=False))
            self.assertIsNotNone(probs_grad, "grad_probs 为 None：routing 反向会丢梯度！")
            gp_op = probs_grad.detach().to(torch.float32)
            gp_gd = golden_probs_grad.detach().to(torch.float32)
            l2_rel = torch.linalg.vector_norm(gp_op - gp_gd) / (torch.linalg.vector_norm(gp_gd) + 1e-12)
            print(f"[grad_probs] L2 相对误差 = {l2_rel.item():.3e} (need < 1e-3)")
            self.assertTrue(l2_rel.item() < 1e-3,
                            f"grad_probs L2 相对误差 {l2_rel.item():.3e} 超过 1e-3：整体梯度偏差过大！")

    def test_gts_swiglu_limit_probs_fallback(self):
        # 回退：probs=None / 开关关 / limit<=0 应返回 None（开关用独立 swiglu1_probs）
        x = torch.rand((16, 32)).to(torch.bfloat16).npu()
        probs = torch.ones((16, 1), dtype=torch.float32).npu()
        is_enable_op("swiglu1_probs", True)
        self.assertIsNone(gts_swiglu_limit_probs(x, 10.0, None))   # probs=None
        self.assertIsNone(gts_swiglu_limit_probs(x, 0.0, probs))   # limit<=0
        is_enable_op("swiglu1_probs", False)
        self.assertIsNone(gts_swiglu_limit_probs(x, 10.0, probs))  # 开关关
        is_enable_op("swiglu1_probs", False)

    def test_swiglu_limit_grad_probs_op(self):
        # 直接调底层算子 npu_swiglu_limit_backward_probs，验 grad_probs(算子内逐行 reduce 算)。
        # 重点：用训练真实 shape [12288,6144]（colLen=3072）确证不切列、grad_probs 不失效。
        #   grad_probs[t]=sum_H(gradout*h) 量级 O(几十~百)，若切列(算子不写值, result2 是脏数据)必失败，
        #   故本测试能真正测出"切列导致 grad_probs 失效"的问题（不会被脏数据蒙混）。
        import gtsops
        import torch.nn.functional as F
        for T, H in [(1024, 3072), (12288, 3072)]:   # test shape + 训练真实 shape
            torch.manual_seed(0)
            x = (torch.rand(T, 2 * H) * 8 - 4).to(torch.bfloat16)
            g = (torch.rand(T, H) * 2 - 1).to(torch.bfloat16)
            probs = (torch.rand(T, 1) * 2 - 0.5).to(torch.float32)
            limit = 10.0
            # golden grad_probs = sum_H(gradout * h)，h = swiglu_limit(x)（fp32）
            xf = x.float().npu(); a, b = torch.chunk(xf, 2, dim=-1)
            ac = torch.clamp(a, max=limit); bc = torch.clamp(b, min=-limit, max=limit)
            h = F.silu(ac) * bc
            gp_golden = (g.float().npu() * h).sum(dim=-1, keepdim=True).cpu()
            # 算子（一次出 grad_x, grad_probs）
            _, gp_op = gtsops.npu_swiglu_limit_backward_probs(
                g.npu().contiguous(), x.npu().contiguous(), -1, float(limit), probs.npu().contiguous())
            gp_op = gp_op.cpu()
            err = (gp_op - gp_golden).abs().max().item()
            rel = err / (gp_golden.abs().max().item() + 1e-6)
            print(f"[grad_probs] shape[{T},{2*H}] golden[:3]={gp_golden[:3,0].tolist()} "
                  f"op[:3]={gp_op[:3,0].tolist()} max_err={err:.3f} rel={rel:.2e}")
            # rel<1e-2 才算对；若切列(脏数据)，±百量级的 grad_probs 必不匹配 → 失败
            self.assertTrue(rel < 1e-2,
                            f"grad_probs 不匹配 shape[{T},{2*H}] rel={rel:.2e}：可能切列导致算子未算 grad_probs！")

    @staticmethod
    def _golden_make_chunk_sort_map(split_sizes, sorted_indices, num_tokens):
        """CPU golden for MindSpeed make_chunk_sort_map.

        row_id_map[t] = output_prefix[inverse_sorted_indices[chunk(t)]]
                        + (t - input_prefix[chunk(t)])
        """
        split_list = [int(v) for v in split_sizes.cpu().tolist()]
        sorted_list = [int(v) for v in sorted_indices.cpu().tolist()]
        num_splits = len(split_list)

        input_prefix = [0]
        for size in split_list:
            input_prefix.append(input_prefix[-1] + size)

        inverse_sorted_indices = [0] * num_splits
        for out_chunk, in_chunk in enumerate(sorted_list):
            inverse_sorted_indices[in_chunk] = out_chunk

        output_prefix = [0]
        for in_chunk in sorted_list:
            output_prefix.append(output_prefix[-1] + split_list[in_chunk])

        row_id_map = torch.empty((num_tokens,), dtype=torch.int32)
        for chunk in range(num_splits):
            start = input_prefix[chunk]
            end = input_prefix[chunk + 1]
            out_start = output_prefix[inverse_sorted_indices[chunk]]
            for token in range(start, end):
                row_id_map[token] = out_start + token - start
        return row_id_map

    @staticmethod
    def _onsite_case():
        """复刻现场 fb_overlap MoE dispatcher 的真实调用参数。

        现场(profiling 0629 + MindSpeed 源码双证):
          num_splits    = num_experts * tp_size = 384
          num_tokens    = 12288 (静态)
          sorted_indices = arange(384).reshape(-1, num_local_experts).T.ravel()
                           (mindspeed/core/fusions/fused_moe_permute.py, block-transpose 排列, 非随机)
          split_sizes    = num_global_tokens_per_local_expert.ravel(), 非均匀且含零。
        这里用确定性(不依赖随机种子)的含零非均匀分布, 总和恰为 12288。
        """
        num_splits, num_local_experts, num_tokens = 384, 8, 12288
        sorted_values = torch.arange(num_splits).reshape(-1, num_local_experts).T.ravel().tolist()
        # 确定性含零非均匀分布: 每 4 个 chunk 放一个 0, 其余按模式取值, 再把余数补到最后一个非零 chunk。
        split_values = [0 if (i % 4 == 3) else (i % 13 + 1) for i in range(num_splits)]
        remainder = num_tokens - sum(split_values)
        split_values[-2] += remainder  # -2 保证是非零位(-1 为 i%4==3 的 0 位), 使总和==num_tokens
        assert sum(split_values) == num_tokens and split_values[-2] >= 0
        return split_values, sorted_values

    def test_gts_make_chunk_sort_map(self):
        """Adapter-level test for WindMakeChunkSortMap.

        This exercises gts_custom_op.gts_make_chunk_sort_map directly, including:
          - switch-off fallback returns None;
          - int32/int64 split/sorted inputs;
          - uneven chunks and zero-size chunks;
          - output equals MindSpeed row_id_map semantics.
        """
        cases = [
            ([7], [0]),
            ([3, 5], [1, 0]),
            ([0, 4, 2, 0, 7, 1, 0, 3], [4, 0, 6, 2, 7, 1, 3, 5]),
            ([i % 7 for i in range(384)], list(reversed(range(384)))),
            self._onsite_case(),
        ]
        for dtype in (torch.int32, torch.int64):
            for split_values, sorted_values in cases:
                split_sizes = torch.tensor(split_values, dtype=dtype, device="npu")
                sorted_indices = torch.tensor(sorted_values, dtype=dtype, device="npu")
                num_tokens = int(sum(split_values))

                is_enable_op("make_chunk_sort_map", False)
                self.assertIsNone(gts_make_chunk_sort_map(split_sizes, sorted_indices, num_tokens))

                golden = self._golden_make_chunk_sort_map(split_sizes.cpu(), sorted_indices.cpu(), num_tokens)
                is_enable_op("make_chunk_sort_map", True)
                out = gts_make_chunk_sort_map(split_sizes, sorted_indices, num_tokens)
                is_enable_op("make_chunk_sort_map", False)

                self.assertIsNotNone(out, "gts_make_chunk_sort_map returned None with switch enabled")
                self.assertEqual(out.dtype, torch.int32)
                self.assertEqual(tuple(out.shape), (num_tokens,))
                self.assertTrue(torch.equal(out.cpu(), golden),
                                f"make_chunk_sort_map mismatch dtype={dtype} splits={len(split_values)}")

    # ===== gts_scale_rope: op-adapter 接口验证(q/kv/o 三处, 含 freqs→cos/sin 转换的端到端) =====
    #   验证: ① gts_scale_rope 接口数值正确 ② 含 freqs 转换(不像底层算子测试把转换提到循环外)
    #   golden = 现场 deepseek_utils.apply_rotary_emb(独立的复数式 RoPE, 检验算子与现场一致)。
    @staticmethod
    def _apply_rotary_emb(x, freqs_cis, inverse=False):
        # 拷贝自 deepseek_utils.py:8-28(复数式 RoPE)。freqs 按 x 的 seq 维(=size(1))广播。
        od = x.dtype
        xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        if inverse:
            freqs_cis = freqs_cis.conj()
        if xc.ndim == 3:
            freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        return torch.view_as_real(xc * freqs_cis).flatten(-2).to(od)

    @staticmethod
    def _rel(a, b):
        return ((a.float() - b.float()).abs().max()
                / a.float().abs().max().clamp_min(1e-6)).item()

    @staticmethod
    def _golden_scale_rope(x, freqs, rope, scale, inverse):
        """纯 torch 可导 golden(现场链路): u=x*scale; 后 rope 维做复数式 RoPE。
        转到 seq 在 dim1 的 layout(apply_rotary_emb 要求 freqs 按 size(1)广播), 算完转回。
        返回保留 autograd 图, 供反向对拍直接 .backward()。"""
        u = x if scale is None else (x * scale)             # 不 detach: 反向要经 scale/x 回传
        ut = u.transpose(0, 1)                              # [s,b,..] -> [b,s,..], seq 到 dim1
        rot = TestGts._apply_rotary_emb(ut[..., -rope:], freqs, inverse)
        # 整行输出(不用 slice-assign, 避免 golden 自身引入 CopySlices 反向, 与算子"整行输出"对齐)
        gold_t = torch.cat([ut[..., :-rope], rot], dim=-1)
        return gold_t.transpose(0, 1)                       # 转回 [s,b,..]

    def _check_scale_rope(self, x, freqs, rope, scale, inverse, tag):
        # ---- 前向 ----
        is_enable_op("rope", True)
        out = gts_scale_rope(x, freqs, rope, scale=scale, inverse=inverse)
        is_enable_op("rope", False)
        self.assertIsNotNone(out, f"{tag}: gts_scale_rope 返回 None(开关未生效)")
        with torch.no_grad():
            gold = self._golden_scale_rope(x, freqs, rope, scale, inverse)
        pr = self._rel(gold[..., :-rope], out[..., :-rope])   # 直通段
        rr = self._rel(gold[..., -rope:], out[..., -rope:])   # rope 段

        # ---- 反向(接口级: 验 gts_scale_rope 经 freqs转换/reshape/view/inverse 后梯度仍正确) ----
        #   同一上游梯度 g 分别反传 算子链 与 golden链, 比 grad_x(及 grad_scale)。
        #   golden 用纯 torch autograd(复数 RoPE 可导), 是可信参照。
        g = torch.randn_like(out)
        xA = x.detach().clone().requires_grad_(True)
        sA = None if scale is None else scale.detach().clone().requires_grad_(True)
        is_enable_op("rope", True)
        outA = gts_scale_rope(xA, freqs, rope, scale=sA, inverse=inverse)
        is_enable_op("rope", False)
        outA.backward(g)

        xB = x.detach().clone().requires_grad_(True)
        sB = None if scale is None else scale.detach().clone().requires_grad_(True)
        self._golden_scale_rope(xB, freqs, rope, sB, inverse).backward(g)

        gxr = self._rel(xB.grad, xA.grad)
        gsr = 0.0 if scale is None else self._rel(sB.grad, sA.grad)
        smsg = "" if scale is None else f"  grad_scale rel={gsr:.2e}"
        print(f"  [{tag}] fwd 直通 rel={pr:.2e}  rope rel={rr:.2e} | bwd grad_x rel={gxr:.2e}{smsg}")
        self.assertTrue(pr < 2e-2 and rr < 2e-2, f"{tag}: 前向数值不符 (直通{pr:.1e}, rope{rr:.1e})")
        self.assertTrue(gxr < 2e-2, f"{tag}: 反向 grad_x 不符 ({gxr:.1e})")
        self.assertTrue(gsr < 2e-2, f"{tag}: 反向 grad_scale 不符 ({gsr:.1e})")

    def test_gts_scale_rope_q(self):
        # q: [s,b,n,d]=[4096,1,64,512], scale=rsqrt 外露, forward, global freqs
        S, B, N, D, ROPE, HALF = 4096, 1, 64, 512, 64, 32
        q = torch.randn(S, B, N, D, dtype=torch.bfloat16, device="npu")
        rsqrt = (torch.rand(S, B, N, 1, dtype=torch.bfloat16, device="npu") * 0.5 + 0.5)
        freqs = torch.polar(torch.ones(S, HALF, device="npu"), torch.randn(S, HALF, device="npu"))
        self._check_scale_rope(q, freqs, ROPE, rsqrt, False, "q")

    def test_gts_scale_rope_kv(self):
        # kv: [s,b,d]=[2048,1,512] 无 head 维, scale=None(ones), forward, local freqs
        S, B, D, ROPE, HALF = 2048, 1, 512, 64, 32
        kv = torch.randn(S, B, D, dtype=torch.bfloat16, device="npu")
        freqs = torch.polar(torch.ones(S, HALF, device="npu"), torch.randn(S, HALF, device="npu"))
        self._check_scale_rope(kv, freqs, ROPE, None, False, "kv")

    def test_gts_scale_rope_o(self):
        # o: [s,b,n,d]=[4096,1,64,512], scale=None, inverse, global freqs
        S, B, N, D, ROPE, HALF = 4096, 1, 64, 512, 64, 32
        o = torch.randn(S, B, N, D, dtype=torch.bfloat16, device="npu")
        freqs = torch.polar(torch.ones(S, HALF, device="npu"), torch.randn(S, HALF, device="npu"))
        self._check_scale_rope(o, freqs, ROPE, None, True, "o")

    def test_gts_scale_rope_shapes(self):
        # shape-agnostic 验证: batch 变化(MBS 训练中会调)、不同 seq/head/rope_dim。
        #   seq 反算 seq=row//(b*n), 必须对 b>1 也正确(这是当初 seq=0 bug 的根因)。
        HALF_OF = lambda r: r // 2
        cases = [
            # (S, B, N, D, ROPE, scale?, inverse, tag)
            (1024, 2, 64, 512, 64, True,  False, "q_B2"),       # batch=2(MBS 变化!), q 带 scale
            (1024, 4, 32, 512, 64, True,  False, "q_B4_N32"),   # batch=4, head=32
            (2048, 2, 512,    64, False, False, "kv_B2"),       # kv 无 head 维, batch=2
            (4096, 1, 64, 512, 64, False, True,  "o_inv"),      # o inverse 复核
            (512,  1, 16, 256, 32, True,  False, "small_d256_rope32"),  # 小 shape, D=256 rope=32
            (8192, 1, 8,  512, 128, False, False, "longseq_rope128"),   # 长 seq, rope=128
        ]
        for c in cases:
            if len(c) == 8:
                S, B, N, D, ROPE, has_scale, inverse, tag = c
                x = torch.randn(S, B, N, D, dtype=torch.bfloat16, device="npu")
            else:  # kv 形态(无 head 维): (S,B,D,ROPE,has_scale,inverse,tag)
                S, B, D, ROPE, has_scale, inverse, tag = c
                x = torch.randn(S, B, D, dtype=torch.bfloat16, device="npu")
            scale = (torch.rand(*x.shape[:-1], 1, dtype=torch.bfloat16, device="npu") * 0.5 + 0.5) if has_scale else None
            freqs = torch.polar(torch.ones(S, HALF_OF(ROPE), device="npu"),
                                torch.randn(S, HALF_OF(ROPE), device="npu"))
            self._check_scale_rope(x, freqs, ROPE, scale, inverse, tag)

    def test_wind_overlap_transform_forward_backward(self):
        def overlap_transform(tensor, compress_ratio, head_dim, value=0):
            """Eager reference: overlap_transform for a single tensor.

            Args:
                tensor: [B, S, R, 2D] float32
                compress_ratio: int (R)
                head_dim: int (D)
                value: fill value for padding row (0 for kv, -inf for score)
            Returns:
                [B, S, 2R, D] float32
            """
            R, D = compress_ratio, head_dim
            b, s, _, _ = tensor.size()
            new_tensor = tensor.new_full((b, s, 2 * R, D), value)
            new_tensor[:, :, R:] = tensor[:, :, :, D:]
            new_tensor[:, 1:, :R] = tensor[:, :-1, :, :D]
            return new_tensor

        def get_forward_golden( kv, score, compress_ratio, head_dim, kv_value, score_value):
            return overlap_transform(kv, compress_ratio, head_dim, kv_value), overlap_transform(score, compress_ratio, head_dim, score_value)

        def run_forward_backward_test(shape, dtype, compress_ratio=4, head_dim=128, kv_value=0.0, score_value=0.0):            # 创建输入（开启梯度）
            kv = torch.rand(shape, dtype=dtype, requires_grad=True, device="npu")
            score = torch.rand(shape, dtype=dtype, requires_grad=True, device="npu")
            # 独立副本用于 golden 参考
            kv_ref = kv.clone().detach().requires_grad_(True)
            score_ref = score.clone().detach().requires_grad_(True)

            # ----- 前向 -----
            kv_out, score_out = gts_wind_overlap_transform(
                kv, score, compress_ratio, head_dim, kv_value, score_value
            )
            golden_kv_out, golden_score_out = get_forward_golden(
                kv_ref, score_ref, compress_ratio, head_dim, kv_value, score_value
            )

            # ----- 反向 -----
            (kv_out.sum() + score_out.sum()).backward()
            (golden_kv_out.sum() + golden_score_out.sum()).backward()

            # ----- 比较（仅非性能模式 且 第一次迭代）-----
            print("------  golden_kv_out vs kv_out  ------")
            self.assertTrue(compare(golden_kv_out.cpu(), kv_out.cpu(), dtype))
            print("------  golden_score_out vs score_out  ------")
            self.assertTrue(compare(golden_score_out.cpu(), score_out.cpu(), dtype))
            print("------  kv_ref grad vs kv grad  ------")
            self.assertTrue(compare(kv_ref.grad.cpu(), kv.grad.cpu(), dtype))
            print("------  score_ref grad vs score grad  ------")
            self.assertTrue(compare(score_ref.grad.cpu(), score.grad.cpu(), dtype))

        run_forward_backward_test([1,1024,4,256], torch.float32, 4, 128, 0.0, float("-inf"))
        run_forward_backward_test([1,1024,4,1024], torch.float32, 4, 512, 0.0, float("-inf"))

if __name__ == "__main__":
    run_tests()
