# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import torch_npu
from typing import Tuple

_GTS_GRAPH_OP_ENABLED = None

def is_enable_op(op_name: str, inject=None) -> bool:
    global _GTS_GRAPH_OP_ENABLED
    if _GTS_GRAPH_OP_ENABLED is None:
        try:
            import gtsspeed.ops
            _GTS_GRAPH_OP_ENABLED = True
            print("===Success: Register gts ops")
        except ImportError:
            _GTS_GRAPH_OP_ENABLED = False
            print("===Warning: Failed to register gts ops")

    if not _GTS_GRAPH_OP_ENABLED:
        return False

    """
    检查是否启用了指定的算子环境变量。
    传入 'mm' -> 拼接为 'GTS_ENABLE_OP_MM'。
    """
    # 1. 将传入的字符串变为大写，并拼接前缀
    env_name = f"GTS_ENABLE_OP_{op_name.replace('-', '_').upper()}"

    if inject is not None:
        os.environ[env_name] = ("true" if inject else "false")
        return inject

    # 2. 判断环境变量是否存在且值为 true
    return os.getenv(env_name, "false").lower() == "true"

is_enable_op("")
if _GTS_GRAPH_OP_ENABLED:
    import gtsspeed.ops

# [Add by GTS-AILab] [feature GTSOps] swiglu_limit 融合算子(clamp+swiglu)，前向+反向自动求导。
# 接入点为 fused_swiglu_with_limit（自带老路径 fallback），故用 None 回退：开关关 / limit<=0 / 非支持 dtype 时返回 None。
# 反向 PyTorch autograd 自动触发（gtsspeed.ops.wind_swiglu_limit = NpuSwiGluLimit.apply），无需单独函数。
def gts_swiglu_limit(x: torch.Tensor, limit: float):
    if (not is_enable_op('swiglu1') or limit <= 0
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return None
    # v2 融合算子（反向已删 round-trip，精度对齐线上）。
    # 签名 NpuSwiGluLimit.apply(x, limit, y=None, dim=-1, backend)；传 "v2"。
    return gtsspeed.ops.wind_swiglu_limit(x, float(limit), None, -1, "v2")


# swiglu_limit + probs 融合算子（MoE 专家 FFN：swiglu 后逐 token 乘 routing 权重 probs）。
# 正向接纳前向同事的 with_probs v3（取 h*probs 输出），反向把 grad_h = grad_out * probs
# 和 grad_probs reduce 都收进 backward_probs。
# probs=None 或开关关 / limit<=0 / 非支持 dtype 时返回 None，回退老路；probs 应为 [T,1] fp32。
# 开关用独立的 'swiglu1_probs'（GTS_ENABLE_OP_SWIGLU1_PROBS / --enable-gts-swiglu1-probs），
# 与无 probs 的 gts_swiglu_limit('swiglu1') 解耦：可单独灰度 MoE probs 融合，不影响已上线的 dense swiglu_limit。
def gts_swiglu_limit_probs(x: torch.Tensor, limit: float, probs: torch.Tensor):
    if (not is_enable_op('swiglu1_probs') or limit <= 0 or probs is None
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return None
    return gtsspeed.ops.wind_swiglu_limit_probs(
        x, None, probs, float(limit), -1, True, True, "v3"
    )

def gts_rmsnorm_without_weight(
    x: torch.Tensor,
    norm_eps: float = 1e-6,
    outDType: int | None  = None
) -> torch.Tensor:
    if not is_enable_op('rms0'):
        return None

    return gtsspeed.ops.wind_rms_norm_without_weight(x, norm_eps, (x.dtype if outDType is None else outDType))


def gts_make_chunk_sort_map(split_sizes: torch.Tensor, sorted_indices: torch.Tensor, num_tokens: int):
    """融合 MoE chunk-sort 的 row_id_map 构造。

    替换 mindspeed.lite.ops.triton.sort_chunks_by_idx.make_chunk_sort_map 中的
    cumsum/arange/scatter/index/cumsum + _make_chunk_sort_map_kernel 元数据链。
    返回 None 时调用方回退原 MindSpeed 路径。
    """
    if not is_enable_op('make_chunk_sort_map'):
        return None
    if split_sizes.device.type != 'npu' or sorted_indices.device.type != 'npu':
        return None
    if split_sizes.dim() != 1 or sorted_indices.dim() != 1:
        return None
    if split_sizes.numel() != sorted_indices.numel():
        return None
    if split_sizes.dtype not in (torch.int32, torch.int64) or sorted_indices.dtype not in (torch.int32, torch.int64):
        return None

    op = getattr(gtsspeed.ops, 'wind_make_chunk_sort_map', None)
    if op is None:
        raise RuntimeError(
            "GTS_ENABLE_OP_MAKE_CHUNK_SORT_MAP=true but gtsspeed.ops.wind_make_chunk_sort_map is not available"
        )
    return op(split_sizes, sorted_indices, int(num_tokens))


# ===== WindScaleRope: 融合 (可选)scale*x + interleave RoPE, 覆盖 q/kv/o 三处 =====
#   语义: out = rope(x*scale); 前 (D-rope) 维直通, 后 rope 维复数式旋转。
#   q:  scale=rsqrt(外露), forward,  global freqs
#   kv: scale=1,           forward,  local  freqs (无 head 维 [s,b,d])
#   o:  scale=1,           inverse,  global freqs (sin 翻号)
#   freqs_cis: 复数 [S, rope/2](torch.polar 生成), 内部拆成 cos/sin 实数喂算子(算子不支持复数)。
# 缓存 cos/sin_fwd/sin_bwd, 命中则不重算 real/imag/repeat/stack。
#   freqs_cis 来自 rotary 表的切片 self.freqs_cis[start_pos:...]; 切片每 step 是新对象(id 变),
#   但源表与 start_pos/len 不变时共享底层存储、data_ptr 相同, 故用 (data_ptr,shape,dtype,device)
#   作 key 可跨 step 命中。存源 tensor 引用(hit[0])并校验 data_ptr 一致, 防 ptr 释放后被别的张量
#   复用导致脏命中; 不匹配则重算。LRU 限 8 个最近切片(start_pos 可能随变长而变)。
_CS_CACHE = {}  # (data_ptr,shape,dtype,device) -> (freqs_cis_ref, cos, sin_fwd, sin_bwd)
_CS_CACHE_MAX = 8

# kv/o 的 scale=1 用全 1 tensor; 直接 torch.ones 每 step 产生一个 OnesLike 算子。
#   按 (T, device, dtype) 缓存复用同一个 ones, 后续 step 不再产生该算子。
_ONES_CACHE = {}  # (T, device, dtype) -> ones[T,1]


def _get_ones(T, device, dtype):
    """返回缓存的 [T,1] 全 1 tensor(scale=1 用), 避免每 step 的 torch.ones。"""
    key = (T, str(device), dtype)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones(T, 1, device=device, dtype=dtype)
        _ONES_CACHE[key] = t
    return t


def _freqs_to_cos_sin_full(freqs_cis):
    """复数 freqs_cis [S, HALF] -> (cos, sin_fwd, sin_bwd)。host 一次产出 3 份, 带层内缓存。
    cos: 每复数重复 2 份(不分 fwd/inverse)。
    sin_fwd: sinSigned 偶位=-sin 奇位=+sin(forward 乘 freqs)。
    sin_bwd: = -sin_fwd, 偶位=+sin 奇位=-sin(= inverse 乘 conj 的 sinSigned)。
      -> q forward 用 sin_fwd; o(inverse) 的 sin 即 sin_bwd; 反向各用对方。
    用 stack 一次构造, 无 slice-assign 散射 ViewCopy。
    """
    # GTS_ROPE_NO_CACHE=1 禁用缓存(定位缓存是否引入脏数据)。
    _no_cache = os.getenv('GTS_ROPE_NO_CACHE', '0') == '1'
    key = (freqs_cis.data_ptr(), tuple(freqs_cis.shape), freqs_cis.dtype, str(freqs_cis.device))
    hit = None if _no_cache else _CS_CACHE.get(key)
    if hit is not None and hit[0].data_ptr() == freqs_cis.data_ptr():
        return hit[1], hit[2], hit[3]
    c = freqs_cis.real.float()
    s = freqs_cis.imag.float()
    cos = c.repeat_interleave(2, dim=-1).contiguous()
    sin_fwd = torch.stack([-s, s], dim=-1).reshape(c.size(0), -1).contiguous()  # 偶-s 奇+s
    sin_bwd = torch.stack([s, -s], dim=-1).reshape(c.size(0), -1).contiguous()  # 偶+s 奇-s = -sin_fwd
    if not _no_cache:
        if len(_CS_CACHE) >= _CS_CACHE_MAX:   # LRU: 满则删最早插入的(dict 保插入序)
            _CS_CACHE.pop(next(iter(_CS_CACHE)))
        _CS_CACHE[key] = (freqs_cis, cos, sin_fwd, sin_bwd)
    return cos, sin_fwd, sin_bwd


def _freqs_to_cos_sin(freqs_cis, inverse=False):
    """兼容旧接口: 返回 (cos, sin)。inverse=True 时 sin = sin_bwd(conj 的 sinSigned)。"""
    cos, sin_fwd, sin_bwd = _freqs_to_cos_sin_full(freqs_cis)
    return cos, (sin_bwd if inverse else sin_fwd)


def gts_scale_rope(x, freqs_cis, rope_head_dim, scale=None, inverse=False):
    """融合 (可选)scale + RoPE。返回 None 时调用方回退原路(apply_rotary_emb)。
    Args:
        x:    [*, D] bf16, 任意前置维度(q [s,b,n,d] / kv [s,b,d] / o [b,s,n,d] 均可, 内部摊平 [T,D])
        freqs_cis: 复数 [S, rope_head_dim/2]
        rope_head_dim: 旋转维度(后 rope_head_dim 维做 RoPE)
        scale: [*, 1] bf16(q 传 rsqrt) 或 None(kv/o, 内部用 ones)
        inverse: True 则乘 conj(freqs)(o 段), 内部 sin 翻号
    """
    if not is_enable_op('rope'):
        return None
    D = x.shape[-1]
    T = x.numel() // D
    cos, sin_fwd, sin_bwd = _freqs_to_cos_sin_full(freqs_cis)
    # forward 的 sin: q/kv 用 sin_fwd; o(inverse) 用 sin_bwd。其反向 sin 即对方(传给算子省图上 Neg)。
    sin, sin_back = (sin_bwd, sin_fwd) if inverse else (sin_fwd, sin_bwd)
    if scale is None:
        scale2 = _get_ones(T, x.device, x.dtype)
    else:
        scale2 = scale.reshape(T, 1)
    # 优先传 sin_back(反向用预存 -sin, 省图上 Neg); 旧版 wind_scale_rope 不接该参数时 fallback。
    try:
        out = gtsspeed.ops.wind_scale_rope(x.reshape(T, D), scale2, cos, sin, sin_back)
    except TypeError:
        out = gtsspeed.ops.wind_scale_rope(x.reshape(T, D), scale2, cos, sin)
    return out.view(x.shape)


def gts_wind_overlap_transform(kv: torch.Tensor, score: torch.Tensor, compress_ratio, head_dim, kv_value, score_value) -> Tuple[torch.Tensor, torch.Tensor]:
    if not is_enable_op('compressor0'):
        return None, None
    return gtsspeed.ops.wind_fused_overlap_transform(kv, score, compress_ratio, head_dim, kv_value, score_value)
