# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu

def is_available():
    torch_npu.npu.get_device_properties()
    print(True)

# release api
from .wind_fused_overlap_transform import wind_fused_overlap_transform
from .wind_make_chunk_sort_map import wind_make_chunk_sort_map
from .wind_rms_norm_without_weight import wind_rms_norm_without_weight
from .wind_scale_rope import wind_scale_rope
from .wind_swiglu_limit import wind_swiglu_limit, wind_swiglu_limit_probs

