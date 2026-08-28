/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_template_grad_tiling_key.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHARED_GRAD_TEMPLATE_TILING_KEY_H
#define SPARSE_ATTN_SHARED_GRAD_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SASG_LAYOUT_BSND 0
#define SASG_LAYOUT_TND 1
#define SASG_SWA_MODE 0
#define SASG_CFA_MODE 1
#define SASG_SCFA_MODE 2
#define ASCENDC_TPL_4_BW 4

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseAttnSharedkvGrad, // 算子OpType
ASCENDC_TPL_UINT_DECL(LAYOUT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SASG_LAYOUT_BSND, SASG_LAYOUT_TND),
ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SASG_SWA_MODE, SASG_CFA_MODE, SASG_SCFA_MODE),
);

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, SASG_LAYOUT_BSND, SASG_LAYOUT_TND),
    ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, SASG_SWA_MODE, SASG_CFA_MODE, SASG_SCFA_MODE),
    ),
);

#endif // TEMPLATE_TILING_KEY