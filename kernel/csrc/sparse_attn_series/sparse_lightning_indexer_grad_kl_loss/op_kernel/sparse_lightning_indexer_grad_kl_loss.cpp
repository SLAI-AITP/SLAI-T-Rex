/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file sparse_lightning_indexer_grad_kl_loss.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_lightning_indexer_grad_kl_loss_template_tiling_key.h"
#include "sparse_lightning_indexer_grad_kl_loss_base.h"
#include "sparse_lightning_indexer_grad_kl_loss_common.h"

using namespace AscendC;

#define SLI_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        templateClass<SLIType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, key, queryIndex, keyIndex, weight, sparseIndices,                          \
	            softmaxMax, softmaxMum, queryRope, keyRope,                                       \
                actualSeqLengthsQuery, actualSeqLengthsKey,                                       \
                dQueryIndex, dKeyIndex, dWeight, loss,                                            \
                user, tiling_data, &tPipe);                                                       \
        op.Process();                                                                             \
    } while (0)

template< bool HasRope, int TopKRange, int LayoutT_QT, int LayoutT_KT, int SparseMode, bool Deterministic>
 __global__ __aicore__ void
sparse_lightning_indexer_grad_kl_loss(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                      __gm__ uint8_t *queryIndex, __gm__ uint8_t *keyIndex,
                                      __gm__ uint8_t *weight, __gm__ uint8_t *sparseIndices,
                                      __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxMum,
                                      __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                                      __gm__ uint8_t* actualSeqLengthsQuery,
                                      __gm__ uint8_t* actualSeqLengthsKey,
                                      __gm__ uint8_t *dQueryIndex, __gm__ uint8_t *dKeyIndex,
                                      __gm__ uint8_t *dWeight, __gm__ uint8_t *loss,
                                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::SparseLightningIndexerGradKLLossTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16) {
        SLI_OP_IMPL(SparseLightningIndexerGradKLLossBase, optiling::SparseLightningIndexerGradKLLossTilingData,
            half, half, half, static_cast<SLITopKRange>(TopKRange),
            static_cast<SLILayout>(LayoutT_QT), static_cast<SLILayout>(LayoutT_KT),
            SLISparseMode::RightDown, HasRope, Deterministic);
    }
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16) {
        SLI_OP_IMPL(SparseLightningIndexerGradKLLossBase, optiling::SparseLightningIndexerGradKLLossTilingData,
            bfloat16_t, bfloat16_t, bfloat16_t, static_cast<SLITopKRange>(TopKRange),
            static_cast<SLILayout>(LayoutT_QT), static_cast<SLILayout>(LayoutT_KT),
            SLISparseMode::RightDown, HasRope, Deterministic);
    }
}