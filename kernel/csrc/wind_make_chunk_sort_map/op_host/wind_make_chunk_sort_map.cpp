/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <register/op_def_registry.h>

namespace ops {
class WindMakeChunkSortMap : public OpDef {
public:
    explicit WindMakeChunkSortMap(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> indexTypes = {ge::DT_INT32, ge::DT_INT64};
        std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("split_sizes").ParamType(REQUIRED).DataType(indexTypes).Format(formats);
        this->Input("sorted_indices").ParamType(REQUIRED).DataType(indexTypes).Format(formats);
        this->Output("row_id_map").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32}).Format(formats);
        this->Attr("num_tokens").AttrType(REQUIRED).Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("prebuildPattern.value", "Opaque")
                .ExtendCfgInfo("coreType.value", "AiCore")
                .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(WindMakeChunkSortMap);
} // namespace ops
