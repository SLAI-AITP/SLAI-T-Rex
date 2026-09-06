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
// autogen by gtsgen WindFusedOverlapTransform -f inputs=[kv;score];outputs=[kv_out;score_out];attrs=[compress_ratio:i:4;head_dim:i:128;value:f:0.0] -d fp32 -s ascend910b,ascend910c
class WindFusedOverlapTransform : public OpDef {
public:
    explicit WindFusedOverlapTransform(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> types = {ge::DT_FLOAT};
        std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("kv").ParamType(REQUIRED).DataType(types).Format(formats).AutoContiguous();
        this->Input("score").ParamType(OPTIONAL).DataType(types).Format(formats).AutoContiguous();
        this->Output("kv_out").ParamType(REQUIRED).DataType(types).Format(formats);
        this->Output("score_out").ParamType(REQUIRED).DataType(types).Format(formats);
        this->Attr("compress_ratio").AttrType(REQUIRED).Int(4U);
        this->Attr("head_dim").AttrType(REQUIRED).Int(128U);
        this->Attr("kv_value").AttrType(REQUIRED).Float(0.0);
        this->Attr("score_value").AttrType(REQUIRED).Float(0.0);

        OpAICoreConfig aicore_conf;
        aicore_conf.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("prebuildPattern.value", "Opaque")
                .ExtendCfgInfo("coreType.value", "AiCore")
                .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_conf);
        this->AICore().AddConfig("ascend910_93", aicore_conf);
    }
};

OP_ADD(WindFusedOverlapTransform);
} // namespace ops

