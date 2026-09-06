/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <string>
#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>
#include <securec.h>
// #include <toolchain/slog.h>
//#include <common/util/error_manager/error_manager.h>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <exe_graph/runtime/infer_datatype_context.h>

namespace ops {
namespace utils {

struct LogBase {
public:
    static constexpr const int maxLogLen = 16000;
    static constexpr const int msgHdrLen = 200;

    static inline uint64_t GetTid()
    {
        return static_cast<uint64_t>(syscall(__NR_gettid));
    }

    static inline const char* GetStr(const std::string& str)
    {
        return str.c_str();
    }

    static inline const char* GetStr(const char* str)
    {
        return str;
    }

    static inline const std::string& GetOpInfo(const std::string& str)
    {
        return str;
    }

    static inline const char* GetOpInfo(const char* str)
    {
        return str;
    }

    static inline std::string GetOpInfo(const gert::TilingContext* context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::TilingParseContext* context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferShapeContext* context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferDataTypeContext* context)
    {
        return GetOpInfoFromContext(context);
    }

private:
    template <class T>
    static inline std::string GetOpInfoFromContext(T context)
    {
        if (context == nullptr) {
            return "nil:nil";
        }
        std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
        opInfo += ":";
        opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
        return opInfo;
    }
};

}  // namespace utils
}  // namespace ops

// 使用本宏前需预定义标识子模块名称的 OPS_UTILS_LOG_SUB_MOD_NAME
// 如: #define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING" 或通过 CMake 传递预定义宏
#ifdef OP_TILING_LIB
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING"
#elif OP_PROTO_LIB
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_PROTO"
#else
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_UNKNOWN"
#endif
#define OPS_LOG_STUB(MOD_ID, LOG_LEVEL, OPS_DESC, FMT, ...)                                                 \
DlogSub(static_cast<int>(MOD_ID), (OPS_UTILS_LOG_SUB_MOD_NAME), (LOG_LEVEL), "[%s][%lu] OpName:[%s] " #FMT, \
    __FUNCTION__, ops::utils::LogBase::GetTid(),                                                            \
    ops::utils::LogBase::GetStr(ops::utils::LogBase::GetOpInfo(OPS_DESC)), ##__VA_ARGS__)

#define OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)                                                               \
do {                                                                                                        \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool"); \
    if (__builtin_expect((COND), 0)) {                                                                      \
        LOG_FUNC;                                                                                           \
        EXPR;                                                                                               \
    }                                                                                                       \
} while (0)

#define REPORT_INNER_ERROR(...)
#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)    \
do {                                                            \
    OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__); \
    REPORT_INNER_ERROR(ERR_CODE_STR, FMT, ##__VA_ARGS__);       \
} while (0)

#define REPORT_CALL_ERROR(...)
#define OPS_CALL_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)     \
do {                                                            \
    OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__); \
    REPORT_CALL_ERROR(ERR_CODE_STR, FMT, ##__VA_ARGS__);        \
} while (0)

#define OPS_LOG_STUB_D(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_DEBUG, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_I(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_INFO, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_W(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_WARN, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_E(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_EVENT(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_EVENT, OPS_DESC, FMT, ##__VA_ARGS__)
