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

#include "../utils/slog.h"
#include "inner/dfx_base.h"

#ifdef ENABLE_LOG_ON
#define OP_LOG(fmt, args...)    printf(fmt "\n", ##args)
#else
#define OP_LOG(fmt, args...)    dlog_debug(OP, "",  ##args)
#endif

/* OP日志 */
#define OP_LOGD(opname, ...) dlog_debug(OP, __VA_ARGS__)
#define OP_LOGI(opname, ...) dlog_info(OP, __VA_ARGS__)
#define OP_LOGW(opname, ...) dlog_warn(OP, __VA_ARGS__)
#define OP_LOGE(opname, ...) dlog_error(OP, __VA_ARGS__)
#define OP_LOG_D OP_LOGD
#define OP_LOG_I OP_LOGI
#define OP_LOG_E OP_LOGE

/* 基础日志 */
#define OPS_LOG_D(OPS_DESC, ...) dlog_debug(OP, __VA_ARGS__)
#define OPS_LOG_I(OPS_DESC, ...) dlog_info(OP, __VA_ARGS__)
#define OPS_LOG_W(OPS_DESC, ...) dlog_warn(OP, __VA_ARGS__)
#define OPS_LOG_E(OPS_DESC, ...) dlog_error(OP, __VA_ARGS__)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) OPS_LOG_STUB_E(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_EVENT(OPS_DESC, ...) OPS_LOG_STUB_EVENT(OPS_DESC, __VA_ARGS__)

/* 全量日志
 * 输出超长日志, 若日志超长, 则会被分为多行输出 */
#define OPS_LOG_D_FULL(OPS_DESC, ...) OPS_LOG_D(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I_FULL(OPS_DESC, ...) OPS_LOG_I(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W_FULL(OPS_DESC, ...) OPS_LOG_W(OPS_DESC, __VA_ARGS__)

/* 条件日志 */
#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_D(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_I(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_W(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_E(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_EVENT(OP_DESC, __VA_ARGS__), EXPR)

#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, EXPR)                     \
if (__builtin_expect((PTR) == nullptr, 0)) {                       \
    OPS_LOG_STUB_E(OPS_DESC, "%s is nullptr!", #PTR);              \
    OPS_CALL_ERR_STUB("EZ9999", OPS_DESC, "%s is nullptr!", #PTR); \
    EXPR;                                                          \
}

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    OP_LOGE(name, "%s is nullptr!", #ptr);                                                       \
    return ge::GRAPH_FAILED;                                                                     \
}

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)
#endif
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                  \
static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");    \
do {                                                                                                            \
    if (unlikely(condition)) {                                                                                  \
        OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                   \
        return return_value;                                                                                    \
    }                                                                                                           \
} while (0)

#define OP_LOGW_IF(condition, op_name, fmt, ...)                                                                \
static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");    \
do {                                                                                                            \
    if (unlikely(condition)) {                                                                                  \
        OP_LOGW(op_name, fmt, ##__VA_ARGS__);                                                                   \
    }                                                                                                           \
} while (0)

#define OP_LOGI_IF_RETURN(condition, return_value, op_name, fmt, ...)                                           \
static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");    \
do {                                                                                                            \
    if (unlikely(condition)) {                                                                                  \
        OP_LOGI(op_name, fmt, ##__VA_ARGS__);                                                                   \
        return return_value;                                                                                    \
    }                                                                                                           \
} while (0)

#define OPS_CHECK(COND, LOG_FUNC, EXPR) \
if (COND) {                             \
    LOG_FUNC;                           \
    EXPR;                               \
}
#define OP_CHECK OPS_CHECK
#define OP_CHECK_IF OPS_CHECK
#define OP_CHECK_NULL_WITH_CONTEXT