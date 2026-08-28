/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_PYBIND_H
#define GTSOPS_PYBIND_H
#include <functional>
#include <vector>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <syslog.h>

inline void fsMakeSyslog(const char *content)
{
    if (content != nullptr) {
        syslog(LOG_MAKEPRI(LOG_LOCAL0, LOG_INFO), "%s", content);
    }
}

#pragma GCC visibility push(hidden)
class Pybind final {
    using BindFunc = std::function<void(::pybind11::module&)>;

public:
    Pybind() = default;
    ~Pybind() = default;
    static Pybind& GetInstance()
    {
        static Pybind instance;
        return instance;
    }

    bool Add(const std::string& name, const BindFunc& func)
    {
        funcs.push_back(func);
        return true;
    }

    void Init(::pybind11::module& m) const
    {
        for (const auto& func : funcs) {
            func(m);
        }
    }

    void ReadJitCompile()
    {
        py::module torch_npu = pybind11::module::import("torch_npu");
        py::module torch = pybind11::module::import("torch.npu");
        py::object get_option = torch.attr("is_jit_compile_false");
        c10::optional<bool> jitCompile = get_option().cast<c10::optional<bool>>();
        if (jitCompile.has_value()) {
            isJitCompileEnable = !jitCompile.value();
        } else {
            isJitCompileEnable = false;
        }
    }

    bool IsJitCompileEnable()
    {
        static bool isRead = false;
        if (!isRead) {
            ReadJitCompile();
            isRead = true;
        }
        return isJitCompileEnable;
    }

private:
    std::vector<BindFunc> funcs;
    bool isJitCompileEnable = false;
};

#define PYBIND_FUNC(...) [__VA_ARGS__](::pybind11::module_ & m)
#define PYBIND_METHOD(name, func) static bool __##gtsops_bind_##__COUNTER__ = Pybind::GetInstance().Add(name, func)
#define IS_ACLNN_MODE() (Pybind::GetInstance().IsJitCompileEnable() == false)
#define IS_GRAPH_MODE() (Pybind::GetInstance().IsJitCompileEnable() == true)

#endif // GTSOPS_PYBIND_H
