# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
add_library(intf_pub INTERFACE)
target_compile_options(intf_pub INTERFACE
    -fPIC
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++1z>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-ftrapv -fstack-check>
    $<$<COMPILE_LANGUAGE:C>:-pthread -Wfloat-equal -Wshadow -Wformat=2 -Wno-deprecated -Wextra>
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(intf_pub INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)
target_include_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCENDC_INCLUDE_DIR}/kernel)
target_link_options(intf_pub INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
    $<$<CONFIG:Release>:-s>
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
