# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
if(ENABLE_CROSS_COMPILE)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(CROSS_COMPILE_PLATFORM aarch64)
    else()
        set(CROSS_COMPILE_PLATFORM x86_64)
    endif()
    set(PLATFORM ${CMAKE_SYSTEM_PROCESSOR})
    set(CMAKE_COMPILE_COMPILER_LIBRARY ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/devlib/linux/${CROSS_COMPILE_PLATFORM}/)
    set(CMAKE_COMPILE_RUNTIME_LIBRARY ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/devlib/${CROSS_COMPILE_PLATFORM}/)
    set(CMAKE_SYSTEM_PROCESSOR ${CROSS_COMPILE_PLATFORM})
    set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_CROSS_PLATFORM_COMPILER})
else()
    set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})
endif()

set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")

set(ASCENDC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ASCENDC_CMAKE_UTIL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util)

if (NOT DEFINED vendor_name)
    set(vendor_name customize CACHE STRING "")
endif()
if (NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "")
endif()
if (NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
    set(ASCEND_PYTHON_EXECUTABLE python3 CACHE STRING "")
endif()
if (NOT DEFINED ASCEND_COMPUTE_UNIT)
    message(FATAL_ERROR "ASCEND_COMPUTE_UNIT not set in CMakePreset.json ! 
")
endif()
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen)
set(ASCEND_FRAMEWORK_TYPE tensorflow)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
set(IMPL_OUT_DIR ${CMAKE_BINARY_DIR}/impl)
set(BINARY_OUT_DIR ${CMAKE_BINARY_DIR}/binary)
execute_process(COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
                COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS})

# Device tiling sink support - locate CANN cmake scripts
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/cmake/)
    set(ASCENDC_CMAKE_SCRIPTS_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/cmake/)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_template/ascendc/customize/cmake/)
    set(ASCENDC_CMAKE_SCRIPTS_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_template/ascendc/customize/cmake/)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize/cmake/)
    set(ASCENDC_CMAKE_SCRIPTS_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize/cmake/)
else()
    message(WARNING "ASCENDC_CMAKE_SCRIPTS_PATH not found, device tiling sink will not be available")
endif()

set(ASCENDC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
