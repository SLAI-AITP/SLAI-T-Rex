# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(gtsspeed)

set(CMAKE_CXX_STANDARD 17)
option(USE_CXX11_ABI "Build CXX11 ABI." ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC -fwrapv -Wall -Wno-sign-compare -Wno-deprecated-declarations -Wno-return-type -Wno-enum-compare -Wno-unused-function -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" -DPYBIND11_STDLIB=\\\"_libstdcpp\\\" -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_EXTENSION_NAME=gtsops")
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__=\\\"$(notdir $<)\\\"")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__=\\\"$(shell basename $<)\\\"")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__=\\\"$(notdir $(abspath $<))\\\"")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -D__FILENAME__=\\\"$(notdir $(abspath $<))\\\"")
#add_compile_options(-D__FILENAME__=\"$(notdir $(abspath $@))\")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__=\\\"csrc\\\"")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+i8mm+dotprod+fp16fml+sve")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-sanitize=alignment -fsanitize-recover=address,all -gdwarf-3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -ftrapv -fstack-protector-strong")
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,--strip-all")
endif ()
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SKIP_RPATH TRUE)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}" CACHE PATH "Library output dir.")

if (NOT SITE_PACKAGES_PATH)
    set(SITE_PACKAGES_PATH "$ENV{SITE_PACKAGES_PATH}")
endif ()

message("=========CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")
message("=========SITE_PACKAGES_PATH:${SITE_PACKAGES_PATH}")
message("=========PYTHON_INCLUDE_PATH:$ENV{PYTHON_INCLUDE_PATH}")
message("=========LIBRARY_OUTPUT_PATH:${LIBRARY_OUTPUT_PATH}")

if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTSOPS_BUILD_RELEASE=1")
else()
    set(ALPHA_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/alpha/*.cpp")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/inc/
    $ENV{ASCEND_HOME_PATH}/include/
    $ENV{ASCEND_HOME_PATH}/include/aclnn/
    $ENV{PYTHON_INCLUDE_PATH}/
    ${SITE_PACKAGES_PATH}/torch/include/
    ${SITE_PACKAGES_PATH}/torch/include/torch/csrc/api/include
    ${SITE_PACKAGES_PATH}/torch_npu/include
    ${SITE_PACKAGES_PATH}/torch_npu/include/third_party/acl/inc/
)

file(GLOB GTSOPS_SRCS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/inc/*.cpp
    ${ALPHA_SRCS}
    ${MINDIE_TEST_SRCS}
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64/
    ${SITE_PACKAGES_PATH}/torch/lib
    ${SITE_PACKAGES_PATH}/torch_npu/lib
)

message("=========${GTSOPS_SRCS}")
add_library(gtsops SHARED ${GTSOPS_SRCS})
target_include_directories(gtsops SYSTEM PUBLIC)
target_compile_options(gtsops PUBLIC ${GTSOPS_DEFINITIONS})
target_link_libraries(gtsops c10 torch torch_cpu torch_python torch_npu c_sec ${MINDIE_TEST_LIBS})
install(TARGETS gtsops LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib.linux-*/gtsspeed)