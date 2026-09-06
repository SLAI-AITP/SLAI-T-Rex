# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(ascendc)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/kernel)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tiling)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wno-builtin-macro-redefined")
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

option(BUILD_ENABLE_LOGS "Build enable logs." ON)
if (BUILD_ENABLE_LOGS)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_LOG_ON=1")
endif()

# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0)

set(ALL_OP_SRC)
set(ALL_OP_LIST)
file(GLOB excludedirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${EXCLUDE_OPS_LIST})
message("*******************exclude dirs: ${excludedirs}")
set(PUBLIC_DIRECTORY cmake scripts include build_out ${excludedirs})
file(GLOB subdirectories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${BUILD_OPS_LIST})

if (NOT subdirectories)
    message(FATAL_ERROR "no op directory found. BUILD_OPS_LIST: ${BUILD_OPS_LIST}")
endif()

foreach(subdirectory ${subdirectories})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory})
        if(NOT subdirectory IN_LIST PUBLIC_DIRECTORY)
            list(APPEND ALL_OP_LIST ${subdirectory})
            file(GLOB_RECURSE OP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory}/op_host/*.cpp")
            # 忽略算子op_host/op_api目录的编译（自动生成aclnn接口）
            list(FILTER OP_SRC EXCLUDE REGEX "/op_host/op_api/")
            list(APPEND ALL_OP_SRC ${OP_SRC})
        endif()
    else()
        message(STATUS "${subdirectory} not found")
    endif()
endforeach()

message("*******************build dirs: ${ALL_OP_LIST}")
message("*******************build srcs: ${ALL_OP_SRC}")
opbuild(OPS_SRC ${ALL_OP_SRC}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

add_library(op_proto SHARED
    ${ALL_OP_SRC}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/temp/fs_make_tmp.cpp
)

target_compile_definitions(op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(op_proto PRIVATE
        -fvisibility=hidden
        -DLOG_CPP
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(op_proto PRIVATE
        ${CMAKE_COMPILE_COMPILER_LIBRARY}
        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()

target_link_libraries(op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
        dl
)
set_target_properties(op_proto PROPERTIES OUTPUT_NAME
    cust_opsproto_rt2.0
)

add_library(optiling SHARED
    ${ALL_OP_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/temp/fs_make_tmp.cpp
)

if(TARGET tiling_sink_task)
    add_dependencies(optiling tiling_sink_task)
endif()

target_compile_definitions(optiling PRIVATE
    OP_TILING_LIB
)

target_compile_options(optiling PRIVATE
    -fvisibility=hidden
    -DLOG_CPP
)

if(ENABLE_CROSS_COMPILE)
    target_link_directories(optiling PRIVATE
        ${CMAKE_COMPILE_COMPILER_LIBRARY}
        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()

target_link_libraries(optiling PRIVATE
    intf_pub
    exe_graph
    register
    tiling_api
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive
    dl
)

set_target_properties(optiling PROPERTIES OUTPUT_NAME
    cust_opmaster_rt2.0
)

# support outer opapi
include_directories($ENV{ASCEND_HOME_PATH}/include/aclnn)
set(OPAPI_L0_SRCS)
foreach(subdirectory ${subdirectories})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory}/op_api/)
        file(GLOB opapi_l2_src ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory}/op_api/level2/aclnn_*)
        file(COPY ${opapi_l2_src} DESTINATION ${ASCEND_AUTOGEN_PATH}/)

        file(GLOB opapi_l0_src ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory}/op_api/level0/*)
        list(APPEND OPAPI_L0_SRCS ${opapi_l0_src})
        file(COPY ${opapi_l0_src} DESTINATION ${ASCEND_AUTOGEN_PATH}/)
    endif()
endforeach()

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
list(APPEND aclnn_src ${CMAKE_CURRENT_SOURCE_DIR}/include/temp/fs_make_tmp.cpp)
add_library(opapi SHARED
    ${aclnn_src} ${OPAPI_L0_SRCS}
)

# [add for gts] mc2 dependences
target_include_directories(opapi PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn)
target_include_directories(opapi PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/platform)
target_include_directories(opapi PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof)
target_compile_definitions(opapi PRIVATE
    -D_FORTIFY_SOURCE=2
)

if(ENABLE_CROSS_COMPILE)
    target_link_directories(opapi PRIVATE
        ${CMAKE_COMPILE_COMPILER_LIBRARY}
        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()

target_link_libraries(opapi PRIVATE
    intf_pub
    ascendcl
    nnopbase
    $ENV{ASCEND_HOME_PATH}/lib64/libopapi.so
    -Wl,-z,relro,-z,now
)

set_target_properties(opapi PROPERTIES OUTPUT_NAME
    cust_opapi
)

add_custom_target(optiling_compat ALL
    COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:optiling>
        ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

install(TARGETS op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
install(TARGETS optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
# Package op whitelist files (*.whitelist.*) to vendor/op_whitelist/<op_name>/
foreach(op_name ${ALL_OP_LIST})
    file(GLOB WHITELIST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_host/*.whitelist.*)
    foreach(WHITELIST_FILE ${WHITELIST_FILES})
        install(FILES ${WHITELIST_FILE}
                DESTINATION packages/vendors/${vendor_name}/op_whitelist/${op_name})
    endforeach()
endforeach()
install(TARGETS opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)

# Install device tiling sink library
get_property(TILINGSINK_OUTPUT GLOBAL PROPERTY ASCENDC_DEVICE_SINK_TARGET_OUTPUT)
if(TILINGSINK_OUTPUT)
    install(FILES ${TILINGSINK_OUTPUT}
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_master_device/lib)
endif()

add_custom_target(binary)

foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
        OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
        IGNORE ignore_${compute_unit}
    )

    # [add for gts] ignore unsupportted ascend hardware
    if ("${ignore_${compute_unit}}" STREQUAL "true")
        continue()
    endif ()

    add_custom_target(ops_config_gen_${compute_unit}
        COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_ops_config.py -p ${BINARY_OUT_DIR}/${compute_unit}/bin -s ${compute_unit}
    )

    add_dependencies(binary ops_config_gen_${compute_unit})

    install(FILES ${BINARY_OUT_DIR}/${compute_unit}/bin/binary_info_config.json
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel/config/${compute_unit} OPTIONAL
    )

    add_ops_impl_target(
        TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        OP_NAME_LIST ${ALL_OP_LIST}
        COMPUTE_UNIT ${compute_unit}
        OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
    )

    if (${ENABLE_BINARY_PACKAGE})
        generate_compilation_command(
            OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            COMPUTE_UNIT ${compute_unit}
        )

        add_bin_compile_target(
            TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            COMPUTE_UNIT ${compute_unit}
        )
    endif()
endforeach()

if (${ENABLE_SOURCE_PACKAGE})
    foreach(op_name ${ALL_OP_LIST})
        file(GLOB KERNEL_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/*
        )
        install(FILES ${KERNEL_FILES}
                DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
        )
        file(GLOB INCLUDE_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
        )
        install(FILES ${INCLUDE_FILES}
                DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
        )
        file(GLOB INCLUDE_UTILS_FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/*
        )
        install(FILES ${INCLUDE_UTILS_FILES}
                DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic/utils
        )
    endforeach()
endif()

# modify vendor_name in install.sh and upgrade.sh
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/scripts -m 750
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/scripts/* ${CMAKE_BINARY_DIR}/scripts/
    COMMAND sed -i "s/vendor_name=customize/vendor_name=${vendor_name}/g" ${CMAKE_BINARY_DIR}/scripts/*
)
add_custom_target(modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh)
install(DIRECTORY ${CMAKE_BINARY_DIR}/scripts/ DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/custom.proto DESTINATION packages OPTIONAL)

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(gen_version_info ALL
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.info
        DESTINATION packages/vendors/${vendor_name}/)

# CPack config
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "${vendor_name}_${SYSTEM_INFO}.run")
set(CPACK_GENERATOR External)
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/makeself.cmake)
set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
include(CPack)
