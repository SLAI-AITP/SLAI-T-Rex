# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
include(ExternalProject)

function(npu_op_device_tiling_library target_name target_type)
    if (NOT DEFINED ASCENDC_CMAKE_SCRIPTS_PATH)
        message(WARNING "ASCENDC_CMAKE_SCRIPTS_PATH not defined, skipping device tiling library ${target_name}")
        return()
    endif()
    message(STATUS "Ascendc device tiling library generating: ${target_name}")
    string(TOUPPER ${target_type} _upper_target_type)
    set(support_types SHARED STATIC)
    if(NOT _upper_target_type IN_LIST support_types)
        message(FATAL_ERROR "target_type ${_upper_target_type} does not support, the support list is ${support_types}")
    endif()
    set(SOURCES)
    foreach(_source ${ARGN})
        get_filename_component(absolute_source "${_source}" ABSOLUTE)
        list(APPEND SOURCES ${absolute_source})
    endforeach()
    string(REPLACE ";" " " EP_SOURCES "${SOURCES}")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E echo "cmake_minimum_required(VERSION 3.16.0)\nproject(cust_tiling_sink)
        \ninclude_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
        \ninclude(${ASCENDC_CMAKE_SCRIPTS_PATH}/device_task.cmake)
        \ntarget_compile_options(cust_opmaster PRIVATE -std=c++1z -Wno-builtin-macro-redefined)\n"
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
        RESULT_VARIABLE result
    )

    ExternalProject_Add(tiling_sink_task
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
            -DTARGET=${target_name}
            -DOPTION=${_upper_target_type}
            -DSRC=${EP_SOURCES}
            <SOURCE_DIR>
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${ASCENDC_INSTALL_PREFIX}
        INSTALL_COMMAND ""
        BUILD_ALWAYS TRUE
    )
    ExternalProject_Get_Property(tiling_sink_task BINARY_DIR)
    set(TILINGSINK_LIB "")
    if ("${_upper_target_type}" STREQUAL "SHARED")
        set(TILINGSINK_LIB "${BINARY_DIR}/libcust_opmaster.so")
    else()
        set(TILINGSINK_LIB "${BINARY_DIR}/libcust_opmaster.a")
    endif()
    set_property(GLOBAL PROPERTY ASCENDC_DEVICE_SINK_TARGET_OUTPUT ${TILINGSINK_LIB})
    get_property(tmp_device_sink_target GLOBAL PROPERTY ASCENDC_DEVICE_SINK_TARGET)
    list(APPEND tmp_device_sink_target ${target_name})
    set_property(GLOBAL PROPERTY ASCENDC_DEVICE_SINK_TARGET ${tmp_device_sink_target})
    set_property(GLOBAL PROPERTY _ASC_TGT_${target_name}_TYPE "DEVICE_SINK")
endfunction()

function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++17 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0 -DLOG_CPP
                  -I ${ASCENDC_INCLUDE_DIR} -I ${ASCENDC_INCLUDE_DIR}/tiling -I ${ASCENDC_INCLUDE_DIR}/kernel
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR;IGNORE" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)

  # [add for gtsops] check config is exist
  execute_process(COMMAND test ! -f ${OPINFO_OPS_INFO} RESULT_VARIABLE EXEC_RESULT)
  set(${OPINFO_IGNORE} "false" PARENT_SCOPE)
  if (NOT ${EXEC_RESULT})
      set(${OPINFO_IGNORE} "true" PARENT_SCOPE)
      return()
  endif ()

  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path} -m 750
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASCENDC_CMAKE_UTIL_DIR}/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  install(FILES ${OPINFO_OUTPUT}
          DESTINATION ${OPINFO_INSTALL_DIR}
  )
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
       "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

set(OP_COMPILE_OPTION "")
function(add_ops_impl_target)
    cmake_parse_arguments(OPIMPL "" "OPS_INFO;COMPUTE_UNIT;TOP_DIR" "OPS_BATCH;OPS_ITERATE;OP_NAME_LIST" ${ARGN})

    set(_OUT_DIR ${IMPL_OUT_DIR}/${OPIMPL_COMPUTE_UNIT})
    foreach(op_name ${OPIMPL_OP_NAME_LIST})
        set(TARGET_NAME ascendc_impl_gen_${OPIMPL_COMPUTE_UNIT}_${op_name})
        if (${op_name} STREQUAL "flash_attention_s" OR ${op_name} STREQUAL "incre_flash_attention_s")
            set(OP_COMPILE_OPTION "-mllvm -cce-aicore-fp-ceiling=2,")
        else()
          set(OP_COMPILE_OPTION "")
        endif ()
        add_custom_command(OUTPUT ${_OUT_DIR}/dynamic/${op_name}.py
            COMMAND mkdir -p ${_OUT_DIR} -m 750
            COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_impl_build.py
                    ${OPIMPL_OPS_INFO}
                    \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
                    ${OPIMPL_TOP_DIR}/${op_name}/op_kernel
                    ${_OUT_DIR}
                    ${ASCEND_AUTOGEN_PATH}
                    ${OP_COMPILE_OPTION}
            DEPENDS ${OPIMPL_OPS_INFO}
                    ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_impl_build.py
        )

        add_custom_target(${TARGET_NAME} ALL
            DEPENDS ${_OUT_DIR}/dynamic/${op_name}.py
        )
    endforeach()

    if (${ENABLE_SOURCE_PACKAGE})
        install(DIRECTORY ${_OUT_DIR}/dynamic
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl
        )
    endif()
endfunction()

function(generate_compilation_command)
    cmake_parse_arguments(GEN "" "OPS_INFO;COMPUTE_UNIT" "" ${ARGN})

    set(_OUT_DIR ${BINARY_OUT_DIR}/${GEN_COMPUTE_UNIT})
    set(GEN_OUT_DIR ${_OUT_DIR}/gen)

    file(MAKE_DIRECTORY ${GEN_OUT_DIR})
    
    execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
                          ${GEN_OPS_INFO} ${GEN_OUT_DIR} ${GEN_COMPUTE_UNIT}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
    )
    if (${EXEC_RESULT})
        message("ops binary compile scripts gen info: ${EXEC_INFO}")
        message("ops binary compile scripts gen error: ${EXEC_ERROR}")
        message(FATAL_ERROR "ops binary compile scripts gen failed!")
    endif()
endfunction()

function(add_bin_compile_target)
    cmake_parse_arguments(BINCMP "" "COMPUTE_UNIT;TOP_DIR" "" ${ARGN})

    set(_INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel)
    set(_OUT_DIR ${BINARY_OUT_DIR}/${BINCMP_COMPUTE_UNIT})

    set(BIN_OUT_DIR ${_OUT_DIR}/bin)
    set(GEN_OUT_DIR ${_OUT_DIR}/gen)
    file(MAKE_DIRECTORY ${BIN_OUT_DIR})

    file(GLOB bin_scripts ${GEN_OUT_DIR}/*.sh)
    foreach(bin_script ${bin_scripts})
        get_filename_component(bin_file ${bin_script} NAME_WE)
        string(REPLACE "-" ";" bin_sep ${bin_file})
        list(GET bin_sep 0 op_type)
        list(GET bin_sep 1 op_file)
        list(GET bin_sep 2 op_index)

        set(TARGET_NAME ascendc_bin_${BINCMP_COMPUTE_UNIT}_${op_file})
        if(NOT TARGET ${TARGET_NAME})
            add_custom_target(${TARGET_NAME})
            add_dependencies(ops_config_gen_${BINCMP_COMPUTE_UNIT} ${TARGET_NAME})

            set(SRC_OUT_DIR ${_OUT_DIR}/src/${op_file})
            file(MAKE_DIRECTORY ${SRC_OUT_DIR})
            file(MAKE_DIRECTORY ${BIN_OUT_DIR}/${op_file})

            add_custom_target(${TARGET_NAME}_copy
                COMMAND cp -rf ${BINCMP_TOP_DIR}/${op_file}/op_kernel/*.* ${SRC_OUT_DIR}
                COMMAND cp -rf ${BINCMP_TOP_DIR}/${op_file}/op_kernel/*.h ${SRC_OUT_DIR} 2>/dev/null || true
                COMMAND cp -rn ${BINCMP_TOP_DIR}/${op_file}/op_kernel/arch* ${SRC_OUT_DIR} 2>/dev/null || true
                # Copy ALL op_kernel subdirs, not only arch*: some ops nest kernels under
                # non-arch dirs (e.g. a16w4_msd/, gmm_infra/) and #include "<subdir>/xxx.h".
                # The arch* line above covers arch* only; this glob picks up the rest (-n: keep arch* copy).
                COMMAND cp -rn ${BINCMP_TOP_DIR}/${op_file}/op_kernel/*/ ${SRC_OUT_DIR} 2>/dev/null || true
                COMMAND cp -rf ${IMPL_OUT_DIR}/${BINCMP_COMPUTE_UNIT}/dynamic/${op_file}.py      ${SRC_OUT_DIR}/${op_type}.py
            )
    
            add_dependencies(${TARGET_NAME}_copy ascendc_impl_gen_${BINCMP_COMPUTE_UNIT}_${op_file})
            
            install(DIRECTORY ${BIN_OUT_DIR}/${op_file}
                DESTINATION ${_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT} OPTIONAL
            )

            install(FILES ${BIN_OUT_DIR}/${op_file}.json
                DESTINATION ${_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}/ OPTIONAL
            )
        endif()


        add_custom_target(${TARGET_NAME}_${op_index}
                        COMMAND export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && bash ${bin_script} ${SRC_OUT_DIR}/${op_type}.py ${BIN_OUT_DIR}/${op_file}
                        WORKING_DIRECTORY ${GEN_OUT_DIR}
        )
        add_dependencies(${TARGET_NAME}_${op_index} ${TARGET_NAME}_copy)
        add_dependencies(${TARGET_NAME} ${TARGET_NAME}_${op_index})
    endforeach()
endfunction()
