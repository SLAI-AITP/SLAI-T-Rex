# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
string(TIMESTAMP Year "%Y")
string(TIMESTAMP Month "%m")
string(TIMESTAMP Day "%d")
set(FullDate "${Year}-${Month}-${Day}")

if (EXISTS $ENV{ASCEND_HOME_PATH}/tools/ascend_project/cmake/)
  set(CMAKE_MAKESELF_PATH $ENV{ASCEND_HOME_PATH}/tools/ascend_project/cmake/util/makeself/)
elseif (EXISTS $ENV{ASCEND_HOME_PATH}/tools/op_project_template/ascendc/customize/cmake/)
  set(CMAKE_MAKESELF_PATH $ENV{ASCEND_HOME_PATH}/tools/op_project_template/ascendc/customize/cmake/util/makeself/)
elseif (EXISTS $ENV{ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake/)
  set(CMAKE_MAKESELF_PATH $ENV{ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake/util/makeself/)
endif()

execute_process(COMMAND chmod +x ${CMAKE_MAKESELF_PATH}/makeself.sh)
execute_process(COMMAND ${CMAKE_MAKESELF_PATH}/makeself.sh
        --header ${CMAKE_MAKESELF_PATH}/makeself-header.sh
        --help-header ./help.info
        --nocomp --nomd5 --nocrc --target makeself-gtsops --packaging-date ${FullDate}
        ./ ${CPACK_PACKAGE_FILE_NAME} version:1.0 ./install.sh
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
)

if (NOT "${EXEC_RESULT}x" STREQUAL "0x")
  message(FATAL_ERROR "CPack Command error: ${EXEC_RESULT}\n${EXEC_ERROR}")
endif()
execute_process(COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES} ${CPACK_PACKAGE_DIRECTORY}/
                COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES} to ${CPACK_PACKAGE_DIRECTORY}/"
                WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
)
