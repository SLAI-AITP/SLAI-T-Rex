#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e
export SCRIPT_PATH=$(cd "$(dirname $0)"; pwd)
export CODE_ROOT_PATH=$(cd "$(dirname $0)/../"; pwd)

function link_op_interface() {
(
  local patch_path=${CODE_ROOT_PATH}/patch/
  test -d ${mindspeed_path}/gts || ln -s ${patch_path}/gts ${mindspeed_path}/gts
)
}

function apply_git_patch() {
  local _code_path=${opensource_path}/$1
  test ! -d ${_code_path} && echo "=== [ERROR] path ${_code_path} is not exist" && exit 1
  test ! -d ${CODE_ROOT_PATH}/patch/${1} && echo "=== [ERROR] patch/${1} not exist" && exit 1
  test -f ${_code_path}/.patch.enable && echo "skip ${1} patch" && return 0

  cd ${_code_path}
  git apply ${CODE_ROOT_PATH}/patch/${1}/*.patch
  date +%Y-%m-%d\ %H:%M:%S > ${_code_path}/.patch.enable
}

function main() {
  link_op_interface
  # apply_git_patch Megatron-LM
  apply_git_patch MindSpeed
  apply_git_patch MindSpeed-LLM
}

[ $# -eq 0 ] && echo "=== [ERROR] please input: <opensource path>" && exit 1
opensource_path=$(realpath $1)
main