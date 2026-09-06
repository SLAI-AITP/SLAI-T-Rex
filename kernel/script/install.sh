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

# install gtsops
function install_gtsops() {
  temp_dir=$(mktemp -d)
  cd ${temp_dir}
  bash ${CODE_ROOT_PATH}/package/gtsops_*.run --install-path=${CODE_ROOT_PATH}
  cd -
  rm -fr ${temp_dir}
}

main() {
  install_gtsops
}

main