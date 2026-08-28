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

export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}
export ENV_PIPELINE_STARTTIME=$(date +%Y%m%d%H%M%S)
export PYTHON_INCLUDE_PATH=$(realpath $(which python) | sed "s|bin|include|")
export SITE_PACKAGES_PATH="$(realpath $(which python) | sed "s|bin|lib|")/site-packages/"
pip3 install ${PIP_CONF} -v --no-build-isolation -e . || { echo "error: editable install failed"; exit 1; }
exit 0