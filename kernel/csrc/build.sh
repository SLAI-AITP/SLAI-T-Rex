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
script_path=$(realpath $(dirname $0))
code_root_path=$(cd "$(dirname $0)/../"; pwd)

echo ASCEND_COMPUTE_UNIT=$ASCEND_COMPUTE_UNIT
echo ASCEND_HOME_PATH=$ASCEND_HOME_PATH
source  $ASCEND_HOME_PATH/bin/setenv.bash || echo "0"
export GTS_BUILD_TARGET=${GTS_BUILD_TARGET:-build_out}
mkdir -p ${GTS_BUILD_TARGET} -m 750
rm -rf ${GTS_BUILD_TARGET}/*
test -d makeself-gtsops && rm -rf makeself-gtsops
cd ${GTS_BUILD_TARGET}

begintime=$(date +%s)
function steplog() {
    [ ! -z ${time} ] && echo -e ">>>>>>>>>>>>>>Duration $* \e[31m$[ $(date +%s) - time ]\e[0ms"

    echo -e "\n<<<<<<<<<<<<<<<<<<<\e[1;35m$*\e[0m<<<<<<<<<<<<<<<<<<<"
    time=$(date +%s)
}

CMAKE_OPTS="${CMAKE_OPTS} -DBUILD_ENABLE_LOGS=OFF"
if [ ! -z "${BUILD_OPS_LIST}" ]; then
  CMAKE_OPTS="${CMAKE_OPTS} -DBUILD_OPS_LIST=${BUILD_OPS_LIST}"
fi

if [ ! -z "${BUILD_SOC}" ]; then
  [ $(echo ${BUILD_SOC} | grep -c "^ascend[0-9_]\{3\}[a-z_]\{0,1\}") -ne 1 ] && echo "invalid soc: ${BUILD_SOC}, support: ascend310p, ascend910b, ascend910_93" && exit 1
  CMAKE_OPTS="${CMAKE_OPTS} -DASCEND_COMPUTE_UNIT=${BUILD_SOC}"
fi

if [ "#${BUILD_TYPE}" == "#debug" ]; then
  CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_BUILD_TYPE=debug"
else
  CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_BUILD_TYPE=release"
fi

if [ "#${GTS_BUILD_TARGET}" != "#" ]; then
  CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_INSTALL_PREFIX=${script_path}/${GTS_BUILD_TARGET} -B ."
fi

steplog "stpe1: load cmake"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
echo --preset=default -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH -DASCEND_COMPUTE_UNIT=$ASCEND_COMPUTE_UNIT ${CMAKE_OPTS}
if [ "z$ASCEND_COMPUTE_UNIT" != "z" ]; then
    cmake .. --preset=default -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH -DASCEND_COMPUTE_UNIT=$ASCEND_COMPUTE_UNIT ${CMAKE_OPTS}
else
    cmake .. --preset=default -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH ${CMAKE_OPTS}
fi
target=package
if [ "$1"x != ""x ]; then target=$1; fi

steplog "step2: build host package"
cmake --build . --target $target -j64 --verbose
if [ $? -ne 0 ]; then exit 1; fi

package_name=$(grep -A 2 "vendor_name" ../CMakePresets.json | tail -n 1 | grep "value" | awk -F\" '{print $(NF-1)}')
if [ $target = "package" ]; then
    steplog "step3: install host package"
    ./${package_name}_*.run
    if [ $? -ne 0 ]; then exit 1; fi
    steplog "step4: build binary"
    cmake --build . --target binary -j64 --verbose
    if [ $? -ne 0 ]; then exit 1; fi
    steplog "step5: build host+device package"
    rm ${package_name}_*.run
    cmake --build . --target $target -j64 --verbose
fi

echo -e "Total Spend: \e[31m$[ $(date +%s) - begintime ]\e[0ms"