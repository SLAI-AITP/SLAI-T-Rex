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
export GTSBUILD_IGNORE_PACKAGE=${IGNORE_BUILD_PACKAGE:-false}

help() {
  echo "Usage: gts fast build tools
  bash build.sh [ops(op)|speed(sp)]
    --version             package version, default 1.0
    -o|--ops \"a;b;c\"    build ops"
  exit 0
}

export BUILD_ARGS=""
export BUILD_OPS=false
export PACKAGE_VERSION=1.0
export ASCEND_OP_LIST=""
unset CPLUS_INCLUDE_PATH
unset CXXFLAGS
parse_opts() {
  [ $# -eq 0 ] && help

  while [ $# -gt 0 ]; do
    case "$1" in
      init)
        init
        ;;
      clean)
        clean
        ;;
      ops|op)
        export BUILD_OPS=true
        ;;
      speed|sp)
        export BUILD_SPEED=true
        ;;
      malloc)
        export BUILD_MALLOC=true
        ;;
      all)
        export BUILD_OPS=true
        export BUILD_SPEED=true
        export BUILD_MALLOC=true
        ;;
      --version)
        export PACKAGE_VERSION=$2
        shift
        ;;
      -o|--op|--ops)
        [ ! -z ${BUILD_OPS_LIST} ] && export BUILD_OPS_LIST="${BUILD_OPS_LIST};$2" || export BUILD_OPS_LIST="$2"
        shift
        ;;
      --soc)
        export BUILD_SOC=$(echo "$2" | sed "s|--with-||" | tr 'A-Z' 'a-z' | tr '-' '_')
        [ "#${BUILD_SOC:0:6}" != "#ascend" ] && BUILD_SOC="ascend${BUILD_SOC}"
        BUILD_SOC=$(echo ${BUILD_SOC} | sed "s|\(ascend[0-9_]\{3\}[a-z]\{0,1\}\)[0-9]\{1\}|\1|")
        [ $(echo ${BUILD_SOC} | grep -c "^ascend[0-9_]\{3\}[a-z_]\{0,1\}") -ne 1 ] && echo "invalid soc: ${BUILD_SOC}, support: 310p,310P,ascend310p,Ascend910B3,ascend910_93" && exit 1
        shift
        ;;
      *)
        help
        ;;
    esac
    shift
  done

  export GTS_BUILD_TARGET=${GTS_BUILD_TARGET:-build_out}
  [ $(echo ${BUILD_OPS_LIST} | grep -c "[A-Z]") -ne 0 ] && echo "please check build ops:${BUILD_OPS_LIST}, eg:swi_glu" && exit 1
  return 0
}

print_log()
{
  [ -z "${begintime}" ] && begintime=$(date +%s)

  if [ "$1" == "begin" ]; then
    shift
    echo -e ">>>>>>>> begin $*" >> ${LOG_FILE}
  else
    Status="\e[31merror\e[0m"
    [ $1 -eq 0 ] && Status="\e[32msuccess\e[0m"
    shift
    echo -e ">>>>>>>> $* ${Status}, Duration \e[31m$[ $(date +%s) - begintime ]\e[0ms" >> ${LOG_FILE}
  fi

  begintime=$(date +%s)
}

clean()
{
  cd ${CODE_ROOT_PATH}/csrc/
  [ -d "${GTS_BUILD_TARGET}" ] && rm -fr ${GTS_BUILD_TARGET} ]
  cd ${CODE_ROOT_PATH}/op-speed
  [ -d "build" ] && rm -fr build ]
  [ -d "dist" ] && rm -fr dist ]
  find . -type f -name "*.so" | xargs rm -f
  find . -type d -name "*gtsspeed.egg-info" | xargs rm -fr
  echo "clean success"
  exit 0
}

build_ops()
{
  print_log "begin" "building ops"
  cd ${CODE_ROOT_PATH}/csrc/
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  unset ASCEND_CUSTOM_OPP_PATH
  local install_path=${CODE_ROOT_PATH}
  export ASCEND_CUSTOM_OPP_PATH=${install}
  bash build.sh "${BUILD_ARGS}"
  print_log $? "built ops"

  local vendor_name=$(grep -w -A 2 "vendor_name" ${CODE_ROOT_PATH}/csrc/CMakePresets.json | grep -w "value" | awk -F\" '{print $(NF-1)}')
  local linux_kernel_name=$(cat /etc/os-release | grep -w "ID" | sed "s|\"||g" | awk -F= '{print $NF}')
  local package_name=${vendor_name}_${linux_kernel_name}_$(uname -i)
  bash ${GTS_BUILD_TARGET}/${package_name}*.run --install-path=${install_path}
  mv ${GTS_BUILD_TARGET}/${package_name}*.run ${GTS_BUILD_TARGET}/${package_name}_${PACKAGE_VERSION}.run
  print_log $? "installed ops"
}

build_malloc()
{
  cd ${CODE_ROOT_PATH}
  test -d ${CODE_ROOT_PATH}/thrid_party/malloc/tcmalloc && echo "ignore tcmalloc build" && return 0

  test -d thrid_party || mkdir -p thrid_party
  cd thrid_party
  if [ ! -d ${gperftools-2.18.1} ]; then
    git clone --branch gperftools-2.18.1 https://github.com/gperftools/gperftools.git gperftools-2.18.1
    cd gperftools-2.18.1
    ./configure --prefix ${CODE_ROOT_PATH}/thrid_party/malloc/tcmalloc --enable-minimal
    make install -j80
  fi
}

build_op_speed()
{
  print_log "begin" "building op_speed"
  cd ${CODE_ROOT_PATH}/op-speed
  bash build.sh
  print_log $? "built op_speed"
}

main()
{
  local LOG_FILE=${CODE_ROOT_PATH}/logs/.$(date -d today +'%Y%m%d%H%M%S.%6N').log
  test -d ${CODE_ROOT_PATH}/logs/ || mkdir -p ${CODE_ROOT_PATH}/logs/
  parse_opts $@

  background_pids=""
  [ "${BUILD_OPS}" == "true" ] && {
    build_ops &
    background_pids="$! ${background_pids}"
  }

  [ "${BUILD_SPEED}" == "true" ] && build_op_speed
  [ "${BUILD_MALLOC}" == "true" ] && build_malloc
  for pid in ${background_pids}; do
    wait ${pid}
  done

  echo "----------------path----------------"
  echo "${BUILD_PATH}"
  echo "----------------build summary----------------"
  test ! -f ${LOG_FILE} || cat ${LOG_FILE}
}

main $@
