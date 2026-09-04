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

export CURRENT_PATH=$(dirname ${BASH_SOURCE[0]})
export SCRIPT_PATH=$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)
export CODE_ROOT_PATH=$(cd "$(dirname ${BASH_SOURCE[0]})/.."; pwd)
export GTS_CUSTOM_OPP_PATH=${CODE_ROOT_PATH}/vendors/gtsops
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GTS_CUSTOM_OPP_PATH}/op_api/lib
export PYTHONPATH=${CODE_ROOT_PATH}/op-speed:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}:${GTS_CUSTOM_OPP_PATH}

function main() {
  while [ $# -gt 0 ]; do
    case $1 in
      --set-*)
        export GTS_ARGS="$GTS_ARGS $1 $2"
        shift 1
        ;;
      --enable-gts-*)
          export GTS_ENABLE_OP_$(echo "$1" | sed "s|--enable-gts-||" | tr 'a-z' 'A-Z' | tr '-' '_')=true
          export GTS_DISABLE_OP_$(echo "$1" | sed "s|--enable-gts-||" | tr 'a-z' 'A-Z' | tr '-' '_')=false
          ;;
      --disable-gts-*)
          export GTS_ENABLE_OP_$(echo "$1" | sed "s|--disable-gts-||" | tr 'a-z' 'A-Z' | tr '-' '_')=false
          export GTS_DISABLE_OP_$(echo "$1" | sed "s|--disable-gts-||" | tr 'a-z' 'A-Z' | tr '-' '_')=true
          ;;
    esac
    shift
  done
  env | grep "GTS_ENABLE_OP"
}

function enable_malloc() {
  local malloc_path=${CODE_ROOT_PATH}/third_party/
  [ ! -d "${malloc_path}/malloc" ] && malloc_path=/usr/local/
  [ ! -d "${malloc_path}/malloc" ] && malloc_path=${GTS_INSTALL_PATH}

  local malloc_file=$(find ${malloc_path}/malloc/$1/ -type f -name "lib${1}*.so*" 2>/dev/null | head -n 1)
  if [ -f "${malloc_file}" ]; then
    case ${1} in
        mimalloc)
            #export MIMALLOC_VERBOSE=1
            export MIMALLOC_ALLOW_LARGE_OS_PAGES=1
            ;;
        tcmalloc)
          # 释放策略：快速回收空闲内存，防止物理内存吃满
          export TCMALLOC_RELEASE_RATE=10.0
          export TCMALLOC_AGGRESSIVE_MEMORY_DEALLOCATION=1

          # 线程缓存：适当提高，应对大量 Python 线程及 NPU 回调线程
          export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=1073741824   # 1 GB
          export TCMALLOC_PER_THREAD_CACHE_BYTES=8388608            # 8 MB

          # 批量传输优化
          export TCMALLOC_TRANSFER_NUM_OBJ=256
          ;;
    esac

    export LD_PRELOAD=${malloc_file}:${LD_PRELOAD}
    echo "====[malloc]=== $1 enable LD_PRELOAD=${LD_PRELOAD}"
    return
  fi

  echo "===[malloc]=== $1 is not found, please check!!!"
}

function replace_transformer_ops() {
  test ! -d ${CURRENT_PATH}/package/$1 && echo "=== [ERROR] package $1 is not exist" && exit 1
  bash ${CURRENT_PATH}/custom_transformer/install.sh $1 /usr/local/Ascend/vendors/custom_transformer/ "${2:-sparse_attn_sharedkv_grad}"
}

export PYTHONDONTWRITEBYTECODE=1
main --enable-gts-bind-cpus --enable-gts-swiglu1 --enable-gts-swiglu1-probs --enable-gts-permute --enable-gts-index-muls
# 其他暂不开启特性：--enable-gts-rms0 --enable-gts-indexer-q-detach --enable-gts-rope --enable-gts-compressor0 --enable-gts-make-chunk-sort-map

enable_malloc tcmalloc # support: mimalloc tcmalloc jemalloc
