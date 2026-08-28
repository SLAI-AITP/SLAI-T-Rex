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


if [[ -z "$1" ]]; then
    echo -e "[ERROR] No source dir provided"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo -e "[ERROR] No destination dir provided"
    exit 1
fi

src=$1
dest_file=$2/npu_supported_ops.json

if [ -f "$dest_file" ];then
    chmod u+w $dest_file
fi

echo $*

add_ops() {
    name=$1
    isHeavy=$2
    file=$3
    grep -w "\"$name\"" ${file} >/dev/null
    if [ $? == 0 ];then
        return
    fi
    echo "  \"${name}\": {" >> ${file}
    echo "    \"isGray\": false," >> ${file}
    echo "    \"isHeavy\": ${isHeavy}" >> ${file}
    echo "  }," >> ${file}
}

echo "{" > ${dest_file}
ini_files=$(find ${src} -name "*.ini")
for file in ${ini_files} ; do
    name=$(grep '^\[' ${file} | sed 's/\[//g' | sed 's/]//g' | sed 's/\r//g')
    grep 'heavyOp.flag' ${file} >/dev/null
    if [ $? == 0 ];then
        isHeavy=$(grep 'heavyOp.flag' ${file} | awk -F= '{print $2}')
    else
        isHeavy="false"
    fi
    for op in ${name} ; do
        add_ops ${op} "false" ${dest_file}
    done
done
echo "}" >> ${dest_file}
file_count=$(cat ${dest_file} | wc -l)
line=$(($file_count-1))
sed -i "${line}{s/,//g}" ${dest_file}

chmod 640 "${dest_file}"
echo -e "[INFO] Succed generated ${dest_file}"

exit 0

