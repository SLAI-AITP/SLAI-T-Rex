#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import glob
import json
import argparse
import const_var


def load_json(json_file: str):
    with open(json_file, encoding='utf-8') as file:
        json_content = json.load(file)
    return json_content


def get_specified_suffix_file(root_dir, suffix):
    specified_suffix = os.path.join(root_dir, '**/*.{}'.format(suffix))
    all_suffix_files = glob.glob(specified_suffix, recursive=True)
    return all_suffix_files


def add_dict_key(dict_to_add, key, value):
    if value is not None:
        dict_to_add[key] = value


def correct_format_mode(format_mode):
    if format_mode == 'FormatDefault':
        return 'nd_agnostic'
    if format_mode == 'FormatAgnostic':
        return 'static_nd_agnostic'
    if format_mode == 'FormatFixed':
        return 'normal'
    return format_mode


def get_input_or_output_config(in_or_out):
    param_config = {}
    add_dict_key(param_config, 'name', in_or_out.get('name'))
    add_dict_key(param_config, 'index', in_or_out.get('index'))
    add_dict_key(param_config, 'paramType', in_or_out.get('paramType'))
    add_dict_key(param_config, 'dtypeMode', in_or_out.get('dtype_match_mode'))
    add_dict_key(param_config, 'formatMode', correct_format_mode(in_or_out.get('format_match_mode')))
    return param_config


def get_inputs_or_outputs_config(inputs_or_outputs):
    if inputs_or_outputs is None:
        return None
    result = []
    for param in inputs_or_outputs:
        if isinstance(param, list):
            result.append([get_input_or_output_config(param[0])])
        else:
            result.append(get_input_or_output_config(param))
    return result


def get_params_config(support_info):
    params = {
        'inputs': get_inputs_or_outputs_config(support_info.get('inputs')),
        'outputs': get_inputs_or_outputs_config(support_info.get('outputs')),
    }
    if support_info.get('attrs') is not None:
        params['attrs'] = [
            {key: attr[key] for key in ('name', 'mode') if key in attr}
            for attr in support_info.get('attrs')
        ]
    return params


def add_simplified_config(op_type, key, core_type, objfile, support_info, config):
    simple_cfg = config.get('binary_info_config.json')
    op_cfg = simple_cfg.get(op_type)
    if not op_cfg:
        op_cfg = {}
        op_cfg['dynamicRankSupport'] = True
        op_cfg['simplifiedKeyMode'] = support_info.get('simplifiedKeyMode', 0)
        add_dict_key(op_cfg, 'optionalInputMode', support_info.get('optionalInputMode'))
        add_dict_key(op_cfg, 'optionalOutputMode', support_info.get('optionalOutputMode'))
        op_cfg['params'] = get_params_config(support_info)
        op_cfg['binaryList'] = []
        simple_cfg[op_type] = op_cfg
    bin_list = op_cfg.get('binaryList')
    bin_list.append({'coreType': core_type, 'simplifiedKey': key, 'binPath': objfile})
    bin_list.sort(key=lambda x: x.get('binPath')) # BEP整改：增加排序逻辑保障顺序


def add_op_config(op_file, bin_info, config):
    op_cfg = config.get(op_file)
    if not op_cfg:
        op_cfg = {}
        op_cfg['binList'] = []
        config[op_file] = op_cfg
    op_cfg.get('binList').append(bin_info)
    op_cfg.get('binList').sort(key=lambda x: x.get("binInfo", {}).get('jsonFilePath')) # BEP整改：增加排序逻辑保障顺序


def gen_ops_config(json_file, soc, config):
    core_type_map = {"MIX": 0, "AiCore": 1, "VectorCore": 2}
    contents = load_json(json_file)
    if ('binFileName' not in contents) or ('supportInfo' not in contents):
        return
    json_base_name = os.path.basename(json_file)
    op_dir = os.path.basename(os.path.dirname(json_file))
    support_info = contents.get('supportInfo')
    bin_name = contents.get('binFileName')
    bin_suffix = contents.get('binFileSuffix')
    core_type = core_type_map.get(contents.get("coreType"))
    bin_file_name = bin_name + bin_suffix
    op_type = bin_name.split('_')[0]
    op_file = op_dir + '.json'
    bin_info = {}
    keys = support_info.get('simplifiedKey')
    if keys:
        bin_info['simplifiedKey'] = keys
        for key in keys:
            add_simplified_config(op_type, key, core_type, os.path.join(soc, op_dir, bin_file_name),
                                  support_info, config)
    bin_info['staticKey'] = support_info.get('staticKey')
    bin_info['int64Mode'] = support_info.get('int64Mode')
    bin_info['inputs'] = support_info.get('inputs')
    bin_info['outputs'] = support_info.get('outputs')
    bin_info['deterministic'] = support_info.get('deterministic')
    add_dict_key(bin_info, 'simplifiedKeyMode', support_info.get('simplifiedKeyMode'))
    add_dict_key(bin_info, 'dynamicParamMode', support_info.get('dynamicParamMode'))
    add_dict_key(bin_info, 'opMode', support_info.get('opMode'))
    add_dict_key(bin_info, 'optionalInputMode', support_info.get('optionalInputMode'))
    add_dict_key(bin_info, 'optionalOutputMode', support_info.get('optionalOutputMode'))
    if support_info.get('attrs'):
        bin_info['attrs'] = support_info.get('attrs')
    bin_info['binInfo'] = {'jsonFilePath': os.path.join(soc, op_dir, json_base_name)}
    add_op_config(op_file, bin_info, config)


def gen_all_config(root_dir, soc):
    suffix = 'json'
    config = {}
    config['binary_info_config.json'] = {}
    all_json_files = get_specified_suffix_file(root_dir, suffix)
    for _json in all_json_files:
        gen_ops_config(_json, soc, config)
    for cfg_key in config.keys():
        cfg_file = os.path.join(root_dir, cfg_key)
        with os.fdopen(os.open(cfg_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            json.dump(config.get(cfg_key), fd, indent='  ', sort_keys=True)


def args_prase():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        nargs='?',
                        required=True,
                        help='Parse the path of the json file.')
    parser.add_argument('-s',
                        '--soc',
                        nargs='?',
                        required=True,
                        help='Parse the soc_version of ops.')
    return parser.parse_args()


def main():
    args = args_prase()
    gen_all_config(args.path, args.soc)


if __name__ == '__main__':
    main()
