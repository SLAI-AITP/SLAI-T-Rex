#!/bin/bash
set -euo pipefail

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-convert_ckpt_v2.py}"
MCORE_LOAD_DIR="${MCORE_LOAD_DIR:-./model_weights/deepseek4_flash_mcore}"
HF_SAVE_DIR="${HF_SAVE_DIR:-./model_from_hf/deepseek4_flash_hf}"

if [[ -f "${ASCEND_ENV}" ]]; then
  source "${ASCEND_ENV}"
fi

python "${CONVERT_SCRIPT}" \
  --load-model-type mg \
  --save-model-type hf \
  --model-type-hf deepseek4 \
  --load-dir "${MCORE_LOAD_DIR}" \
  --save-dir "${HF_SAVE_DIR}" \
  --noop-layers 43 \
  --mtp-num-layers 1 \
  --moe-grouped-gemm \
  --expert-tensor-parallel-size 1 \

# 当前仅支持开启 gemm 并且 etp=1 的情景
# 如果使用base模型，请将--model-type-hf 设置为 deepseek4_base
