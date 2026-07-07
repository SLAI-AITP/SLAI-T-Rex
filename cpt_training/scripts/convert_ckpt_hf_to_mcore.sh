#!/bin/bash
set -euo pipefail

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-convert_ckpt_v2.py}"
HF_LOAD_DIR="${HF_LOAD_DIR:-./model_from_hf/deepseek4_flash_hf}"
MCORE_SAVE_DIR="${MCORE_SAVE_DIR:-./model_weights/deepseek4_flash_mcore}"

if [[ -f "${ASCEND_ENV}" ]]; then
  source "${ASCEND_ENV}"
fi

python "${CONVERT_SCRIPT}" \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir "${HF_LOAD_DIR}" \
  --save-dir "${MCORE_SAVE_DIR}" \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 4 \
  --target-expert-parallel-size 32 \
  --noop-layers 43 \
  --mtp-num-layers 1 \
  --moe-grouped-gemm
