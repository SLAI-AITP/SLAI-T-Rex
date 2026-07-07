#!/bin/bash
# Convert a DeepSeek-V4 FP8 HuggingFace checkpoint to BF16 HuggingFace format.
#
# Required:
#   INPUT_FP8_HF_PATH    Source FP8 HuggingFace checkpoint directory.
#   OUTPUT_BF16_HF_PATH  Target BF16 HuggingFace checkpoint directory.
#   MINDSPEED_LLM_DIR    MindSpeed-LLM repository root.

set -euo pipefail

MINDSPEED_LLM_DIR="${MINDSPEED_LLM_DIR:-}"

INPUT_FP8_HF_PATH="${INPUT_FP8_HF_PATH:-/path/to/deepseek4_fp8_hf}"
OUTPUT_BF16_HF_PATH="${OUTPUT_BF16_HF_PATH:-/path/to/deepseek4_bf16_hf}"
QUANT_TYPE="${QUANT_TYPE:-bfloat16}"

if [[ -z "${MINDSPEED_LLM_DIR}" || ! -d "${MINDSPEED_LLM_DIR}" ]]; then
    echo "[ERROR] Set MINDSPEED_LLM_DIR to your MindSpeed-LLM repository root." >&2
    exit 1
fi

if [[ "${INPUT_FP8_HF_PATH}" == /path/to/* || "${OUTPUT_BF16_HF_PATH}" == /path/to/* ]]; then
    echo "[ERROR] Set INPUT_FP8_HF_PATH and OUTPUT_BF16_HF_PATH before running." >&2
    exit 1
fi

if ! python -c "import torchao" >/dev/null 2>&1; then
    echo "[ERROR] Python package 'torchao' is required. Install it in your runtime environment first." >&2
    exit 1
fi

cd "${MINDSPEED_LLM_DIR}"

if [[ ! -f tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py ]]; then
    echo "[ERROR] Missing tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py under MINDSPEED_LLM_DIR." >&2
    exit 1
fi

python tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py \
    --input_fp8_hf_path "${INPUT_FP8_HF_PATH}" \
    --output_hf_path "${OUTPUT_BF16_HF_PATH}" \
    --quant_type "${QUANT_TYPE}"
