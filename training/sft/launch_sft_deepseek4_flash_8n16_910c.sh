#!/bin/bash
# Launcher for DeepSeek-V4-Flash SFT on 8 nodes x 16 NPUs.
#
# This wrapper is intentionally dataset-agnostic. Pass DATA_PATH for training.
# Set PREPARE_INDEXMAP=1 when your packed SFT dataset needs a precomputed
# shuffle index before distributed SFT starts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_LLM_DIR="${MINDSPEED_LLM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SFT_TRAIN_SCRIPT="${SFT_TRAIN_SCRIPT:-${SCRIPT_DIR}/train_sft_deepseek4_flash_8k.sh}"

TRAIN_ITERS="${TRAIN_ITERS:-250}"
GBS="${GBS:-128}"
INDEXMAP_SEED="${INDEXMAP_SEED:-1234}"

infer_node_rank() {
    if [[ -n "${NODE_RANK:-}" ]]; then
        echo "${NODE_RANK}"
    elif [[ -n "${VC_TASK_INDEX:-}" ]]; then
        echo "${VC_TASK_INDEX}"
    elif [[ "${HOSTNAME:-}" =~ -([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "0"
    fi
}

NODE_RANK="$(infer_node_rank)"
export NODE_RANK TRAIN_ITERS GBS MINDSPEED_LLM_DIR
export PYTHONPATH="${MINDSPEED_LLM_DIR}/Megatron-LM:${PYTHONPATH:-}"

if [[ -n "${BASHRC:-}" && -f "${BASHRC}" ]]; then
    source "${BASHRC}"
fi

if [[ -n "${PRE_INSTALL_SCRIPT:-}" ]]; then
    if [[ ! -f "${PRE_INSTALL_SCRIPT}" ]]; then
        echo "[ERROR] PRE_INSTALL_SCRIPT not found: ${PRE_INSTALL_SCRIPT}" >&2
        exit 1
    fi
    bash "${PRE_INSTALL_SCRIPT}"
fi

export SFT_RUN_ID="${SFT_RUN_ID:-${RUN_ID:-${VC_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M)}}}}"
echo "SFT_RUN_ID=${SFT_RUN_ID}"
echo "NODE_RANK=${NODE_RANK}"

if [[ "${PREPARE_INDEXMAP:-0}" == "1" || -n "${INDEXMAP_SCRIPT:-}" || -n "${INDEXMAP_DATA_PREFIX:-}" ]]; then
    INDEXMAP_SCRIPT="${INDEXMAP_SCRIPT:-${SCRIPT_DIR}/prepare_sft_indexmap.py}"
    INDEXMAP_DATA_PREFIX="${INDEXMAP_DATA_PREFIX:-${DATA_PATH:-}}"
    if [[ -z "${INDEXMAP_DATA_PREFIX}" ]]; then
        echo "[ERROR] indexmap preparation needs INDEXMAP_DATA_PREFIX or DATA_PATH." >&2
        exit 1
    fi
    if [[ ! -f "${INDEXMAP_SCRIPT}" ]]; then
        echo "[ERROR] INDEXMAP_SCRIPT not found: ${INDEXMAP_SCRIPT}" >&2
        exit 1
    fi

    indexmap_args=(
        --data-prefix "${INDEXMAP_DATA_PREFIX}"
        --split-name "${INDEXMAP_SPLIT_NAME:-train}"
        --train-iters "${TRAIN_ITERS}"
        --global-batch-size "${GBS}"
        --seed "${INDEXMAP_SEED}"
    )
    if [[ -n "${INDEXMAP_DOC_COUNT:-}" ]]; then
        indexmap_args+=(--doc-count "${INDEXMAP_DOC_COUNT}")
    fi
    if [[ -n "${INDEXMAP_SOURCE_JSONL:-}" ]]; then
        indexmap_args+=(--source-jsonl "${INDEXMAP_SOURCE_JSONL}")
    fi
    if [[ "${INDEXMAP_NO_SHUFFLE:-0}" == "1" ]]; then
        indexmap_args+=(--no-shuffle)
    fi
    if [[ "${INDEXMAP_PADDED_SAMPLES:-0}" == "1" ]]; then
        indexmap_args+=(--padded-samples)
    fi

    if [[ "${NODE_RANK}" != "0" ]]; then
        indexmap_args+=(--wait --timeout "${INDEXMAP_TIMEOUT:-1800}")
    fi

    python "${INDEXMAP_SCRIPT}" "${indexmap_args[@]}"
fi

cd "${MINDSPEED_LLM_DIR}"
bash "${SFT_TRAIN_SCRIPT}"
