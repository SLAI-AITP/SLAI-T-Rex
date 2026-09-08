#!/usr/bin/env bash

# Resolve external MindSpeed repositories without embedding machine-specific paths.

resolve_mindspeed_runtime() {
    local entrypoint="${1:?MindSpeed-LLM entrypoint is required}"
    local helper_dir project_root requested candidate

    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    project_root="$(cd "${helper_dir}/../.." && pwd)"
    export SLAI_T_REX_ROOT="${project_root}"

    requested="${MINDSPEED_LLM_DIR:-${MINDSPEED_LLM_HOME:-}}"
    if [[ -n "${requested}" ]]; then
        if [[ ! -f "${requested}/${entrypoint}" ]]; then
            echo "[ERROR] ${entrypoint} was not found under MINDSPEED_LLM_DIR: ${requested}" >&2
            return 1
        fi
        MINDSPEED_LLM_DIR="$(cd "${requested}" && pwd -P)"
    else
        for candidate in \
            "${project_root}/MindSpeed-LLM" \
            "${project_root}/../MindSpeed-LLM" \
            /workspace/MindSpeed-LLM; do
            if [[ -f "${candidate}/${entrypoint}" ]]; then
                MINDSPEED_LLM_DIR="$(cd "${candidate}" && pwd -P)"
                break
            fi
        done
    fi

    if [[ -z "${MINDSPEED_LLM_DIR:-}" ]]; then
        echo "[ERROR] Cannot locate MindSpeed-LLM (${entrypoint})." >&2
        echo "[ERROR] Export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM." >&2
        return 1
    fi

    requested="${MINDSPEED_DIR:-${MINDSPEED_HOME:-}}"
    if [[ -n "${requested}" ]]; then
        if [[ ! -d "${requested}/mindspeed" ]]; then
            echo "[ERROR] Python package 'mindspeed' was not found under MINDSPEED_DIR: ${requested}" >&2
            return 1
        fi
        MINDSPEED_DIR="$(cd "${requested}" && pwd -P)"
    else
        for candidate in \
            "${MINDSPEED_LLM_DIR}/../MindSpeed" \
            "${project_root}/MindSpeed" \
            "${project_root}/../MindSpeed" \
            /workspace/MindSpeed; do
            if [[ -d "${candidate}/mindspeed" ]]; then
                MINDSPEED_DIR="$(cd "${candidate}" && pwd -P)"
                break
            fi
        done
    fi

    if [[ -z "${MINDSPEED_DIR:-}" ]]; then
        echo "[ERROR] Cannot locate MindSpeed." >&2
        echo "[ERROR] Export MINDSPEED_DIR=/path/to/MindSpeed." >&2
        return 1
    fi

    requested="${MEGATRON_DIR:-${MEGATRON_HOME:-}}"
    if [[ -n "${requested}" ]]; then
        if [[ ! -d "${requested}/megatron" ]]; then
            echo "[ERROR] Python package 'megatron' was not found under MEGATRON_DIR: ${requested}" >&2
            return 1
        fi
        MEGATRON_DIR="$(cd "${requested}" && pwd -P)"
    else
        for candidate in \
            "${MINDSPEED_LLM_DIR}" \
            "${MINDSPEED_LLM_DIR}/Megatron-LM" \
            "${MINDSPEED_LLM_DIR}/../Megatron-LM" \
            "${project_root}/Megatron-LM" \
            "${project_root}/../Megatron-LM" \
            /workspace/Megatron-LM; do
            if [[ -d "${candidate}/megatron" ]]; then
                MEGATRON_DIR="$(cd "${candidate}" && pwd -P)"
                break
            fi
        done
    fi

    if [[ -z "${MEGATRON_DIR:-}" ]]; then
        echo "[ERROR] Cannot locate Megatron-LM." >&2
        echo "[ERROR] Export MEGATRON_DIR=/path/to/Megatron-LM." >&2
        return 1
    fi

    export MINDSPEED_LLM_DIR MINDSPEED_DIR MEGATRON_DIR
    export MINDSPEED_LLM_HOME="${MINDSPEED_LLM_DIR}"
    export MINDSPEED_HOME="${MINDSPEED_DIR}"
    export MEGATRON_HOME="${MEGATRON_DIR}"
    export PYTHONPATH="${MINDSPEED_LLM_DIR}:${MINDSPEED_DIR}:${MEGATRON_DIR}:${PYTHONPATH:-}"
}
