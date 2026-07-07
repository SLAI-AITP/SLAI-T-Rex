#!/bin/bash
# Generic MindSpeed-LLM data conversion helper.
#
# The script does not encode any dataset-specific presets. Convert one dataset
# with --input/--output-prefix, or convert many datasets with --manifest.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  convert_data.sh --input <path> --output-prefix <prefix> [options]
  convert_data.sh --manifest <manifest.tsv> [options]

Manifest format, one dataset per line:
  name<TAB>input_path<TAB>output_prefix[<TAB>workers[<TAB>n_subs[<TAB>log_interval]]]

Blank lines and lines starting with # are ignored. Whitespace-separated
manifests also work as long as paths do not contain spaces.

Options:
  --name <name>            Dataset name used for logs/cache in single-input mode.
  --tokenizer <path>       HuggingFace tokenizer/model path.
  --mindspeed-llm-dir <d>  MindSpeed-LLM repository root.
  --workers <n>            Default HuggingFace datasets map workers. Default: 4.
  --n-subs <n>             Default preprocess_data.py --n-subs value. Default: 1.
  --seq-length <n>         Sequence length. Default: 4096.
  --json-keys <keys>       JSON key(s) passed to preprocess_data.py. Default: text.
  --handler-name <name>    Handler class name. Default: GeneralPretrainHandler.
  --log-interval <n>       Default preprocess_data.py log interval. Default: 1000.
  --cache-base <dir>       HuggingFace datasets cache root.
  --log-dir <dir>          Conversion log directory.
  --parallel <n>           Max concurrent conversions for manifest mode. Default: 1.
  --overwrite              Rebuild existing .bin/.idx outputs.
  --no-append-eod          Do not pass --append-eod.
  -h, --help               Show this help.

Examples:
  bash convert_data.sh \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/processed/my_dataset \
    --tokenizer /path/to/tokenizer \
    --workers 8 --n-subs 16

  bash convert_data.sh \
    --manifest /path/to/convert_manifest.tsv \
    --tokenizer /path/to/tokenizer \
    --parallel 4
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_LLM_DIR="${MINDSPEED_LLM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

INPUT="${INPUT:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
MANIFEST="${MANIFEST:-}"
NAME="${NAME:-}"

TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer_or_model}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
WORKERS="${WORKERS:-4}"
N_SUBS="${N_SUBS:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
JSON_KEYS="${JSON_KEYS:-text}"
HANDLER_NAME="${HANDLER_NAME:-GeneralPretrainHandler}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
CACHE_BASE="${CACHE_BASE:-}"
LOG_DIR="${LOG_DIR:-}"
PARALLEL="${PARALLEL:-1}"
OVERWRITE="${OVERWRITE:-0}"
APPEND_EOD="${APPEND_EOD:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --tokenizer|--tokenizer-name-or-path) TOKENIZER_PATH="$2"; shift 2 ;;
        --mindspeed-llm-dir) MINDSPEED_LLM_DIR="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --n-subs) N_SUBS="$2"; shift 2 ;;
        --seq-length) SEQ_LENGTH="$2"; shift 2 ;;
        --json-keys) JSON_KEYS="$2"; shift 2 ;;
        --handler-name) HANDLER_NAME="$2"; shift 2 ;;
        --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
        --cache-base) CACHE_BASE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --no-append-eod) APPEND_EOD=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

init_runtime() {
    local output_root="$1"

    CACHE_BASE="${CACHE_BASE:-${output_root}/.hf_cache}"
    LOG_DIR="${LOG_DIR:-${output_root}/logs}"

    if [[ -f "${ASCEND_ENV}" ]]; then
        source "${ASCEND_ENV}"
    fi

    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export HF_DATASETS_CACHE="${CACHE_BASE}"

    mkdir -p "${output_root}" "${CACHE_BASE}" "${LOG_DIR}"
    cd "${MINDSPEED_LLM_DIR}"
}

precreate_nsub_dirs() {
    local input_path="$1"
    local output_prefix="$2"
    local n_subs="$3"

    [[ "${n_subs}" -gt 1 && -f "${input_path}" ]] || return 0

    local num_samples gap count last_idx last_tag target_prefixname k sub_prefix
    num_samples="$(wc -l < "${input_path}")"
    gap=$(( num_samples / n_subs ))
    if [[ "${gap}" -le 0 ]]; then
        echo "[ERROR] n_subs=${n_subs} is larger than sample count=${num_samples}: ${input_path}" >&2
        return 1
    fi

    count=$(( num_samples / gap ))
    if [[ $(( num_samples % gap )) -ne 0 ]]; then
        count=$(( count + 1 ))
    fi

    last_idx=$(( count - 1 ))
    last_tag="$(printf "%03d" "${last_idx}")"
    target_prefixname="$(basename "${output_prefix}")"
    for (( k=0; k<count; k++ )); do
        sub_prefix="${output_prefix//${target_prefixname}/$(printf "%03d" "${k}")_of_${last_tag}_${target_prefixname}}"
        mkdir -p "$(dirname "${sub_prefix}")"
    done
}

convert_one() {
    local name="$1"
    local input_path="$2"
    local output_prefix="$3"
    local workers="$4"
    local n_subs="$5"
    local log_interval="$6"

    local final_prefix="${output_prefix}_text_document"
    local log_file="${LOG_DIR}/${name}.log"
    local dataset_cache="${CACHE_BASE}/${name}"
    local append_args=()

    mkdir -p "$(dirname "${output_prefix}")" "${dataset_cache}"
    precreate_nsub_dirs "${input_path}" "${output_prefix}" "${n_subs}"

    if [[ "${OVERWRITE}" != "1" && -f "${final_prefix}.bin" && -f "${final_prefix}.idx" ]]; then
        log "[SKIP] ${name}: ${final_prefix}.bin/.idx already exists"
        return 0
    fi

    if [[ "${OVERWRITE}" == "1" ]]; then
        rm -f "${final_prefix}.bin" "${final_prefix}.idx"
    fi

    if [[ "${APPEND_EOD}" == "1" ]]; then
        append_args+=(--append-eod)
    fi

    log "[START] ${name}"
    log "  input        = ${input_path}"
    log "  output       = ${final_prefix}"
    log "  workers      = ${workers}"
    log "  n_subs       = ${n_subs}"
    log "  seq_length   = ${SEQ_LENGTH}"
    log "  cache        = ${dataset_cache}"
    log "  log          = ${log_file}"

    if ! python ./preprocess_data.py \
        --input "${input_path}" \
        --tokenizer-name-or-path "${TOKENIZER_PATH}" \
        --tokenizer-type PretrainedFromHF \
        --handler-name "${HANDLER_NAME}" \
        --output-prefix "${output_prefix}" \
        --json-keys "${JSON_KEYS}" \
        --workers "${workers}" \
        --n-subs "${n_subs}" \
        --log-interval "${log_interval}" \
        --cache-dir "${dataset_cache}" \
        --seq-length "${SEQ_LENGTH}" \
        "${append_args[@]}" \
        2>&1 | tee "${log_file}"; then
        log "[ERROR] ${name} failed"
        return 1
    fi

    log "[DONE] ${name}"
    ls -lh "${final_prefix}".*
}

wait_for_slot() {
    local running

    while true; do
        running="$(jobs -pr | wc -l)"
        if [[ "${running}" -lt "${PARALLEL}" ]]; then
            return 0
        fi
        sleep 1
    done
}

wait_all_jobs() {
    local failed=0
    local pid

    for pid in "$@"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done

    if [[ "${failed}" != "0" ]]; then
        exit 1
    fi
}

run_single() {
    if [[ -z "${INPUT}" || -z "${OUTPUT_PREFIX}" ]]; then
        echo "[ERROR] single conversion requires --input and --output-prefix" >&2
        usage
        exit 1
    fi

    NAME="${NAME:-$(basename "${OUTPUT_PREFIX}")}"
    init_runtime "$(dirname "${OUTPUT_PREFIX}")"
    convert_one "${NAME}" "${INPUT}" "${OUTPUT_PREFIX}" "${WORKERS}" "${N_SUBS}" "${LOG_INTERVAL}"
}

run_manifest() {
    if [[ -z "${MANIFEST}" ]]; then
        echo "[ERROR] manifest conversion requires --manifest" >&2
        usage
        exit 1
    fi
    if [[ ! -f "${MANIFEST}" ]]; then
        echo "[ERROR] manifest not found: ${MANIFEST}" >&2
        exit 1
    fi

    local output_root
    output_root="$(dirname "${MANIFEST}")"
    init_runtime "${output_root}"

    local pids=()
    local line name input_path output_prefix workers n_subs log_interval

    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line//[[:space:]]/}" ]] && continue
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue

        read -r name input_path output_prefix workers n_subs log_interval <<< "${line}"
        if [[ -z "${name:-}" || -z "${input_path:-}" || -z "${output_prefix:-}" ]]; then
            echo "[ERROR] invalid manifest line: ${line}" >&2
            exit 1
        fi

        workers="${workers:-${WORKERS}}"
        n_subs="${n_subs:-${N_SUBS}}"
        log_interval="${log_interval:-${LOG_INTERVAL}}"

        wait_for_slot
        convert_one "${name}" "${input_path}" "${output_prefix}" "${workers}" "${n_subs}" "${log_interval}" &
        pids+=("$!")
    done < "${MANIFEST}"

    wait_all_jobs "${pids[@]}"
}

if [[ -n "${MANIFEST}" ]]; then
    run_manifest
else
    run_single
fi
