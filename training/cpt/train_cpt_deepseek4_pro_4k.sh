#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Pro 4K continual pre-training template for Ascend clusters.
#
# Required runtime inputs:
#   TRAIN_DATA_PATH  Weighted MindSpeed indexed dataset prefixes.
#   TOKENIZER_PATH   HuggingFace tokenizer/model directory.
#   CKPT_LOAD_DIR    MCore checkpoint directory to continue from.
#
# Repository locations are resolved by training/common/resolve_runtime.sh.

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || die "${name} must be a positive integer, got: ${value}"
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] \
        || die "${name} must be a non-negative integer, got: ${value}"
}

require_path() {
    local name="$1"
    local value="${!name:-}"
    [[ -n "${value}" && "${value}" != /path/to/* ]] \
        || die "${name} must be set"
}

first_host_from_list() {
    local first_host="${1%%,*}"
    first_host="${first_host#"${first_host%%[![:space:]]*}"}"
    first_host="${first_host%"${first_host##*[![:space:]]}"}"
    printf '%s' "${first_host}"
}

derive_node_rank() {
    local host="${HOSTNAME:-$(hostname)}"

    if [[ -n "${NODE_RANK:-}" ]]; then
        printf '%s' "${NODE_RANK}"
    elif [[ -n "${VC_TASK_INDEX:-}" ]]; then
        printf '%s' "${VC_TASK_INDEX}"
    elif [[ -n "${MA_TASK_INDEX:-}" ]]; then
        printf '%s' "${MA_TASK_INDEX}"
    elif [[ "${host}" =~ -worker-([0-9]+)$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    elif [[ "${host}" =~ -([0-9]+)$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        return 1
    fi
}

source_if_present() {
    local env_file="$1"
    if [[ -n "${env_file}" && -f "${env_file}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${env_file}"
        set -u
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/resolve_runtime.sh"
resolve_mindspeed_runtime pretrain_deepseek4.py

source_if_present "${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
source_if_present "${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export HCCL_OP_EXTENSION_MODE="${HCCL_OP_EXTENSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-7200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_ALGO="${HCCL_ALGO:-alltoall=level0:NA;level1:pipeline}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-}"
require_path TRAIN_DATA_PATH
require_path TOKENIZER_PATH
require_path CKPT_LOAD_DIR

[[ -d "${TOKENIZER_PATH}" ]] || die "TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}"
[[ -d "${CKPT_LOAD_DIR}" ]] || die "CKPT_LOAD_DIR does not exist: ${CKPT_LOAD_DIR}"

NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
NNODES="${NNODES:-${VC_WORKER_NUM:-${MA_NUM_HOSTS:-32}}}"
if ! NODE_RANK="$(derive_node_rank)"; then
    [[ "${NNODES}" == "1" ]] || die "cannot determine NODE_RANK"
    NODE_RANK=0
fi

SCHEDULER_HOSTS="${VC_WORKER_HOSTS:-${MA_WORKER_HOSTS:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "${MASTER_ADDR}" && -n "${SCHEDULER_HOSTS}" ]]; then
    MASTER_ADDR="$(first_host_from_list "${SCHEDULER_HOSTS}")"
fi
if [[ -z "${MASTER_ADDR}" && -n "${MA_VJ_NAME:-}" \
    && -n "${MA_HJ_NAME:-}" && -n "${MA_NAMESPACE:-}" ]]; then
    MASTER_ADDR="${MA_VJ_NAME}-worker-0.${MA_HJ_NAME}.${MA_NAMESPACE}.svc.cluster.local"
fi
if [[ -z "${MASTER_ADDR}" && "${NNODES}" == "1" ]]; then
    MASTER_ADDR=127.0.0.1
fi
MASTER_PORT="${MASTER_PORT:-8192}"

TP=2
PP=8
EP=64
ETP=1
CP=1
NUM_LAYERS=64
NUM_EXPERTS=384
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GBS="${GBS:-256}"
TRAIN_ITERS="${TRAIN_ITERS:-45}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${TRAIN_ITERS}}"
LR="${LR:-1.0e-6}"
MIN_LR="${MIN_LR:-1.0e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
NO_SHARED_STORAGE="${NO_SHARED_STORAGE:-1}"

for item in \
    "NPUS_PER_NODE:${NPUS_PER_NODE}" \
    "NNODES:${NNODES}" \
    "MASTER_PORT:${MASTER_PORT}" \
    "SEQ_LEN:${SEQ_LEN}" \
    "MBS:${MBS}" \
    "GBS:${GBS}" \
    "TRAIN_ITERS:${TRAIN_ITERS}" \
    "LR_DECAY_ITERS:${LR_DECAY_ITERS}" \
    "SAVE_INTERVAL:${SAVE_INTERVAL}"; do
    require_positive_integer "${item%%:*}" "${item#*:}"
done
require_nonnegative_integer NODE_RANK "${NODE_RANK}"
require_nonnegative_integer WARMUP_ITERS "${WARMUP_ITERS}"
[[ "${NO_SHARED_STORAGE}" == "0" || "${NO_SHARED_STORAGE}" == "1" ]] \
    || die "NO_SHARED_STORAGE must be 0 or 1"
[[ -n "${MASTER_ADDR}" ]] || die "cannot determine MASTER_ADDR; export it explicitly"
[[ "${NODE_RANK}" -lt "${NNODES}" ]] \
    || die "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}"
[[ "${MASTER_PORT}" -le 65535 ]] || die "MASTER_PORT must be at most 65535"
[[ "${WARMUP_ITERS}" -lt "${LR_DECAY_ITERS}" ]] \
    || die "WARMUP_ITERS must be smaller than LR_DECAY_ITERS"

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
[[ "${WORLD_SIZE}" -eq 512 ]] \
    || die "TP2/PP8/EP64 requires 512 ranks; got ${WORLD_SIZE}"
[[ $((WORLD_SIZE % (TP * PP * CP))) -eq 0 ]] \
    || die "WORLD_SIZE is incompatible with TP*PP*CP"
[[ $((WORLD_SIZE % (ETP * EP * PP))) -eq 0 ]] \
    || die "WORLD_SIZE is incompatible with ETP*EP*PP"
DP=$((WORLD_SIZE / (TP * PP * CP)))
[[ $((GBS % (MBS * DP))) -eq 0 ]] \
    || die "GBS must be divisible by MBS*dense-DP=$((MBS * DP))"

RUN_ID="${RUN_ID:-${VC_JOB_ID:-${SLURM_JOB_ID:-deepseek4-pro-cpt}}}"
[[ "${RUN_ID}" != *[[:space:]/]* ]] \
    || die "RUN_ID must not contain whitespace or '/': ${RUN_ID}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SLAI_T_REX_ROOT}/outputs/cpt}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_ID}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/logs/${RUN_ID}}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${OUTPUT_ROOT}/data_cache/${RUN_ID}}"
mkdir -p "${CKPT_SAVE_DIR}" "${TENSORBOARD_DIR}" "${RUN_DIR}" "${DATA_CACHE_PATH}"

torchrun_args=(
    --nproc-per-node="${NPUS_PER_NODE}"
    --nnodes="${NNODES}"
    --node-rank="${NODE_RANK}"
    --master-addr="${MASTER_ADDR}"
    --master-port="${MASTER_PORT}"
)

model_args=(
    --transformer-impl local
    --spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec
    --mtp-spec mindspeed_llm.tasks.models.spec.deepseek4_spec mtp_spec
    --manual-gc
    --manual-gc-interval 50
    --noop-layers 3,62,63
    --use-distributed-optimizer
    --use-flash-attn
    --use-mcore-models
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --num-layers-per-virtual-pipeline-stage 4
    --expert-model-parallel-size "${EP}"
    --expert-tensor-parallel-size "${ETP}"
    --sequence-parallel
    --context-parallel-size "${CP}"
    --context-parallel-algo ulysses_cp_algo
    --num-layers "${NUM_LAYERS}"
    --hidden-size 7168
    --ffn-hidden-size 7168
    --num-attention-heads 128
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "${TOKENIZER_PATH}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 163840
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --make-vocab-size-divisible-by 1
    --lr "${LR}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.02
    --hidden-dropout 0.0
    --position-embedding-type g2
    --normalization RMSNorm
    --use-fused-rotary-pos-emb
    --use-rotary-position-embeddings
    --use-fused-swiglu
    --use-fused-rmsnorm
    --swiglu
    --swiglu-limit 10
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr "${MIN_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --lr-warmup-iters "${WARMUP_ITERS}"
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --initial-loss-scale 65536
    --vocab-size 129280
    --padded-vocab-size 129280
    --rotary-base 10000
    --norm-epsilon 1e-6
    --no-load-optim
    --no-load-rng
    --bf16
    --distributed-timeout-minutes 120
    --gemm-gradient-accumulation-fusion
)
if [[ "${NO_SHARED_STORAGE}" == "1" ]]; then
    model_args+=(--no-shared-storage)
fi

dsa_args=(
    --enable-dsa-indexer
    --index-n-heads 64
    --index-head-dim 128
    --index-topk 1024
    --enable-mhc
    --hc-mult 4
    --kv-compress
    --norm-eps 1e-6
    --use-triton-sinkhorn
    --use-triton-mhc
    --use-triton-rmsnorm-without-weight
    --use-fused-lightning-indexer-loss
    --use-fused-lightning-indexer
    --use-sparse-flash-attn
)

mla_args=(
    --multi-latent-attention
    --qk-pos-emb-head-dim 64
    --qk-head-dim 512
    --q-lora-rank 1536
    --o-lora-rank 1024
    --kv-lora-rank 512
    --v-head-dim 128
    --qk-layernorm
    --mla-fa-without-pad
)

compress_ratios=(
    128 128 4 0 128 4 128 4 128 4 128 4 128 4 128 4
    128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4
    128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4
    128 4 128 4 128 4 128 4 128 4 128 4 128 4 0 0 0
)
[[ "${#compress_ratios[@]}" -eq $((NUM_LAYERS + 1)) ]] \
    || die "compress-ratios must contain NUM_LAYERS + one MTP entry"
csa_args=(
    --use-g2-attention
    --o-groups 16
    --g2-window-size 128
    --rope-head-dim 64
    --original-seq-len 65536
    --rope-factor 16
    --compress-rope-theta 160000.0
    --max-batch-size 4
    --compress-ratios "${compress_ratios[@]}"
    --use-g2-indexer-loss
)

moe_args=(
    --fix-router
    --moe-grouped-gemm
    --moe-permutation-async-comm
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq 1
    --first-k-dense-replace -1
    --num-experts "${NUM_EXPERTS}"
    --moe-router-topk 6
    --moe-ffn-hidden-size 3072
    --moe-router-load-balancing-type none
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-router-topk-scaling-factor 2.5
    --seq-aux
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-shared-expert-intermediate-size 3072
    --moe-router-dtype fp32
    --n-hash-layers 3
    --moe-alltoall-overlap-comm
    --moe-permute-fusion
)

mtp_args=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.3
    --mtp-mem-efficient-logits
)

memory_args=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --swap-optimizer
)

rope_args=(
    --beta-fast 32
    --beta-slow 1
    --rope-scaling-factor 40
    --rope-scaling-mscale 1.0
    --rope-scaling-mscale-all-dim 1.0
    --rope-scaling-original-max-position-embeddings "${SEQ_LEN}"
    --rope-scaling-type yarn
)

train_data_flat="${TRAIN_DATA_PATH//$'\n'/ }"
read -r -a train_data_tokens <<< "${train_data_flat}"
data_args=(
    --data-path "${train_data_tokens[@]}"
    --split 100,0,0
    --data-cache-path "${DATA_CACHE_PATH}"
)

output_args=(
    --log-interval 1
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${TRAIN_ITERS}"
    --eval-iters 0
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --save "${CKPT_SAVE_DIR}"
    --load "${CKPT_LOAD_DIR}"
    --ckpt-format torch
    --no-save-optim
    --no-save-rng
    --log-throughput
)

cd "${MINDSPEED_LLM_DIR}"
LOG_FILE="${RUN_DIR}/node-${NODE_RANK}.log"

echo "===== DeepSeek-V4-Pro 4K CPT ====="
echo "run_id=${RUN_ID}"
echo "framework=${MINDSPEED_LLM_DIR}"
echo "data=${TRAIN_DATA_PATH}"
echo "load=${CKPT_LOAD_DIR}"
echo "save=${CKPT_SAVE_DIR}"
echo "ranks=${WORLD_SIZE} TP=${TP} PP=${PP} EP=${EP} GBS=${GBS} iters=${TRAIN_ITERS}"

torchrun "${torchrun_args[@]}" "${MINDSPEED_LLM_DIR}/pretrain_deepseek4.py" \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${output_args[@]}" \
    "${mla_args[@]}" \
    "${rope_args[@]}" \
    "${moe_args[@]}" \
    "${dsa_args[@]}" \
    "${csa_args[@]}" \
    "${memory_args[@]}" \
    "${mtp_args[@]}" \
    --distributed-backend nccl 2>&1 | tee "${LOG_FILE}"
