#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Pro full SFT for the final release261 image.
# The image must already contain:
#   1. MindSpeed/MindSpeed-LLM/Megatron-LM release261 sources;
#   2. the applied GTS framework patch and built GTS operators;
#   3. installed ops-transformer SparseFlashMla/LightningIndexer operators.
# This launcher never builds operators or applies framework patches at runtime.

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] \
        || die "${name} must be a positive integer, got: ${value}"
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] \
        || die "${name} must be a non-negative integer, got: ${value}"
}

require_boolean() {
    local name="$1"
    local value="$2"
    [ "$value" = "0" ] || [ "$value" = "1" ] \
        || die "${name} must be 0 or 1, got: ${value}"
}

require_path() {
    local name="$1"
    local value="${!name:-}"
    [ -n "${value}" ] && [[ "${value}" != /path/to/* ]] \
        || die "${name} must be set"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/resolve_runtime.sh"
resolve_mindspeed_runtime posttrain_gpt.py
MODEL_ROOT="${MINDSPEED_LLM_DIR}"
MINDSPEED_ROOT="${MINDSPEED_DIR}"
MEGATRON_ROOT="${MEGATRON_DIR}"

first_host_from_list() {
    local host_list="$1"
    local first_host="${host_list%%,*}"
    first_host="${first_host#"${first_host%%[![:space:]]*}"}"
    first_host="${first_host%"${first_host##*[![:space:]]}"}"
    printf '%s' "$first_host"
}

derive_node_rank() {
    local host="${HOSTNAME:-$(hostname)}"

    if [ -n "${NODE_RANK:-}" ]; then
        printf '%s' "$NODE_RANK"
    elif [ -n "${VC_TASK_INDEX:-}" ]; then
        printf '%s' "$VC_TASK_INDEX"
    elif [ -n "${MA_TASK_INDEX:-}" ]; then
        printf '%s' "$MA_TASK_INDEX"
    elif [[ "$host" =~ -worker-([0-9]+)$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    elif [[ "$host" =~ -([0-9]+)$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        return 1
    fi
}

source_if_present() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        set +u
        # shellcheck disable=SC1090
        source "$env_file"
        set -u
        return 0
    fi
    return 1
}

resolve_gts_env_script() {
    if [ -n "${GTS_ENV_SCRIPT:-}" ]; then
        [ -f "$GTS_ENV_SCRIPT" ] \
            || die "GTS_ENV_SCRIPT does not exist: ${GTS_ENV_SCRIPT}"
        printf '%s' "$GTS_ENV_SCRIPT"
        return 0
    fi

    local candidate
    for candidate in \
        "${SLAI_T_REX_ROOT}/kernel/script/set_env.sh" \
        /opt/gts/script/set_env.sh \
        /opt/kernel/script/set_env.sh \
        /opt/mindspeed/gts/script/set_env.sh \
        /opt/mindspeed/kernel/script/set_env.sh \
        /workspace/GTSOps/script/set_env.sh \
        /workspace/kernel/script/set_env.sh \
        /usr/local/gtsops/script/set_env.sh; do
        if [ -f "$candidate" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

activate_gts_runtime() {
    local env_script
    if [ -n "${GTS_ENV_SCRIPT:-}" ]; then
        [ -f "$GTS_ENV_SCRIPT" ] \
            || die "GTS_ENV_SCRIPT does not exist: ${GTS_ENV_SCRIPT}"
        env_script="$GTS_ENV_SCRIPT"
    else
        env_script="$(resolve_gts_env_script || true)"
    fi

    if [ -n "$env_script" ]; then
        echo "===== activate image-bundled GTS runtime ====="
        echo "GTS_ENV_SCRIPT=${env_script}"
        export GTS_ARGS="${GTS_ARGS:-}"
        export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
        export LD_PRELOAD="${LD_PRELOAD:-}"
        source_if_present "$env_script"
        export GTS_ENV_SCRIPT="$env_script"
    else
        [ -n "${GTS_CUSTOM_OPP_PATH:-}" ] \
            || die "cannot locate image-bundled GTS set_env.sh and GTS_CUSTOM_OPP_PATH is unset; export GTS_ENV_SCRIPT"
        echo "GTS runtime is already configured by the image environment"
    fi
}

precompile_mindspeed_extensions() {
    echo "===== precompile/load MindSpeed runtime extensions ====="
    python3 -u - <<'PY'
import torch
import torch_npu

from mindspeed.op_builder import GMMOpBuilder
from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder
from mindspeed.ops.npu_groupmatmul_add import groupmatmul_add_op_builder
from mindspeed.ops.npu_matmul_add import matmul_add_op_builder
from mindspeed.ops.npu_moe_token_permute import moe_token_permute_op_builder
from mindspeed.ops.npu_moe_token_unpermute import moe_token_unpermute_op_builder
from mindspeed.ops.npu_rotary_position_embedding import rope_op_builder

print(f"[precompile] torch={torch.__version__} torch_npu={torch_npu.__version__}", flush=True)
builders = (
    ("moe_token_permute", moe_token_permute_op_builder),
    ("moe_token_unpermute", moe_token_unpermute_op_builder),
    ("rotary_position_embedding", rope_op_builder),
    ("gmm", GMMOpBuilder()),
    ("groupmatmul_add", groupmatmul_add_op_builder),
    ("matmul_add", matmul_add_op_builder),
    ("fused_adamw_v2", FusedAdamWV2OpBuilder()),
)
for name, builder in builders:
    print(f"[precompile] loading {name}", flush=True)
    builder.load()
print("[precompile] all MindSpeed runtime extensions are ready", flush=True)
PY
}

# Load CANN and the installed ops-transformer environment. ModelArts may
# override the image entrypoint, so the launcher sources them explicitly.
set +u
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash ]; then
    source /usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
elif [ -f /usr/local/Ascend/vendors/custom_transformer/bin/set_env.bash ]; then
    source /usr/local/Ascend/vendors/custom_transformer/bin/set_env.bash
fi
set -u

unset TORCH_DEVICE_BACKEND_AUTOLOAD || true
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

GTS_ENABLE="${GTS_ENABLE:-1}"
GMM_PRECOMPILE="${GMM_PRECOMPILE:-0}"
USE_FUSED_MHC="${USE_FUSED_MHC:-1}"
GTS_ARGS="${GTS_ARGS:-}"

if [ -n "${NPUS_PER_NODE:-}" ]; then
    NPUS_PER_NODE="$NPUS_PER_NODE"
elif [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a visible_devices <<< "$ASCEND_VISIBLE_DEVICES"
    NPUS_PER_NODE="${#visible_devices[@]}"
else
    NPUS_PER_NODE=16
fi

NNODES="${NNODES:-${VC_WORKER_NUM:-${MA_NUM_HOSTS:-32}}}"
if ! NODE_RANK="$(derive_node_rank)"; then
    [ "$NNODES" = "1" ] || die "cannot determine NODE_RANK"
    NODE_RANK=0
fi

SCHEDULER_HOSTS="${VC_WORKER_HOSTS:-${MA_WORKER_HOSTS:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
if [ -z "$MASTER_ADDR" ] && [ -n "$SCHEDULER_HOSTS" ]; then
    MASTER_ADDR="$(first_host_from_list "$SCHEDULER_HOSTS")"
fi
if [ -z "$MASTER_ADDR" ] && [ -n "${MA_VJ_NAME:-}" ] \
    && [ -n "${MA_HJ_NAME:-}" ] && [ -n "${MA_NAMESPACE:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-worker-0.${MA_HJ_NAME}.${MA_NAMESPACE}.svc.cluster.local"
fi
MASTER_PORT="${MASTER_PORT:-8192}"

# Parallelism and model shape must match the Pro TP2/PP8/EP64 checkpoint.
TP=2
PP=8
EP=64
ETP=1
CP=1
CP_TYPE="ulysses_cp_algo"
NUM_LAYERS=64
NUM_EXPERTS=384
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GBS="${GBS:-1024}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
require_positive_integer TRAIN_ITERS "$TRAIN_ITERS"
require_positive_integer LR_DECAY_ITERS "$LR_DECAY_ITERS"
if [ -z "${WARMUP_ITERS+x}" ]; then
    if [ "$TRAIN_ITERS" -gt 500 ]; then
        WARMUP_ITERS=500
    else
        WARMUP_ITERS=$((TRAIN_ITERS / 4))
    fi
fi
if [ -z "${SAVE_INTERVAL+x}" ]; then
    if [ "$TRAIN_ITERS" -lt 500 ]; then
        SAVE_INTERVAL="$TRAIN_ITERS"
    else
        SAVE_INTERVAL=500
    fi
fi
LR="${LR:-1.0e-5}"
MIN_LR="${MIN_LR:-1.0e-7}"
SWIGLU_LIMIT="${SWIGLU_LIMIT:-10}"
FIX_ROUTER="${FIX_ROUTER:-1}"
NO_SHARED_STORAGE="${NO_SHARED_STORAGE:-1}"

# The deterministic fallback keeps all workers in the same output directory.
# Set RUN_ID explicitly when launching repeated jobs outside a scheduler.
RUN_ID="${RUN_ID:-${SFT_RUN_ID:-${MA_VJ_NAME:-${VC_JOB_ID:-deepseek4-pro-sft}}}}"
[[ "$RUN_ID" != *[[:space:]/]* ]] \
    || die "RUN_ID must not contain whitespace or '/': ${RUN_ID}"

DATA_PATH="${DATA_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-}"
require_path DATA_PATH
require_path TOKENIZER_PATH
require_path CKPT_LOAD_DIR

OUTPUT_ROOT="${OUTPUT_ROOT:-${SLAI_T_REX_ROOT}/outputs/sft}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${CKPT_SAVE_DIR}/logs}"

for item in \
    "NPUS_PER_NODE:$NPUS_PER_NODE" \
    "NNODES:$NNODES" \
    "MASTER_PORT:$MASTER_PORT" \
    "SEQ_LEN:$SEQ_LEN" \
    "MBS:$MBS" \
    "GBS:$GBS" \
    "TRAIN_ITERS:$TRAIN_ITERS" \
    "LR_DECAY_ITERS:$LR_DECAY_ITERS" \
    "SAVE_INTERVAL:$SAVE_INTERVAL"; do
    require_positive_integer "${item%%:*}" "${item#*:}"
done
require_nonnegative_integer NODE_RANK "$NODE_RANK"
require_nonnegative_integer WARMUP_ITERS "$WARMUP_ITERS"
require_boolean FIX_ROUTER "$FIX_ROUTER"
require_boolean NO_SHARED_STORAGE "$NO_SHARED_STORAGE"
require_boolean GTS_ENABLE "$GTS_ENABLE"
require_boolean GMM_PRECOMPILE "$GMM_PRECOMPILE"
require_boolean USE_FUSED_MHC "$USE_FUSED_MHC"

[ -n "$MASTER_ADDR" ] \
    || die "cannot determine MASTER_ADDR; export it or provide VC_WORKER_HOSTS/MA_WORKER_HOSTS"
[ "$NODE_RANK" -lt "$NNODES" ] \
    || die "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}"
[ "$NPUS_PER_NODE" -eq 16 ] \
    || die "this Pro recipe expects 16 NPUs per node, got ${NPUS_PER_NODE}"
[ "$MASTER_PORT" -le 65535 ] || die "MASTER_PORT must be at most 65535"
[ "$WARMUP_ITERS" -lt "$LR_DECAY_ITERS" ] \
    || die "WARMUP_ITERS must be smaller than LR_DECAY_ITERS"

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
[ "$WORLD_SIZE" -eq 512 ] \
    || die "TP2/PP8/EP64 Pro checkpoint requires 512 ranks: use 32 nodes x 16 NPUs (got ${WORLD_SIZE})"
[ $((WORLD_SIZE % (TP * PP * CP))) -eq 0 ] \
    || die "WORLD_SIZE is incompatible with TP*PP*CP"
[ $((WORLD_SIZE % (ETP * EP * PP))) -eq 0 ] \
    || die "WORLD_SIZE is incompatible with ETP*EP*PP"
[ $((NUM_EXPERTS % EP)) -eq 0 ] || die "NUM_EXPERTS must be divisible by EP"
DP=$((WORLD_SIZE / (TP * PP * CP)))
[ $((GBS % (MBS * DP))) -eq 0 ] \
    || die "GBS must be divisible by MBS*dense-DP=$((MBS * DP))"

[ -f "${MODEL_ROOT}/posttrain_gpt.py" ] \
    || die "release261 framework entrypoint not found: ${MODEL_ROOT}/posttrain_gpt.py"
[ -d "${MINDSPEED_ROOT}/mindspeed" ] \
    || die "release261 MindSpeed source not found: ${MINDSPEED_ROOT}/mindspeed"
[ -d "${MEGATRON_ROOT}/megatron" ] \
    || die "release261 Megatron source not found: ${MEGATRON_ROOT}/megatron"
[ -d "$CKPT_LOAD_DIR" ] || die "source checkpoint directory not found: $CKPT_LOAD_DIR"
[ -f "${TOKENIZER_PATH%/}/tokenizer.json" ] \
    || die "tokenizer.json not found under: $TOKENIZER_PATH"
for suffix in packed_input_ids_document packed_labels_document packed_attention_mask_document; do
    [ -f "${DATA_PATH}_${suffix}.bin" ] \
        || die "SFT dataset file not found: ${DATA_PATH}_${suffix}.bin"
    [ -f "${DATA_PATH}_${suffix}.idx" ] \
        || die "SFT dataset file not found: ${DATA_PATH}_${suffix}.idx"
done

SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
if [ -n "${SOURCE_MANIFEST}" ]; then
    [ -f "${SOURCE_MANIFEST}" ] || die "SOURCE_MANIFEST does not exist: ${SOURCE_MANIFEST}"
    grep -q 'MindSpeed .* v26.1.0_core_r0.12.1$' "${SOURCE_MANIFEST}" \
        || die "SOURCE_MANIFEST does not contain MindSpeed v26.1.0_core_r0.12.1"
    grep -q 'MindSpeed-LLM .* v26.1.0$' "${SOURCE_MANIFEST}" \
        || die "SOURCE_MANIFEST does not contain MindSpeed-LLM v26.1.0"
fi

if [ "$GTS_ENABLE" = "1" ]; then
    activate_gts_runtime
fi

export MODEL_ROOT MINDSPEED_ROOT MEGATRON_ROOT GTS_ENABLE USE_FUSED_MHC
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -u - <<'PY'
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path

from packaging.version import Version

expected = {
    "torch": "2.7.1",
    "torch-npu": "2.7.1.post8",
    "transformers": "5.2.0",
    "triton-ascend": "3.2.2",
}
for name, wanted in expected.items():
    actual = metadata.version(name)
    comparable = Version(actual).base_version if name == "torch" else actual
    if comparable != wanted:
        raise RuntimeError(f"{name}: expected {wanted}, got {actual}")
    print(f"{name}={actual}", flush=True)

model_root = Path(os.environ["MODEL_ROOT"])
csa_path = model_root / "mindspeed_llm/tasks/models/transformer/deepseek4/csa.py"
csa_source = csa_path.read_text()
for marker in ("gts_rmsnorm_without_weight", "gts_scale_rope", "is_enable_op"):
    if marker not in csa_source:
        raise RuntimeError(f"GTS framework patch marker {marker!r} is missing from {csa_path}")
print(f"GTS framework patch markers ready: {csa_path}", flush=True)

import megatron.core.datasets.helpers_cpp as helpers_cpp
print(f"helpers_cpp={helpers_cpp.__file__}", flush=True)

custom_ops = importlib.import_module("cann_ops_transformer.ops")
required_sparse_ops = (
    "sparse_flash_mla",
    "sparse_flash_mla_metadata",
    "sparse_flash_mla_grad",
    "sparse_flash_mla_grad_metadata",
    "lightning_indexer",
    "lightning_indexer_metadata",
    "sparse_lightning_indexer_kl_loss_grad",
    "sparse_lightning_indexer_kl_loss_grad_metadata",
)
missing = [name for name in required_sparse_ops if not hasattr(custom_ops, name)]
if missing:
    raise RuntimeError(f"ops-transformer SparseAttn interfaces are missing: {missing}")
print("ops-transformer SparseFlashMla/LightningIndexer interfaces ready", flush=True)

if os.environ["USE_FUSED_MHC"] == "1":
    required_mhc_ops = ("mhc_pre_sinkhorn", "mhc_post")
    missing = [name for name in required_mhc_ops if not hasattr(custom_ops, name)]
    if missing:
        raise RuntimeError(
            "USE_FUSED_MHC=1 but ops-transformer MHC interfaces are missing: "
            f"{missing}. The image must provide the MHC interfaces used by the "
            "reference Pro SFT recipe, or USE_FUSED_MHC must be explicitly set to 0."
        )

if os.environ["GTS_ENABLE"] == "1":
    gtsops = importlib.import_module("gtsops")
    gtsspeed_ops = importlib.import_module("gtsspeed.ops")
    gts_custom = importlib.import_module("mindspeed.gts.ops.gts_custom_op")
    importlib.import_module("mindspeed.gts.ops.gts_rope")

    enabled_features = sorted(
        name.removeprefix("GTS_ENABLE_OP_").lower()
        for name, value in os.environ.items()
        if name.startswith("GTS_ENABLE_OP_") and value.lower() == "true"
    )
    feature_symbols = {
        "swiglu1": ("wind_swiglu_limit",),
        "swiglu1_probs": ("wind_swiglu_limit_probs",),
        "rms0": ("wind_rms_norm_without_weight",),
        "rope": ("wind_scale_rope",),
        "compressor0": ("wind_fused_overlap_transform",),
        "make_chunk_sort_map": ("wind_make_chunk_sort_map",),
    }
    required_gts_functions = sorted({
        symbol
        for feature in enabled_features
        for symbol in feature_symbols.get(feature, ())
    })
    missing = [
        name for name in required_gts_functions
        if not hasattr(gtsspeed_ops, name)
    ]
    if missing:
        raise RuntimeError(
            "GTS interfaces required by the enabled set_env.sh features are missing: "
            f"{missing}"
        )
    print(
        "GTS features enabled by set_env.sh: "
        + (", ".join(enabled_features) if enabled_features else "none"),
        flush=True,
    )

    origins = {
        "gtsops": Path(gtsops.__file__).resolve(),
        "gtsspeed.ops": Path(gtsspeed_ops.__file__).resolve(),
        "mindspeed.gts": Path(gts_custom.__file__).resolve(),
    }
    for name, origin in origins.items():
        print(f"{name}={origin}", flush=True)
    print("GTS Python, custom OPP and feature switches ready", flush=True)
PY

if [ "$GMM_PRECOMPILE" = "1" ]; then
    precompile_mindspeed_extensions
fi

mkdir -p "$CKPT_SAVE_DIR" "$RUN_DIR"
cd "$MODEL_ROOT"

echo "===== DeepSeek-V4-Pro 4K SFT ====="
echo "run_id=${RUN_ID}"
echo "host=${HOSTNAME:-unknown} node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "world_size=${WORLD_SIZE} npu_per_node=${NPUS_PER_NODE} TP=${TP} PP=${PP} EP=${EP} CP=${CP} DP=${DP}"
echo "framework=${MODEL_ROOT}"
echo "gts_enable=${GTS_ENABLE} gts_env=${GTS_ENV_SCRIPT:-preconfigured}"
echo "fused_mhc=${USE_FUSED_MHC}"
echo "data=${DATA_PATH}"
echo "load=${CKPT_LOAD_DIR}"
echo "save=${CKPT_SAVE_DIR}"
echo "iters=${TRAIN_ITERS} warmup=${WARMUP_ITERS} save_interval=${SAVE_INTERVAL} gbs=${GBS} lr=${LR}"

torchrun_args=(
    --nproc-per-node="$NPUS_PER_NODE"
    --nnodes="$NNODES"
    --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR"
    --master-port="$MASTER_PORT"
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
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --num-layers-per-virtual-pipeline-stage 4
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size "$ETP"
    --sequence-parallel
    --context-parallel-size "$CP"
    --context-parallel-algo "$CP_TYPE"
    --num-layers "$NUM_LAYERS"
    --hidden-size 7168
    --ffn-hidden-size 7168
    --num-attention-heads 128
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "$TOKENIZER_PATH"
    --seq-length "$SEQ_LEN"
    --max-position-embeddings 163840
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --make-vocab-size-divisible-by 1
    --lr "$LR"
    --train-iters "$TRAIN_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.02
    --hidden-dropout 0.0
    --position-embedding-type deepseek4
    --normalization RMSNorm
    --use-fused-swiglu
    --use-fused-rmsnorm
    --swiglu
    --swiglu-limit "$SWIGLU_LIMIT"
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr "$MIN_LR"
    --weight-decay 1e-2
    --lr-warmup-iters "$WARMUP_ITERS"
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
if [ "$NO_SHARED_STORAGE" = "1" ]; then
    model_args+=(--no-shared-storage)
fi

dsa_args=(
    --enable-dsa-indexer
    --index-n-heads 64
    --index-head-dim 128
    --index-topk 1024
    --enable-mhc
    --hc-mult 4
    --hc-sinkhorn-iters 20
    --kv-compress
    --use-triton-mhc
    --use-fused-lightning-indexer-loss
    --use-fused-lightning-indexer
    --use-sparse-flash-attn
    --indexer-loss-coeff 1.0
)
if [ "$USE_FUSED_MHC" = "1" ]; then
    dsa_args+=(--use-fused-mhc)
fi

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
[ "${#compress_ratios[@]}" -eq $((NUM_LAYERS + 1)) ] \
    || die "compress-ratios must contain NUM_LAYERS + one MTP entry"
csa_args=(
    --o-groups 16
    --sliding-window-size 128
    --original-seq-len 65536
    --rope-factor 16
    --compress-rope-theta 160000.0
    --max-batch-size 4
    --compress-ratios "${compress_ratios[@]}"
)

moe_args=(
    --moe-grouped-gemm
    --moe-permutation-async-comm
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq 1
    --first-k-dense-replace -1
    --num-experts "$NUM_EXPERTS"
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
if [ "$FIX_ROUTER" = "1" ]; then
    moe_args+=(--fix-router)
fi

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
    --rope-scaling-original-max-position-embeddings "$SEQ_LEN"
    --rope-scaling-type yarn
)

data_args=(
    --data-path "$DATA_PATH"
    --split 100,0,0
)

output_args=(
    --log-interval 1
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$TRAIN_ITERS"
    --eval-iters 0
    --save "$CKPT_SAVE_DIR"
    --load "$CKPT_LOAD_DIR"
    --ckpt-format torch
    --no-save-optim
    --no-save-rng
    --log-throughput
)

finetune_args=(
    --finetune
    --stage sft
    --is-instruction-dataset
    --prompt-type deepseek4
)

gts_args=()
if [ -n "${GTS_ARGS//[[:space:]]/}" ]; then
    read -r -a gts_args <<< "$GTS_ARGS"
fi

LOG_FILE="${RUN_DIR}/node-${NODE_RANK}.log"
torchrun "${torchrun_args[@]}" "$MODEL_ROOT/posttrain_gpt.py" \
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
    "${finetune_args[@]}" \
    "${gts_args[@]}" \
    --distributed-backend nccl 2>&1 | tee "$LOG_FILE"
