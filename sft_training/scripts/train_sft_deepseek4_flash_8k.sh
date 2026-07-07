#!/bin/bash
# DeepSeek-V4-Flash 8K SFT training template.
#
# Required runtime inputs:
#   DATA_PATH       MindSpeed instruction dataset prefix.
#   TOKENIZER_PATH  HuggingFace tokenizer/model path.
#   CKPT_LOAD_DIR   MCore checkpoint directory to finetune from.

set -o pipefail
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_LLM_DIR="${MINDSPEED_LLM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MINDSPEED_DIR="${MINDSPEED_DIR:-${MINDSPEED_LLM_DIR}/../MindSpeed}"

required_var() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "${value}" || "${value}" == /path/to/* ]]; then
        echo "[ERROR] ${name} must be set." >&2
        exit 1
    fi
}

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

resolve_master_addr() {
    if [[ -n "${MASTER_ADDR:-}" ]]; then
        echo "${MASTER_ADDR}"
    elif [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
        echo "${VC_WORKER_HOSTS%%,*}"
    elif [[ -n "${MA_VJ_NAME:-}" && -n "${MA_HJ_NAME:-}" && -n "${MA_NAMESPACE:-}" ]]; then
        echo "${MA_VJ_NAME}-worker-0.${MA_HJ_NAME}.${MA_NAMESPACE}.svc.cluster.local"
    else
        echo "127.0.0.1"
    fi
}

DATA_PATH="${DATA_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer_or_model}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-/path/to/mcore_checkpoint}"
required_var DATA_PATH
required_var TOKENIZER_PATH
required_var CKPT_LOAD_DIR

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
CUSTOM_TRANSFORMER_ENV="${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash}"
if [[ -f "${CANN_ENV}" ]]; then
    source "${CANN_ENV}"
fi
if [[ -f "${CUSTOM_TRANSFORMER_ENV}" ]]; then
    source "${CUSTOM_TRANSFORMER_ENV}"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "${CONDA_SH:-${HOME}/miniconda3/bin/activate}" "${CONDA_ENV}"
fi

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-3000}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-7200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_ALGO="${HCCL_ALGO:-alltoall=level0:NA;level1:pipeline}"
export PYTHONUNBUFFERED=1

JOB_NAME="${JOB_NAME:-deepseek4_sft_8k_gbs128}"
DESCRIPTION="${DESCRIPTION:-DeepSeek-V4-Flash 8K SFT, weighted/user-provided data}"
NODE_RANK="$(infer_node_rank)"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
NNODES="${NNODES:-${VC_WORKER_NUM:-8}}"
MASTER_ADDR="$(resolve_master_addr)"
MASTER_PORT="${MASTER_PORT:-8192}"
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

if [[ -n "${RUN_ID:-}" ]]; then
    TIMESTAMP="${TIMESTAMP:-external}"
else
    TIMESTAMP="${TIMESTAMP:-${SFT_RUN_ID:-${VC_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M)}}}}"
    RUN_ID="${JOB_NAME}_${TIMESTAMP}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${MINDSPEED_LLM_DIR}/outputs/deepseek4_sft}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_ID}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${RUN_ID}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${OUTPUT_ROOT}/archive}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${OUTPUT_ROOT}/model_registry}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ARCHIVE_ROOT}/${RUN_ID}}"
TRAIN_LOG="${TRAIN_LOG:-logs/${RUN_ID}_rank${NODE_RANK}.log}"
MODEL_REGISTRY_TOOL="${MODEL_REGISTRY_TOOL:-${MINDSPEED_LLM_DIR}/../model_registry_tools/register_manifest.py}"

mkdir -p "${CKPT_SAVE_DIR}" "${TENSORBOARD_DIR}" "${ARCHIVE_DIR}" logs

TP="${TP:-1}"
PP="${PP:-4}"
EP="${EP:-32}"
CP="${CP:-1}"
CP_TYPE="${CP_TYPE:-ulysses_cp_algo}"
NUM_LAYERS="${NUM_LAYERS:-44}"
SEQ_LEN="${SEQ_LEN:-8192}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
LR="${LR:-5.0e-6}"
MIN_LR="${MIN_LR:-5.0e-8}"
TRAIN_ITERS="${TRAIN_ITERS:-250}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-22}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
CKPT_STEP="${CKPT_STEP:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

DSA_ARGS="
    --enable-dsa-indexer \
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 512 \
    --enable-mhc \
    --hc-mult 4 \
    --kv-compress \
    --norm-eps 1e-6 \
    --use-triton-sinkhorn \
    --use-triton-mhc \
    --use-triton-rmsnorm-without-weight \
    --use-fused-lightning-indexer-loss \
    --use-fused-lightning-indexer \
    --use-sparse-flash-attn \
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 512 \
    --q-lora-rank 1024 \
    --o-lora-rank 1024 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-fa-without-pad \
"

CA_ARGS="
    --use-g2-attention \
    --o-groups 8 \
    --g2-window-size 128 \
    --rope-head-dim 64 \
    --original-seq-len 65536 \
    --rope-factor 16 \
    --compress-rope-theta 160000.0 \
    --max-batch-size 4 \
    --compress-ratios 0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 \
    --use-g2-indexer-loss \
"

MOE_ARGS="
    --fix-router \
    --moe-grouped-gemm \
    --moe-layer-freq 1 \
    --moe-token-dispatcher-type alltoall \
    --moe-permutation-async-comm \
    --moe-permute-fusion \
    --first-k-dense-replace -1 \
    --num-experts 256 \
    --moe-router-topk 6 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 1 \
    --moe-router-topk-scaling-factor 1.5 \
    --seq-aux \
    --moe-aux-loss-coeff 0.001 \
    --moe-router-score-function sqrtsoftplus \
    --moe-router-enable-expert-bias \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-dtype fp32 \
    --n-hash-layers 3 \
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
"

MEM_ARGS="
    --mtp-mem-efficient-logits \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --swap-optimizer \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 8192 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --noop-layers 43 \
    --transformer-impl local \
    --spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec \
    --mtp-spec mindspeed_llm.tasks.models.spec.deepseek4_spec mtp_spec \
    --manual-gc \
    --manual-gc-interval 50 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr ${LR} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type g2 \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --no-shared-storage \
    --no-gradient-accumulation-fusion \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard \
    --save ${CKPT_SAVE_DIR} \
    --load ${CKPT_LOAD_DIR} \
    --ckpt-step ${CKPT_STEP} \
    --ckpt-format torch \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL:-2000} \
    --eval-iters ${EVAL_ITERS:-0} \
    --no-save-optim \
    --no-save-rng \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type ${PROMPT_TYPE:-deepseek4} \
"

cd "${MINDSPEED_LLM_DIR}"
export PYTHONPATH="${MINDSPEED_DIR}:${MINDSPEED_LLM_DIR}/Megatron-LM:${PYTHONPATH:-}"

if [[ "${NODE_RANK}" == "0" ]]; then
    cp "$0" "${ARCHIVE_DIR}/train_script.sh"
    export DESCRIPTION RUN_ID TIMESTAMP NNODES NPUS_PER_NODE WORLD_SIZE TP PP EP CP
    export NUM_LAYERS SEQ_LEN MBS GBS TRAIN_ITERS LR_WARMUP_ITERS LR MIN_LR
    export SAVE_INTERVAL CKPT_STEP DATA_PATH TOKENIZER_PATH CKPT_LOAD_DIR CKPT_SAVE_DIR
    export TENSORBOARD_DIR ARCHIVE_DIR
    python3 <<'PY'
import json
import os

def env_int(name):
    return int(os.environ[name])

def env_float(name):
    return float(os.environ[name])

cfg = {
    "description": os.environ.get("DESCRIPTION", ""),
    "job_type": "sft",
    "run_id": os.environ["RUN_ID"],
    "timestamp": os.environ["TIMESTAMP"],
    "nnodes": env_int("NNODES"),
    "npus_per_node": env_int("NPUS_PER_NODE"),
    "world_size": env_int("WORLD_SIZE"),
    "tp": env_int("TP"),
    "pp": env_int("PP"),
    "ep": env_int("EP"),
    "cp": env_int("CP"),
    "num_layers": env_int("NUM_LAYERS"),
    "seq_len": env_int("SEQ_LEN"),
    "mbs": env_int("MBS"),
    "gbs": env_int("GBS"),
    "train_iters": env_int("TRAIN_ITERS"),
    "warmup_iters": env_int("LR_WARMUP_ITERS"),
    "lr": env_float("LR"),
    "min_lr": env_float("MIN_LR"),
    "save_interval": env_int("SAVE_INTERVAL"),
    "ckpt_step": os.environ["CKPT_STEP"],
    "data_path": os.environ["DATA_PATH"],
    "tokenizer": os.environ["TOKENIZER_PATH"],
    "ckpt_load": os.environ["CKPT_LOAD_DIR"],
    "ckpt_save": os.environ["CKPT_SAVE_DIR"],
    "hf_output_dir": os.environ["CKPT_SAVE_DIR"] + "_hf",
    "tensorboard": os.environ["TENSORBOARD_DIR"],
    "archive": os.environ["ARCHIVE_DIR"],
}
with open(os.path.join(os.environ["ARCHIVE_DIR"], "config.json"), "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("config.json saved")
PY

    if [[ -n "${MODEL_REGISTRY_TOOL}" && -f "${MODEL_REGISTRY_TOOL}" ]]; then
        python3 "${MODEL_REGISTRY_TOOL}" training \
            --archive-root "${ARCHIVE_ROOT}" \
            --registry-root "${REGISTRY_ROOT}" \
            --config "${ARCHIVE_DIR}/config.json" \
            --job-type sft \
            --ckpt-step "${CKPT_STEP}" \
            --hf-output-dir "${CKPT_SAVE_DIR}_hf" \
            || echo "[WARN] model registry training registration failed"
    fi
fi

python -m torch.distributed.launch ${DISTRIBUTED_ARGS} posttrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${MLA_ARGS} \
    ${ROPE_ARGS} \
    ${MOE_ARGS} \
    ${DSA_ARGS} \
    ${CA_ARGS} \
    ${MEM_ARGS} \
    ${MTP_ARGS} \
    ${FINETUNE_ARGS} \
    --distributed-backend nccl 2>&1 | tee "${TRAIN_LOG}"
TRAIN_RC=${PIPESTATUS[0]}

if [[ "${NODE_RANK}" == "0" ]]; then
    echo "============================================"
    echo "  Training finished, archive: ${ARCHIVE_DIR}"
    echo "============================================"

    grep "iteration" "${TRAIN_LOG}" | grep "lm loss" > "${ARCHIVE_DIR}/metrics.tsv" 2>/dev/null || true
    export ARCHIVE_DIR DATA_PATH TRAIN_ITERS GBS SEQ_LEN
    python3 <<'PY' 2>/dev/null || true
import json
import os
import re

archive = os.environ["ARCHIVE_DIR"]
try:
    with open(os.path.join(archive, "metrics.tsv")) as f:
        lines = f.readlines()
    data = {"iter": [], "lm_loss": [], "lr": []}
    for line in lines:
        m = re.search(r"iteration\s+(\d+)", line)
        if m:
            data["iter"].append(int(m.group(1)))
        m = re.search(r"lm loss:\s*([\d.]+E[+-]\d+)", line)
        if m:
            data["lm_loss"].append(float(m.group(1)))
        m = re.search(r"learning rate:\s*([\d.]+E[+-]\d+)", line)
        if m:
            data["lr"].append(float(m.group(1)))
    with open(os.path.join(archive, "metrics.json"), "w") as f:
        json.dump(data, f)
    print(f"Metrics: {len(data['iter'])} iters")
except Exception:
    print("(metrics extraction skipped)")
PY

    python3 <<'PY' 2>/dev/null || echo "(data_stats skipped)"
from megatron.core.datasets.indexed_dataset import IndexedDataset
import json
import os

p = os.environ["DATA_PATH"]
if os.path.exists(p + ".bin"):
    ds = IndexedDataset(p)
    stats = {
        "total_tokens": int(sum(ds.sequence_lengths)),
        "total_seqs": len(ds.sequence_lengths),
        "consumed_est": int(os.environ["TRAIN_ITERS"]) * int(os.environ["GBS"]) * int(os.environ["SEQ_LEN"]),
    }
    with open(os.path.join(os.environ["ARCHIVE_DIR"], "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Data stats: {stats['total_tokens']:,} tokens")
else:
    print("(data_stats skipped: no .bin found)")
PY

    [[ -d "${TENSORBOARD_DIR}" ]] && cp -r "${TENSORBOARD_DIR}" "${ARCHIVE_DIR}/tensorboard" 2>/dev/null || true
    ls -lh "${ARCHIVE_DIR}"/
fi

if [[ "${TRAIN_RC:-0}" -ne 0 ]]; then
    echo "[ERROR] Training failed with exit code ${TRAIN_RC}; skip HF conversion."
    exit "${TRAIN_RC}"
fi

if [[ "${NODE_RANK}" == "0" && "${ENABLE_HF_CONVERT:-0}" == "1" ]]; then
    if [[ -z "${HF_CONVERT_SCRIPT:-}" || ! -f "${HF_CONVERT_SCRIPT}" ]]; then
        echo "[WARN] ENABLE_HF_CONVERT=1 but HF_CONVERT_SCRIPT is missing; skip conversion."
        exit 0
    fi

    echo "=============================================="
    echo "[Convert] MCore checkpoint -> HF"
    echo "  Load: ${CKPT_SAVE_DIR}"
    echo "  Save: ${CKPT_SAVE_DIR}_hf"
    echo "=============================================="

    if ! python "${HF_CONVERT_SCRIPT}" \
        --load-dir "${CKPT_SAVE_DIR}" \
        --hf-cfg-dir "${TOKENIZER_PATH}" \
        --save-dir "${CKPT_SAVE_DIR}_hf" \
        --noop-layers 43 \
        --mtp-num-layers 1; then
        echo "[ERROR] HF conversion failed; skip artifact registration."
        exit 1
    fi

    echo "[Convert] Done: ${CKPT_SAVE_DIR}_hf"

    if [[ -n "${MODEL_REGISTRY_TOOL}" && -f "${MODEL_REGISTRY_TOOL}" && -d "${CKPT_SAVE_DIR}_hf" ]]; then
        SFT_SOURCE_ITER="${TRAIN_ITERS}"
        if [[ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
            SFT_SOURCE_ITER="$(cat "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "${TRAIN_ITERS}")"
        fi
        python3 "${MODEL_REGISTRY_TOOL}" artifact \
            --archive-root "${ARCHIVE_ROOT}" \
            --registry-root "${REGISTRY_ROOT}" \
            --source-training-run-id "${RUN_ID}" \
            --source-mcore-ckpt-dir "${CKPT_SAVE_DIR}" \
            --source-iter "${SFT_SOURCE_ITER}" \
            --hf-dir "${CKPT_SAVE_DIR}_hf" \
            --converter-script "${HF_CONVERT_SCRIPT}" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            || echo "[WARN] model registry artifact registration failed"
    fi
fi
