#!/bin/bash
set -o pipefail

# DeepSeek-V4-Flash 4K CPT training script
#
# Required runtime input:
#   TRAIN_DATA_PATH  Weighted MindSpeed indexed dataset prefixes.
#
# Outputs:
#   Checkpoints, TensorBoard events, and archive metadata under OUTPUT_ROOT.

JOB_NAME=${JOB_NAME:-"cpt_weighted_mix_gbs128"}
DESCRIPTION="${DESCRIPTION:-Weighted CPT data mixture, GBS128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_LLM_DIR="${MINDSPEED_LLM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MINDSPEED_DIR="${MINDSPEED_DIR:-${MINDSPEED_LLM_DIR}/../MindSpeed}"

DATA_ROOT="${DATA_ROOT:-/path/to/processed_datasets}"
VALID_DATA_PATH="${VALID_DATA_PATH:-${DATA_ROOT}/validation/val_text_document}"

TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/DeepSeek-V4-Flash}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-/path/to/DeepSeek-V4-Flash-BF16-mcore}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MINDSPEED_LLM_DIR}/outputs/deepseek4_posttraining}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"

# total consumed tokens = TRAIN_ITERS * GBS * SEQ_LEN.
TRAIN_ITERS=280
WARMUP_ITERS=10
LR=3.0e-6
MIN_LR=3.0e-7
WEIGHT_DECAY=1e-2
GBS=128
MBS=1
SAVE_INTERVAL=140
EVAL_INTERVAL=50
EVAL_ITERS=10

NNODES=${NNODES:-${VC_WORKER_NUM:-8}}
TP=1
PP=4
EP=32
CP=1
SEQ_LEN=4096
NUM_LAYERS=44

if [[ -z "${TRAIN_DATA_PATH//[[:space:]]/}" ]]; then
    echo "[ERROR] TRAIN_DATA_PATH is empty."
    echo "[ERROR] Set it before launch, for example:"
    echo "  export TRAIN_DATA_PATH=\$'0.7 /path/to/data_a_text_document\\n0.3 /path/to/data_b_text_document'"
    exit 1
fi

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
CUSTOM_TRANSFORMER_ENV="${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash}"
if [[ -f "${CANN_ENV}" ]]; then
    source "${CANN_ENV}"
fi
if [[ -f "${CUSTOM_TRANSFORMER_ENV}" ]]; then
    source "${CUSTOM_TRANSFORMER_ENV}"
fi

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

NPUS_PER_NODE=${NPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-$(echo "${VC_WORKER_HOSTS:-127.0.0.1}" | cut -d ',' -f 1)}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# Prefer a scheduler-provided run id so all nodes write to the same directory.
if [[ -n "${RUN_ID:-}" ]]; then
    TIMESTAMP="${TIMESTAMP:-external}"
else
    TIMESTAMP="${TIMESTAMP:-${VC_JOB_ID:-${BATCH_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M)}}}}"
    RUN_ID="${JOB_NAME}_${TIMESTAMP}"
fi

CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_ID}}"
TB_DIR="${TB_DIR:-${OUTPUT_ROOT}/tensorboard/${RUN_ID}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${OUTPUT_ROOT}/archive}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${OUTPUT_ROOT}/model_registry}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ARCHIVE_ROOT}/${RUN_ID}}"
TRAIN_LOG="logs/${RUN_ID}.log"
MODEL_REGISTRY_TOOL="${MODEL_REGISTRY_TOOL:-${MINDSPEED_LLM_DIR}/../model_registry_tools/register_manifest.py}"

CP_TYPE='ulysses_cp_algo'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq 1 \
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
    --moe-permute-fusion \
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
    --rope-scaling-original-max-position-embeddings 4096 \
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
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr ${LR} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --lr-decay-iters ${TRAIN_ITERS} \
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
    --lr-warmup-iters ${WARMUP_ITERS} \
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
    --tensorboard-dir ${TB_DIR} \
    --log-timers-to-tensorboard \
    --log-throughput \
    --ckpt-format torch \
"

DATA_ARGS="
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --no-save-optim \
    --no-save-rng \
"

mkdir -p "$ARCHIVE_DIR"

python3 -c "
import json, os, re

train_list = []
for line in '''$TRAIN_DATA_PATH'''.strip().split('\n'):
    line = line.strip()
    if not line or line.startswith('#'): continue
    parts = line.split(None, 1)
    if len(parts) == 2:
        try: train_list.append({'weight': float(parts[0]), 'path': parts[1]})
        except: train_list.append({'weight': 1.0, 'path': line})
    else:
        train_list.append({'weight': 1.0, 'path': parts[0]})

cfg = {
    'description': '$DESCRIPTION',
    'job_type': 'cpt',
    'job_name': '$JOB_NAME',
    'run_id': '$RUN_ID',
    'timestamp': '$TIMESTAMP',
    'nnodes': $NNODES,
    'npus_per_node': $NPUS_PER_NODE,
    'world_size': $WORLD_SIZE,
    'tp': $TP, 'pp': $PP, 'ep': $EP, 'cp': $CP,
    'num_layers': $NUM_LAYERS,
    'seq_len': $SEQ_LEN,
    'mbs': $MBS, 'gbs': $GBS,
    'train_iters': $TRAIN_ITERS,
    'warmup_iters': $WARMUP_ITERS,
    'lr': $LR, 'min_lr': $MIN_LR,
    'weight_decay': $WEIGHT_DECAY,
    'save_interval': $SAVE_INTERVAL,
    'eval_interval': $EVAL_INTERVAL,
    'eval_iters': $EVAL_ITERS,
    'train_data': train_list,
    'valid_data': '$VALID_DATA_PATH'.strip(),
    'tokenizer': '$TOKENIZER_PATH',
    'ckpt_load': '$CKPT_LOAD_DIR',
    'ckpt_save': '$CKPT_SAVE_DIR',
    'tensorboard': '$TB_DIR',
    'archive': '$ARCHIVE_DIR'
}
os.makedirs('$ARCHIVE_DIR', exist_ok=True)
with open('$ARCHIVE_DIR/config.json', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print('config.json saved')
"

if [[ "${NODE_RANK}" == "0" ]] && [[ -f "${MODEL_REGISTRY_TOOL}" ]]; then
    python3 "${MODEL_REGISTRY_TOOL}" training \
        --archive-root "${ARCHIVE_ROOT}" \
        --registry-root "${REGISTRY_ROOT}" \
        --config "${ARCHIVE_DIR}/config.json" \
        --job-type cpt \
        || echo "[WARN] model registry training registration failed"
fi

cp "$0" "$ARCHIVE_DIR/train_script.sh"

echo "============================================"
echo "  Job: $JOB_NAME  |  ID: $RUN_ID"
echo "  Archive: $ARCHIVE_DIR"
echo "============================================"

COMPLETED_ITER_FILE="${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt"
if [[ -f "${COMPLETED_ITER_FILE}" ]]; then
    COMPLETED_ITER=$(tr -cd '0-9' < "${COMPLETED_ITER_FILE}" || true)
    COMPLETED_ITER=${COMPLETED_ITER:-0}
    if (( COMPLETED_ITER >= TRAIN_ITERS )) && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "[INFO] checkpoint already completed: ${CKPT_SAVE_DIR} (iter ${COMPLETED_ITER}/${TRAIN_ITERS})"
        echo "[INFO] skip training. Set FORCE_RERUN=1 only if you intentionally want to overwrite/retrain."
        exit 0
    fi
    if [[ "${RESUME_FROM_SAVE:-0}" != "1" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "[ERROR] found incomplete checkpoint: ${CKPT_SAVE_DIR} (iter ${COMPLETED_ITER}/${TRAIN_ITERS})"
        echo "[ERROR] refusing to restart from base checkpoint and risk overwriting partial output."
        echo "[ERROR] set FORCE_RERUN=1 to intentionally rerun, or RESUME_FROM_SAVE=1 after adding resume logic."
        exit 1
    fi
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    source "${CONDA_SH:-${HOME}/miniconda3/bin/activate}" "${CONDA_ENV}"
fi

cd "${MINDSPEED_LLM_DIR}"
mkdir -p "$CKPT_SAVE_DIR"
mkdir -p logs

export PYTHONPATH="${MINDSPEED_DIR}:${MINDSPEED_LLM_DIR}/Megatron-LM:${PYTHONPATH:-}"
MINDSPEED_DATA_CACHE="${MINDSPEED_DATA_CACHE:-/tmp/mindspeed_data_cache}"
rm -rf "${MINDSPEED_DATA_CACHE}"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_deepseek4.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $DSA_ARGS \
    $CA_ARGS \
    $MEM_ARGS \
    $MTP_ARGS \
    --distributed-backend nccl 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo "[ERROR] training command failed with exit code ${TRAIN_EXIT}"
    exit ${TRAIN_EXIT}
fi

if [[ $NODE_RANK == "0" ]]; then
    echo ""
    echo "============================================"
    echo "  Training finished; archiving to $ARCHIVE_DIR"
    echo "============================================"

    cp "$TRAIN_LOG" "$ARCHIVE_DIR/train.log"

    grep "iteration" "$TRAIN_LOG" | grep "lm loss" > "$ARCHIVE_DIR/metrics.tsv"

    python3 -c "
import json, re
with open('${ARCHIVE_DIR}/metrics.tsv') as f:
    lines = f.readlines()
data = {'iter':[], 'lm_loss':[], 'lr':[], 'throughput':[], 'grad_norm':[]}
for line in lines:
    m = re.search(r'iteration\s+(\d+)', line)
    if m: data['iter'].append(int(m.group(1)))
    m = re.search(r'lm loss:\s*([\d.]+E[+-]\d+)', line)
    if m: data['lm_loss'].append(float(m.group(1)))
    m = re.search(r'learning rate:\s*([\d.]+E[+-]\d+)', line)
    if m: data['lr'].append(float(m.group(1)))
    m = re.search(r'throughput.*?:\s*([\d.]+)', line)
    if m: data['throughput'].append(float(m.group(1)))
    m = re.search(r'grad norm:\s*([\d.]+)', line)
    if m: data['grad_norm'].append(float(m.group(1)))
with open('${ARCHIVE_DIR}/metrics.json', 'w') as f:
    json.dump(data, f)
print(f'Metrics: {len(data[\"iter\"])} iters extracted')
" 2>/dev/null || echo "(metrics extraction skipped)"

    if [ -d "$TB_DIR" ]; then
        cp -r "$TB_DIR" "$ARCHIVE_DIR/tensorboard"
    fi

    python3 -c "
from megatron.core.datasets.indexed_dataset import IndexedDataset
import json, os

ARCHIVE='${ARCHIVE_DIR}'
with open(f'{ARCHIVE}/config.json') as f:
    cfg = json.load(f)

train_list = cfg.get('train_data', [])
train_iters = cfg['train_iters']
tokens_per_iter = cfg['gbs'] * cfg['seq_len']
total_consumed = train_iters * tokens_per_iter

datasets = [(d['weight'], d['path']) for d in train_list]
total_weight = sum(w for w, _ in datasets)

stats = {'train': {}, 'valid': {}, 'summary': {}}
for w, p in datasets:
    name = os.path.basename(p).replace('_text_document', '')
    if os.path.exists(p + '.bin'):
        ds = IndexedDataset(p)
        t = int(sum(ds.sequence_lengths))
        n = len(ds.sequence_lengths)
        consumed = int(total_consumed * w / total_weight) if total_weight > 0 else 0
        stats['train'][name] = {
            'total_tokens': t, 'total_seqs': n,
            'weight': w, 'weight_pct': round(100 * w / total_weight, 1),
            'consumed_tokens_est': consumed,
            'epochs_est': round(consumed / t, 2) if t > 0 else 0
        }

val = cfg.get('valid_data', '').strip()
if val and os.path.exists(val + '.bin'):
    ds = IndexedDataset(val)
    stats['valid'] = {'total_tokens': int(sum(ds.sequence_lengths)), 'total_seqs': len(ds.sequence_lengths)}

stats['summary'] = {
    'tokens_per_iter': tokens_per_iter,
    'total_iters': train_iters,
    'total_consumed': total_consumed
}
with open(f'{ARCHIVE}/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('Data stats saved')
"

    echo ""
    echo "============================================"
    echo "  Archive complete"
    ls -lh "$ARCHIVE_DIR"/
    echo "============================================"
fi
