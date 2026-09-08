# CPT Training

MindSpeed-LLM 4K CPT launchers for DeepSeek-V4-Flash and DeepSeek-V4-Pro on Ascend. Conversion scripts: [`../convert`](../convert/).

The launchers resolve MindSpeed-LLM, MindSpeed, and Megatron in this order:

1. `MINDSPEED_LLM_DIR` / `MINDSPEED_DIR` / `MEGATRON_DIR`;
2. repositories vendored in or cloned next to SLAI-T-Rex;
3. `/workspace/MindSpeed-LLM`, `/workspace/MindSpeed`, and `/workspace/Megatron-LM`.

```bash
export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
```

**Data** (`{"text": "..."}` JSONL):

```bash
cd SLAI-T-Rex/training/cpt
bash ../convert/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --input /path/to/cpt_corpus.jsonl \
  --output-prefix /path/to/processed/or_cpt_corpus \
  --tokenizer "$TOKENIZER_PATH" \
  --handler-name GeneralPretrainHandler \
  --json-keys text --seq-length 4096 --workers 8 --n-subs 16
```

**Checkpoint:** FP8 HF → BF16 (`../convert/convert_ckpt_fp8_to_bf16.sh`) → MCore (`../convert/convert_ckpt_hf_to_mcore.sh`).

**DeepSeek-V4-Flash** (defaults: `SEQ_LEN=4096`, `GBS=128`, `TP=1,PP=4,EP=32`):

```bash
export CKPT_LOAD_DIR=/path/to/mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export TRAIN_DATA_PATH=$'1.0 /path/to/processed/or_cpt_corpus_text_document'
export NNODES=8 NPUS_PER_NODE=16 MASTER_ADDR=MASTER_NODE_HOST NODE_RANK=0
bash train_cpt_deepseek4_flash_4k.sh
```

**DeepSeek-V4-Pro** (defaults: `SEQ_LEN=4096`, `GBS=256`, `TP=2,PP=8,EP=64`):

```bash
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Pro
export CKPT_LOAD_DIR=/path/to/deepseek4_pro_mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export TRAIN_DATA_PATH=$'0.8 /path/to/processed/domain_text_document\n0.2 /path/to/processed/general_text_document'

export NNODES=32 NPUS_PER_NODE=16
export MASTER_ADDR=MASTER_NODE_HOST MASTER_PORT=8192 NODE_RANK=0
bash train_cpt_deepseek4_pro_4k.sh
```

The Pro launcher does not enable or inject GTS operators. It expects the standard MindSpeed-LLM runtime and any operators required by the selected DeepSeek-V4 implementation to already be installed.

MCore → HF: `../convert/convert_ckpt_mcore_to_hf.sh`.
