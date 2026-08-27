# CPT Training

MindSpeed-LLM 4K CPT launcher for Ascend 910C. Conversion scripts: [`../convert`](../convert/).

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

**Train** (defaults: `SEQ_LEN=4096`, `GBS=128`, `TP=1,PP=4,EP=32`):

```bash
export CKPT_LOAD_DIR=/path/to/mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export TRAIN_DATA_PATH=$'1.0 /path/to/processed/or_cpt_corpus_text_document'
export NNODES=8 NPUS_PER_NODE=16 MASTER_ADDR=MASTER_NODE_HOST NODE_RANK=0
bash train_cpt_deepseek4_flash_4k.sh
```

MCore → HF: `../convert/convert_ckpt_mcore_to_hf.sh`.
