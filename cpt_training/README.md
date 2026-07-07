# 2. CPT Training

This module provides a MindSpeed-LLM example for continual pre-training (CPT) on OR-domain text corpora. The scripts are model/runtime templates: edit paths through environment variables, keep generated checkpoints and logs outside this repository, and run them inside a prepared MindSpeed-LLM environment.

## Files

```text
cpt_training/
├── scripts/
│   ├── convert_data.sh                         # JSONL/text -> MindSpeed indexed dataset
│   ├── train_cpt_deepseek4_flash_4k.sh                        # DeepSeek-V4-Flash CPT launcher
│   ├── convert_ckpt_hf_to_mcore.sh
│   └── convert_ckpt_mcore_to_hf.sh
└── README.md
```

## 0. Environment

Required external repositories:

```bash
export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
```

Common runtime variables:

```bash
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export CUSTOM_TRANSFORMER_ENV=/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
export CONDA_ENV=your_conda_env_name
```

## 1. Convert CPT Corpus

Input data should contain a text field, for example:

```json
{"text": "A clean OR-domain document or paragraph ..."}
```

Convert one dataset:

```bash
cd ORproject/cpt_training

bash scripts/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --input /path/to/cpt_corpus.jsonl \
  --output-prefix /path/to/processed/or_cpt_corpus \
  --tokenizer "$TOKENIZER_PATH" \
  --handler-name GeneralPretrainHandler \
  --json-keys text \
  --seq-length 4096 \
  --workers 8 \
  --n-subs 16
```

The training prefix passed later is usually:

```text
/path/to/processed/or_cpt_corpus_text_document
```

Manifest batch conversion is also supported:

```bash
bash scripts/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --manifest /path/to/convert_manifest.tsv \
  --tokenizer "$TOKENIZER_PATH" \
  --parallel 4
```

Manifest format:

```text
# name input_path output_prefix workers n_subs log_interval
or_books /path/to/books.jsonl /path/to/processed/or_books 8 16 1000
or_notes /path/to/notes.jsonl /path/to/processed/or_notes 8 16 1000
```

## 2. Convert HF Checkpoint to MCore

If your source checkpoint is FP8 HF format, first convert it to BF16 with:

```bash
cd ORproject/model_download_deployment

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export INPUT_FP8_HF_PATH=/path/to/deepseek4_fp8_hf
export OUTPUT_BF16_HF_PATH=/path/to/deepseek4_bf16_hf

bash scripts/convert_ckpt_fp8_to_bf16.sh
```

Then use the BF16 HF checkpoint as `HF_LOAD_DIR`:

```bash
cd ORproject/cpt_training

export HF_LOAD_DIR=/path/to/deepseek4_flash_bf16_hf
export MCORE_SAVE_DIR=/path/to/deepseek4_flash_mcore

bash scripts/convert_ckpt_hf_to_mcore.sh
```

## 3. Run CPT

`TRAIN_DATA_PATH` follows MindSpeed weighted dataset syntax. Use one line per dataset:

```bash
cd ORproject/cpt_training

export CKPT_LOAD_DIR=/path/to/deepseek4_flash_mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export VALID_DATA_PATH=/path/to/processed/validation_text_document

export TRAIN_DATA_PATH=$'0.7 /path/to/processed/or_books_text_document\n0.3 /path/to/processed/or_notes_text_document'

export NNODES=8
export NPUS_PER_NODE=16
export MASTER_ADDR=MASTER_NODE_HOST
export MASTER_PORT=6000
export NODE_RANK=0

bash scripts/train_cpt_deepseek4_flash_4k.sh
```

Important defaults in `scripts/train_cpt_deepseek4_flash_4k.sh`:

```text
SEQ_LEN=4096
GBS=128
MBS=1
TRAIN_ITERS=280
LR=3.0e-6
MIN_LR=3.0e-7
TP=1, PP=4, EP=32, CP=1
```

Override them with environment variables only after you have confirmed the hardware layout and batch size.

## 4. Outputs

```text
$OUTPUT_ROOT/checkpoints/<run_id>/      MCore checkpoints
$OUTPUT_ROOT/tensorboard/<run_id>/      TensorBoard events
$OUTPUT_ROOT/archive/<run_id>/          copied script, config.json, logs, metrics
$OUTPUT_ROOT/model_registry/            optional registry metadata
```

The script refuses to silently overwrite an incomplete checkpoint. Use `FORCE_RERUN=1` only when you intentionally want to rerun from scratch.

## 5. Convert MCore Back to HF

```bash
cd ORproject/cpt_training

export MCORE_LOAD_DIR=/path/to/cpt_checkpoint
export HF_SAVE_DIR=/path/to/cpt_checkpoint_hf

bash scripts/convert_ckpt_mcore_to_hf.sh
```
