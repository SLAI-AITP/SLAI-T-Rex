# CPT Training

This module provides **SLAI T-Rex** MindSpeed-LLM templates for OR-oriented continued pre-training (CPT) on Ascend 910C environments.

The scripts are launch templates, not a self-contained training stack. Keep MindSpeed-LLM, MindSpeed, CANN, custom operators, private checkpoints, logs, and generated datasets outside this repository.

## Files

```text
cpt_training/
├── scripts/
│   ├── convert_data.sh                    # JSONL/text -> MindSpeed indexed dataset
│   ├── train_cpt_deepseek4_flash_4k.sh    # DeepSeek-V4-Flash CPT launcher
│   ├── convert_ckpt_hf_to_mcore.sh        # HuggingFace -> MindSpeed/Megatron-Core
│   └── convert_ckpt_mcore_to_hf.sh        # MindSpeed/Megatron-Core -> HuggingFace
└── README.md
```

## Environment

Prepare the external runtime first:

```bash
export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
```

Common variables:

```bash
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export CUSTOM_TRANSFORMER_ENV=/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
export CONDA_ENV=your_conda_env_name
```

## Convert CPT Corpus

Input JSONL should expose a text field:

```json
{"text": "A solver-verified OR document, formulation note, or cleaned domain paragraph ..."}
```

Convert one dataset:

```bash
cd SLAI-T-Rex/cpt_training

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

The training prefix is usually:

```text
/path/to/processed/or_cpt_corpus_text_document
```

Batch conversion through a manifest is also supported:

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
or_solver_docs /path/to/solver_docs.jsonl /path/to/processed/or_solver_docs 8 16 1000
```

## Convert Checkpoint to MCore

If the source checkpoint is FP8 HuggingFace format, first prepare a BF16 HuggingFace checkpoint:

```bash
cd SLAI-T-Rex/model_download_deployment

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export INPUT_FP8_HF_PATH=/path/to/deepseek4_fp8_hf
export OUTPUT_BF16_HF_PATH=/path/to/deepseek4_bf16_hf

bash scripts/convert_ckpt_fp8_to_bf16.sh
```

Then convert BF16 HuggingFace to MindSpeed/Megatron-Core:

```bash
cd SLAI-T-Rex/cpt_training

export HF_LOAD_DIR=/path/to/deepseek4_flash_bf16_hf
export MCORE_SAVE_DIR=/path/to/deepseek4_flash_mcore

bash scripts/convert_ckpt_hf_to_mcore.sh
```

## Run CPT

`TRAIN_DATA_PATH` follows MindSpeed weighted dataset syntax:

```bash
cd SLAI-T-Rex/cpt_training

export CKPT_LOAD_DIR=/path/to/deepseek4_flash_mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export VALID_DATA_PATH=/path/to/processed/validation_text_document

export TRAIN_DATA_PATH=$'0.7 /path/to/processed/or_books_text_document\n0.3 /path/to/processed/or_solver_docs_text_document'

export NNODES=8
export NPUS_PER_NODE=16
export MASTER_ADDR=MASTER_NODE_HOST
export MASTER_PORT=6000
export NODE_RANK=0

bash scripts/train_cpt_deepseek4_flash_4k.sh
```

Important defaults:

```text
SEQ_LEN=4096
GBS=128
MBS=1
TRAIN_ITERS=280
LR=3.0e-6
MIN_LR=3.0e-7
TP=1, PP=4, EP=32, CP=1
```

Override them only after matching the hardware topology, sequence packing, and checkpoint parallelism.

## Outputs

```text
$OUTPUT_ROOT/checkpoints/<run_id>/      MCore checkpoints
$OUTPUT_ROOT/tensorboard/<run_id>/      TensorBoard events
$OUTPUT_ROOT/archive/<run_id>/          copied script, config.json, logs, metrics
$OUTPUT_ROOT/model_registry/            optional registry metadata
```

The launcher avoids silently overwriting incomplete checkpoints. Set `FORCE_RERUN=1` only when you intentionally want to rerun from scratch.

## Convert MCore Back to HuggingFace

```bash
cd SLAI-T-Rex/cpt_training

export MCORE_LOAD_DIR=/path/to/cpt_checkpoint
export HF_SAVE_DIR=/path/to/cpt_checkpoint_hf

bash scripts/convert_ckpt_mcore_to_hf.sh
```
