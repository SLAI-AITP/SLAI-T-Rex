# SFT Training

This module provides **SLAI T-Rex** MindSpeed-LLM templates for supervised fine-tuning (SFT) DeepSeek-V4-Flash on OR modeling data.

It consumes OpenAI-style `messages` JSONL from `../sft_data_construction`, converts it into a MindSpeed packed instruction dataset, and launches an 8K SFT job on Ascend 910C environments.

## Files

```text
sft_training/
├── scripts/
│   ├── convert_data.sh                         # OpenAI/ShareGPT/Alpaca JSONL -> MindSpeed instruction dataset
│   ├── launch_sft_deepseek4_flash_8n16_910c.sh # multi-node wrapper with optional indexmap preparation
│   ├── train_sft_deepseek4_flash_8k.sh         # DeepSeek-V4-Flash 8K SFT launcher
│   └── convert_ckpt_mcore_to_hf.sh             # MindSpeed/Megatron-Core -> HuggingFace
├── tools/
│   └── prepare_sft_indexmap.py                 # prebuild packed SFT shuffle index map
└── README.md
```

## Environment

Prepare external dependencies:

```bash
export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
```

Common variables:

```bash
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CKPT_LOAD_DIR=/path/to/source_mcore_checkpoint
export OUTPUT_ROOT=/path/to/training_outputs/sft
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export CUSTOM_TRANSFORMER_ENV=/usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
export CONDA_ENV=your_conda_env_name
```

## Prepare SFT JSONL

From the data module:

```bash
cd SLAI-T-Rex/sft_data_construction
python3 -m or_data_distill run --config configs/run.local.yaml
python3 -m or_data_distill validate-sft --input runs/sft_data_demo/sft.jsonl
```

Expected row format:

```json
{
  "messages": [
    {"role": "user", "content": "problem statement"},
    {"role": "assistant", "content": "modeling answer"}
  ]
}
```

The report's strongest SFT setting uses cleaned OR data with concise modeling checklists. This repository exposes the conversion and launch path; private large-scale cleaned datasets are not included.

## Convert SFT JSONL

Use `SharegptStyleInstructionHandler` with OpenAI-style field mapping:

```bash
cd SLAI-T-Rex/sft_training

export SFT_JSONL=../sft_data_construction/runs/sft_data_demo/sft.jsonl
export SFT_PREFIX=/path/to/processed/or_sft/openai

bash scripts/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --input "$SFT_JSONL" \
  --output-prefix "$SFT_PREFIX" \
  --tokenizer "$TOKENIZER_PATH" \
  --handler-name SharegptStyleInstructionHandler \
  --prompt-type deepseek4 \
  --map-keys '{"messages":"messages","tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}' \
  --seq-length 8192 \
  --workers 8 \
  --n-subs 16 \
  --no-append-eod
```

The converted `DATA_PATH` is the output prefix:

```bash
export DATA_PATH="$SFT_PREFIX"
```

For Alpaca-style data, switch to:

```bash
--handler-name AlpacaStyleInstructionHandler \
--map-keys '{"prompt":"instruction","query":"input","response":"output","system":"system","history":"history"}'
```

## Optional: Prebuild Packed Index Map

Large multi-node SFT jobs can race while creating the packed shuffle index map. The wrapper can prepare it once on rank 0 while other ranks wait:

```bash
export PREPARE_INDEXMAP=1
export INDEXMAP_SOURCE_JSONL="$SFT_JSONL"
export INDEXMAP_DATA_PREFIX="$DATA_PATH"
export INDEXMAP_SPLIT_NAME=train
```

If the source JSONL is unavailable, provide the number of training examples:

```bash
export INDEXMAP_DOC_COUNT=100000
```

## Run SFT

Multi-node wrapper:

```bash
cd SLAI-T-Rex/sft_training

export DATA_PATH="$SFT_PREFIX"
export RUN_ID=slai_trex_sft_demo

export NNODES=8
export NPUS_PER_NODE=16
export MASTER_ADDR=MASTER_NODE_HOST
export MASTER_PORT=8192
export NODE_RANK=0

bash scripts/launch_sft_deepseek4_flash_8n16_910c.sh
```

Direct inner launcher:

```bash
bash scripts/train_sft_deepseek4_flash_8k.sh
```

Important defaults:

```text
SEQ_LEN=8192
GBS=128
MBS=1
TRAIN_ITERS=250
LR=5.0e-6
MIN_LR=5.0e-8
TP=1, PP=4, EP=32, CP=1
PROMPT_TYPE=deepseek4
```

## Outputs

```text
$OUTPUT_ROOT/checkpoints/<run_id>/      MCore SFT checkpoints
$OUTPUT_ROOT/tensorboard/<run_id>/      TensorBoard events
$OUTPUT_ROOT/archive/<run_id>/          copied script, config.json, logs, metrics
$OUTPUT_ROOT/model_registry/            optional registry metadata
```

## Convert to HuggingFace

Enable conversion in the launcher:

```bash
export ENABLE_HF_CONVERT=1
export HF_CONVERT_SCRIPT=/path/to/parallel_convert_mg2hf.py
```

Or convert manually:

```bash
export MCORE_LOAD_DIR=/path/to/sft_checkpoint
export HF_SAVE_DIR=/path/to/sft_checkpoint_hf

bash scripts/convert_ckpt_mcore_to_hf.sh
```
