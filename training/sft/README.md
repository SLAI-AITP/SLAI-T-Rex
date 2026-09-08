# SFT Training

MindSpeed-LLM SFT launchers for DeepSeek-V4-Flash and DeepSeek-V4-Pro. Conversion scripts: [`../convert`](../convert/). Input: OpenAI-style `messages` JSONL from [`../../data/sft`](../../data/sft/).

The launchers share `../common/resolve_runtime.sh`, so they can use repositories cloned next to SLAI-T-Rex or paths supplied through `MINDSPEED_LLM_DIR`, `MINDSPEED_DIR`, and `MEGATRON_DIR`.

```bash
export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CKPT_LOAD_DIR=/path/to/mcore
export OUTPUT_ROOT=/path/to/training_outputs/sft
```

```bash
cd SLAI-T-Rex/training/sft
export SFT_JSONL=../../data/sft/runs/sft_data_demo/sft.jsonl
export DATA_PATH=/path/to/processed/or_sft/openai

bash ../convert/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --input "$SFT_JSONL" \
  --output-prefix "$DATA_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --handler-name SharegptStyleInstructionHandler \
  --prompt-type deepseek4 \
  --map-keys '{"messages":"messages","tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}' \
  --seq-length 8192 --workers 8 --n-subs 16 --no-append-eod

export NNODES=8 NPUS_PER_NODE=16 MASTER_ADDR=MASTER_NODE_HOST NODE_RANK=0
bash launch_sft_deepseek4_flash_8n16_910c.sh
```

Flash defaults: `SEQ_LEN=8192`, `GBS=128`, `TP=1,PP=4,EP=32`. Multi-node indexmap races: `PREPARE_INDEXMAP=1`. HF export: `../convert/convert_ckpt_mcore_to_hf.sh`.

For DeepSeek-V4-Pro 4K SFT, provide the packed dataset prefix and Pro checkpoint explicitly:

```bash
export DATA_PATH=/path/to/packed_dataset/or_sft
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Pro
export CKPT_LOAD_DIR=/path/to/deepseek4_pro_mcore
export OUTPUT_ROOT=/path/to/training_outputs/sft

export NNODES=32 NPUS_PER_NODE=16
export MASTER_ADDR=MASTER_NODE_HOST MASTER_PORT=8192 NODE_RANK=0
bash train_sft_deepseek4_pro_4k.sh
```

Pro defaults: `SEQ_LEN=4096`, `GBS=1024`, `TP=2,PP=8,EP=64`. Set `GTS_ENABLE=0` to disable GTS runtime activation and argument injection; the current Pro recipe still expects the compatible DeepSeek-V4 framework patch and sparse-attention operators described in the script header. Set `SOURCE_MANIFEST` only when an image source manifest should be verified.
