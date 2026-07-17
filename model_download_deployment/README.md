# Model Download and Deployment

This module contains checkpoint preparation and deployment notes for **SLAI T-Rex**.

The current public script prepares DeepSeek-V4-family HuggingFace checkpoints by converting FP8 weights to BF16 HuggingFace format. The resulting checkpoint can then be converted to MindSpeed/Megatron-Core format for CPT or SFT, or used by compatible BF16 inference stacks.

## Files

```text
model_download_deployment/
├── scripts/
│   └── convert_ckpt_fp8_to_bf16.sh   # DeepSeek-V4 FP8 HF checkpoint -> BF16 HF checkpoint
└── README.md
```

## FP8 HuggingFace to BF16 HuggingFace

Some DeepSeek-V4-family HuggingFace checkpoints may be distributed in FP8 format. Before MindSpeed/Megatron-Core conversion or BF16 inference, prepare a BF16 HuggingFace checkpoint:

```bash
python3 -m pip install torchao
```

```bash
cd SLAI-T-Rex/model_download_deployment

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export INPUT_FP8_HF_PATH=/path/to/deepseek4_fp8_hf
export OUTPUT_BF16_HF_PATH=/path/to/deepseek4_bf16_hf

bash scripts/convert_ckpt_fp8_to_bf16.sh
```

The output directory remains a HuggingFace checkpoint. Use it as `HF_LOAD_DIR` for:

```text
../cpt_training/scripts/convert_ckpt_hf_to_mcore.sh
```

## Deployment Boundary

The technical report evaluates a full post-training chain from CPT to SFT, HF export, serving, and benchmark evaluation. This repository currently releases only the checkpoint preparation script and training-side conversion hooks.

Future additions should include:

- released model list and download commands;
- HuggingFace export checks;
- local inference and OpenAI-compatible serving examples;
- Ascend-specific deployment notes;
- provenance manifests linking training checkpoint, exported HF artifact, served model, and benchmark score.
