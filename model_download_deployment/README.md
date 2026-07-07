# 5. 模型下载与部署

本模块用于放置模型下载、权重准备和部署相关脚本。当前已提供 DeepSeek-V4 FP8 HuggingFace checkpoint 转 BF16 HuggingFace checkpoint 的准备脚本，便于后续继续进行 MindSpeed/Megatron-Core 权重转换、CPT/SFT 或 BF16 推理部署。

## 文件说明

```text
model_download_deployment/
├── scripts/
│   └── convert_ckpt_fp8_to_bf16.sh   # DeepSeek-V4 FP8 HF checkpoint -> BF16 HF checkpoint
└── README.md
```

## FP8 权重转 BF16

部分 DeepSeek-V4 系列 HuggingFace 权重可能以 FP8 形式发布。继续转换到 MindSpeed/Megatron-Core，或用于部分 BF16 推理流程前，可以先转为 BF16 HF checkpoint。

依赖：

```bash
python -m pip install torchao
```

运行：

```bash
cd ORproject/model_download_deployment

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export INPUT_FP8_HF_PATH=/path/to/deepseek4_fp8_hf
export OUTPUT_BF16_HF_PATH=/path/to/deepseek4_bf16_hf

bash scripts/convert_ckpt_fp8_to_bf16.sh
```

输出目录 `OUTPUT_BF16_HF_PATH` 仍是 HuggingFace checkpoint，可继续作为 `cpt_training/scripts/convert_ckpt_hf_to_mcore.sh` 的 `HF_LOAD_DIR`，转换为 MindSpeed/Megatron-Core 格式。

## 后续计划

- 已发布模型列表；
- 模型下载命令；
- 本地推理服务启动命令；
- OpenAI-compatible API 部署示例；
- 不同硬件环境下的部署注意事项。
