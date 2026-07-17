<p align="center">
  <img src="assets/slai-trex-logo.png" width="960" alt="SLAI T-Rex logo">
</p>

<p align="center">
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="SLAI%20T-Rex.pdf"><img alt="Paper" src="https://img.shields.io/badge/Paper-SLAI%20T--Rex-B34A78?logo=readthedocs&logoColor=white"></a>
  <a href="https://www.modelscope.cn/models/SLAIAITP/DeepSeek-V4-Flash-OR"><img alt="ModelScope checkpoint" src="https://img.shields.io/badge/ModelScope-Checkpoint-624AFF?logo=modelscope&logoColor=white"></a>
  <a href="https://github.com/SLAI-AITP/SLAI-T-Rex"><img alt="GitHub repository" src="https://img.shields.io/badge/GitHub-SLAI--T--Rex-181717?logo=github&logoColor=white"></a>
  <br>
  <img alt="Ascend 910C" src="https://img.shields.io/badge/Ascend-910C-C7000B?logo=huawei&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Operations%20Research-orange">
</p>

# SLAI T-Rex

**SLAI T-Rex** is the open-source companion repository for the technical report:

**Paper:** [SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD](SLAI%20T-Rex.pdf)  
**Model:** [SLAIAITP/DeepSeek-V4-Flash-OR](https://www.modelscope.cn/models/SLAIAITP/DeepSeek-V4-Flash-OR)  
**Code:** [SLAI-AITP/SLAI-T-Rex](https://github.com/SLAI-AITP/SLAI-T-Rex)

The report studies full-parameter post-training for the DeepSeek-V4 family on Ascend CloudMatrix384 SuperPOD with Ascend 910C NPUs. The repository exposes the reproducible parts of that work: OR-oriented data construction, MindSpeed-LLM CPT/SFT launch templates, checkpoint preparation, and documentation for the end-to-end post-training workflow.

SLAI T-Rex has two connected goals:

- **System scaling:** optimize trillion-parameter-class MoE training on Ascend SuperPOD through parallelism, communication orchestration, CPU-NPU coordination, and AscendC kernel optimization.
- **Domain specialization:** adapt DeepSeek-V4-Flash to Operations Research (OR) with solver-grounded CPT data, self-distilled SFT data, contract-aware cleaning, and benchmark evaluation.

## Report Highlights

<p align="center">
  <img src="assets/overview_infra_and_acc.png" width="960" alt="SLAI T-Rex system and benchmark overview">
</p>

- **34.22% MFU on DeepSeek-V4-Pro training**, a **2.93x** improvement over the open-source baseline recipe on Ascend SuperPOD.
- **AuraKernel**, an AscendC kernel optimization workflow for bottleneck operators in sparse attention, RMS normalization, lightning indexer gradients, RoPE, limited SwiGLU, and mHC-related chains.
- **OR CPT-SFT specialization** for DeepSeek-V4-Flash, combining collected OR resources, solver-verified synthetic documents, self-distilled SFT samples, and Clean-CoT quality gates.
- **10K high-quality SFT samples** across four OR task categories and three problem representations.
- **71.81% average zero-shot Pass@1** across NL4OPT, OptiBench, B4O-Feasible, and B4O-ORGEval, outperforming GPT-5.4-Mini by 3.98 points and the base DeepSeek-V4-Flash by 11.27 points in the reported comparison.
- **CPT-to-SFT transfer gain:** under identical SFT conditions, CPT-initialized SFT improves B4O-Feasible to 71.22% and B4O-ORGEval to 59.39%.

## Repository Status

| Module | Status | Purpose |
| --- | --- | --- |
| [cpt_data_construction](cpt_data_construction/) | design note | OR-CPT engine scope: solver-verified document synthesis and provenance requirements |
| [cpt_training](cpt_training/) | scripts included | MindSpeed-LLM CPT data conversion, checkpoint conversion, and 4K training launcher |
| [sft_data_construction](sft_data_construction/) | runnable | OR SFT self-distillation toolkit with seed IR, synthetic IR, rendering, quality gate, resume, cache, and multi-endpoint generation |
| [sft_training](sft_training/) | scripts included | MindSpeed-LLM SFT data conversion and 8K multi-node training launcher |
| [model_download_deployment](model_download_deployment/) | script included | DeepSeek-V4 FP8 HuggingFace checkpoint to BF16 HuggingFace checkpoint preparation |
| [docs](docs/) | index | extended notes and future dataset/model cards |
| [examples](examples/) | index | end-to-end workflow entry points |

## Workflow

```text
DeepSeek-V4 checkpoint
  -> FP8/BF16 checkpoint preparation
  -> HF <-> MindSpeed/Megatron-Core conversion
  -> OR-CPT data construction
  -> CPT on Ascend 910C
  -> self-distilled OR SFT data
  -> Clean-CoT / contract-aware filtering
  -> SFT on Ascend 910C
  -> HF export, serving, and OR benchmark evaluation
```

The runnable repository focuses on the public data and training scaffolding. Large-scale production inputs, private cluster configuration, and proprietary evaluation artifacts are intentionally excluded.

## Quick Start

Clone the renamed repository:

```bash
git clone https://github.com/SLAI-AITP/SLAI-T-Rex.git
cd SLAI-T-Rex
```

Install the SFT data construction toolkit:

```bash
cd sft_data_construction
python3 -m pip install -e .
```

Validate the public seed pool:

```bash
python3 -m or_data_distill validate-sft --input seeds/public_seed.jsonl
```

Run a dry-run without calling an LLM:

```bash
python3 -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

Generate SFT data with an OpenAI-compatible backend:

```bash
cp configs/run.example.yaml configs/run.local.yaml
export LLM_API_KEY=YOUR_KEY_IF_NEEDED
python3 -m or_data_distill run --config configs/run.local.yaml
```

Convert the generated SFT JSONL and launch the MindSpeed-LLM SFT template:

```bash
cd ../sft_training

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CKPT_LOAD_DIR=/path/to/source_mcore_checkpoint
export OUTPUT_ROOT=/path/to/training_outputs/sft

bash scripts/convert_data.sh \
  --mindspeed-llm-dir "$MINDSPEED_LLM_DIR" \
  --input ../sft_data_construction/runs/sft_data_demo/sft.jsonl \
  --output-prefix /path/to/processed/or_sft/openai \
  --tokenizer "$TOKENIZER_PATH" \
  --handler-name SharegptStyleInstructionHandler \
  --prompt-type deepseek4 \
  --map-keys '{"messages":"messages","tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}' \
  --seq-length 8192 \
  --workers 8 \
  --n-subs 16 \
  --no-append-eod

export DATA_PATH=/path/to/processed/or_sft/openai
bash scripts/launch_sft_deepseek4_flash_8n16_910c.sh
```

For CPT scripts, see [cpt_training/README.md](cpt_training/README.md). For checkpoint preparation, see [model_download_deployment/README.md](model_download_deployment/README.md).

## SFT Data Construction

The public SFT toolkit implements the data flywheel described in the report:

```text
problem-answer seeds
  -> generic modeling IR
  -> synthetic modeling IR
  -> rendered problem
  -> rendered answer
  -> quality gate
  -> OpenAI-style SFT JSONL
```

Key capabilities:

- public seed pool under `sft_data_construction/seeds/`;
- DP, DT, and DPS problem representations;
- controllable OR buckets for domain, structure, difficulty, data interface, and answer style;
- accepted-target top-up with `target_count`, `generation_oversample`, and `max_rounds`;
- request cache and resumable generation;
- multi-endpoint OpenAI-compatible generation with `llm.base_urls` and `workers_per_api`;
- similarity filtering against seeds and current run outputs;
- accepted synthetic pools for later flywheel rounds;
- surplus pools for valid samples generated after the quota is full.

## Training Templates

The training scripts are templates for an already prepared Ascend/MindSpeed environment. They do not vendor MindSpeed-LLM, MindSpeed, CANN, custom operators, cluster launchers, or private checkpoints.

CPT defaults in `cpt_training/scripts/train_cpt_deepseek4_flash_4k.sh`:

```text
SEQ_LEN=4096
GBS=128
MBS=1
TRAIN_ITERS=280
LR=3.0e-6
MIN_LR=3.0e-7
TP=1, PP=4, EP=32, CP=1
```

SFT defaults in `sft_training/scripts/train_sft_deepseek4_flash_8k.sh`:

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

Adjust these values only after matching the hardware layout, checkpoint format, parallelism strategy, and data packing configuration.

## Repository Layout

```text
SLAI-T-Rex/
├── cpt_data_construction/       # OR-CPT engine design scope and release notes
├── cpt_training/                # MindSpeed-LLM CPT conversion and launch templates
├── sft_data_construction/       # Runnable OR SFT data distillation toolkit
├── sft_training/                # MindSpeed-LLM SFT conversion and launch templates
├── model_download_deployment/   # Checkpoint preparation and deployment notes
├── docs/                        # Extended documentation index
├── examples/                    # End-to-end workflow index
├── assets/                      # README icons kept for compatibility
├── SLAI T-Rex.pdf               # Technical report PDF
├── README.md                    # English entry
├── README_en.md                 # English compatibility entry
└── README_zh.md                 # Chinese entry
```

## Citation

```bibtex
@techreport{slai2026trex,
  title  = {SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD},
  author = {Dongfang Li and Xiaodong Luo and Ruoyu Sun and Xuhui Chen and
            Linyuan Qiu and Jian Meng and Zhengxuan Lu and Yiting Wang and
            Yucheng Xie and Tao Guo and Tianxiang Fang and Jing Li and
            Sihang Chen and Shihao Hong and Chang Liu and Weihua Dai and
            Zirong Zeng and Ziwei Zhu and Zhuohan Wang and Zhengjun Yue and
            Igor Vasilyev and Min Liu and Weijian Sun and Xin Chen and
            Yingmeng Gao and Jinhua Zhou and Taolue Chen and Chenwei Wu and
            Dong Zhang and Wenlong Jin and Jinmin Xiang and Barkova Maria and
            Ushakov Anton and Xianfei Jin and Tian Ding and Zhihang Lin and
            Qian Chen and Linxin Yang and Mingzhe Yang and Bingwei Zhang and
            Hongzhang Yang and Fangxue Zhang and Shijun Qin and Jie Yu and
            Cuihua Hu and Tolstykh Vasiliy and Nosov Ivan and Abdullin Amir and
            Zhichen Zhou and Xin Zhang and Zhixiong Ning and Xutong Zhao and
            Junjie Huang and Jiajun Liu and Weiyan Kong and Zheng Zhang and
            Wenhan Luo and Lin Hu and Yangbo Guo and Li Zeng and Shihao Zeng and
            Baotian Hu and Min Zhang and Haizhou Li and Zhiquan Luo},
  year   = {2026},
  url    = {https://github.com/SLAI-AITP/SLAI-T-Rex}
}
```
