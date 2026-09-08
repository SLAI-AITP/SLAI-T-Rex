<p align="center">
  <img src="assets/slai-trex-logo.png" width="960" alt="SLAI T-Rex logo">
</p>

<p align="center">
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="docs/SLAI-T-Rex.pdf"><img alt="Paper" src="https://img.shields.io/badge/Paper-SLAI%20T--Rex-B34A78?logo=readthedocs&logoColor=white"></a>
  <a href="https://www.modelscope.cn/models/SLAIAITP/DeepSeek-V4-Flash-OR"><img alt="ModelScope checkpoint" src="https://img.shields.io/badge/ModelScope-Checkpoint-624AFF?logo=modelscope&logoColor=white"></a>
  <a href="https://github.com/SLAI-AITP/SLAI-T-Rex"><img alt="GitHub repository" src="https://img.shields.io/badge/GitHub-SLAI--T--Rex-181717?logo=github&logoColor=white"></a>
  <br>
  <img alt="Ascend 910C" src="https://img.shields.io/badge/Ascend-910C-C7000B?logo=huawei&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Operations%20Research-orange">
</p>

# SLAI T-Rex

Open-source companion to [SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD](docs/SLAI-T-Rex.pdf).

- **Model:** [SLAIAITP/DeepSeek-V4-Flash-OR](https://www.modelscope.cn/models/SLAIAITP/DeepSeek-V4-Flash-OR)
- **Code:** [SLAI-AITP/SLAI-T-Rex](https://github.com/SLAI-AITP/SLAI-T-Rex)

The report studies full-parameter post-training of DeepSeek-V4 on Ascend CloudMatrix384 SuperPOD (910C). This repository ships the reproducible pieces: OR CPT/SFT data construction, MindSpeed-LLM launch templates, checkpoint conversion, and OR benchmark evaluation.

Two connected goals:

- **System scaling:** trillion-parameter-class MoE training on Ascend SuperPOD (parallelism, communication, CPU–NPU coordination, AscendC kernels).
- **Domain specialization:** adapt DeepSeek-V4-Flash to Operations Research with solver-grounded CPT, self-distilled SFT, and benchmark evaluation.

<p align="center">
  <img src="assets/overview_infra_and_acc.png" width="960" alt="SLAI T-Rex system and benchmark overview">
</p>

## Highlights

- **34.22% MFU** on DeepSeek-V4-Pro, **2.93x** vs the open baseline recipe on Ascend SuperPOD.
- **AuraKernel:** AscendC optimizations for sparse attention, RMSNorm, lightning indexer gradients, RoPE, limited SwiGLU, and mHC-related chains.
- **OR CPT–SFT** for DeepSeek-V4-Flash: collected OR resources, solver-verified synthetic documents, self-distilled SFT, Clean-CoT gates.
- **10K** high-quality SFT samples (four OR task categories, three problem representations).
- **71.81%** average zero-shot Pass@1 on NL4OPT / OptiBench / B4O-Feasible / B4O-ORGEval (+3.98 vs GPT-5.4-Mini, +11.27 vs base DeepSeek-V4-Flash in the report).
- CPT→SFT transfer (same SFT): B4O-Feasible **71.22%**, B4O-ORGEval **59.39%**.

## Layout

```text
SLAI-T-Rex/
├── data/cpt/            OR-CPT engine (solver-verified synthesis)
├── data/sft/            OR SFT distillation toolkit
├── training/convert/    data and checkpoint conversion
├── training/common/     shared MindSpeed runtime discovery
├── training/cpt/        DeepSeek-V4 Flash/Pro 4K CPT launchers
├── training/sft/        DeepSeek-V4 Flash 8K and Pro 4K SFT launchers
├── eval/                OR benchmarks
├── docs/                technical report PDF
└── assets/
```

| Module | Status | Role |
| --- | --- | --- |
| [data/cpt](data/cpt/) | runnable | OptMATH generators → Gurobi → NL backtranslation → CPT JSONL |
| [data/sft](data/sft/) | runnable | seed / synthetic IR → render → quality gate → OpenAI-style SFT JSONL |
| [training/convert](training/convert/) | scripts | MindSpeed data conversion; HF / MCore / FP8 checkpoints |
| [training/cpt](training/cpt/) | scripts | MindSpeed-LLM 4K CPT |
| [training/sft](training/sft/) | scripts | MindSpeed-LLM 8K SFT |
| [eval](eval/) | runnable | NL4OPT, OptiBench, B4O-Feasible, B4O-ORGEval |

```text
checkpoint prep -> CPT data -> CPT train -> SFT data -> SFT train -> eval
```

Large-scale production inputs, private cluster configs, and proprietary eval artifacts are not included. Training scripts assume an existing Ascend / MindSpeed environment (MindSpeed-LLM, CANN, and cluster launchers are not vendored).

## Quick start

SFT data dry-run (no LLM):

```bash
git clone https://github.com/SLAI-AITP/SLAI-T-Rex.git
cd SLAI-T-Rex/data/sft
python3 -m pip install -e .
python3 -m or_data_distill validate-sft --input seeds/public_seed.jsonl
python3 -m or_data_distill run --config examples/configs/demo.yaml --dry-run
```

Real SFT generation: copy `configs/run.example.yaml`, set `llm.base_url` / `model`, then `python3 -m or_data_distill run --config configs/run.local.yaml`.

CPT engine smoke (needs `gurobipy` + an OpenAI-compatible endpoint for LLM stages):

```bash
cd SLAI-T-Rex/data/cpt
python3 -m pip install -e .
python3 -m or_cpt_engine.cli.main --help
```

Training, conversion, and eval commands live in the module READMEs above.

## Training defaults

| Recipe | Seq / GBS | Iters / LR | Parallelism |
| --- | --- | --- | --- |
| Flash CPT | 4096 / 128 | 280 / 3e-6 | TP=1, PP=4, EP=32 |
| Flash SFT | 8192 / 128 | 250 / 5e-6 | TP=1, PP=4, EP=32 |
| Pro CPT | 4096 / 256 | 45 / 1e-6 | TP=2, PP=8, EP=64 |
| Pro SFT | 4096 / 1024 | 2000 / 1e-5 | TP=2, PP=8, EP=64 |

Training paths are provided through environment variables. The launchers discover MindSpeed repositories cloned next to SLAI-T-Rex or use explicit `MINDSPEED_LLM_DIR`, `MINDSPEED_DIR`, and `MEGATRON_DIR` values. Match hardware layout, checkpoint format, tokenizer, and packing before changing the defaults.

## Citation

```bibtex
@misc{li2026slaitrexfullparameterposttraining,
      title={SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD}, 
      author={Dongfang Li and Xiaodong Luo and Ruoyu Sun and Xuhui Chen and Linyuan Qiu and Jian Meng and Zhengxuan Lu and Yiting Wang and Yucheng Xie and Tao Guo and Tianxiang Fang and Jing Li and Sihang Chen and Shihao Hong and Chang Liu and Weihua Dai and Zirong Zeng and Ziwei Zhu and Zhuohan Wang and Zhengjun Yue and Igor Vasilyev and Min Liu and Weijian Sun and Xin Chen and Yingmeng Gao and Jinhua Zhou and Taolue Chen and Chenwei Wu and Dong Zhang and Wenlong Jin and Jinmin Xiang and Barkova Maria and Ushakov Anton and Xianfei Jin and Tian Ding and Zhihang Lin and Qian Chen and Linxin Yang and Mingzhe Yang and Bingwei Zhang and Hongzhang Yang and Fangxue Zhang and Shijun Qin and Jie Yu and Cuihua Hu and Tolstykh Vasiliy and Nosov Ivan and Abdullin Amir and Zhicheng Zhou and Xin Zhang and Zhixiong Ning and Xutong Zhao and Junjie Huang and Jiajun Liu and Weiyan Kong and Zheng Zhang and Wenhan Luo and Lin Hu and Yangbo Guo and Li Zeng and Shihao Zhang and Baotian Hu and Min Zhang and Haizhou Li and Zhiquan Luo},
      year={2026},
      eprint={2607.20145},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2607.20145}, 
}
```
