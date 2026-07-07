<p align="center">
  <img src="assets/icons/orproject.svg" width="720" alt="ORProject logo">
</p>

<p align="center">
  <b>English</b> | <a href="README.md">中文</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Status" src="https://img.shields.io/badge/Status-SFT%20Data%20%2B%20Training%20Templates-brightgreen">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Operations%20Research-orange">
</p>

# ORProject: A Post-training Pipeline for OR-focused LLMs

ORProject is an open-source post-training project for large language models in Operations Research and Mathematical Optimization. The repository is organized as a five-stage pipeline, from domain corpus construction to continual pre-training, SFT data construction, SFT training, and finally model download and deployment.

**Stage 3: SFT Data Construction** is ready to run. **Stage 2: CPT Training** and **Stage 4: SFT Training** now include MindSpeed-LLM launch templates. The remaining stages are scaffolded and will be filled in future releases.

<p align="center">
  <img src="assets/icons/workflow.svg" width="960" alt="ORProject workflow">
</p>

## 1. CPT Data Construction

<img src="assets/icons/cpt-data.svg" width="36" alt=""> Entry: [cpt_data_construction](cpt_data_construction/)

Status: placeholder.

Planned contents:

- OR-domain raw corpus collection;
- document cleaning, deduplication, and normalization;
- domain-relevance filtering;
- tokenizer-length packing;
- CPT dataset cards and release scripts.

## 2. CPT Training

<img src="assets/icons/cpt-train.svg" width="36" alt=""> Entry: [cpt_training](cpt_training/)

Status: MindSpeed-LLM training template included.

This stage provides DeepSeek-V4-Flash-style CPT scripts for:

- converting CPT text corpora into MindSpeed indexed datasets;
- converting checkpoints between HuggingFace and MindSpeed/Megatron-Core formats;
- launching multi-node CPT jobs;
- saving checkpoints, TensorBoard events, archives, and optional registry metadata.

Minimal example:

```bash
cd ORproject/cpt_training

export MINDSPEED_LLM_DIR=/path/to/MindSpeed-LLM
export MINDSPEED_DIR=/path/to/MindSpeed
export TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash
export CKPT_LOAD_DIR=/path/to/deepseek4_flash_mcore
export OUTPUT_ROOT=/path/to/training_outputs/cpt
export TRAIN_DATA_PATH=$'1.0 /path/to/processed/or_corpus_text_document'

bash scripts/train_cpt_deepseek4_flash_4k.sh
```

See [cpt_training/README.md](cpt_training/README.md) for details.

## 3. SFT Data Construction

<img src="assets/icons/sft-data.svg" width="36" alt=""> Entry: [sft_data_construction](sft_data_construction/)

Status: ready to run.

This stage expands a small set of high-quality OR modeling problem-answer seeds into larger SFT datasets. The core flow is:

```text
problem-answer seeds
  -> generic modeling IR
  -> synthetic modeling IR
  -> rendered problem
  -> rendered answer
  -> quality gate
  -> SFT JSONL
```

Supported features:

- public seed pool: `sft_data_construction/seeds/public_seed.jsonl`;
- DP / DT / DPS problem modes;
- generic OR buckets: domain, structure, difficulty, data interface, answer style;
- Synthetic IR -> problem -> answer generation;
- target-bucket control to avoid domain collapse;
- request caching;
- accepted-target top-up: `target_count` means desired accepted rows;
- multi-round generation: `generation_oversample` + `max_rounds`;
- resumable runs: `resume: true`;
- multi-endpoint generation: `llm.base_urls` + `workers_per_api`;
- streaming outputs;
- similarity filtering;
- surplus pools for valid samples produced after the quota is filled.

### 3.1 Installation

```bash
cd ORproject/sft_data_construction
python -m pip install -e .
```

You can also run without installation:

```bash
PYTHONPATH=src python -m or_data_distill --help
```

### 3.2 Validate the Public Seed Pool

```bash
cd ORproject/sft_data_construction
python -m or_data_distill validate-sft --input seeds/public_seed.jsonl
```

Expected output:

```json
{
  "rows": 200,
  "rows_with_issues": 0
}
```

### 3.3 Dry-run

```bash
python -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

Dry-run does not call an LLM. It only writes request payloads and a manifest, which is useful for checking config parsing.

### 3.4 Generate SFT Data with a Real API

Copy a local config:

```bash
cp configs/run.example.yaml configs/run.local.yaml
```

Edit `configs/run.local.yaml`:

```yaml
run:
  run_id: sft_data_demo
  output_root: runs
  target_count: 200
  max_rounds: 3
  generation_oversample: 1.5
  concurrency: 16
  resume: true

paths:
  seeds: seeds/public_seed.jsonl
  synthetic_pool: []

llm:
  base_url: http://YOUR_HOST:PORT/v1
  model: your-model-name
  api_key_env: LLM_API_KEY
  temperature: 0.7
  top_p: 0.9
  max_tokens: 4096
  timeout_seconds: 240
  disable_proxy: true
```

Run:

```bash
export LLM_API_KEY=YOUR_KEY_IF_NEEDED
python -m or_data_distill run --config configs/run.local.yaml
```

Inspect and validate:

```bash
python tools/inspect_run.py --run-dir runs/sft_data_demo
python -m or_data_distill validate-sft --input runs/sft_data_demo/sft.jsonl
```

Key outputs:

```text
runs/<run_id>/sft.jsonl                       accepted SFT rows
runs/<run_id>/accepted_synthetic_pool.jsonl   Synthetic IR pool for later flywheel runs
runs/<run_id>/surplus_sft.jsonl               valid rows generated after quota is full
runs/<run_id>/surplus_synthetic_pool.jsonl    IR for surplus rows
runs/<run_id>/rejected.jsonl                  rejected samples
runs/<run_id>/attempts.jsonl                  per-attempt status
runs/<run_id>/manifest.json                   run summary
```

### 3.5 Data Flywheel

Use an earlier accepted pool as parent context for later expansion:

```yaml
paths:
  seeds: seeds/public_seed.jsonl
  synthetic_pool:
    - runs/sft_data_demo/accepted_synthetic_pool.jsonl

parent_pool:
  parent_pool_mode: hybrid
  synthetic_parent_share: 0.5
  parent_match_top_k: 8
  parent_usage_penalty: 0.25
```

For a more aggressive snowball mode:

```yaml
parent_pool:
  parent_pool_mode: snowball
  synthetic_parent_share: 0.75
```

### 3.6 Multi-API Generation

```yaml
llm:
  base_urls:
    - http://HOST_A:8000/v1
    - http://HOST_B:8000/v1
  workers_per_api: 32
  model: your-model-name
```

Total concurrency is approximately:

```text
len(base_urls) * workers_per_api
```

## 4. SFT Training

<img src="assets/icons/sft-train.svg" width="36" alt=""> Entry: [sft_training](sft_training/)

Status: MindSpeed-LLM training template included.

This stage converts the OpenAI-style `messages` JSONL from Stage 3 into a MindSpeed packed instruction dataset and launches SFT:

```bash
cd ORproject/sft_training

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

See [sft_training/README.md](sft_training/README.md) for details.

## 5. Model Download and Deployment

<img src="assets/icons/model-deploy.svg" width="36" alt=""> Entry: [model_download_deployment](model_download_deployment/)

Status: an FP8 -> BF16 checkpoint preparation script is included; model download, inference, and serving examples are still planned.

Included:

- `model_download_deployment/scripts/convert_ckpt_fp8_to_bf16.sh`: converts a DeepSeek-V4 FP8 HuggingFace checkpoint to a BF16 HuggingFace checkpoint.

Planned contents:

- released model list;
- model download commands;
- local inference serving commands;
- OpenAI-compatible API deployment examples;
- hardware-specific deployment notes.

## Repository Layout

```text
ORproject/
├── cpt_data_construction/       # 1. CPT data construction, placeholder
├── cpt_training/                # 2. CPT training, MindSpeed template
├── sft_data_construction/       # 3. SFT data construction, ready
├── sft_training/                # 4. SFT training, MindSpeed template
├── model_download_deployment/   # 5. model download and deployment, includes FP8 -> BF16 checkpoint preparation
├── assets/icons/                # README icons and workflow figure
├── docs/                        # extended documentation, placeholder
├── examples/                    # end-to-end examples, placeholder
├── README.md                    # Chinese version
└── README_en.md                 # English version
```

## Citation

Citation information will be added after the CPT data construction and model release modules are completed.
