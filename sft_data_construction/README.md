# SFT Data Construction

This module is the runnable OR SFT data distillation toolkit used by **SLAI T-Rex**.

It turns a compact set of high-quality optimization modeling seeds into OpenAI-style SFT JSONL through intermediate representations, controlled synthetic generation, problem/answer rendering, similarity checks, and quality gates.

## Pipeline

```text
problem-answer seeds
  -> sanitized seeds
  -> generic modeling IR
  -> synthetic modeling IR
  -> rendered problem
  -> rendered answer
  -> quality gate
  -> SFT JSONL
```

The design mirrors the report's SFT flywheel: accepted synthetic IRs can be reused as parent context in later rounds, gradually expanding the modeling-structure space while keeping the original seeds as anchors.

## Design Goals

- Keep all API endpoints, private seeds, and generated outputs outside git.
- Represent OR problems with explicit modeling dimensions: mode, domain, structure, difficulty, data interface, and answer style.
- Support DP, DT, and DPS problem forms.
- Export clean two-message `user` / `assistant` rows for downstream MindSpeed-LLM SFT.
- Make runs restartable, quota-aware, cached, and auditable.
- Preserve enough metadata to analyze accepted, rejected, surplus, and parent-pool samples.

## Installation

```bash
cd SLAI-T-Rex/sft_data_construction
python3 -m pip install -e .
```

You can also run without installation:

```bash
PYTHONPATH=src python3 -m or_data_distill --help
```

## Data Format

Input seeds may be simple problem-answer JSONL:

```json
{"problem":"...","answer":"...","metadata":{"mode":"DP","domain":"logistics","structure":"transportation","difficulty":"medium"}}
```

Final SFT rows use OpenAI-style messages:

```json
{
  "id": "sft_xxx",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "mode": "DP",
    "domain": "logistics",
    "structure": "transportation",
    "difficulty": "medium",
    "data_interface": "inline_text",
    "answer_style": "math_model_with_code",
    "generation_stage": "silver",
    "quality_status": "accepted"
  }
}
```

## Public Seed Pool

This repository includes a small public seed pool:

```text
seeds/public_seed.jsonl
seeds/public_seed_manifest.json
```

Validate it:

```bash
python3 -m or_data_distill validate-sft --input seeds/public_seed.jsonl
```

Expected output:

```json
{
  "rows": 200,
  "rows_with_issues": 0
}
```

## Dry Run

Dry-run writes request payloads and a manifest without calling an LLM:

```bash
python3 -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

## Run with an OpenAI-Compatible Backend

Copy a local config:

```bash
cp configs/run.example.yaml configs/run.local.yaml
```

Edit the local config:

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

cache:
  enabled: true
  dir: cache/chat

quality:
  problem_similarity_threshold: 0.9
  compare_to_seeds: true
  compare_to_run: true
```

Run:

```bash
export LLM_API_KEY=YOUR_KEY_IF_NEEDED
python3 -m or_data_distill run --config configs/run.local.yaml
```

Inspect and validate:

```bash
python3 tools/inspect_run.py --run-dir runs/sft_data_demo
python3 -m or_data_distill validate-sft --input runs/sft_data_demo/sft.jsonl
```

## Outputs

```text
runs/<run_id>/requests.jsonl                  request payloads
runs/<run_id>/attempts.jsonl                  per-attempt status for resume and audit
runs/<run_id>/synthetic_ir.jsonl              generated synthetic IR records
runs/<run_id>/sft.jsonl                       accepted SFT rows
runs/<run_id>/accepted_synthetic_pool.jsonl   accepted Synthetic IR for later flywheel runs
runs/<run_id>/surplus_sft.jsonl               valid rows produced after the quota was filled
runs/<run_id>/surplus_synthetic_pool.jsonl    reusable surplus IR
runs/<run_id>/rejected.jsonl                  ordinary quality/API/format rejections
runs/<run_id>/manifest.json                   final run summary
```

## Flywheel Runs

Use an earlier accepted pool as parent context:

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

For a more aggressive mode:

```yaml
parent_pool:
  parent_pool_mode: snowball
  synthetic_parent_share: 0.75
```

The previous pool is used only as parent context. New accepted rows are written to the new run directory.

## Multi-Endpoint Generation

```yaml
llm:
  base_urls:
    - http://HOST_A:8000/v1
    - http://HOST_B:8000/v1
  workers_per_api: 32
  model: your-model-name
```

Approximate total concurrency:

```text
len(base_urls) * workers_per_api
```

## Generic Buckets

Problem modes:

- `DP`: all data are in the problem statement.
- `DT`: data are shown in tables.
- `DPS`: data are supplied as attached files or structured payloads.

Structure buckets:

- `assignment`
- `scheduling`
- `routing`
- `transportation`
- `blending`
- `capacity_planning`
- `facility_location`
- `production_planning`
- `portfolio`
- `generic_lp_mip`

Difficulty buckets:

- `small`
- `medium`
- `large`
- `industrial`

These buckets are controls for diversity, not claims about any private source distribution.

## Quality Gate

The default gate checks:

- clean two-message SFT format;
- non-empty problem and answer;
- no placeholder or unfinished text;
- no mechanical artifacts such as `)Skip`;
- complete metadata dimensions;
- DT problems contain a real Markdown table;
- DPS problems include inline file blocks because the default export is messages-only.

Downstream projects can add solver execution, LP structure checks, ORGEval-style validators, or stronger mathematical audits.

## Repository Layout

```text
configs/       reusable run and schema examples
prompts/       generic LLM prompts
src/           Python package
tools/         inspection and seed-subset helpers
examples/      tiny public demo seeds and configs
runs/          local outputs, ignored by git
cache/         local cache, ignored by git
```
