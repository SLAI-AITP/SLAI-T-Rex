# OR Data Distill

A compact research framework for distilling optimization modeling SFT data from a small set of high-quality problem-answer seeds.

The project is intentionally generic. It does not assume any particular evaluation suite, source collection, or prompt format. The recommended workflow is:

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

## Design Goals

- Keep the core method simple enough for academic use.
- Use high-quality modeling examples as the initial seed pool.
- Represent problems through generic OR modeling dimensions: mode, domain, structure, difficulty, data interface, and answer style.
- Support iterative growth through accepted synthetic pools.
- Export clean `user / assistant` SFT rows without source-specific fields.
- Keep all API endpoints and private data outside the repository.

## Data Format

Input seeds may be simple problem-answer JSONL:

```json
{"problem":"...","answer":"...","metadata":{"mode":"DP","domain":"logistics","structure":"transportation","difficulty":"medium"}}
```

Final SFT rows use only:

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

## Installation

```bash
cd /path/to/model_get_release
python -m pip install -e .
```

## Quick Start

Dry-run without calling an LLM:

```bash
python -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

This writes request payloads and a manifest under `runs/demo/`.

Sanitize an existing private SFT file before using it as seed data:

```bash
python -m or_data_distill sanitize-seeds \
  --input seeds/private/source_sft.jsonl \
  --output seeds/sanitized_seed.jsonl
```

Build generic modeling IR from sanitized seeds:

```bash
python -m or_data_distill extract-ir \
  --input seeds/sanitized_seed.jsonl \
  --output seeds/generic_ir.jsonl
```

Run with an OpenAI-compatible backend:

```bash
export LLM_API_KEY=YOUR_KEY_IF_NEEDED
python -m or_data_distill run --config configs/run.example.yaml
```

Set `run.concurrency` in the YAML to run multiple samples in parallel. Each sample still follows the same internal order: synthetic IR, rendered problem, rendered answer, quality gate.

During generation, the runner samples generic target buckets from the configured modes, domains, structures, and difficulties. These targets are passed to the synthetic IR prompt so the generated pool does not collapse into a few common domains.

No API key, endpoint, or private seed file should be committed.

## Production Features

The `run` command is designed to be restartable and quota-aware while keeping the method generic:

- `run.target_count` is the desired number of accepted SFT rows, not merely the number of attempts.
- `run.generation_oversample` submits extra candidates for each remaining gap.
- `run.max_rounds` repeats top-up rounds when rejection leaves a shortfall.
- `run.resume: true` continues from existing `runs/<run_id>/sft.jsonl`, `attempts.jsonl`, and accepted pools.
- `cache.enabled: true` stores LLM responses by request hash under `cache/chat`.
- outputs are streamed as each attempt finishes, so interrupted runs keep completed rows.
- `quality.problem_similarity_threshold` rejects generated problems that are too close to the seed pool or current run.
- `paths.synthetic_pool` lets a later run use accepted Synthetic IR from earlier runs as parent context.
- `parent_pool` controls flywheel behavior through `parent_pool_mode`, `synthetic_parent_share`, `parent_match_top_k`, and `parent_usage_penalty`.
- `llm.base_urls` plus `llm.workers_per_api` supports multi-endpoint generation without changing quota accounting.

Important output files:

```text
runs/<run_id>/sft.jsonl                       accepted SFT rows
runs/<run_id>/accepted_synthetic_pool.jsonl   accepted Synthetic IR for later flywheel runs
runs/<run_id>/surplus_sft.jsonl               valid rows produced after the quota was filled
runs/<run_id>/surplus_synthetic_pool.jsonl    valid surplus IR that can be reused as parent context
runs/<run_id>/rejected.jsonl                  ordinary quality/API/format rejections
runs/<run_id>/attempts.jsonl                  per-attempt status for resume and audit
runs/<run_id>/manifest.json                   final run summary
```

To expand from an earlier run:

```bash
python -m or_data_distill run --config configs/run.example.yaml
```

with:

```yaml
paths:
  seeds: seeds/public_seed.jsonl
  synthetic_pool:
    - runs/previous_run/accepted_synthetic_pool.jsonl
```

The previous pool is only used as parent context. New accepted rows still go into the new run directory.

## Public Seed Subset

This repository includes a small public seed pool at:

```text
seeds/public_seed.jsonl
```

It contains diverse generated optimization modeling examples in the same simple seed format:

```json
{"id":"...","problem":"...","answer":"...","metadata":{"mode":"DP","domain":"production","structure":"scheduling","difficulty":"medium"}}
```

The accompanying `seeds/public_seed_manifest.json` records only aggregate selection statistics. It does not store source paths, private endpoints, or lineage fields.

To build a similar public seed subset from an existing accepted problem-answer archive:

```bash
python tools/build_public_seed_subset.py \
  --answers path/to/accepted_answers.jsonl \
  --synthetic-pools path/to/accepted_synthetic_pool.jsonl \
  --output seeds/public_seed.jsonl \
  --manifest seeds/public_seed_manifest.json \
  --count 200
```

The builder keeps only generic fields, folds small attached data files into the problem text, maps metadata to the public schema, filters unfinished or environment-specific artifacts, and performs stratified diversity sampling.

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

These buckets are generic controls for diversity, not labels tied to any source.

## Quality Gate

The default gate checks:

- clean two-message SFT format;
- non-empty problem and answer;
- no placeholder or unfinished text;
- no mechanical artifacts such as `)Skip`;
- complete metadata dimensions.
- DT problems contain a real Markdown table.
- DPS problems include inline file blocks, because the default SFT export is messages-only.

Optional solver execution and stronger mathematical checks can be added by downstream projects, but the default repository keeps the core method lightweight.

## Repository Layout

```text
configs/       reusable run and schema examples
prompts/       generic LLM prompts
src/           Python package
tools/         small shell/Python helpers
examples/      tiny public demo seeds and configs
runs/          local outputs, ignored by git
cache/         local cache, ignored by git
```
