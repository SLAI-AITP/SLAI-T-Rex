# CPT Data Construction

This module is the **OR-CPT engine** used by **SLAI T-Rex**: a solver-verified synthesis pipeline that turns parameterized OptMATH generators into CPT training documents.

It was imported from the `or_cpt_engine` package of the internal CPT data factory. That source tree also contains two SFT→CPT cleaners (`cpt_cleaner` and `or_sft2cpt_cleaner`). Those clean existing chat/SFT rows; they are not the paper's CPT construction path and are **not** vendored here, except for three small utility modules that the engine already imports.

## Pipeline

```text
OR source generators
  -> generator contracts / profiles
  -> parameterized optimization instance
  -> Gurobi solver validation
  -> instance quality gate
  -> NL backtranslation
  -> forward modeling
  -> forward solver evaluation
  -> CPT document rendering
  -> train JSONL export
```

Formal production flow:

```text
scan-generators
  -> smoke-test-generators
  -> audit-generators
  -> plan-production
  -> run-seed-factory
  -> run-cpt-from-seeds
```

## Installation

```bash
cd SLAI-T-Rex/data/cpt
python3 -m pip install -e .
```

Gurobi / `gurobipy` is required for seed factory and forward evaluation. LLM stages need an OpenAI-compatible endpoint.

## Configure the LLM pool

`engine_configs/llm.yaml` reads `configs/model_api_pool2.yaml`. Copy the example and point it at your service:

```bash
cp configs/model_api_pool.example.yaml configs/model_api_pool2.yaml
```

You can also set `OR_LLM_BASE_URL` and `OR_LLM_API_KEY`.

## Smoke the generator path (no LLM)

Run from this directory so `engine_configs/` resolves:

```bash
cd SLAI-T-Rex/data/cpt
python3 -m or_cpt_engine.cli.main --help
```

Example seed audit (CPU + Gurobi only):

```bash
python3 -m or_cpt_engine.cli.main audit-generators \
  --output engine_runs/seed_audit_smoke \
  --samples-per-generator 2 \
  --run-id seed_audit_smoke
```

Full command list: [docs/OR_CPT_ENGINE_CLI_命令手册.md](docs/OR_CPT_ENGINE_CLI_命令手册.md).  
Design and stage notes: [docs/OR_CPT_ENGINE_使用说明.md](docs/OR_CPT_ENGINE_使用说明.md).

## Layout

```text
or_cpt_engine/       engine package, prompts, vendored OptMATH seed generators
engine_configs/      generator profiles, contracts, quality thresholds
configs/             local LLM endpoint pool (not committed with private hosts)
cpt_cleaner/utils/   json/code/throughput helpers required by the engine
docs/                engine manuals
tests/               or_cpt_engine unit tests
```

Runtime outputs (`engine_runs/`, `logs/`) stay local and are gitignored.

## What was not imported

| Source | Why it stayed out |
| --- | --- |
| `cpt_cleaner/` pipeline (parse/repair/grade) | SFT→CPT cleaner, superseded in spirit by `or_sft2cpt_cleaner` |
| `or_sft2cpt_cleaner/` | Independent SFT/Alpaca cleaner; ~1.2G with run artifacts |
| `data/`, `engine_runs/` | Private corpora and production dumps |
| `configs/model_api_pool*.yaml` from source | Internal endpoint IPs |

For runnable SFT data generation, use [data/sft](../sft/). CPT training launchers are in [training/cpt](../../training/cpt/).
