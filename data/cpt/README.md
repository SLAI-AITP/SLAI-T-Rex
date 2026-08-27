# CPT Data Construction

Solver-verified OR-CPT factory: OptMATH generators → Gurobi checks → NL backtranslation → forward modeling → CPT documents.

```text
scan-generators -> smoke-test-generators -> audit-generators
  -> plan-production -> run-seed-factory -> run-cpt-from-seeds
```

## Install

```bash
cd SLAI-T-Rex/data/cpt
python3 -m pip install -e .
```

Needs `gurobipy` for seed factory / forward eval, and an OpenAI-compatible endpoint for LLM stages.

```bash
cp configs/model_api_pool.example.yaml configs/model_api_pool2.yaml
# edit base_url / model_name, or set OR_LLM_BASE_URL and OR_LLM_API_KEY
```

## Run

Commands must be launched from this directory (`engine_configs/` is resolved relative to cwd):

```bash
python3 -m or_cpt_engine.cli.main --help

python3 -m or_cpt_engine.cli.main audit-generators \
  --output engine_runs/seed_audit_smoke \
  --samples-per-generator 2 \
  --run-id seed_audit_smoke
```

YAML under `engine_configs/` controls generators, LLM, and quality thresholds. Outputs go to `engine_runs/` (gitignored).
