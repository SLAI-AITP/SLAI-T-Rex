# OR Benchmark Evaluation

This directory contains the standalone benchmark evaluation module for ORProject. It includes the benchmark data, prompts, evaluation code, summary tooling, and ready-to-run launch scripts.

The default evaluation suite covers:

- `NL4OPT`
- `OptiBench`
- `B40 feasible`
- `B40 ORGEval`

## Directory Layout

```text
or_benchmark_eval/
├── data/                  # Benchmark data
├── evaluation/            # Evaluation entry points for each benchmark
├── evaluate/              # B40 ORGEval LP-structure equivalence tools
├── scripts/               # 1-pass / 5-shot / 4-acc / 16-acc launch and summary scripts
├── requirements.txt       # Python dependencies for evaluation
└── README.md
```

The prompts used by the evaluation are stored in:

- `evaluation/nl4opt/prompt_template.txt`
- `evaluation/optibench/prompt_template.txt`
- `evaluation/bench4opt/prompt_template.txt`
- `evaluation/bench4opt/prompt_template_lp_structure.txt`

These are the actual prompts used by the launch scripts. They ask the model to produce Python code compatible with Gurobi 12.x.

## Environment Setup

Python 3.10 is recommended.

```bash
cd or_benchmark_eval

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

The evaluation executes or validates Gurobi models, so the machine must have a working `gurobipy` installation and a valid Gurobi license.

You can verify the Gurobi installation with:

```bash
python - <<'PY'
import gurobipy as gp
print(gp.gurobi.version())
PY
```

## Model Service Configuration

The launch scripts use an OpenAI-compatible chat completions endpoint. Configure the endpoint, model name, and optional API key through environment variables:

```bash
export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none
```

Set `MODEL_NAME` explicitly. This avoids ambiguity when different services expose different model discovery behavior.

## Launch Scripts

The four convenience scripts share these defaults:

- `CONCURRENCY=16`
- `MAX_TOKENS=9600`
- `TEMPERATURE=0.6`
- `TOP_P=1.0`
- `SEED=42`
- `BACKGROUND=true`

When `BACKGROUND=true`, each run writes a generated command file, metadata, PID, and full log under `logs/full_<run_name>/`.

### 1-pass

```bash
export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none
export CONCURRENCY=16
export MAX_TOKENS=9600

bash scripts/run_eval_1pass.sh
```

### 5-shot

```bash
export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none
export CONCURRENCY=16
export MAX_TOKENS=9600
export FEW_SHOT_SOURCE=your-few-shot-source.jsonl

bash scripts/run_eval_5shot.sh
```

For 5-shot evaluation, `FEW_SHOT_SOURCE` must point to a JSON or JSONL file containing reference examples. B40 ORGEval also supports comma-separated JSON result files. The sampler excludes the current sample ID when selecting examples, so a sample is not allowed to use itself as a shot.

By default, 5-shot uses similarity-based retrieval:

```bash
FEW_SHOT_STRATEGY=similar
```

You can override this if needed:

```bash
export FEW_SHOT_STRATEGY=random
```

### 4-acc

```bash
export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none
export CONCURRENCY=16
export MAX_TOKENS=9600

bash scripts/run_eval_acc4.sh
```

### 16-acc

```bash
export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none
export CONCURRENCY=16
export MAX_TOKENS=9600

bash scripts/run_eval_acc16.sh
```

### Foreground Smoke Test

Use a small slice in the foreground before launching a full run:

```bash
BACKGROUND=false START=0 END=2 bash scripts/run_eval_1pass.sh
```

## Output Files

Each launch prints paths like:

```text
RUN_NAME=your-model_mode_tokens_timestamp
RESULT=./results/your-run-name
LOG=./logs/full_your-run-name/run.log
COMMAND=./logs/full_your-run-name/command.sh
PID=background-pid
```

Main result files:

```text
results/your-run-name/nl4opt/your-model_solver.json
results/your-run-name/optibench/your-model_solver.json
results/your-run-name/bench4opt/your-model_solver.json
results/your-run-name/bench4opt/your-model_orgeval.json
results/your-run-name/summary/your-run-name/summary.md
results/your-run-name/summary/your-run-name/summary.json
```

Monitor a background run:

```bash
RUN_NAME=your-run-name
tail -f "logs/full_${RUN_NAME}/run.log"
ps -fp "$(cat "logs/full_${RUN_NAME}/pid")"
```

## Common Options

```bash
OPENAI_BASE_URL=your-model-service-base-url MODEL_NAME=your-model-name \
CONCURRENCY=8 MAX_TOKENS=6000 RUN_TAG=debug \
bash scripts/run_eval_1pass.sh
```

| Variable | Description | Default |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible model service base URL | Required |
| `MODEL_NAME` / `MODEL` | Model identifier served by the endpoint | Required |
| `OPENAI_API_KEY` | API key. Use `none` if the endpoint does not require authentication | `none` |
| `CONCURRENCY` | Request concurrency for each benchmark | `16` |
| `MAX_TOKENS` | Maximum completion tokens | `9600` |
| `TEMPERATURE` | Sampling temperature | `0.6` |
| `TOP_P` | Nucleus sampling parameter | `1.0` |
| `SEED` | Request seed | `42` |
| `RUN_TAG` | Timestamp-like suffix used when building `RUN_NAME` | Current timestamp |
| `RUN_NAME` | Custom result directory name | Auto-generated |
| `RESULT_ROOT` | Custom result root | `results/${RUN_NAME}` |
| `LOG_DIR` | Custom log directory | `logs/full_${RUN_NAME}` |
| `BACKGROUND` | Run in the background | `true` |
| `RERUN_FLAG` | Pass `--rerun` to ignore existing partial results | `true` |
| `BENCH4OPT_EVAL_WORKERS` | Local B40 validation concurrency | Same as `CONCURRENCY` |
| `FEW_SHOT_SOURCE` | Few-shot example source path | Empty |
| `FEW_SHOT_STRATEGY` | Few-shot selection strategy | `similar` for 5-shot, otherwise `random` |
| `START` / `END` | Optional sample slice | `0` / `none` |

## Unified Entry Point

The convenience scripts call `scripts/run_eval.sh` internally. You can also call the unified entry point directly:

```bash
bash scripts/run_eval.sh \
  --models "$MODEL_NAME" \
  --targets nl4opt_solver optibench_solver bench4opt_feasible_solver bench4opt_orgeval \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "${OPENAI_API_KEY:-none}" \
  --result_root results/manual_run \
  --summary_root results/manual_run/summary \
  --summary_tag manual_run \
  --temperature 0.6 \
  --top_p 1.0 \
  --max_tokens 9600 \
  --few_shot 0 \
  --acc_samples 1 \
  --rerun
```

## Regenerate Summary Only

If the raw JSON result files already exist, regenerate the summary without rerunning model inference:

```bash
RUN_NAME=your-run-name
python scripts/run_eval.py \
  --skip_run \
  --result_root "results/${RUN_NAME}" \
  --summary_root "results/${RUN_NAME}/summary" \
  --summary_tag "$RUN_NAME" \
  --models "$MODEL_NAME" \
  --targets nl4opt_solver optibench_solver bench4opt_feasible_solver bench4opt_orgeval
```

## Notes

- `B40 feasible` evaluates whether generated solver code can solve the instance and recover the expected objective value.
- `B40 ORGEval` evaluates whether generated code can build an LP whose structure is equivalent to the reference LP.
- These two B40 metrics are related but not identical: one checks solver execution and objective recovery, while the other checks LP construction and graph-structure equivalence.
- Keep endpoint URLs, API keys, and deployment-specific settings outside the repository. Prefer environment variables or local shell wrappers for private configuration.
