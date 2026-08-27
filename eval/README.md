# OR Benchmark Evaluation

NL4OPT, OptiBench, B4O-Feasible (solver objective), B4O-ORGEval (LP structure). Needs Python 3.10+, `gurobipy`, and Gurobi license. Prompts ask for Gurobi 12.x Python.

```bash
cd eval
python -m pip install -r requirements.txt

export OPENAI_BASE_URL=your-model-service-base-url
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=none

BACKGROUND=false START=0 END=2 bash scripts/run_eval_1pass.sh
bash scripts/run_eval_1pass.sh
```

Other modes: `run_eval_5shot.sh` (set `FEW_SHOT_SOURCE`), `run_eval_acc4.sh`, `run_eval_acc16.sh`. Shared flags: `CONCURRENCY`, `MAX_TOKENS`, `RUN_NAME`. Results under `results/<run>/`, logs under `logs/full_<run>/`.
