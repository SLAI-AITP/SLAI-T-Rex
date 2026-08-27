#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODE=${MODE:-1pass}
case "$MODE" in
  1pass)
    DEFAULT_FEW_SHOT=0
    DEFAULT_ACC_SAMPLES=1
    DEFAULT_FEW_SHOT_STRATEGY=random
    ;;
  5shot)
    DEFAULT_FEW_SHOT=5
    DEFAULT_ACC_SAMPLES=1
    DEFAULT_FEW_SHOT_STRATEGY=similar
    ;;
  acc4)
    DEFAULT_FEW_SHOT=0
    DEFAULT_ACC_SAMPLES=4
    DEFAULT_FEW_SHOT_STRATEGY=random
    ;;
  acc16)
    DEFAULT_FEW_SHOT=0
    DEFAULT_ACC_SAMPLES=16
    DEFAULT_FEW_SHOT_STRATEGY=random
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use one of: 1pass, 5shot, acc4, acc16." >&2
    exit 2
    ;;
esac

if [ -z "${OPENAI_BASE_URL:-}" ]; then
  echo "Set OPENAI_BASE_URL to your model service base URL." >&2
  exit 2
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export OPENAI_BASE_URL

MODEL_NAME="${MODEL_NAME:-${MODEL:-}}"
if [ -z "$MODEL_NAME" ]; then
  echo "Set MODEL_NAME to the model identifier served by your endpoint." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ALIAS="${MODEL_ALIAS:-$(printf '%s' "$MODEL_NAME" | sed 's#[^A-Za-z0-9._-]#_#g')}"
MAX_TOKENS="${MAX_TOKENS:-9600}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-16}"
FEW_SHOT="${FEW_SHOT:-$DEFAULT_FEW_SHOT}"
FEW_SHOT_STRATEGY="${FEW_SHOT_STRATEGY:-$DEFAULT_FEW_SHOT_STRATEGY}"
FEW_SHOT_SOURCE="${FEW_SHOT_SOURCE:-}"
ACC_SAMPLES="${ACC_SAMPLES:-$DEFAULT_ACC_SAMPLES}"
START="${START:-0}"
END="${END:-none}"
RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}_${MODE}_${MAX_TOKENS}_${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-results/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-logs/full_${RUN_NAME}}"
BACKGROUND="${BACKGROUND:-true}"
RERUN_FLAG="${RERUN_FLAG:-true}"
PASS_OPENAI_API_KEY_ARG="${PASS_OPENAI_API_KEY_ARG:-false}"
WRITE_OPENAI_API_KEY_TO_COMMAND="${WRITE_OPENAI_API_KEY_TO_COMMAND:-false}"
BENCH4OPT_EVAL_WORKERS="${BENCH4OPT_EVAL_WORKERS:-$CONCURRENCY}"

if [ "$FEW_SHOT" -gt 0 ] && [ -z "$FEW_SHOT_SOURCE" ]; then
  echo "Set FEW_SHOT_SOURCE to a JSON/JSONL few-shot example file when FEW_SHOT > 0." >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

COMMAND_FILE="${LOG_DIR}/command.sh"
if [ "$WRITE_OPENAI_API_KEY_TO_COMMAND" = "true" ]; then
  COMMAND_OPENAI_API_KEY_LINE="export OPENAI_API_KEY='$OPENAI_API_KEY'"
else
  COMMAND_OPENAI_API_KEY_LINE='export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"'
fi

RERUN_LINE=
if [ "$RERUN_FLAG" = "true" ]; then
  RERUN_LINE='  --rerun \'
fi

FEW_SHOT_SOURCE_LINE=
if [ -n "$FEW_SHOT_SOURCE" ]; then
  FEW_SHOT_SOURCE_LINE="  --few_shot_source '$FEW_SHOT_SOURCE' \\"
fi

cat > "$COMMAND_FILE" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
$COMMAND_OPENAI_API_KEY_LINE
export OPENAI_BASE_URL='$OPENAI_BASE_URL'
export EVAL_CONCURRENCY='$CONCURRENCY'
export NL4OPT_WORKERS='${NL4OPT_WORKERS:-$CONCURRENCY}'
export OPTIBENCH_WORKERS='${OPTIBENCH_WORKERS:-$CONCURRENCY}'
export BENCH4OPT_SOLVER_WORKERS='${BENCH4OPT_SOLVER_WORKERS:-$CONCURRENCY}'
export BENCH4OPT_ORGEVAL_WORKERS='${BENCH4OPT_ORGEVAL_WORKERS:-$CONCURRENCY}'
export BENCH4OPT_EVAL_WORKERS='$BENCH4OPT_EVAL_WORKERS'

bash scripts/run_eval.sh \\
  --models '$MODEL_NAME' \\
  --targets nl4opt_solver optibench_solver bench4opt_feasible_solver bench4opt_orgeval \\
  --openai_api_key "\${OPENAI_API_KEY:-none}" \\
  --pass_openai_api_key_arg '$PASS_OPENAI_API_KEY_ARG' \\
  --openai_base_url '$OPENAI_BASE_URL' \\
  --result_root '$RESULT_ROOT' \\
  --summary_root '$RESULT_ROOT/summary' \\
  --summary_tag '$RUN_NAME' \\
  --temperature '$TEMPERATURE' \\
  --top_p '$TOP_P' \\
  --max_tokens '$MAX_TOKENS' \\
  --seed '$SEED' \\
  --few_shot '$FEW_SHOT' \\
  --few_shot_strategy '$FEW_SHOT_STRATEGY' \\
$FEW_SHOT_SOURCE_LINE
  --acc_samples '$ACC_SAMPLES' \\
  --start '$START' \\
  --end '$END' \\
$RERUN_LINE
  --verbose
SH

chmod +x "$COMMAND_FILE"

cat > "${LOG_DIR}/run_metadata.txt" <<EOF
run_name=${RUN_NAME}
mode=${MODE}
model=${MODEL_NAME}
openai_base_url=${OPENAI_BASE_URL}
result_root=${REPO_ROOT}/${RESULT_ROOT}
log_file=${REPO_ROOT}/${LOG_DIR}/run.log
command_file=${REPO_ROOT}/${COMMAND_FILE}
targets=nl4opt_solver optibench_solver bench4opt_feasible_solver bench4opt_orgeval
temperature=${TEMPERATURE}
top_p=${TOP_P}
max_tokens=${MAX_TOKENS}
seed=${SEED}
concurrency=${CONCURRENCY}
bench4opt_eval_workers=${BENCH4OPT_EVAL_WORKERS}
few_shot=${FEW_SHOT}
few_shot_strategy=${FEW_SHOT_STRATEGY}
few_shot_source=${FEW_SHOT_SOURCE}
acc_samples=${ACC_SAMPLES}
rerun=${RERUN_FLAG}
pass_openai_api_key_arg=${PASS_OPENAI_API_KEY_ARG}
write_openai_api_key_to_command=${WRITE_OPENAI_API_KEY_TO_COMMAND}
EOF

echo "RUN_NAME=${RUN_NAME}"
echo "MODE=${MODE}"
echo "MODEL=${MODEL_NAME}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "RESULT=${REPO_ROOT}/${RESULT_ROOT}"
echo "LOG=${REPO_ROOT}/${LOG_DIR}/run.log"
echo "COMMAND=${REPO_ROOT}/${COMMAND_FILE}"

if [ "$BACKGROUND" = "false" ]; then
  exec bash "$COMMAND_FILE"
fi

setsid bash "$COMMAND_FILE" > "${LOG_DIR}/run.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "${LOG_DIR}/pid"
echo "PID=${PID}"
