#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODELS=${MODELS:-"deepseek-v4-flash"}
TARGETS=${TARGETS:-"nl4opt_solver optibench_solver bench4opt_feasible_solver bench4opt_orgeval"}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
PASS_OPENAI_API_KEY_ARG=${PASS_OPENAI_API_KEY_ARG:-true}
CONDA_ENV=${CONDA_ENV:-}
RESULT_ROOT=${RESULT_ROOT:-results}
SUMMARY_ROOT=${SUMMARY_ROOT:-}
SUMMARY_TAG=${SUMMARY_TAG:-}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-9600}
SEED=${SEED:-42}
FEW_SHOT=${FEW_SHOT:-0}
FEW_SHOT_SEED=${FEW_SHOT_SEED:-}
FEW_SHOT_STRATEGY=${FEW_SHOT_STRATEGY:-random}
FEW_SHOT_SOURCE=${FEW_SHOT_SOURCE:-}
ACC_SAMPLES=${ACC_SAMPLES:-1}
START=${START:-0}
END=${END:-}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-}
NL4OPT_WORKERS=${NL4OPT_WORKERS:-${EVAL_CONCURRENCY:-16}}
OPTIBENCH_WORKERS=${OPTIBENCH_WORKERS:-${EVAL_CONCURRENCY:-16}}
BENCH4OPT_SOLVER_WORKERS=${BENCH4OPT_SOLVER_WORKERS:-${EVAL_CONCURRENCY:-16}}
BENCH4OPT_ORGEVAL_WORKERS=${BENCH4OPT_ORGEVAL_WORKERS:-${EVAL_CONCURRENCY:-16}}
BENCH4OPT_EVAL_WORKERS=${BENCH4OPT_EVAL_WORKERS:-${EVAL_CONCURRENCY:-1}}
BENCH4OPT_MAX_SAMPLES=${BENCH4OPT_MAX_SAMPLES:-}
NL4OPT_DATASET=${NL4OPT_DATASET:-./data/NL4OPT}
OPTIBENCH_DATASET=${OPTIBENCH_DATASET:-./data/optibench}
BENCH4OPT_DATASET=${BENCH4OPT_DATASET:-data/bench4opt}
BENCH4OPT_FEASIBLE_DATASET=${BENCH4OPT_FEASIBLE_DATASET:-data/bench4opt_feasible}
RERUN=${RERUN:-false}
SKIP_RUN=${SKIP_RUN:-false}
SKIP_SUMMARY=${SKIP_SUMMARY:-false}
SKIP_MISSING_RESULTS=${SKIP_MISSING_RESULTS:-false}
VERBOSE=${VERBOSE:-false}
FAIL_FAST=${FAIL_FAST:-false}

FAILED_COMMANDS=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_eval.sh [options]

Targets:
  nl4opt_solver
  optibench_solver
  bench4opt_feasible_solver
  bench4opt_orgeval

Common options:
  --models MODEL [MODEL ...]
  --targets TARGET [TARGET ...]
  --openai_api_key KEY
  --openai_base_url URL
  --result_root DIR
  --summary_root DIR
  --summary_tag TAG
  --temperature VALUE              Default: 0.6
  --top_p VALUE                    Default: 1.0
  --max_tokens VALUE               Default: 9600
  --seed VALUE                     Default: 42
  --few_shot VALUE                 Default: 0
  --few_shot_seed VALUE            Default: --seed
  --few_shot_strategy VALUE        random or similar, default: random
  --few_shot_source PATH           User-provided few-shot JSON/JSONL file
  --acc_samples VALUE              Default: 1
  --start VALUE                    Default: 0
  --end VALUE                      Pass none to disable
  --bench4opt_max_samples VALUE
  --nl4opt_dataset PATH            Default: ./data/NL4OPT
  --optibench_dataset PATH         Default: ./data/optibench
  --bench4opt_dataset PATH         Default: data/bench4opt
  --bench4opt_feasible_dataset PATH Default: data/bench4opt_feasible
  --rerun
  --skip_run
  --skip_summary
  --skip_missing_results
  --pass_openai_api_key_arg VALUE  Default: true
  --verbose
  --fail_fast
EOF
}

is_true() {
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

safe_model_name() {
    printf '%s' "$1" | sed 's#[^A-Za-z0-9._-]#-#g'
}

range_suffix() {
    if [ "$START" != "0" ] || [ -n "$END" ]; then
        end_text=${END:-none}
        printf '_%s_%s' "$START" "$end_text"
    fi
}

result_path_for_target() {
    target_key=$1
    model_name=$2
    safe_model=$(safe_model_name "$model_name")
    slice_suffix=$(range_suffix)

    case "$target_key" in
        nl4opt_solver)
            printf '%s/nl4opt/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        optibench_solver)
            printf '%s/optibench/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        bench4opt_feasible_solver)
            printf '%s/bench4opt/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        bench4opt_orgeval)
            max_suffix=
            if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
                max_suffix="_max${BENCH4OPT_MAX_SAMPLES}"
            fi
            printf '%s/bench4opt/%s%s%s_orgeval.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix" "$max_suffix"
            ;;
        *)
            printf '%s' ""
            ;;
    esac
}

ensure_parent_dir() {
    mkdir -p "$(dirname "$1")"
}

run_python() {
    if [ -n "$CONDA_ENV" ]; then
        conda run -n "$CONDA_ENV" python "$@"
    else
        python "$@"
    fi
}

run_command() {
    label=$1
    shift

    echo
    echo "=== Running ${label} ==="
    masked_args=
    mask_next=false
    for arg in "$@"; do
        if [ "$mask_next" = "true" ]; then
            arg="***masked***"
            mask_next=false
        elif [ "$arg" = "--openai_api_key" ]; then
            mask_next=true
        fi
        masked_args="${masked_args} ${arg}"
    done
    echo "python${masked_args}"

    if run_python "$@"; then
        return 0
    fi

    rc=$?
    FAILED_COMMANDS=$((FAILED_COMMANDS + 1))
    echo "[WARN] ${label} failed with exit code ${rc}" >&2
    if is_true "$FAIL_FAST"; then
        exit "$rc"
    fi
    return 0
}

append_common_generation_args() {
    set -- "$@" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --few_shot "$FEW_SHOT" \
        --few_shot_seed "$FEW_SHOT_SEED" \
        --acc_samples "$ACC_SAMPLES" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --start "$START"

    if is_true "$PASS_OPENAI_API_KEY_ARG"; then
        set -- "$@" --openai_api_key "$OPENAI_API_KEY"
    fi
    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if [ -n "$FEW_SHOT_SOURCE" ]; then
        set -- "$@" --few_shot_source "$FEW_SHOT_SOURCE"
    fi
    if is_true "$RERUN"; then
        set -- "$@" --rerun
    fi

    printf '%s\n' "$@"
}

run_nl4opt_solver() {
    model_name=$1
    save_path=$2
    set -- -m evaluation.nl4opt.run_evaluation_solver \
        --dataset_name "$NL4OPT_DATASET" \
        --split test \
        --batch_size "$NL4OPT_WORKERS" \
        --num_workers "$NL4OPT_WORKERS" \
        --model_name "$model_name" \
        --timeout 360.0 \
        --tolerance 1e-6 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"
    args=$(append_common_generation_args "$@")
    # shellcheck disable=SC2086
    run_command "nl4opt solver / ${model_name}" $args
}

run_optibench_solver() {
    model_name=$1
    save_path=$2
    set -- -m evaluation.optibench.run_evaluation_solver \
        --dataset_name "$OPTIBENCH_DATASET" \
        --split test \
        --batch_size "$OPTIBENCH_WORKERS" \
        --num_workers "$OPTIBENCH_WORKERS" \
        --model_name "$model_name" \
        --timeout 360.0 \
        --tolerance 1e-6 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"
    args=$(append_common_generation_args "$@")
    # shellcheck disable=SC2086
    run_command "optibench solver / ${model_name}" $args
}

run_bench4opt_feasible_solver() {
    model_name=$1
    save_path=$2
    set -- -m evaluation.bench4opt.run_evaluation_solver \
        --data_dir "$BENCH4OPT_FEASIBLE_DATASET" \
        --model_name "$model_name" \
        --max_workers "$BENCH4OPT_SOLVER_WORKERS" \
        --timeout 360 \
        --tolerance 1e-6 \
        --save_every 32 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"
    args=$(append_common_generation_args "$@")
    # shellcheck disable=SC2086
    run_command "bench4opt feasible solver / ${model_name}" $args
}

run_bench4opt_orgeval() {
    model_name=$1
    save_path=$2
    set -- -m evaluation.bench4opt.run_evaluation_fast \
        --model_name "$model_name" \
        --save_path "$save_path" \
        --data_dir "$BENCH4OPT_DATASET" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --max_workers "$BENCH4OPT_ORGEVAL_WORKERS" \
        --eval_workers "$BENCH4OPT_EVAL_WORKERS" \
        --eval_timeout 180 \
        --save_every 32 \
        --seed "$SEED" \
        --few_shot "$FEW_SHOT" \
        --few_shot_seed "$FEW_SHOT_SEED" \
        --few_shot_strategy "$FEW_SHOT_STRATEGY" \
        --acc_samples "$ACC_SAMPLES" \
        --start "$START" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS"
    if is_true "$PASS_OPENAI_API_KEY_ARG"; then
        set -- "$@" --openai_api_key "$OPENAI_API_KEY"
    fi
    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
        set -- "$@" --max_samples "$BENCH4OPT_MAX_SAMPLES"
    fi
    if [ -n "$FEW_SHOT_SOURCE" ]; then
        set -- "$@" --few_shot_source "$FEW_SHOT_SOURCE"
    fi
    run_command "bench4opt orgeval / ${model_name}" "$@"
}

run_summary() {
    set -- "$SCRIPT_DIR/run_eval.py" --skip_run --result_root "$RESULT_ROOT"

    if [ -n "$SUMMARY_ROOT" ]; then
        set -- "$@" --summary_root "$SUMMARY_ROOT"
    fi
    if [ -n "$SUMMARY_TAG" ]; then
        set -- "$@" --summary_tag "$SUMMARY_TAG"
    fi
    if [ "$START" != "0" ]; then
        set -- "$@" --start "$START"
    fi
    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
        set -- "$@" --bench4opt_max_samples "$BENCH4OPT_MAX_SAMPLES"
    fi
    if is_true "$SKIP_MISSING_RESULTS"; then
        set -- "$@" --skip_missing_results
    fi

    set -- "$@" \
        --nl4opt_dataset "$NL4OPT_DATASET" \
        --optibench_dataset "$OPTIBENCH_DATASET" \
        --bench4opt_dataset "$BENCH4OPT_DATASET" \
        --bench4opt_feasible_dataset "$BENCH4OPT_FEASIBLE_DATASET" \
        --models

    for model_name in $MODELS; do
        set -- "$@" "$model_name"
    done

    set -- "$@" --targets
    for target_key in $TARGETS; do
        set -- "$@" "$target_key"
    done

    run_command "summary" "$@"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            shift
            MODELS=
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                MODELS="${MODELS:+$MODELS }$1"
                shift
            done
            continue
            ;;
        --targets)
            shift
            TARGETS=
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                TARGETS="${TARGETS:+$TARGETS }$1"
                shift
            done
            continue
            ;;
        --openai_api_key) OPENAI_API_KEY=$2; shift 2 ;;
        --openai_base_url) OPENAI_BASE_URL=$2; shift 2 ;;
        --conda_env) CONDA_ENV=$2; shift 2 ;;
        --result_root) RESULT_ROOT=$2; shift 2 ;;
        --summary_root) SUMMARY_ROOT=$2; shift 2 ;;
        --summary_tag) SUMMARY_TAG=$2; shift 2 ;;
        --temperature) TEMPERATURE=$2; shift 2 ;;
        --top_p) TOP_P=$2; shift 2 ;;
        --max_tokens) MAX_TOKENS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --few_shot) FEW_SHOT=$2; shift 2 ;;
        --few_shot_seed) FEW_SHOT_SEED=$2; shift 2 ;;
        --few_shot_strategy) FEW_SHOT_STRATEGY=$2; shift 2 ;;
        --few_shot_source) FEW_SHOT_SOURCE=$2; shift 2 ;;
        --acc_samples) ACC_SAMPLES=$2; shift 2 ;;
        --start) START=$2; shift 2 ;;
        --end)
            if [ "$2" = "none" ]; then END=; else END=$2; fi
            shift 2
            ;;
        --bench4opt_max_samples) BENCH4OPT_MAX_SAMPLES=$2; shift 2 ;;
        --nl4opt_dataset) NL4OPT_DATASET=$2; shift 2 ;;
        --optibench_dataset) OPTIBENCH_DATASET=$2; shift 2 ;;
        --bench4opt_dataset) BENCH4OPT_DATASET=$2; shift 2 ;;
        --bench4opt_feasible_dataset) BENCH4OPT_FEASIBLE_DATASET=$2; shift 2 ;;
        --rerun) RERUN=true; shift ;;
        --skip_run) SKIP_RUN=true; shift ;;
        --skip_summary) SKIP_SUMMARY=true; shift ;;
        --skip_missing_results) SKIP_MISSING_RESULTS=true; shift ;;
        --pass_openai_api_key_arg) PASS_OPENAI_API_KEY_ARG=$2; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --fail_fast) FAIL_FAST=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "$MODELS" ]; then
    echo "MODELS is empty" >&2
    exit 1
fi

if [ -z "$FEW_SHOT_SEED" ]; then
    FEW_SHOT_SEED=$SEED
fi

if ! is_true "$SKIP_RUN"; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY is required unless --skip_run is used" >&2
        exit 1
    fi
    if [ -z "$OPENAI_BASE_URL" ]; then
        echo "OPENAI_BASE_URL is required unless --skip_run is used" >&2
        exit 1
    fi
fi

if ! is_true "$SKIP_RUN"; then
    for target_key in $TARGETS; do
        for model_name in $MODELS; do
            save_path=$(result_path_for_target "$target_key" "$model_name")
            ensure_parent_dir "$save_path"
            case "$target_key" in
                nl4opt_solver) run_nl4opt_solver "$model_name" "$save_path" ;;
                optibench_solver) run_optibench_solver "$model_name" "$save_path" ;;
                bench4opt_feasible_solver) run_bench4opt_feasible_solver "$model_name" "$save_path" ;;
                bench4opt_orgeval) run_bench4opt_orgeval "$model_name" "$save_path" ;;
                *) echo "Unknown target: $target_key" >&2; exit 1 ;;
            esac
        done
    done
fi

if ! is_true "$SKIP_SUMMARY"; then
    run_summary
fi

if [ "$FAILED_COMMANDS" -gt 0 ]; then
    echo "Completed with ${FAILED_COMMANDS} failed command(s)" >&2
    exit 1
fi
