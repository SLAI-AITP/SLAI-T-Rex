#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-${1:-configs/run.example.yaml}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p runs logs
RUN_ID="$(python - "${CONFIG}" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print((data.get("run") or {}).get("run_id") or "run")
PY
)"

cmd=(python -m or_data_distill run --config "${CONFIG}")
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

{
  echo "Started at $(date '+%F %T')"
  echo "Config: ${CONFIG}"
  printf "Command:"
  printf " %q" "${cmd[@]}"
  echo
} > "logs/${RUN_ID}.log"

setsid env PYTHONUNBUFFERED=1 "${cmd[@]}" >> "logs/${RUN_ID}.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "logs/${RUN_ID}.pid"
echo "Started ${RUN_ID} with pid ${pid}"
echo "Log: logs/${RUN_ID}.log"

