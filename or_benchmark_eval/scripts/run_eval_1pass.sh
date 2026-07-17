#!/usr/bin/env bash
set -euo pipefail

MODE=1pass exec "$(dirname "$0")/launch_eval.sh"
