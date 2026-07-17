#!/usr/bin/env bash
set -euo pipefail

MODE=acc16 exec "$(dirname "$0")/launch_eval.sh"
