#!/usr/bin/env bash
set -euo pipefail

MODE=5shot exec "$(dirname "$0")/launch_eval.sh"
