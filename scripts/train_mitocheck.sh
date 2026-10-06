#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export MITOV2_ROOT="${MITOV2_ROOT:-/work/tl441/mitov2}"
exec "${PYTHON:-python}" src/train_mitocheck.py "$@"
