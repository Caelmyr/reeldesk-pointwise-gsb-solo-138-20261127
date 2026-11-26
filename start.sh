#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$root"
export REELDESK_DATA="${REELDESK_DATA:-$root/data}"
export REELDESK_HOST="${REELDESK_HOST:-127.0.0.1}"
export REELDESK_PORT="${REELDESK_PORT:-8080}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${REELDESK_PYTHON:-python3}" -B -m backend.server
