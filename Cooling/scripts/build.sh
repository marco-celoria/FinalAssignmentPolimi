#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PRESET="${PRESET:-generic-x86-nogpu}"
JOBS="${JOBS:-}"

if [[ -z "${JOBS}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS="$(sysctl -n hw.ncpu)"
    else
        JOBS=4
    fi
fi

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" --parallel "${JOBS}"
