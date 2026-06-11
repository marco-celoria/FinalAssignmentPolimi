#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Project build entry point
#
# Usage:
#   ./scripts/build.sh
#
# Optional environment overrides:
#   PRESET=leonardo-a100 JOBS=32 ./scripts/build.sh
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd -- "${PROJECT_ROOT}"

PRESET="${PRESET:-generic-x86-nogpu}"
JOBS="${JOBS:-}"

# ------------------------------------------------------------
# Validate preset
# ------------------------------------------------------------
#
# Security/stability rationale:
# - PRESET is environment-controlled.
# - Quoting already prevents shell injection.
# - This whitelist prevents accidental invalid presets and
#   option-like or path-like values from being passed to CMake.
#
case "${PRESET}" in
    generic-x86-nogpu|generic-x86-nvidia|macos-arm64|leonardo-a100)
        ;;
    *)
        echo "[ERROR] Unknown PRESET: ${PRESET}" >&2
        echo "[INFO] Valid presets are:" >&2
        echo "       generic-x86-nogpu" >&2
        echo "       generic-x86-nvidia" >&2
        echo "       macos-arm64" >&2
        echo "       leonardo-a100" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------
# Determine parallel build jobs
# ------------------------------------------------------------

if [[ -z "${JOBS}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS="$(sysctl -n hw.ncpu)"
    else
        JOBS=4
    fi
fi

# ------------------------------------------------------------
# Validate JOBS
# ------------------------------------------------------------
#
# Security/stability rationale:
# - JOBS is environment-controlled.
# - It is not shell-injection dangerous because it is quoted.
# - However, validating it avoids confusing CMake behavior from
#   values such as "0", "-1", "abc", or "--help".
#
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] JOBS must be a positive integer, got: ${JOBS}" >&2
    exit 1
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] CMake preset: ${PRESET}"
echo "[INFO] Build jobs:   ${JOBS}"

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" --parallel "${JOBS}"
