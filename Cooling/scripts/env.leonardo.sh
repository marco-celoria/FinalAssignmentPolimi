#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Leonardo-specific environment setup
# ============================================================

if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] Environment modules are not available on this system." >&2
    return 1 2>/dev/null || exit 1
fi

module purge

module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7

export BUILD_OPENMP="${BUILD_OPENMP:-ON}"
export BUILD_CUDA="${BUILD_CUDA:-ON}"
export STRICT_CUDA="${STRICT_CUDA:-1}"
export NATIVE_ARCH="${NATIVE_ARCH:-OFF}"
export FAST_MATH_CUDA="${FAST_MATH_CUDA:-OFF}"
export REPRODUCIBLE_FP="${REPRODUCIBLE_FP:-MODERATE}"

if command -v nvcc >/dev/null 2>&1; then
    export CUDACXX="$(command -v nvcc)"
    echo "[INFO] CUDACXX=${CUDACXX}"
else
    echo "[WARN] nvcc not found after loading cuda module"
fi

if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}/cooling_venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/cooling_venv/bin/activate"
elif [[ -d "cooling_venv" ]]; then
    # shellcheck source=/dev/null
    source "cooling_venv/bin/activate"
fi
