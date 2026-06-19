#!/usr/bin/env bash

# ============================================================
# Leonardo-specific environment setup
#
# Intended usage:
#   source scripts/env.leonardo.sh
#   ./scripts/build.sh
#
# This file is meant to be sourced, not executed.
# ============================================================

# ------------------------------------------------------------
# Resolve project root from this script location
# ------------------------------------------------------------
#
# Security rationale:
# - Do not trust an externally supplied PROJECT_ROOT when sourcing
#   files such as virtualenv activation scripts.
# - source executes shell code, so the path should be derived from
#   this repository, not from the user's ambient environment.
# ------------------------------------------------------------

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${_SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

# ------------------------------------------------------------
# Check environment modules
# ------------------------------------------------------------

if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] Environment modules are not available on this system." >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Load Leonardo software stack
# ------------------------------------------------------------
#
# module purge is intentional here: it gives a reproducible module
# environment for this project. Be aware that this modifies the
# current shell when the file is sourced.
#
# The Leonardo preset expects:
# - PARTICLES_BUILD_HDF5=ON
# - PARTICLES_BUILD_OPENMP=ON
# - PARTICLES_BUILD_CUDA=ON
# - PARTICLES_STRICT_CUDA=ON
# - PARTICLES_CUDA_ARCHITECTURES=80
# ------------------------------------------------------------

module purge

module load gcc/12.2.0
module load cuda/12.2
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7

# ------------------------------------------------------------
# Select Leonardo build preset
# ------------------------------------------------------------

export PRESET="${PRESET:-leonardo-a100}"

# ------------------------------------------------------------
# CUDA compiler
# ------------------------------------------------------------

if command -v nvcc >/dev/null 2>&1; then
    CUDACXX="$(command -v nvcc)"
    export CUDACXX
    echo "[INFO] CUDACXX=${CUDACXX}"
else
    echo "[ERROR] nvcc not found after loading cuda module." >&2
    echo "[ERROR] CUDA was expected on Leonardo; check the cuda module." >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Basic tool checks
# ------------------------------------------------------------

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake not found after loading cmake module." >&2
    return 1 2>/dev/null || exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found after loading gcc module." >&2
    return 1 2>/dev/null || exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[WARN] python3 not found after loading python module."
fi

# ------------------------------------------------------------
# HDF5 discovery hints
# ------------------------------------------------------------
#
# The HDF5 module should normally provide enough information for CMake.
# Still, print these if they exist to help debugging.
# ------------------------------------------------------------

if [[ -n "${HDF5_ROOT:-}" ]]; then
    echo "[INFO] HDF5_ROOT=${HDF5_ROOT}"
fi

if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    echo "[INFO] CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
fi

# ------------------------------------------------------------
# Optional Python virtual environment
# ------------------------------------------------------------
#
# Only source the virtualenv from the repository root computed above.
# Do not use an externally supplied PROJECT_ROOT.
# ------------------------------------------------------------

if [[ -d "${PROJECT_ROOT}/particles_venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/particles_venv/bin/activate"
    echo "[INFO] Activated Python virtualenv: ${PROJECT_ROOT}/particles_venv"
fi

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] PRESET=${PRESET}"
echo "[INFO] Leonardo environment ready."
