#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# macOS Apple Silicon environment setup
# ============================================================

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERROR] scripts/env.macos.sh should only be used on macOS." >&2
    return 1 2>/dev/null || exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "[ERROR] Homebrew not found. Install Homebrew first: https://brew.sh" >&2
    return 1 2>/dev/null || exit 1
fi

BREW_PREFIX="$(brew --prefix)"
echo "[INFO] Homebrew prefix: ${BREW_PREFIX}"

if [[ "$(uname -m)" == "arm64" && "${BREW_PREFIX}" != "/opt/homebrew" ]]; then
    echo "[WARN] Apple Silicon detected but Homebrew prefix is not /opt/homebrew."
    echo "[WARN] You may be using Rosetta/x86_64 Homebrew, which can cause link problems."
fi

missing=0

for pkg in cmake hdf5 libomp; do
    if ! brew list "${pkg}" >/dev/null 2>&1; then
        echo "[WARN] Missing Homebrew package: ${pkg}"
        missing=1
    fi
done

if [[ "${missing}" == "1" ]]; then
    echo
    echo "[INFO] Install missing dependencies with:"
    echo "       brew install cmake hdf5 libomp"
    echo
    echo "[INFO] Optional, if AppleClang OpenMP detection still fails:"
    echo "       brew install llvm"
    echo
fi

export HDF5_ROOT="$(brew --prefix hdf5)"
export OPENMP_ROOT="$(brew --prefix libomp)"

echo "[INFO] HDF5_ROOT=${HDF5_ROOT}"
echo "[INFO] OPENMP_ROOT=${OPENMP_ROOT}"

export BUILD_OPENMP="${BUILD_OPENMP:-ON}"
export BUILD_CUDA="OFF"
export NATIVE_ARCH="${NATIVE_ARCH:-OFF}"
export FAST_MATH_CUDA="OFF"

USE_HOMEBREW_LLVM="${USE_HOMEBREW_LLVM:-0}"

if [[ "${USE_HOMEBREW_LLVM}" == "1" ]]; then
    if ! brew list llvm >/dev/null 2>&1; then
        echo "[ERROR] USE_HOMEBREW_LLVM=1 but Homebrew llvm is not installed." >&2
        echo "[ERROR] Run: brew install llvm" >&2
        return 1 2>/dev/null || exit 1
    fi

    LLVM_PREFIX="$(brew --prefix llvm)"
    LLVM_CLANGXX="${LLVM_PREFIX}/bin/clang++"

    if [[ ! -x "${LLVM_CLANGXX}" ]]; then
        echo "[ERROR] Expected compiler not found: ${LLVM_CLANGXX}" >&2
        return 1 2>/dev/null || exit 1
    fi

    export PATH="${LLVM_PREFIX}/bin:${PATH}"
    export CXX_COMPILER="${LLVM_CLANGXX}"

    echo "[INFO] Using Homebrew LLVM clang++: ${CXX_COMPILER}"
else
    echo "[INFO] Using default C++ compiler unless CXX_COMPILER is already set."
fi

if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}/particles_venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/particles_venv/bin/activate"
    echo "[INFO] Activated Python venv: ${PROJECT_ROOT}/particles_venv"
elif [[ -d "particles_venv" ]]; then
    # shellcheck source=/dev/null
    source "particles_venv/bin/activate"
    echo "[INFO] Activated Python venv: $(pwd)/particles_venv"
fi
