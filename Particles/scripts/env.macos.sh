#!/usr/bin/env bash

# ============================================================
# macOS Apple Silicon environment setup
#
# Intended usage:
#   source scripts/env.macos.sh
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
#
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${_SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

# ------------------------------------------------------------
# Platform checks
# ------------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERROR] scripts/env.macos.sh should only be used on macOS." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "[ERROR] This environment script is intended for native Apple Silicon arm64 shells." >&2
    echo "[ERROR] Current architecture reported by uname -m: $(uname -m)" >&2
    echo "[ERROR] If you are on Apple Silicon, make sure your terminal is not running under Rosetta." >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Homebrew checks
# ------------------------------------------------------------

if ! command -v brew >/dev/null 2>&1; then
    echo "[ERROR] Homebrew not found. Install Homebrew first: https://brew.sh" >&2
    return 1 2>/dev/null || exit 1
fi

BREW_PREFIX="$(brew --prefix)"
echo "[INFO] Homebrew prefix: ${BREW_PREFIX}"

if [[ "${BREW_PREFIX}" != "/opt/homebrew" ]]; then
    echo "[ERROR] Apple Silicon detected, but Homebrew prefix is not /opt/homebrew." >&2
    echo "[ERROR] Detected Homebrew prefix: ${BREW_PREFIX}" >&2
    echo "[ERROR] This usually means you are using an x86_64/Rosetta Homebrew installation." >&2
    echo "[ERROR] Please use native arm64 Homebrew to avoid compiler and linker problems." >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Dependency checks
# ------------------------------------------------------------
#
# Required:
# - cmake: needed by scripts/build.sh
# - hdf5: required by CMakeLists.txt via find_package(HDF5 REQUIRED ...)
#
# Optional but recommended:
# - libomp: needed if OpenMP support is desired with AppleClang.
#   The macOS preset uses PARTICLES_BUILD_OPENMP=AUTO, so the build can
#   still proceed without OpenMP, but the OpenMP target may be skipped.
#

missing_required=0

for pkg in cmake hdf5; do
    if ! brew list "${pkg}" >/dev/null 2>&1; then
        echo "[ERROR] Missing required Homebrew package: ${pkg}" >&2
        missing_required=1
    fi
done

if [[ "${missing_required}" == "1" ]]; then
    echo >&2
    echo "[INFO] Install required dependencies with:" >&2
    echo "       brew install cmake hdf5" >&2
    echo >&2
    return 1 2>/dev/null || exit 1
fi

if ! brew list libomp >/dev/null 2>&1; then
    echo "[WARN] Missing optional Homebrew package: libomp"
    echo "[WARN] OpenMP detection may fail, and the OpenMP target may not be built."
    echo "[INFO] To enable OpenMP support, install it with:"
    echo "       brew install libomp"
fi

# ------------------------------------------------------------
# Dependency roots
# ------------------------------------------------------------

HDF5_ROOT="$(brew --prefix hdf5)"
export HDF5_ROOT

echo "[INFO] HDF5_ROOT=${HDF5_ROOT}"

if brew list libomp >/dev/null 2>&1; then
    OPENMP_ROOT="$(brew --prefix libomp)"
    export OPENMP_ROOT
    echo "[INFO] OPENMP_ROOT=${OPENMP_ROOT}"
fi

# Help CMake discover Homebrew packages.
#
# CMAKE_PREFIX_PATH is a standard CMake search path variable.
# Keep any existing value, but prepend the project-relevant prefixes.
#
if [[ -n "${OPENMP_ROOT:-}" ]]; then
    export CMAKE_PREFIX_PATH="${HDF5_ROOT}:${OPENMP_ROOT}:${CMAKE_PREFIX_PATH:-}"
else
    export CMAKE_PREFIX_PATH="${HDF5_ROOT}:${CMAKE_PREFIX_PATH:-}"
fi

echo "[INFO] CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"

# ------------------------------------------------------------
# Select macOS build preset
# ------------------------------------------------------------
#
# The CMake preset remains the source of truth for build options:
# - PARTICLES_BUILD_OPENMP=AUTO
# - PARTICLES_BUILD_CUDA=OFF
# - CMAKE_OSX_ARCHITECTURES=arm64
#
# This makes the usual workflow correct:
#   source scripts/env.macos.sh
#   ./scripts/build.sh
#

export PRESET="${PRESET:-macos-arm64}"

# ------------------------------------------------------------
# Optional Homebrew LLVM compiler
# ------------------------------------------------------------
#
# USE_HOMEBREW_LLVM=1 switches the compiler to Homebrew LLVM.
#
# Important:
# - CMake reads CC/CXX on the first configure of a build directory.
# - If build/macos-arm64 already exists and was configured with a
#   different compiler, remove it before switching compilers:
#
#     rm -rf build/macos-arm64
#
# - Do not use CXX_COMPILER here; CMake does not automatically consume
#   that environment variable.
#

USE_HOMEBREW_LLVM="${USE_HOMEBREW_LLVM:-0}"

case "${USE_HOMEBREW_LLVM}" in
    0|1)
        ;;
    *)
        echo "[ERROR] USE_HOMEBREW_LLVM must be 0 or 1, got: ${USE_HOMEBREW_LLVM}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

if [[ "${USE_HOMEBREW_LLVM}" == "1" ]]; then
    if ! brew list llvm >/dev/null 2>&1; then
        echo "[ERROR] USE_HOMEBREW_LLVM=1 but Homebrew llvm is not installed." >&2
        echo "[ERROR] Run: brew install llvm" >&2
        return 1 2>/dev/null || exit 1
    fi

    LLVM_PREFIX="$(brew --prefix llvm)"
    LLVM_CLANG="${LLVM_PREFIX}/bin/clang"
    LLVM_CLANGXX="${LLVM_PREFIX}/bin/clang++"

    if [[ ! -x "${LLVM_CLANG}" ]]; then
        echo "[ERROR] Expected compiler not found or not executable: ${LLVM_CLANG}" >&2
        return 1 2>/dev/null || exit 1
    fi

    if [[ ! -x "${LLVM_CLANGXX}" ]]; then
        echo "[ERROR] Expected compiler not found or not executable: ${LLVM_CLANGXX}" >&2
        return 1 2>/dev/null || exit 1
    fi

    export PATH="${LLVM_PREFIX}/bin:${PATH}"
    export CC="${LLVM_CLANG}"
    export CXX="${LLVM_CLANGXX}"

    echo "[INFO] Using Homebrew LLVM C compiler:   ${CC}"
    echo "[INFO] Using Homebrew LLVM C++ compiler: ${CXX}"
else
    echo "[INFO] Using default C/C++ compiler unless CC/CXX are already set."
    if [[ -n "${CC:-}" ]]; then
        echo "[INFO] Existing CC=${CC}"
    fi
    if [[ -n "${CXX:-}" ]]; then
        echo "[INFO] Existing CXX=${CXX}"
    fi
fi

# ------------------------------------------------------------
# Optional Python virtual environment
# ------------------------------------------------------------
#
# Only source the virtualenv from the repository root we computed
# above. Do not use an externally supplied PROJECT_ROOT.
#

if [[ -d "${PROJECT_ROOT}/particles_venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/particles_venv/bin/activate"
    echo "[INFO] Activated Python virtualenv: ${PROJECT_ROOT}/particles_venv"
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] PRESET=${PRESET}"
