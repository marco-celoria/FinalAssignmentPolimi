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
# ------------------------------------------------------------

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
# Required dependency checks
# ------------------------------------------------------------
#
# Required:
# - cmake
#
# Optional:
# - hdf5: used if COOLING_BUILD_HDF5=AUTO and CMake finds it
# - libomp: used if COOLING_BUILD_OPENMP=AUTO and CMake finds it
# - llvm: used only if USE_HOMEBREW_LLVM=1
# ------------------------------------------------------------

if ! brew list cmake >/dev/null 2>&1; then
    echo "[ERROR] Missing required Homebrew package: cmake" >&2
    echo >&2
    echo "[INFO] Install required dependency with:" >&2
    echo "       brew install cmake" >&2
    echo >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Optional HDF5 and OpenMP discovery
# ------------------------------------------------------------

CMAKE_PREFIX_PATH_ITEMS=()

if brew list hdf5 >/dev/null 2>&1; then
    HDF5_ROOT="$(brew --prefix hdf5)"
    export HDF5_ROOT
    CMAKE_PREFIX_PATH_ITEMS+=("${HDF5_ROOT}")
    echo "[INFO] HDF5 found: ${HDF5_ROOT}"
else
    echo "[WARN] Homebrew package hdf5 not found."
    echo "[WARN] The macos-arm64 preset uses COOLING_BUILD_HDF5=AUTO,"
    echo "[WARN] so the build will continue without HDF5 support."
    echo "[INFO] To enable HDF5 support, run:"
    echo "       brew install hdf5"
fi

if brew list libomp >/dev/null 2>&1; then
    OPENMP_ROOT="$(brew --prefix libomp)"
    export OPENMP_ROOT
    CMAKE_PREFIX_PATH_ITEMS+=("${OPENMP_ROOT}")
    echo "[INFO] OpenMP runtime found: ${OPENMP_ROOT}"
else
    echo "[WARN] Homebrew package libomp not found."
    echo "[WARN] OpenMP detection may fail, and the OpenMP target may not be built."
    echo "[INFO] To enable OpenMP support, run:"
    echo "       brew install libomp"
fi

# ------------------------------------------------------------
# Help CMake discover Homebrew packages
# ------------------------------------------------------------
#
# CMAKE_PREFIX_PATH is a standard CMake search path variable.
# Keep any existing value, but prepend project-relevant prefixes.
# ------------------------------------------------------------

if [[ "${#CMAKE_PREFIX_PATH_ITEMS[@]}" -gt 0 ]]; then
    _NEW_CMAKE_PREFIX_PATH="$(IFS=:; echo "${CMAKE_PREFIX_PATH_ITEMS[*]}")"

    if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
        export CMAKE_PREFIX_PATH="${_NEW_CMAKE_PREFIX_PATH}:${CMAKE_PREFIX_PATH}"
    else
        export CMAKE_PREFIX_PATH="${_NEW_CMAKE_PREFIX_PATH}"
    fi

    echo "[INFO] CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
else
    echo "[INFO] No optional Homebrew package prefixes added to CMAKE_PREFIX_PATH."
fi

# ------------------------------------------------------------
# Select macOS build preset
# ------------------------------------------------------------
#
# The CMake preset remains the source of truth for build options:
# - COOLING_BUILD_HDF5=AUTO
# - COOLING_BUILD_OPENMP=AUTO
# - COOLING_BUILD_CUDA=OFF
# - CMAKE_OSX_ARCHITECTURES=arm64
# ------------------------------------------------------------

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
# ------------------------------------------------------------

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
# Basic tool checks
# ------------------------------------------------------------

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake not found, even though Homebrew cmake was expected." >&2
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------
# Optional Python virtual environment
# ------------------------------------------------------------
#
# Only source the virtualenv from the repository root computed above.
# Do not use an externally supplied PROJECT_ROOT.
# ------------------------------------------------------------

if [[ -d "${PROJECT_ROOT}/cooling_venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/cooling_venv/bin/activate"
    echo "[INFO] Activated Python virtualenv: ${PROJECT_ROOT}/cooling_venv"
fi

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] PRESET=${PRESET}"
echo "[INFO] macOS environment ready."
