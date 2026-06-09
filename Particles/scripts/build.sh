#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Portable build script for CoolingSimulation
#
# Supports:
#   - Linux/HPC with OpenMP and optional CUDA
#   - macOS Apple Silicon with OpenMP-only
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# ------------------------------------------------------------
# User-configurable options
# ------------------------------------------------------------

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-}"

CLEAN="${CLEAN:-0}"
VERBOSE="${VERBOSE:-0}"
INSTALL="${INSTALL:-0}"

BUILD_OPENMP="${BUILD_OPENMP:-ON}"
BUILD_CUDA="${BUILD_CUDA:-AUTO}"
NATIVE_ARCH="${NATIVE_ARCH:-OFF}"
FAST_MATH_CUDA="${FAST_MATH_CUDA:-OFF}"
STRICT_CUDA="${STRICT_CUDA:-0}"

CUDA_ARCH="${CUDA_ARCH:-AUTO}"

INSTALL_PREFIX="${INSTALL_PREFIX:-${PROJECT_ROOT}/install}"

ENV_SCRIPT="${ENV_SCRIPT:-}"
CXX_COMPILER="${CXX_COMPILER:-}"
CUDA_COMPILER="${CUDA_COMPILER:-}"

HDF5_ROOT="${HDF5_ROOT:-}"
OPENMP_ROOT="${OPENMP_ROOT:-}"

CMAKE_PRESET="${CMAKE_PRESET:-}"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

detect_jobs() {
    if [[ -n "${BUILD_JOBS}" ]]; then
        echo "${BUILD_JOBS}"
        return
    fi

    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    elif [[ -f /proc/cpuinfo ]]; then
        grep -c ^processor /proc/cpuinfo
    else
        echo 4
    fi
}

normalize_on_off() {
    case "$1" in
        ON|On|on|1|TRUE|True|true|YES|Yes|yes)    echo "ON" ;;
        OFF|Off|off|0|FALSE|False|false|NO|No|no) echo "OFF" ;;
        *) echo "[ERROR] Invalid ON/OFF value: $1" >&2; exit 1 ;;
    esac
}

normalize_on_off_auto() {
    case "$1" in
        ON|On|on|1|TRUE|True|true|YES|Yes|yes)    echo "ON" ;;
        OFF|Off|off|0|FALSE|False|false|NO|No|no) echo "OFF" ;;
        AUTO|Auto|auto)                           echo "AUTO" ;;
        *) echo "[ERROR] Invalid ON/OFF/AUTO value: $1" >&2; exit 1 ;;
    esac
}

is_macos() {
    [[ "$(uname -s)" == "Darwin" ]]
}

BUILD_JOBS="$(detect_jobs)"
BUILD_OPENMP="$(normalize_on_off "${BUILD_OPENMP}")"
BUILD_CUDA="$(normalize_on_off_auto "${BUILD_CUDA}")"
NATIVE_ARCH="$(normalize_on_off "${NATIVE_ARCH}")"
FAST_MATH_CUDA="$(normalize_on_off "${FAST_MATH_CUDA}")"

# ------------------------------------------------------------
# Optional environment loading
# ------------------------------------------------------------

if [[ -n "${ENV_SCRIPT}" ]]; then
    if [[ -f "${ENV_SCRIPT}" ]]; then
        echo "[INFO] Loading environment from ${ENV_SCRIPT}"
        # shellcheck source=/dev/null
        source "${ENV_SCRIPT}"
    else
        echo "[ERROR] ENV_SCRIPT was set but file does not exist: ${ENV_SCRIPT}" >&2
        exit 1
    fi
fi

# Re-normalize after ENV_SCRIPT in case the env script overrides variables.
BUILD_OPENMP="$(normalize_on_off "${BUILD_OPENMP}")"
BUILD_CUDA="$(normalize_on_off_auto "${BUILD_CUDA}")"
NATIVE_ARCH="$(normalize_on_off "${NATIVE_ARCH}")"
FAST_MATH_CUDA="$(normalize_on_off "${FAST_MATH_CUDA}")"

# ------------------------------------------------------------
# macOS CUDA guard
# ------------------------------------------------------------

if is_macos && [[ "${BUILD_CUDA}" != "OFF" ]]; then
    echo "[WARN] macOS detected. CUDA C++ target is not supported on native Apple Silicon/macOS."
    echo "[WARN] Forcing BUILD_CUDA=OFF."
    BUILD_CUDA="OFF"
fi

# ------------------------------------------------------------
# Sanity checks and compiler discovery
# ------------------------------------------------------------

if [[ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]]; then
    echo "[ERROR] CMakeLists.txt not found in ${PROJECT_ROOT}" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake not found in PATH" >&2
    exit 1
fi

if [[ -n "${CUDA_COMPILER}" ]]; then
    export CUDACXX="${CUDA_COMPILER}"
fi

if [[ "${BUILD_CUDA}" != "OFF" ]]; then
    if [[ -z "${CUDACXX:-}" ]] && command -v nvcc >/dev/null 2>&1; then
        export CUDACXX="$(command -v nvcc)"
    fi
fi

if [[ "${BUILD_CUDA}" == "ON" && "${STRICT_CUDA}" == "1" ]]; then
    if [[ -z "${CUDACXX:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
        echo "[ERROR] BUILD_CUDA=ON but no CUDA compiler was found." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------
# Build directory
# ------------------------------------------------------------

REAL_BUILD_DIR="${BUILD_DIR}"

if [[ -n "${CMAKE_PRESET}" ]]; then
    REAL_BUILD_DIR="build/${CMAKE_PRESET}"
fi

# ------------------------------------------------------------
# Print configuration
# ------------------------------------------------------------

echo "============================================================"
echo "CoolingSimulation build"
echo "============================================================"
echo "Project root:        ${PROJECT_ROOT}"
echo "Build dir:           ${REAL_BUILD_DIR}"
echo "CMake preset:        ${CMAKE_PRESET:-none}"
echo "Build type:          ${BUILD_TYPE}"
echo "Build jobs:          ${BUILD_JOBS}"
echo "Build OpenMP:        ${BUILD_OPENMP}"
echo "Build CUDA:          ${BUILD_CUDA}"
echo "CUDA arch:           ${CUDA_ARCH}"
echo "Native arch:         ${NATIVE_ARCH}"
echo "CUDA fast math:      ${FAST_MATH_CUDA}"
echo "FP policy:           moderate (built-in)"
echo "Install:             ${INSTALL}"
echo "Install prefix:      ${INSTALL_PREFIX}"
echo "CXX compiler:        ${CXX_COMPILER:-default}"
echo "CUDA compiler:       ${CUDA_COMPILER:-${CUDACXX:-default}}"
echo "HDF5_ROOT:           ${HDF5_ROOT:-not set}"
echo "OPENMP_ROOT:         ${OPENMP_ROOT:-not set}"
echo "ENV_SCRIPT:          ${ENV_SCRIPT:-none}"
echo "============================================================"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

if [[ "${CLEAN}" == "1" ]]; then
    echo "[INFO] Removing build directory: ${REAL_BUILD_DIR}"
    rm -rf "${REAL_BUILD_DIR}"
fi

# ------------------------------------------------------------
# Configure and build
# ------------------------------------------------------------

if [[ -n "${CMAKE_PRESET}" ]]; then
    echo "[INFO] Configuring with preset: ${CMAKE_PRESET}"

    PRESET_ARGS=(
        --preset "${CMAKE_PRESET}"
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    )

    if [[ -n "${HDF5_ROOT}" ]]; then
        PRESET_ARGS+=(-DHDF5_ROOT="${HDF5_ROOT}")
    fi

    if [[ -n "${OPENMP_ROOT}" ]]; then
        PRESET_ARGS+=(-DOpenMP_ROOT="${OPENMP_ROOT}")
    fi

    if [[ -n "${CXX_COMPILER}" ]]; then
        PRESET_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX_COMPILER}")
    fi

    if [[ -n "${CUDA_COMPILER}" ]]; then
        PRESET_ARGS+=(-DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}")
    fi

    cmake "${PRESET_ARGS[@]}"

    echo "[INFO] Building with preset: ${CMAKE_PRESET}"
    BUILD_ARGS=(--build --preset "${CMAKE_PRESET}" --parallel "${BUILD_JOBS}")

    if [[ "${VERBOSE}" == "1" ]]; then
        BUILD_ARGS+=(--verbose)
    fi

    cmake "${BUILD_ARGS[@]}"
else
    echo "[INFO] Configuring manually"

    CMAKE_ARGS=(
        -S "${PROJECT_ROOT}"
        -B "${BUILD_DIR}"
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
        -DCOOLING_BUILD_OPENMP="${BUILD_OPENMP}"
        -DCOOLING_BUILD_CUDA="${BUILD_CUDA}"
        -DCOOLING_NATIVE_ARCH="${NATIVE_ARCH}"
        -DCOOLING_FAST_MATH_CUDA="${FAST_MATH_CUDA}"
        -DCOOLING_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        -DCOOLING_STRICT_CUDA="${STRICT_CUDA}"
    )

    if [[ -n "${HDF5_ROOT}" ]]; then
        CMAKE_ARGS+=(-DHDF5_ROOT="${HDF5_ROOT}")
    fi

    if [[ -n "${OPENMP_ROOT}" ]]; then
        CMAKE_ARGS+=(-DOpenMP_ROOT="${OPENMP_ROOT}")
    fi

    if [[ -n "${CXX_COMPILER}" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX_COMPILER}")
    fi

    if [[ -n "${CUDA_COMPILER}" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}")
    fi

    cmake "${CMAKE_ARGS[@]}"

    echo "[INFO] Building manually"
    BUILD_ARGS=(--build "${BUILD_DIR}" --parallel "${BUILD_JOBS}")

    if [[ "${VERBOSE}" == "1" ]]; then
        BUILD_ARGS+=(--verbose)
    fi

    cmake "${BUILD_ARGS[@]}"
fi

# ------------------------------------------------------------
# Optional install
# ------------------------------------------------------------

if [[ "${INSTALL}" == "1" ]]; then
    echo "[INFO] Installing to ${INSTALL_PREFIX}"
    cmake --install "${REAL_BUILD_DIR}"
fi

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

echo
echo "[INFO] Executables found under ${REAL_BUILD_DIR}:"

find "${REAL_BUILD_DIR}" \
    -maxdepth 4 \
    -type f \
    -perm -111 \
    \( \
        -name "cooling_omp" \
        -o -name "cooling_cuda" \
        -o -name "cooling_omp.exe" \
        -o -name "cooling_cuda.exe" \
    \) \
    -print || true

if [[ "${INSTALL}" == "1" ]]; then
    echo
    echo "[INFO] Installed executables:"

    find "${INSTALL_PREFIX}" \
        -maxdepth 4 \
        -type f \
        -perm -111 \
        -print || true
fi
