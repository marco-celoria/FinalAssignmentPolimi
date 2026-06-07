#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Build script for CoolingSimulation
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# ------------------------------------------------------------
# User-configurable options
# ------------------------------------------------------------

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

CLEAN="${CLEAN:-0}"
VERBOSE="${VERBOSE:-0}"
INSTALL="${INSTALL:-0}"

BUILD_OPENMP="${BUILD_OPENMP:-ON}"
BUILD_CUDA="${BUILD_CUDA:-ON}"
NATIVE_ARCH="${NATIVE_ARCH:-ON}"
FAST_MATH_CUDA="${FAST_MATH_CUDA:-OFF}"

CUDA_ARCH="${CUDA_ARCH:-80}"

INSTALL_PREFIX="${INSTALL_PREFIX:-${PROJECT_ROOT}/install}"

# ------------------------------------------------------------
# Print configuration
# ------------------------------------------------------------

echo "============================================================"
echo "CoolingSimulation build"
echo "============================================================"
echo "Project root:       ${PROJECT_ROOT}"
echo "Build dir:          ${BUILD_DIR}"
echo "Build type:         ${BUILD_TYPE}"
echo "Build jobs:         ${BUILD_JOBS}"
echo "Clean:              ${CLEAN}"
echo "Verbose:            ${VERBOSE}"
echo "Install:            ${INSTALL}"
echo
echo "OpenMP target:      ${BUILD_OPENMP}"
echo "CUDA target:        ${BUILD_CUDA}"
echo "Native arch:        ${NATIVE_ARCH}"
echo "CUDA fast math:     ${FAST_MATH_CUDA}"
echo "CUDA arch:          ${CUDA_ARCH}"
echo "Install prefix:     ${INSTALL_PREFIX}"
echo "============================================================"

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------

if [[ -f "${PROJECT_ROOT}/scripts/env.sh" ]]; then
    echo "[INFO] Loading environment from scripts/env.sh"
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/scripts/env.sh"
else
    echo "[WARN] scripts/env.sh not found; assuming modules are already loaded"
fi

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

if [[ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]]; then
    echo "[ERROR] CMakeLists.txt not found in ${PROJECT_ROOT}" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake not found in PATH" >&2
    exit 1
fi

if [[ "${BUILD_CUDA}" == "ON" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[WARN] nvcc not found in PATH. CMake may disable CUDA target."
    fi
fi

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

if [[ "${CLEAN}" == "1" ]]; then
    echo "[INFO] Removing build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# ------------------------------------------------------------
# Configure
# ------------------------------------------------------------

echo "[INFO] Configuring"

cmake -S "${PROJECT_ROOT}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DPARTICLES_BUILD_OPENMP="${BUILD_OPENMP}" \
      -DPARTICLES_BUILD_CUDA="${BUILD_CUDA}" \
      -DPARTICLES_NATIVE_ARCH="${NATIVE_ARCH}" \
      -DPARTICLES_FAST_MATH_CUDA="${FAST_MATH_CUDA}" \
      -DPARTICLES_CUDA_ARCHITECTURES="${CUDA_ARCH}"

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

echo "[INFO] Building"

if [[ "${VERBOSE}" == "1" ]]; then
    cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" --verbose
else
    cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
fi

# ------------------------------------------------------------
# Optional install
# ------------------------------------------------------------

if [[ "${INSTALL}" == "1" ]]; then
    echo "[INFO] Installing to ${INSTALL_PREFIX}"
    cmake --install "${BUILD_DIR}"
fi

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

echo "============================================================"
echo "Build completed successfully."
echo "============================================================"

echo "[INFO] Executables found under ${BUILD_DIR}:"

find "${BUILD_DIR}" \
    -maxdepth 3 \
    -type f \
    -perm -111 \
    \( -name "cooling_omp" -o -name "cooling_cuda" \) \
    -print || true

if [[ "${INSTALL}" == "1" ]]; then
    echo
    echo "[INFO] Installed executables:"
    find "${INSTALL_PREFIX}" \
        -maxdepth 3 \
        -type f \
        -perm -111 \
        -print || true
fi

echo
echo "Useful commands:"
echo
echo "  Normal build:"
echo "    scripts/build.sh"
echo
echo "  Clean rebuild:"
echo "    CLEAN=1 scripts/build.sh"
echo
echo "  Debug build:"
echo "    BUILD_TYPE=Debug CLEAN=1 scripts/build.sh"
echo
echo "  Build only OpenMP:"
echo "    BUILD_CUDA=OFF scripts/build.sh"
echo
echo "  Build only CUDA:"
echo "    BUILD_OPENMP=OFF scripts/build.sh"
echo
echo "  Build for another CUDA architecture:"
echo "    CUDA_ARCH=90 CLEAN=1 scripts/build.sh"
echo
echo "  Build and install:"
echo "    INSTALL=1 scripts/build.sh"
echo
echo "Run examples:"
echo "  ./build/cooling_omp  input/Cooling.inp output/Cooling_cpp.h5"
echo "  ./build/cooling_cuda input/Cooling.inp output/Cooling_cuda.h5"
echo "============================================================"
