#!/usr/bin/env bash

RTOL="${RTOL:-1e-3}"
ATOL="${ATOL:-1e-3}"

VALIDATOR="tools/validate_cooling_h5.py"

echo "Cpp vs Omp"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_cpp.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs Cuda"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs Python"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs Numba"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Omp vs Cuda"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Omp vs Python"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Omp vs Numba"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Omp vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cuda vs Python"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cuda vs Numba"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cuda vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Python vs Numba"
python "${VALIDATOR}" \
  output/Cooling_python.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Python vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_python.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Numba vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_numba.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

