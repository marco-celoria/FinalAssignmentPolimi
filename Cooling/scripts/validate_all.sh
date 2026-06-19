#!/usr/bin/env bash

RTOL="${RTOL:-1e-3}"
ATOL="${ATOL:-1e-3}"

VALIDATOR="tools/validate_cooling_h5.py"

echo
echo "Cpp vs Omp"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_omp.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cpp vs Cuda"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cpp vs Python"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cpp vs Numba"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cpp vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_cpp.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Omp vs Cuda"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Omp vs Python"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Omp vs Numba"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Omp vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_omp.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cuda vs Python"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_python.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cuda vs Numba"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Cuda vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_cuda.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Python vs Numba"
python "${VALIDATOR}" \
  output/Cooling_python.h5 \
  output/Cooling_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Python vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_python.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo
echo "Numba vs NumbaCuda"
python "${VALIDATOR}" \
  output/Cooling_numba.h5 \
  output/Cooling_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

