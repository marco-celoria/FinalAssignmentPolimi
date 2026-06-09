#!/usr/bin/env bash

RTOL="${RTOL:-1e-3}"
ATOL="${ATOL:-1e-3}"

VALIDATOR="tools/validate_particles_h5.py"

echo "Cpp vs Cuda"
python "${VALIDATOR}" \
  output/Particles_cpp.h5 \
  output/Particles_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs Numba"
python "${VALIDATOR}" \
  output/Particles_cpp.h5 \
  output/Particles_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cpp vs NumbaCuda"
python "${VALIDATOR}" \
  output/Particles_cpp.h5 \
  output/Particles_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cuda vs Numba"
python "${VALIDATOR}" \
  output/Particles_cuda.h5 \
  output/Particles_numba.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Cuda vs NumbaCuda"
python "${VALIDATOR}" \
  output/Particles_cuda.h5 \
  output/Particles_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

echo "Numba vs NumbaCuda"
python "${VALIDATOR}" \
  output/Particles_numba.h5 \
  output/Particles_numba_cuda.h5 \
  --rtol="${RTOL}" \
  --atol="${ATOL}"

