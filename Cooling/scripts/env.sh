#!/usr/bin/env bash

module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7

if [[ -d cooling_venv ]]; then
    source cooling_venv/bin/activate
fi
