module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7 
python3 -m venv particles_venv --system-site-packages
source particles_venv/bin/activate
pip install numba
pip install h5py

cmake --preset generic-x86-nogpu
cmake --build --preset generic-x86-nogpu -j
cmake --install build/generic-x86-nogpu

cmake --preset generic-x86-nvidia
cmake --build --preset generic-x86-nvidia -j
cmake --install build/generic-x86-nvidia

source scripts/env.macos.sh
cmake --preset macos-arm64
cmake --build --preset macos-arm64 -j
cmake --install build/macos-arm64

source scripts/env.leonardo.sh
cmake --preset leonardo-a100
cmake --build --preset leonardo-a100 -j
cmake --install build/leonardo-a100
