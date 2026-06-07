module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7 
python3 -m venv cooling_venv --system-site-packages
source cooling_venv/bin/activate
pip install numba
pip install h5py
pip install cupy-cuda12x


cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

