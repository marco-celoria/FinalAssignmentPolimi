#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=200GB
#SBATCH --job-name=run_validate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
##SBATCH --exclusive

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.2
module load gcc/12.2.0 
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7 
source particles_venv/bin/activate
echo "Cpp vs Cuda"
srun python ./tools/validate_particles_h5.py output/Particles_cpp.h5   output/Particles_cuda.h5  --rtol=1e-3 --atol=1e-3
echo "Cpp vs Numba"
srun python ./tools/validate_particles_h5.py output/Particles_cpp.h5   output/Particles_numba.h5 --rtol=1e-3 --atol=1e-3
echo "Cpp vs NumbaCuda"
srun python ./tools/validate_particles_h5.py output/Particles_cpp.h5   output/Particles_numba_cuda.h5 --rtol=1e-3 --atol=1e-3
echo "Numba vs NumbaCuda"
srun python ./tools/validate_particles_h5.py output/Particles_numba.h5 output/Particles_numba_cuda.h5 --rtol=1e-3 --atol=1e-3

