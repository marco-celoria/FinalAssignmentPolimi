#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=50GB
#SBATCH --job-name=run_numbacuda_cooling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
##SBATCH --exclusive

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
source scripts/env.sh
srun python ./src/python/particles_numba_cuda.py ./input/Particles.in ./output/Particles_numba_cuda.h5

