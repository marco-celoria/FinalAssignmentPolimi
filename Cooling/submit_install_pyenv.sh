#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=job_install_cooling_venv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7
python3 -m venv cooling_venv --system-site-packages
source cooling_venv/bin/activate
pip install -r requirements.txt
deactivate

