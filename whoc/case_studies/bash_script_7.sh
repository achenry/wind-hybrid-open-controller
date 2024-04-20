#!/bin/bash
#SBATCH --job-name=debug_floris_case_studies.py
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --account=ssc

module purge
module load conda
module load openmpi
conda activate whoc
echo $SLURM_NTASKS
srun -n $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py debug mpi 7
# srun python run_case_studies.py
