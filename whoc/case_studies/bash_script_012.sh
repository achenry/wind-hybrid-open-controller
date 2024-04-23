#!/bin/bash
#SBATCH --job-name=012_floris_case_studies.py
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

module purge
module load cray-python
module load conda
module load openmpi
conda activate whoc

srun -n $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py nodebug mpi parallel 0 1 2
