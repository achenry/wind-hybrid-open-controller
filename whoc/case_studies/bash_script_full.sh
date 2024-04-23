#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

module purge
#module load cray-python 
module load conda
#module load openmpi
conda activate whoc

#python run_case_studies.py nodebug nompi parallel 0 1 2 3 4 5 6 7
# srun python run_case_studies.py debug nompi 0 1 2 3 4 5 6 7
srun -n $SLURM_NTASKS python run_case_studies.py nodebug dask parallel 0 1 2 3 4 5 6 7
# srun python run_case_studies.py
