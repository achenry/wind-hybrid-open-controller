#!/bin/bash
#SBATCH --job-name=cpu_floris_case_studies.py
#SBATCH --time=00:20:00
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc
##SBATCH --partition=hbw

# salloc --account=ssc --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
# module load mamba
conda activate test_env
# module load PrgEnv-intel # NOTE: DONT NEED THIS WHEN MPI4PY IS INSTALLED WITH MAMBA, SAME GOES FOR LIBRARY LINKING LINE BELOW, ALSO DONT MARK JOB AS EXCLUSIVE
module load intel
module list

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

srun python test_mpi.py
