#!/bin/bash
#SBATCH --job-name=test_mpi.py
#SBATCH --time=00:20:00
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc
#SBATCH --partition=debug

# salloc --account=ssc --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
export PYTHONPATH=$(which python)

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

srun $PYTHONPATH test_mpi.py
