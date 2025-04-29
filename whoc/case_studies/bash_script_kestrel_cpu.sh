#!/bin/bash
#SBATCH --job-name=cpu_floris_case_studies.py
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc
##SBATCH --partition=bigmem
##SBATCH --partition=nvme
# salloc --account=ssc --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
module load mamba
mamba activate wind_forecasting_env

module load PrgEnv-intel

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

#mpirun -np $SLURM_NTASKS python run_case_studies.py 16 17 --exclude_prediction --multiprocessor mpi -rs -rrs --ram_limit 75 --wf_source scada \
python run_case_studies.py 17 --exclude_prediction --multiprocessor cf -rs -rrs --ram_limit 75 --wf_source scada \
       -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ --generate_lut \
       -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
       -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
       -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60_svr.yaml




