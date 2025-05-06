#!/bin/bash
#SBATCH --job-name=cpu_floris_case_studies.py
#SBATCH --time=96:00:00
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc
##SBATCH --partition=hbw

# salloc --account=ssc --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
module load mamba
mamba activate wind_forecasting_env
module load PrgEnv-intel # NOTE: DONT NEED THIS WHEN MPI4PY IS INSTALLED WITH MAMBA, SAME GOES FOR LIBRARY LINKING LINE BELOW, ALSO DONT MARK JOB AS EXCLUSIVE
module list

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

srun python run_case_studies.py 22 --exclude_prediction --multiprocessor mpi -rs -rs --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60_svr.yaml

#srun python run_case_studies.py 19 20 --exclude_prediction --multiprocessor mpi -rs -rrs --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60_svr.yaml
