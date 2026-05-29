#!/bin/bash
#SBATCH --job-name=cpu_floris_case_studies.py
#SBATCH --output=%j_%x.out
#SBATCH --time=72:00:00
##SBATCH --time=48:00:00
#SBATCH --mem=0
##SBATCH --exclusive
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --partition=medmem
#SBATCH --ntasks-per-node=104
#SBATCH --account=awaken

# salloc --account=awaken --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
export PYTHONPATH=$(which python)
#module load conda
#eval "$(conda shell.bash hook)"
module list

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/awaken/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS


srun $PYTHONPATH run_case_studies.py 19 20 22 --multiprocessor mpi -rs --ram_limit 65 --wf_source scada -st auto -ns 30 -sd /projects/awaken/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred_smoothed_svr.yaml -rrs


#srun python run_case_studies.py 21 --multiprocessor mpi -rs --ram_limit 65 --wf_source scada -st auto -ns auto -sd /projects/awaken/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_flasc.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_flasc.yaml -rrs

#srun $PYTHONPATH run_case_studies.py 28 --include_controller_signals --multiprocessor mpi -rs --ram_limit 65 --wf_source scada -st auto -ns 1 -sd /projects/awaken/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml #-rrs


#srun $PYTHONPATH run_case_studies.py 19 20 --multiprocessor mpi -rs -ps -rps -ras --ram_limit 35 --wf_source scada -st auto -ns 30 -sd /projects/awaken/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred_smoothed_svr.yaml #-rrs

#srun python run_case_studies.py 27 --multiprocessor mpi -rs -rrs --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/awaken/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml
