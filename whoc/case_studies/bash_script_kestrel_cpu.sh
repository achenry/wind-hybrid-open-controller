#!/bin/bash
#SBATCH --job-name=cpu_floris_case_studies.py
##SBATCH --time=48:00:00
#SBATCH --time=24:00:00
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --nodes=2
##SBATCH --partition=debug
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

# salloc --account=ssc --time=01:00:00 --partition=debug --nodes=1 --ntasks-per-node=104
module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
export PYTHONPATH=$(which python)
#module load conda
#eval "$(conda shell.bash hook)"
module list

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

#srun python run_case_studies.py 22 --exclude_prediction --multiprocessor mpi -rs --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml -rrs

#srun python run_case_studies.py 21 --exclude_prediction --multiprocessor mpi -rs --ram_limit 65 --wf_source scada -st auto -ns auto -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_flasc.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_flasc.yaml -rrs

#python run_case_studies.py 24 --generate_lut --exclude_prediction --multiprocessor cf -rs --ram_limit 65 --wf_source scada -st auto -ns auto -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_flasc.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60_svr.yaml -rrs

srun $PYTHONPATH run_case_studies.py 19 20 --exclude_prediction --multiprocessor mpi -rs -ps -rps -ras --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml #-rrs

#srun python run_case_studies.py 26 --exclude_prediction --multiprocessor mpi -rs -rrs --ram_limit 65 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml --stoptime 10400
