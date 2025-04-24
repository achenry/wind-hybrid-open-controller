#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc
##SBATCH --partition=bigmem
##SBATCH --partition=nvme
# salloc --account=ssc --time=02:00:00 --nodes=1 --ntasks-per-node=104
module purge
module load mamba
mamba activate wind_forecasting_env

module load PrgEnv-intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

<<<<<<< HEAD
# -sd $TMPDIR
#mpirun -np $SLURM_NTASKS python run_case_studies.py 18 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml
mpirun -np $SLURM_NTASKS python run_case_studies.py 15 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml
mpirun -np $SLURM_NTASKS python run_case_studies.py 16 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml
=======
# for PerfectForecast grid search
mpirun -np $SLURM_NTASKS python run_case_studies.py 18 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada \
        -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ \
        -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
        -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
        -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken.yaml

# for Deterministic Forecasters 
mpirun -np $SLURM_NTASKS python run_case_studies.py 15 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada \
       -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ \
       -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
       -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
       -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken.yaml

# for Probabilistic Forecasters grid search
mpirun -np $SLURM_NTASKS python run_case_studies.py 16 --exclude_prediction --multiprocessor mpi -rs --ram_limit 75 --wf_source scada \
       -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ \
       -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
       -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
       -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml
>>>>>>> fa49e4cd198447c96a49e4ffa9088142f1efdc52

#rm -rf /projects/ssc/ahenry/whoc/floris_case_studies/baseline_controllers_perfect_forecaster_awaken
#mv $TMPDIR/* /projects/ssc/ahenry/whoc/floris_case_studies/
