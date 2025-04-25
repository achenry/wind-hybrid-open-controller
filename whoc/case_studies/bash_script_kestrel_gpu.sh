#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=85G
#SBATCH --account=ssc
##SBATCH --partition=bigmem
##SBATCH --partition=nvme

# salloc --account=ssc --time=01:00:00 --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem-per-cpu=85G --account=ssc --partition=debug
module purge
module load mamba
module load cuda
mamba activate wind_forecasting_env

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/ssc/ahenry/conda/envs/wind_forecasting_env/lib
echo $SLURM_NTASKS

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($SLURM_NTASKS_PER_NODE-1)))
taskset python run_case_studies.py 15 --exclude_prediction --multiprocessor cf -rs --ram_limit 75 --wf_source scada \
       -st auto -ns 10 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ \
       -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
       -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
       -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml

