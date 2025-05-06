#!/bin/bash
#SBATCH --job-name=baseline_wf_cpu
#SBATCH --account=ssc
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
##SBATCH --partition=nvme
#SBATCH --ntasks-per-node=104

# salloc --partition=debug --nodes=1 --ntasks-per-node=104 --time=01:00:00 --mem=0 --account=ssc

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list

export MODELS="kf svr" # "kf persistence sf svr"
export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60_svr.yaml $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred300_svr.yaml"
export DATA_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"

echo "MODELS=${MODELS}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
#echo "TMPDIR=${TMPDIR}"

# prepare training data first
date +"%Y-%m-%d %H:%M:%S"
module purge
module load mamba
# module load PrgEnv-intel
mamba activate wind_forecasting_env

#mpirun -np $SLURM_NTASKS 
python WindForecast.py --ram_limit 65 --model ${MODELS} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --simulation_timestep 1 --save_dir /projects/ssc/ahenry/wind_forecasting/logging --prediction_type distribution --use_tuned_params --use_trained_models --multiprocessor cf --max_splits 10 # --rerun_validation

#python WindForecast.py --model ${MODELS} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --simulation_timestep 1 \
#                                                --save_dir /projects/ssc/ahenry/wind_forecasting/logging  --max_splits 10 --prediction_type distribution \
#                                                --use_tuned_params --use_trained_models --rerun_validation

