#!/bin/bash
#SBATCH --job-name=baseline_wf_gpu
#SBATCH --account=ssc
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=85G

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list

export MODELS="informer autoformer spacetimeformer tactis"
export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred300.yaml"
export DATA_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"

echo "MODELS=${MODELS}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
#echo "TMPDIR=${TMPDIR}"

# prepare training data first
date +"%Y-%m-%d %H:%M:%S"
module purge
module load mamba
module load PrgEnv-intel
mamba activate wind_forecasting_env

python WindForecast.py --model ${MODELS} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --simulation_timestep 1 --checkpoint latest \
	               --multiprocessor cf --max_splits 10 --prediction_type distribution --use_tuned_params --use_trained_models --rerun_validation
