#!/bin/bash
#SBATCH --job-name=baseline_wf
#SBATCH --account=ssc
#SBATCH --output=baseline_wf_%j.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=104

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
echo "NTUNERS=${NTUNERS}"
echo "NTASKS_PER_TUNER=${NTASKS_PER_TUNER}"

echo "=== ENVIRONMENT ==="
module list

export MODELS="kf persistence sf"
export MODEL_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml"
export DATA_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml"

echo "MODEL=${MODEL}"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "DATA_CONFIG=${DATA_CONFIG}"
echo "TMPDIR=${TMPDIR}"

# prepare training data first
date +"%Y-%m-%d %H:%M:%S"
module purge
module load mamba
mamba activate wind_forecasting

python WindForecast.py --model ${MODELS} --model_config ${MODEL_CONFIG} --data_config ${DATA_CONFIG} \\
                       --simulation_timestep 1 --prediction_interval 60 300 \\
                       --multiprocessor cf --max_splits 10 --prediction_type distribution \\
                       --use_tuned_params --use_trained_models