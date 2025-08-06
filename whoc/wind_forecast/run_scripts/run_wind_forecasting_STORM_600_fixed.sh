#!/bin/bash

#SBATCH --partition=cfdg.p          # Partition for H100/A100 GPUs (adjust if needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Requesting 1 task for 1 GPU
#SBATCH --cpus-per-task=4          # CPUs per task (adjust based on inference needs)
#SBATCH --mem-per-cpu=8192          # Memory per CPU (Total Mem = 1 * 4 * 8192 = 32GB)
#SBATCH --gres=gpu:H100:1           # Request 1 H100 GPU (Matches ntasks-per-node)
#SBATCH --time=7-00:00              # Time limit (2 days for comprehensive validation)
#SBATCH --job-name=flasc_valid_600s_tactis_improved
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/flasc_valid_600s_improved_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/flasc_valid_600s_improved_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPU to task

# --- Base Directories ---
export BASE_DIR="/user/taed7566/Forecasting"
export WHOC_DIR="${BASE_DIR}/wind-hybrid-open-controller"
export WF_DIR="${BASE_DIR}/wind-forecasting"
export LOG_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs"
export WHOC_SCRIPT_DIR="${WHOC_DIR}/whoc/wind_forecast"

# --- Validation Settings for 600s Horizon ---
export MODELS="tactis"
export MODEL_CONFIG_PATH_ARG="${WF_DIR}/config/training/storm_configs/training_inputs_juan_flasc_tune_storm_local_db_600.yaml"
export DATA_CONFIG_PATH_ARG="${WF_DIR}/config/preprocessing/preprocessing_inputs_flasc_STORM.yaml"
export CHECKPOINT_ARG="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/tune_tactis_flasc_3_local_tactis/20250730_202152_0_0/epoch=56-step=554496-val_loss=-38.68.ckpt" # Improved 600s checkpoint with random permutations + decoder_num_bins=200
export MAX_STEPS_ARG=3600     # 600s prediction + 600s context + buffer
export PREDICTION_TYPE_ARG="sample"

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/inference_results/flasc_validation_600s_improved_${SLURM_JOB_ID}

# --- Change to Working Directory ---
cd ${WHOC_SCRIPT_DIR} || { echo "ERROR: Failed to change directory to ${WHOC_SCRIPT_DIR}"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WHOC_DIR}:${WF_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# --- Print Job Info ---
echo "--- SLURM JOB INFO (600s Validation Improved) ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM TASKS: ${SLURM_NTASKS}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WHOC_DIR: ${WHOC_DIR}"
echo "WF_DIR: ${WF_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "WHOC_SCRIPT_DIR: ${WHOC_SCRIPT_DIR}"
echo "MODELS: ${MODELS}"
echo "MODEL_CONFIG_PATH_ARG: ${MODEL_CONFIG_PATH_ARG}"
echo "DATA_CONFIG_PATH_ARG: ${DATA_CONFIG_PATH_ARG}"
echo "CHECKPOINT_ARG: ${CHECKPOINT_ARG}"
echo "PREDICTION_TYPE: ${PREDICTION_TYPE_ARG}"
echo "MAX_STEPS: ${MAX_STEPS_ARG}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment 'wf_env_storm'" >&2
  exit 1
fi
echo "Conda environment 'wf_env_storm' activated."
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: ${CAPTURED_LD_LIBRARY_PATH}"
echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "------------------------"

# --- Run Inference Script ---
echo "Starting WindForecast validation for 600s horizon (improved - random permutations + decoder_num_bins=200)..."
date +"%Y-%m-%d %H:%M:%S"

# Execute the Python script with comprehensive validation options
export CUDA_LAUNCH_BLOCKING=1 # For more detailed CUDA error messages
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU ${CUDA_VISIBLE_DEVICES}"

python run_forecaster_validation.py \
    --model ${MODELS} \
    --model_config "${MODEL_CONFIG_PATH_ARG}" \
    --data_config "${DATA_CONFIG_PATH_ARG}" \
    --simulation_timestep 60 \
    --save_dir "${LOG_DIR}/inference_results/flasc_validation_600s_improved_${SLURM_JOB_ID}" \
    --checkpoint "${CHECKPOINT_ARG}" \
    --prediction_type "${PREDICTION_TYPE_ARG}" \
    --use_tuned_params \
    --run_validation \
    --rerun_validation \
    --run_processing \
    --plot \
    --max_steps ${MAX_STEPS_ARG} \
    --ram_limit 85

EXIT_CODE=$?

date +"%Y-%m-%d %H:%M:%S"
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "600s improved validation completed successfully."
else
  echo "ERROR: 600s improved validation failed with exit code ${EXIT_CODE}." >&2
fi
echo "---------------------------------------"

exit ${EXIT_CODE}

# Usage: sbatch /user/taed7566/Forecasting/wind-hybrid-open-controller/whoc/wind_forecast/run_scripts/run_wind_forecasting_STORM_600_fixed.sh