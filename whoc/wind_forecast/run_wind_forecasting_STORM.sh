#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs (adjust if needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Requesting 1 task for 1 GPU
#SBATCH --cpus-per-task=16          # CPUs per task (adjust based on inference needs)
#SBATCH --mem-per-cpu=8192          # Memory per CPU (Total Mem = 1 * 16 * 8192 = 128GB)
#SBATCH --gres=gpu:1           # Request 1 H100 GPU (Matches ntasks-per-node)
#SBATCH --time=1-00:00              # Time limit (e.g., 1 hour for inference)
#SBATCH --job-name=whoc_infer_storm
#SBATCH --output=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/whoc_infer_%j.out
#SBATCH --error=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/whoc_infer_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPU to task

# --- Base Directories ---
export BASE_DIR="/user/taed7566/Forecasting"
export WHOC_DIR="${BASE_DIR}/wind-hybrid-open-controller"
export WF_DIR="${BASE_DIR}/wind-forecasting"
export LOG_DIR="${WF_DIR}/logs"
export WHOC_SCRIPT_DIR="${WHOC_DIR}/whoc/wind_forecast"

# --- Input Arguments ---
# Example Usage: sbatch run_wind_forecasting_STORM.sh tactis \
#                       /user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_flasc.yaml \
#                       /user/taed7566/Forecasting/wind-forecasting/config/preprocessing/preprocessing_inputs_flasc.yaml
export MODELS=${1:-"tactis"}
export MODEL_CONFIG_PATH_ARG=${2:-"${WF_DIR}/config/training/training_inputs_juan_flasc_test_storm.yaml"}
export DATA_CONFIG_PATH_ARG=${3:-"${WF_DIR}/config/preprocessing/preprocessing_inputs_flasc_STORM.yaml"}

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/inference_results/${SLURM_JOB_ID}

# --- Change to Working Directory ---
cd ${WHOC_SCRIPT_DIR} || { echo "ERROR: Failed to change directory to ${WHOC_SCRIPT_DIR}"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WHOC_DIR}:${WF_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# --- Print Job Info ---
echo "--- SLURM JOB INFO ---"
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
echo "PYTHONPATH: ${PYTHONPATH}"
echo "------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
# module load mpi4py/3.1.4-gompi-2023a
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
echo "Starting WindForecast script..."
date +"%Y-%m-%d %H:%M:%S"

# Ensure config paths are absolute
resolve_path() {
  local path_to_resolve=$1
  if [[ "$path_to_resolve" = /* ]]; then
    echo "$path_to_resolve"
  else
    echo "${BASE_DIR}/${path_to_resolve}"
  fi
}
MODEL_CONFIG_PATH_ABS=$(resolve_path "${MODEL_CONFIG_PATH_ARG}")
DATA_CONFIG_PATH_ABS=$(resolve_path "${DATA_CONFIG_PATH_ARG}")

echo "Resolved Model Config Path: ${MODEL_CONFIG_PATH_ABS}"
echo "Resolved Data Config Path: ${DATA_CONFIG_PATH_ABS}"

# Execute the Python script (assuming WindForecast.py is in the current dir: WHOC_SCRIPT_DIR)
# Using CUDA_VISIBLE_DEVICES=0 explicitly, although Slurm binding should handle it
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU ${CUDA_VISIBLE_DEVICES}"

python WindForecast.py \
    --model ${MODELS} \
    --model_config "${MODEL_CONFIG_PATH_ABS}" \
    --data_config "${DATA_CONFIG_PATH_ABS}" \
    --simulation_timestep 1 \
    --save_dir "${LOG_DIR}/inference_results/${SLURM_JOB_ID}" \
    --checkpoint best \
    --prediction_type distribution \
    --use_tuned_params \
    --use_trained_models \
    --rerun_validation

EXIT_CODE=$?

date +"%Y-%m-%d %H:%M:%S"
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "Inference script completed successfully."
else
  echo "ERROR: Inference script failed with exit code ${EXIT_CODE}." >&2
fi
echo "---------------------------------------"

exit ${EXIT_CODE}

# sbatch wind-hybrid-open-controller/whoc/wind_forecast/run_wind_forecasting_STORM.sh