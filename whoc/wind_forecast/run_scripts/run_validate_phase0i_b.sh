#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=06:00:00              # 6h budget for first probabilistic-validation pass with max_splits=1
#SBATCH --job-name=tactis_validate_phase0i_b
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/whoc_validate_phase0i_b_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/whoc_validate_phase0i_b_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# Phase 0i-B validation — CRPS / PICP / PINAW / CWC vs Phase 0h baseline
# =============================================================================
# Probabilistic validation of the unsmoothed-data Phase 0i-B model (ep74)
# using the same harness configuration as Phase 0h (sim_ts=15s to preserve
# samples, prediction_type=sample, max_splits=1).
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --partition=all_gpu.p --export=ALL,LOCAL_PG_PASSWORD run_validate_phase0i_b.sh
# =============================================================================

export BASE_DIR="/user/taed7566/Forecasting"
export WHOC_DIR="${BASE_DIR}/wind-hybrid-open-controller"
export WF_DIR="${BASE_DIR}/wind-forecasting"
export LOG_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs"
export WHOC_SCRIPT_DIR="${WHOC_DIR}/whoc/wind_forecast"

# === Validation inputs ===
export MODEL="tactis"
# Phase 0i-B model config: unsmoothed data + a_max=3
export MODEL_CONFIG="${WF_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_b.yaml"
# Phase 0i-B checkpoint: ep74 (last full save before TIMEOUT, 44 epochs into Stage 2)
export CHECKPOINT="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/train_phase0i_b_unsmoothed_amax3_tactis/20260505_160440_0_0/manual_save_epoch74.ckpt"
# Use awaken preprocessing config (data_path is overridden inside model_config)
export DATA_CONFIG="${WF_DIR}/config/preprocessing/preprocessing_inputs_awaken_STORM.yaml"
export PREDICTION_TYPE="sample"
export SIMULATION_TIMESTEP=15        # match data_module freq=15s so predict_sample preserves all 200 samples

mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/inference_results/${SLURM_JOB_ID}

cd ${WHOC_SCRIPT_DIR} || { echo "ERROR: cd to ${WHOC_SCRIPT_DIR} failed"; exit 1; }

export PYTHONPATH=${WHOC_DIR}:${WF_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=============================================="
echo "TACTIS PHASE 0i-B VALIDATION (CRPS / PICP / PINAW / CWC)"
echo "=============================================="
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)"
echo "----------------------------------------------"
echo "CHECKPOINT:  ${CHECKPOINT}"
echo "MODEL CFG:   ${MODEL_CONFIG} (unsmoothed)"
echo "DATA CFG:    ${DATA_CONFIG}"
echo "SIM_TS:      ${SIMULATION_TIMESTEP}s (matches data_module freq)"
echo "PRED TYPE:   ${PREDICTION_TYPE}"
echo "OUTPUT DIR:  ${LOG_DIR}/inference_results/${SLURM_JOB_ID}"
echo "=============================================="

module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
eval "$(conda shell.bash hook)"
conda activate wf_env_storm

if [ -z "${LOCAL_PG_PASSWORD:-}" ]; then
    echo "WARNING: LOCAL_PG_PASSWORD not set — Optuna lookups will fail (acceptable for inference-only)"
else
    export PGPASSWORD="${LOCAL_PG_PASSWORD}"
fi

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

date +"%Y-%m-%d %H:%M:%S"
echo "=== STARTING run_forecaster_validation.py ==="

python run_forecaster_validation.py \
    --model "${MODEL}" \
    --model_config "${MODEL_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    --simulation_timestep ${SIMULATION_TIMESTEP} \
    --save_dir "${LOG_DIR}/inference_results/${SLURM_JOB_ID}" \
    --checkpoint "${CHECKPOINT}" \
    --prediction_type "${PREDICTION_TYPE}" \
    --max_splits 1 \
    --rerun_validation \
    --plot

EXIT_CODE=$?

date +"%Y-%m-%d %H:%M:%S"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "=== VALIDATION COMPLETED SUCCESSFULLY ==="
else
    echo "=== VALIDATION FAILED (exit ${EXIT_CODE}) ==="
fi

echo "Results location: ${LOG_DIR}/inference_results/${SLURM_JOB_ID}"
exit ${EXIT_CODE}
