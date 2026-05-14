#!/bin/bash

#SBATCH --partition=cfdg.p                 # Storm GPU partition (7-day cap). cfdg.p's only
#SBATCH --nodes=1                          #   H100 node is cfdg002 (H100:4) — single node, 4 GPUs.
#SBATCH --ntasks-per-node=1                # cf (ProcessPoolExecutor) = ONE main process; it forks
#SBATCH --cpus-per-task=32                 #   the worker pool itself. NOT MPI multi-rank.
#SBATCH --gres=gpu:H100:4                  # all 4 H100 on the node
#SBATCH --mem-per-cpu=24000                # 32 x 24000 MB = 768 GB for the node (main + 4 workers
#SBATCH --exclude=cfdg001                  #   each load a DataModule ~60 GB; fits with headroom)
#SBATCH --time=2-00:00:00                  # generous: first run also generates the AWAKEN LUTs at init
#SBATCH --job-name=cs29_tactis_quantile_cp
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/cs29_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/cs29_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block

# =============================================================================
# Case study 29 — baseline_controllers_tactis_quantile_head_cp_awaken
# Controller validation of the quantile-head TACTiS-2 model with CP stddev
# calibration. 4 arms x 30 wind seeds:
#   arm 1: LookupBasedWakeSteering, uncertain=True,  cp_calibrate_stddev=False (raw)
#   arm 2: LookupBasedWakeSteering, uncertain=True,  cp_calibrate_stddev=True  (CP-calibrated)
#   arm 3: LookupBasedWakeSteering, uncertain=False  (deterministic LUT baseline)
#   arm 4: GreedyController                          (no-wake-steering floor)
#
# Uses --multiprocessor cf (ProcessPoolExecutor) NOT mpi: a single main process
# runs initialize_simulations() once (loads the DataModule + auto-generates the 2
# AWAKEN LUTs), then forks a 4-worker pool, one worker per H100 via the gpu_cycler.
# This sidesteps the srun<->MPICH PMI mismatch that made each MPI rank an
# independent rank-0 (4x the DataModule load -> OOM at init).
#
# Submit:  sbatch bash_script_storm_gpu.sh
#   (race the all_gpu.p partition too: sbatch --partition=all_gpu.p --time=24:00:00 bash_script_storm_gpu.sh)
# =============================================================================

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller"
WHOC_DIR="${BASE_DIR}/wind-hybrid-open-controller"
WF_DIR="${BASE_DIR}/wind-forecasting"
export WORK_DIR="${WHOC_DIR}/whoc/case_studies"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CASE_STUDY_OUTPUT_DIR="${OUTPUT_DIR}/floris_case_studies"

mkdir -p ${LOG_DIR}/slurm_logs ${CASE_STUDY_OUTPUT_DIR}
cd ${WORK_DIR} || exit 1

export PYTHONPATH=${BASE_DIR}/floris:${WHOC_DIR}:${WF_DIR}:${BASE_DIR}/pytorch-transformer-ts:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=8
export POLARS_MAX_THREADS=1                # avoid Polars thread contention across the worker pool

# --- Modules (Storm) ---
# No mpi4py module: cf does not use MPI, and the module's python3.11 mpi4py would
# anyway shadow wf_env_storm's python3.12 build (ABI mismatch).
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
eval "$(conda shell.bash hook)"
conda activate wf_env_storm

# --- GPU visibility + worker count for run_case_studies.py ---
# CUDA_VISIBLE_DEVICES feeds the gpu_cycler; SLURM_NTASKS_PER_NODE sets the
# ProcessPoolExecutor's max_workers (run_case_studies.py:199). With cf and
# --ntasks-per-node=1, SLURM sets that env to 1 — override it to the GPU count
# so the pool actually has one worker per GPU.
devices=$SLURM_JOB_GPUS
n_devices=$(( ${#devices} / 2 + 1 ))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( n_devices - 1 )))
export SLURM_NTASKS_PER_NODE=${n_devices}

echo "================================================================"
echo "CASE STUDY 29 — baseline_controllers_tactis_quantile_head_cp_awaken (cf, ${n_devices} workers)"
echo "================================================================"
echo "JOB ID:        ${SLURM_JOB_ID}"
echo "PARTITION:     ${SLURM_JOB_PARTITION}"
echo "NODE:          ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}  ->  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  workers=${SLURM_NTASKS_PER_NODE}"
echo "GPU TYPE:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)"
echo "branch: $(cd ${WHOC_DIR} && git rev-parse --abbrev-ref HEAD) @ $(cd ${WHOC_DIR} && git rev-parse --short HEAD)"
echo "================================================================"

# --- Config files ---
WCNF="${WHOC_DIR}/examples/hercules_input_001.yaml"                                          # has cp_calibrate_stddev registered under MLForecast
DCNF="${WF_DIR}/config/preprocessing/preprocessing_inputs_awaken_STORM.yaml"
MCNF="${WF_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml"

for f in "$WCNF" "$DCNF" "$MCNF"; do
    [ -f "$f" ] || { echo "ERROR: config not found: $f" >&2; exit 1; }
done

date +"%Y-%m-%d %H:%M:%S"
echo "=== STARTING run_case_studies.py 29 (--multiprocessor cf) ==="

# cf = single main process forking a worker pool — run python directly, no srun multi-rank.
python run_case_studies.py 29 \
    --multiprocessor cf \
    -rs \
    --ram_limit 32 \
    --wf_source scada \
    -st auto \
    -ns 30 \
    -sd ${CASE_STUDY_OUTPUT_DIR} \
    -wcnf ${WCNF} \
    -dcnf ${DCNF} \
    -mcnf ${MCNF}

EXIT_CODE=$?
echo "=== CASE STUDY FINISHED (exit ${EXIT_CODE}) ==="
date +"%Y-%m-%d %H:%M:%S"
echo "Results: ${CASE_STUDY_OUTPUT_DIR}/baseline_controllers_tactis_quantile_head_cp_awaken/"
exit ${EXIT_CODE}
