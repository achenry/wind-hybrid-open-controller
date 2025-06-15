#!/bin/bash

#SBATCH --partition=cfds.p                # CPU partition for Storm HPC
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=128              # CPUs per node on Storm (adjust based on cfds.p capacity)
#SBATCH --cpus-per-task=1                 # CPUs per MPI task
#SBATCH --mem-per-cpu=4096                # Memory per CPU in MB
#SBATCH --time=0-01:00                    # Short time limit for testing
#SBATCH --job-name=flasc_test             # Job name for case study 21
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/flasc_test_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/flasc_test_%j.err
#SBATCH --hint=nomultithread              # Disable hyperthreading
#SBATCH --distribution=block:block        # Improve CPU affinity
#SBATCH --no-requeue                      # IMPORTANT: Disable automatic requeue

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller"
WHOC_DIR="${BASE_DIR}/wind-hybrid-open-controller"
WF_DIR="${BASE_DIR}/wind-forecasting"
export WORK_DIR="${WHOC_DIR}/whoc/case_studies"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CASE_STUDY_OUTPUT_DIR="${OUTPUT_DIR}/floris_case_studies"

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${CASE_STUDY_OUTPUT_DIR}

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Shared Environment Variables ---
export PYTHONPATH=${BASE_DIR}/floris:${WHOC_DIR}:${WF_DIR}:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=128

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load git
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."

# Test 1: Simple MPI test
echo "=== Test 1: Simple MPI Hello World ==="
srun --mpi=pmi2 python -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'Hello from rank {rank} of {size}', flush=True)
"
echo "Test 1 exit code: $?"

# Test 2: Try running the case study WITHOUT MPI
echo "=== Test 2: Running case study WITHOUT MPI (sequential) ==="
WCNF="${WHOC_DIR}/examples/hercules_input_001.yaml"
DCNF="${WF_DIR}/config/preprocessing/preprocessing_inputs_flasc_STORM.yaml"
MCNF="${WF_DIR}/config/training/training_inputs_juan_flasc_tune_storm.yaml"

python run_case_studies.py 21 \
    --exclude_prediction \
    -rs \
    -ps \
    --ram_limit 32 \
    --wf_source scada \
    -st 60 \
    -ns auto \
    -sd ${CASE_STUDY_OUTPUT_DIR} \
    -wcnf ${WCNF} \
    -dcnf ${DCNF} \
    -mcnf ${MCNF} \
    -rrs \
    --verbose

echo "Test 2 exit code: $?"

echo "=== All tests completed ==="