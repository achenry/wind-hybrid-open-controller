#!/bin/bash

#SBATCH --partition=cfds.p                # CPU partition for Storm HPC
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=128              # CPUs per node on Storm (adjust based on cfds.p capacity)
#SBATCH --cpus-per-task=1                 # CPUs per MPI task
#SBATCH --mem-per-cpu=4096                # Memory per CPU in MB
#SBATCH --time=5-00:00                   # Time limit
#SBATCH --job-name=flasc_case_study_21_debug    # Job name for case study 21
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/flasc_case_study_21_debug_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-hybrid-open-controller/logs/slurm_logs/flasc_case_study_21_debug_%j.err
#SBATCH --hint=nomultithread              # Disable hyperthreading
#SBATCH --distribution=block:block        # Improve CPU affinity
#SBATCH --no-requeue                      # IMPORTANT: Disable automatic requeue to prevent looping

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
# Add local FLORIS to PYTHONPATH to use development version instead of conda version
export PYTHONPATH=${BASE_DIR}/floris:${WHOC_DIR}:${WF_DIR}:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=128

# Enable MPI debugging
export HYDRA_DEBUG=1
export UCX_LOG_LEVEL=debug
export OMPI_MCA_btl_base_verbose=30

# --- Print Job Info ---
echo "--- SLURM JOB INFO (FLASC Case Study) ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CASE_STUDY_OUTPUT_DIR: ${CASE_STUDY_OUTPUT_DIR}"
echo "------------------------"

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

# Don't load system OpenMPI - use the one from conda environment
echo "Using conda environment's MPI implementation."

# Check MPI configuration
echo "MPI Configuration:"
which mpirun
mpirun --version

# Check if mpi4py is properly installed and compatible
echo "Testing mpi4py compatibility:"
python -c "
try:
    from mpi4py import MPI
    print(f'mpi4py successfully imported')
    print(f'MPI version: {MPI.Get_version()}')
    print(f'MPI library version: {MPI.Get_library_version()}')
    comm = MPI.COMM_WORLD
    print(f'MPI rank: {comm.Get_rank()}, size: {comm.Get_size()}')
except Exception as e:
    print(f'mpi4py import failed: {e}')
    import sys
    sys.exit(1)
"

echo "Total MPI tasks available: $SLURM_NTASKS"

# Test MPI with a simple parallel hello world
echo "Testing MPI parallel execution:"
srun --mpi=pmi2 python -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'Hello from rank {rank} of {size}', flush=True)
comm.Barrier()
if rank == 0:
    print(f'All {size} ranks successfully initialized', flush=True)
"

MPI_TEST_EXIT=$?
if [ $MPI_TEST_EXIT -ne 0 ]; then
    echo "ERROR: MPI test failed with exit code $MPI_TEST_EXIT"
    echo "MPI is not working correctly. Exiting."
    exit 1
fi

# --- Verify FLORIS Installation ---
echo "Checking FLORIS installation:"
python -c "import floris; print(f'FLORIS path: {floris.__file__}'); print(f'Expected: ${BASE_DIR}/floris/floris/__init__.py')"

# --- Configuration Files ---
# Using existing Storm-specific configurations
WCNF="${WHOC_DIR}/examples/hercules_input_001.yaml"  # Wind controller config
DCNF="${WF_DIR}/config/preprocessing/preprocessing_inputs_flasc_STORM.yaml"  # Data preprocessing config (exists)
MCNF="${WF_DIR}/config/training/training_inputs_juan_flasc_tune_storm.yaml"  # Model training config (exists)

# Check if config files exist
if [ ! -f "$WCNF" ]; then
    echo "ERROR: Wind controller config not found: $WCNF"
    exit 1
fi

if [ ! -f "$DCNF" ]; then
    echo "ERROR: Data preprocessing config not found: $DCNF"
    exit 1
fi

if [ ! -f "$MCNF" ]; then
    echo "ERROR: Model training config not found: $MCNF"
    exit 1
fi

# Clean up any temporary files from previous runs
echo "Cleaning up temporary files..."
find ${WF_DIR}/examples/data/preprocessed_flasc_data/ -name "*_tmp.parquet" -delete 2>/dev/null || true

echo "=== STARTING CASE STUDY 21: baseline_controllers_perfect_forecaster_flasc ==="
date +"%Y-%m-%d %H:%M:%S"

# Run case study 21 with MPI parallelization
# Note: Reduced memory limit and use single-threaded Polars to avoid conflicts
export POLARS_MAX_THREADS=1

# First, try running without srun to see if the script itself works
echo "First testing script without MPI..."
python run_case_studies.py 21 \
    --exclude_prediction \
    --multiprocessor mpi \
    -rs \
    -ps \
    --ram_limit 32 \
    --wf_source scada \
    -st 1 \
    -ns 1 \
    -sd ${CASE_STUDY_OUTPUT_DIR} \
    -wcnf ${WCNF} \
    -dcnf ${DCNF} \
    -mcnf ${MCNF} \
    -rrs \
    --verbose

SCRIPT_TEST_EXIT=$?
echo "Script test exit code: $SCRIPT_TEST_EXIT"

if [ $SCRIPT_TEST_EXIT -eq 0 ]; then
    echo "Script works without srun. Now trying with srun..."
    
    # Now run with srun
    srun --mpi=pmi2 python run_case_studies.py 21 \
        --exclude_prediction \
        --multiprocessor mpi \
        -rs \
        -ps \
        --ram_limit 32 \
        --wf_source scada \
        -st auto \
        -ns auto \
        -sd ${CASE_STUDY_OUTPUT_DIR} \
        -wcnf ${WCNF} \
        -dcnf ${DCNF} \
        -mcnf ${MCNF} \
        -rrs
    
    CASE_STUDY_EXIT_CODE=$?
else
    echo "Script failed without srun. Not attempting parallel execution."
    CASE_STUDY_EXIT_CODE=$SCRIPT_TEST_EXIT
fi

echo "=== CASE STUDY FINISHED WITH EXIT CODE: ${CASE_STUDY_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

exit $CASE_STUDY_EXIT_CODE