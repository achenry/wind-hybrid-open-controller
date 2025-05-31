#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --account=ssc
#SBATCH --output=model_tuning_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=0
##SBATCH --time=01:00:00
##SBATCH --partition=debug
#SBATCH --ntasks-per-node=104
##SBATCH --cpus-per-task=1

#  srun -n 1 --exclusive python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr" &
# salloc --account=ssc --job-name=model_tuning  --ntasks=104 --cpus-per-task=1 --time=01:00:00 --partition=debug
# python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr"

export NTASKS_PER_TUNER=26
export MODEL=$1
NTUNERS=$((SLURM_NTASKS / NTASKS_PER_TUNER)) # cast to int

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

export MODEL_CONFIG_PATH=$2
# export MODEL_CONFIG_PATH=/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml
#export MODEL_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_flasc.yaml"
export DATA_CONFIG_PATH="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"
#export DATA_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_flasc.yaml"
#export RESTART_FLAG=""

echo "MODEL=${MODEL}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"

# --- Base Directories ---
export BASE_DIR="/home/ahenry/toolboxes/wind_forecasting_env/wind-hybrid-open-controller"
export WORK_DIR="${BASE_DIR}/whoc/wind_forecast"
export LOG_DIR="${WORK_DIR}/logs"
export RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"
export AUTO_EXIT_WHEN_DONE="true"  # Set to "true" to exit script when all workers finish, "false" to keep running until timeout
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
# Create the job-specific directory for worker logs and final main logs
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}

# --- Print Job Info ---
echo "--- SLURM JOB INFO ---"
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
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "AUTO_EXIT_WHEN_DONE: '${AUTO_EXIT_WHEN_DONE}'"
echo "------------------------"



# prepare training data first
# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env
echo "Conda environment 'wind_forecasting_env' activated."
#module load PrgEnv-intel

echo "=== STARTING DATA PREPARATION ==="
date +"%Y-%m-%d %H:%M:%S"

PYTHONPATH=$(which python)
#srun -n ${SLURM_NTASKS} --export=ALL,WORKER_RANK=0 

export WORKER_RANK=0
python ${WORK_DIR}/tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --seed 0 #--restart_tuning # --reload_data

# --- Parallel Worker Launch using nohup ---
NUM_CPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_CPUS}  # Set total number of workers for tuning
declare -a WORKER_PIDS=()

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

for i in $(seq 1 $((${NTUNERS}))); do
        # if [ $i -eq 1 ]; then #&& [ $j -eq 0 ]; then
        #    export RESTART_FLAG="--restart_tuning"
        # else
        #    export RESTART_FLAG=""
        # fi

        # Create a unique seed for each worker to ensure they explore different areas
        export WORKER_SEED=$((42 + i*10)) #+ j))

        # Calculate worker index for logging
        echo "Saving output for worker ${i} to '${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log'"
        echo "Starting worker ${i} on assigned GPU ${i} with seed ${WORKER_SEED}"
        export WORKER_RANK=${i} #$((i*NUM_WORKERS_PER_CPU + j))

        echo "Starting worker ${WORKER_RANK} with seed ${WORKER_SEED}"

        # Launch worker with environment settings
        #srun -n ${NTASKS_PER_TUNER}
        #taskset -c $start_core-$end_core

        # Launch worker in the background using nohup and a dedicated bash shell
        nohup bash -c "
        
        echo \"Worker ${WORKER_RANK} starting environment setup...\"

        # --- Module loading ---
        module purge
        echo \"Worker ${WORKER_RANK}: Modules loaded.\"

        # --- Activate conda environment ---
        eval \"\$(conda shell.bash hook)\"
        conda activate wind_forecasting_env
        echo \"Worker ${WORKER_RANK}: Conda environment 'wind_forecasting_env' activated.\"

        # --- Calculate start and end cores (assuming i is 1-based) ---
        start_core=$(( ($i - 1) * $NTASKS_PER_TUNER ))
        end_core=$(( $i * $NTASKS_PER_TUNER - 1 ))

        # --- Create the range string ---
        CORES=\"${start_core}-${end_core}\"
        echo \"Using cores ${CORES}\"	


        echo \"Worker ${WORKER_RANK}: Running python script...\"
        taskset -c $start_core-$end_core python ${WORK_DIR}/tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} \
                --multiprocessor cf --seed ${WORKER_SEED} --limit_train_val .1 --mode tune & #${RESTART_FLAG}

        # Check exit status
        status=\$?
        if [ \$status -ne 0 ]; then
                echo \"Worker ${WORKER_RANK} FAILED with status \$status\"
        else
                echo \"Worker ${WORKER_RANK} STARTED RUNNING successfully\"
        fi
        exit \$status
        " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${WORKER_RANK}.out" 2>&1 &

        # Store the process ID
        WORKER_PIDS+=($!)

        # Add a small delay between starting workers on the same GPU
        # to avoid initialization conflicts
        sleep 2
done

echo "Started ${#WORKER_PIDS[@]} worker processes for model ${MODEL}"
echo "Process IDs: ${WORKER_PIDS[@]}"
echo "Check worker logs in ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_*.out"

# Wait for all workers to complete
wait
# done

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="
