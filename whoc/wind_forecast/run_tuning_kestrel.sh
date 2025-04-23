#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --account=ssc
#SBATCH --output=model_tuning_%j.out
##SBATCH --nodes=4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
##SBATCH --time=01:00:00
##SBATCH --partition=debug
##SBATCH --partition=nvme
#SBATCH --ntasks-per-node=104
##SBATCH --cpus-per-task=1

#  srun -n 1 --exclusive python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr" &
# salloc --account=ssc --job-name=model_tuning  --ntasks=104 --cpus-per-task=1 --time=01:00:00 --partition=debug
# python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr"

export NTASKS_PER_TUNER=104
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

# Used to track process IDs for all workers
declare -a WORKER_PIDS=()

export MODEL_CONFIG_PATH=$2
# export MODEL_CONFIG=/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml
#export MODEL_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_flasc.yaml"
export DATA_CONFIG_PATH="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"
#export DATA_CONFIG="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_flasc.yaml"
#export RESTART_FLAG=""

echo "MODEL=${MODEL}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
echo "TMPDIR=${TMPDIR}"

# prepare training data first
module purge
module load mamba
mamba activate wind_forecasting_env
module load PrgEnv-intel

echo "=== STARTING DATA PREPARATION ==="
date +"%Y-%m-%d %H:%M:%S"

PYTHONPATH=$(which python)
#srun -n ${SLURM_NTASKS} --export=ALL,WORKER_RANK=0 
export WORKER_RANK=0
$PYTHONPATH tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --multiprocessor cf --seed 0 --restart_tuning

echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"
# for m in $(seq 0 $((${NUM_MODELS}-1))); do
for i in $(seq 1 $((${NTUNERS}))); do
#    for j in $(seq 0 $((${NUM_WORKERS_PER_CPU}-1))); do
        # The restart flag should only be set for the very first worker (i=0, j=0)
        #iif [ $i -eq 1 ]; then #&& [ $j -eq 0 ]; then
        #    export RESTART_FLAG="--restart_tuning"
        #else
        #    export RESTART_FLAG=""
        #fi

        # Create a unique seed for each worker to ensure they explore different areas
	export WORKER_SEED=$((42 + i*10)) #+ j))

        # Calculate worker index for logging
	export WORKER_RANK=${i} #$((i*NUM_WORKERS_PER_CPU + j))

        echo "Starting worker ${WORKER_RANK} on CPU ${i} with seed ${WORKER_SEED}"
        
        # Launch worker with environment settings
        srun -n ${NTASKS_PER_TUNER} $PYTHONPATH tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --multiprocessor cf --seed ${WORKER_SEED}&

	# nohup bash -c "
        # module purge
        # module load mamba
        # module load PrgEnv-intel
        # mamba activate wind_forecasting_env

        # python tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --multiprocessor cf --seed ${WORKER_SEED} ${RESTART_FLAG}" &

        # Store the process ID
        WORKER_PIDS+=($!)

        # Add a small delay between starting workers on the same GPU
        # to avoid initialization conflicts
        sleep 2
 #   done
done
echo "Started ${#WORKER_PIDS[@]} worker processes for model ${m}"
echo "Process IDs: ${WORKER_PIDS[@]}"

# Wait for all workers to complete
wait
# done

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="
