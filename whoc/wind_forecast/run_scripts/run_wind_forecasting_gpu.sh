#!/bin/bash
#SBATCH --job-name=transformer_wf_gpu
#SBATCH --account=awaken
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
##SBATCH --time=01:00:00
##SBATCH --partition=debug
##SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:2
##SBATCH --mem-per-cpu=20G
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=60G
# salloc --partition=debug --gres=gpu:2 --ntasks-per-node=2 --time=01:00:00 --mem-per-cpu=85G --account=awaken

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}"
echo "SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK}"
echo "SLURM_GPUS=${SLURM_GPUS}"

echo "=== ENVIRONMENT ==="
module list

# export MODELS="informer autoformer spacetimeformer tactis"
#export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred300.yaml"

export MODELS=$1
export MODEL_CONFIG_PATH=$2
export DATA_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"

#echo "MODELS=${MODELS}"
#echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
#echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
#echo "TMPDIR=${TMPDIR}"

# prepare training data first
#date +"%Y-%m-%d %H:%M:%S"
module purge
# module load PrgEnv-intel
#eval "$(conda shell.bash hook)"
ml PrgEnv-intel mamba
#eval "$(conda shell.bash hook)"
mamba activate wind_forecasting_env

export NUMEXPR_MAX_THREADS=128

devices=$SLURM_JOB_GPUS
n_devices=$((${#devices}/2 + 1))
echo "N_GPU_DEVICES=${n_devices}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($n_devices-1)))
#export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Calculate start and end cores (assuming i is 1-based)
#start_core=$(( ($i - 1) * $SLURM_NTASKS_PER_NODE ))
#end_core=$(( $i * $SLURM_NTASKS_PER_NODE - 1 ))

# Create the range string
#CORES="${start_core}-${end_core}"
#echo "Using CPUs ${CORES} out of available {$SLURM_NTASKS_PER_NODE}"
#echo "Using GPUs ${CUDA_VISIBLE_DEVICES}"

# taskset -c $start_core-$end_core 
python ../run_forecaster_validation.py --run_validation --model ${MODELS} --run_name ml --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --simulation_timestep 1 --save_dir /projects/awaken/ahenry/wind_forecasting/logging --checkpoint latest --multiprocessor cf --prediction_type distribution --use_trained_models --max_splits 30 --run_processing #--max_steps 1080 --rerun_validation
