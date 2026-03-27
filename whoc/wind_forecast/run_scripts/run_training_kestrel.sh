#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --account=awaken
#SBATCH --output=model_training_%j.out
##SBATCH --nodes=4
#SBATCH --time=192:00:00
#SBATCH --nodes=1
#SBATCH --mem=0
##SBATCH --time=01:00:00
##SBATCH --partition=debug
##SBATCH --partition=bigmem
##SBATCH --ntasks-per-node=88
##SBATCH --cpus-per-task=1

#  srun -n 1 --exclusive python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr" &
# salloc --account=awaken --job-name=model_tuning  --ntasks=104 --cpus-per-task=1 --time=01:00:00 --partition=debug
# python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr"

export MODEL=$1
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
export DATA_CONFIG_PATH="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"
export TARGET_TURBINE_IDX_FLAG="" #"--target_turbine_indices 74 73 4"


echo "MODEL=${MODEL}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
echo "TMPDIR=${TMPDIR}"

# prepare training data first
module purge
eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env
# module load PrgEnv-intel

# TODO NOTE process gets stuck after writing these .dat files, so run this python first, then the loop
export WORKER_SEED=0 # TODO does nothing atm
echo "=== STARTING TRAINING ==="
date +"%Y-%m-%d %H:%M:%S"
cd ..
python tuning.py --model ${MODEL} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} \
		--multiprocessor cf --seed ${WORKER_SEED} --mode train $TARGET_TURBINE_IDX_FLAG --use_tuned_params # --limit_train_val 0.25

date +"%Y-%m-%d %H:%M:%S"
echo "=== TRAINING COMPLETED ==="
