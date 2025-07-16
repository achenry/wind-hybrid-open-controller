#!/bin/bash
#SBATCH --job-name=baseline_wf_cpu
#SBATCH --account=awaken
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=36:00:00
##SBATCH --time=01:00:00
##SBATCH --partition=debug
#SBATCH --partition=bigmeme
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104

# salloc --partition=debug --nodes=1 --ntasks-per-node=104 --time=01:00:00 --mem=0 --account=awaken

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list

export MODELS="kf persistence sf svr"
export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predLUT.yaml"
export DATA_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"

#export N_PROCESSES=13
#export THREADS_PER_PROCESS=$(($SLURM_CPUS_PER_TASK / $N_PROCESSES))
#export POLARS_MAX_THREADS=$THREADS_PER_PROCESS
#export NUMEXPR_MAX_THREADS=$THREADS_PER_PROCESS

echo "MODELS=${MODELS}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"
echo "N_PROCESSES=${N_PROCESSES}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS}"
#echo "TMPDIR=${TMPDIR}"

# prepare training data first
date +"%Y-%m-%d %H:%M:%S"
module purge
# module load PrgEnv-intel
module load conda #/2022.05
#eval "$(conda shell.bash hook)"
#ml PrgEnv-intel mamba
#eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env

#mpirun -np $SLURM_NTASKS 
python ../run_forecaster_validation.py --ram_limit 65 --model ${MODELS} --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --run_name baseline_forecasters --simulation_timestep 1 \
						--save_dir /projects/awaken/ahenry/wind_forecasting/logging --multiprocessor cf --prediction_type distribution \
						--use_tuned_params --use_trained_models --max_splits 10 --run_validation --rerun_validation --run_processing #--max_steps 1600

