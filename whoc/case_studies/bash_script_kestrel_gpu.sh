#!/bin/bash
#SBATCH --job-name=transfo_floris_case_studies.py
#SBATCH --output=%j_%x.out
#SBATCH --time=96:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=40G
#SBATCH --account=awaken

# salloc --account=awaken --time=01:00:00 --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem-per-cpu=85G --account=awaken --partition=debug

module purge
#ml cuda
ml PrgEnv-intel mamba
#eval "$(mamba shell.bash hook)"
mamba activate wind_forecasting_env

devices=$SLURM_JOB_GPUS
n_devices=$((${#devices}/2 + 1))
echo "N_GPU_DEVICES=${n_devices}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($n_devices-1)))
export CASE_IDX=$1

# Create the range string
echo "Using GPUs ${CUDA_VISIBLE_DEVICES}"

# taskset -c $start_core-$end_core 
srun python run_case_studies.py $CASE_IDX --multiprocessor mpi -rs --ram_limit 75 --wf_source scada \
        -sd /projects/awaken/ahenry/whoc/floris_case_studies/ \
       -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml \
       -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml \
       -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred_smoothed.yaml -st auto -ns 30 \
       -cgs '10,32,566,578,636,645,646,660,664,666,672,673,674,677,679,683,685,710,719,742,745,746,752,753,759,760,777,779,1010,1180' # --stoptime 10400 #-stmp #-rrs 
