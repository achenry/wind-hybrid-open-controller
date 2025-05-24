#!/bin/bash
#SBATCH --job-name=tft_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --output=tft_tuning_%j.log
#SBATCH --error=tft_tuning_%j.err

# Load modules
module purge
module load miniforge
mamba activate wind_forecasting

# Fix CUDA compatibility if needed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/stth7454/software/anaconda/envs/wind_forecasting/lib

# Run the script
python /projects/stth7454/software/wind-hybrid-open-controller/whoc/wind_forecast/tuning.py \
    --model tft \
    --mode tune \
    --model_config /projects/stth7454/software/wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc_steven_HPC.yaml \
    --data_config /projects/stth7454/software/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_flasc_steven_HPC.yaml \
    --max_steps 1200 \
    --max_splits 30 \
    --tune \
    -rd
