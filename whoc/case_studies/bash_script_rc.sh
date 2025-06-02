#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --partition=amem
#SBATCH --qos=mem
##SBATCH --time=01:00:00
##SBATCH --partition=atesting
# salloc --nodes=1 --ntasks-per-node=64 --partition=amilan --time=12:00:00
# load modules
module purge
module load miniforge 
mamba activate wind_forecasting_env
#module load gcc/10.3 openmpi
# module load openmpi/4.1.4
module load intel/2022.1.2 impi/2021.5.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/stth7454/software/anaconda/envs/whoc/lib
module load intel impi


echo $SLURM_NTASKS

mpirun -np $SLURM_NTASKS python run_case_studies.py 18 --multiprocessor mpi -rs --exclude_prediction --ram_limit 75 --wf_source scada -st auto -ns 1 -sd /projects/aohe7145/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/floris_case_studies/ -wcnf /projects/aohe7145/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf /projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_rc_awaken_new.yaml -mcnf /projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_rc_awaken.yaml

#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 120 -ns 1 -p -m mpi -sd /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/floris_case_studies -wcnf /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf floris
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/floris_case_studies -wcnf /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf floris 
