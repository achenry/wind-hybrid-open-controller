#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

module purge
module load mamba
mamba activate wind_forecasting
module load intel
export LD_LIBRARY_PATH=/projects/ssc/ahenry/conda/envs/whoc/lib
echo $SLURM_NTASKS
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 9 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
python run_case_studies.py 18 --wf_source scada --generate_lut -m cf -sd /projects/ssc/ahenry/whoc/floris_case_studies -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml
#mpirun -np $SLURM_NTASKS python run_case_studies.py 5 6 -p -rs -st 3600 -ns 6 -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
