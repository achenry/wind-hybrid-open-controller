#!/bin/bash
#SBATCH --job-name=amr_precursor_3
#SBATCH --time=24:00:00
##SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

# A lot of modules and conda stuff
module purge

#export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier
#export MPICH_COLL_OPT_OFF=mpi_allreduce

export SPACK_MANAGER="/home/ahenry/toolboxes/spack-manager"
source $SPACK_MANAGER/start.sh
spack-start
quick-activate /home/ahenry/toolboxes/whoc_env
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy
spack load amr-wind+helics+openfast

rm logamr
echo "Starting AMR-Wind job at: " $(date) >> logamr
echo $SLURM_NTASKS
# Now go back to scratch folder and launch the job
srun /home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy/amr_wind amr_precursor_original_3.inp
mv post_processing post_processing_3
echo "Finished precursor 3 at:" $(date) >> logamr
