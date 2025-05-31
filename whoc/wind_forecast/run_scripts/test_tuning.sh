
i=1
declare -i WORKER_RANK=${i}
declare -i export NTASKS_PER_TUNER=13
nohup bash -c "
    
    echo \"Worker ${WORKER_RANK} starting environment setup...\"

    # --- Module loading ---
    module purge
    echo \"Worker ${WORKER_RANK}: Modules loaded.\"

    # --- Activate conda environment ---
    eval \"\$(conda shell.bash hook)\"
    conda activate wind_forecasting_env
    echo \"Worker ${WORKER_RANK}: Conda environment 'wind_forecasting_env' activated.\"

    start_core=$(((${WORKER_RANK} - 1) * ${NTASKS_PER_TUNER}))
    end_core=$((${WORKER_RANK} * ${NTASKS_PER_TUNER} - 1))
    
    # --- Create the range string ---
    CORES=\"${start_core}-${end_core}\"
    echo Using cores ${CORES}

" > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${WORKER_RANK}.out" 2>&1 &