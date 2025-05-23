
from typing import Optional, Union
from collections.abc import Iterable
from dataclasses import dataclass
import os
import datetime
from datetime import timedelta
import time
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from memory_profiler import profile
from functools import partial
from sklearn.metrics import mean_squared_error

# from joblib import parallel_backend

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except ImportError as e:
    import traceback
    print(f"ERROR: Failed to import mpi4py. MPI will not be available. Error: {e}")
    print(traceback.format_exc())
    
from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, PercentilePruner, PatientPruner, SuccessiveHalvingPruner, NopPruner
from optuna.trial import TrialState # Added for checking trial status
from optuna.study import MaxTrialsCallback

from floris import FlorisModel


factor = 1.5
# factor = 3.0 # single column
plt.rc('font', size=12*factor)          # controls default text sizes
plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
plt.rc('legend', fontsize=12*factor)    # legend fontsize
plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_palette("Paired")

@dataclass
class WindForecast:
    """Wind speed component forecasting module that provides various prediction methods."""
    context_timedelta: datetime.timedelta
    prediction_timedelta: datetime.timedelta
    measurements_timedelta: datetime.timedelta
    controller_timedelta: Optional[datetime.timedelta]
    fmodel: FlorisModel 
    tid2idx_mapping: dict
    turbine_signature: str
    use_tuned_params: bool
    kwargs: dict
    true_wind_field: Optional[Union[pd.DataFrame, pl.DataFrame]]
    # n_targets_per_turbine: int 
    
    # def read_measurements(self):
    #     """_summary_
    #     Read in new measurements, add to internal container.
    #     """
    #     raise NotImplementedError()

    def __post_init__(self):
        assert (self.context_timedelta % self.measurements_timedelta).total_seconds() == 0, "context_timedelta must be a multiple of measurements_timedelta"
        assert (self.prediction_timedelta % self.measurements_timedelta).total_seconds() == 0, "prediction_timedelta must be a multiple of measurements_timedelta" 
        
        self.train_first = False
        
        self.n_context = int(self.context_timedelta / self.measurements_timedelta) # number of simulation time steps in a context horizon
        self.n_prediction = int(self.prediction_timedelta / self.measurements_timedelta) # number of simulation time steps in a prediction horizon
        
        if self.controller_timedelta:
            assert (self.controller_timedelta % self.measurements_timedelta).total_seconds() == 0, "controller_timedelta must be a multiple of measurements_timedelta"
            self.n_controller = int(self.controller_timedelta / self.measurements_timedelta) # number of simulation time steps in a single controller sampling intervals
        
        self.n_targets_per_turbine = 2 # horizontal and vertical wind speed
        self.last_measurement_time = None
        
        assert all([i in list(self.tid2idx_mapping.values()) for i in np.arange(len(self.tid2idx_mapping))]), f"tid2idx_mapping should map turbine ids to integers {0} to {len(self.tid2idx_mapping)-1}, inclusive."
        
        self.idx2tid_mapping = dict([(v, k) for k, v in self.tid2idx_mapping.items()])
        
        self.outputs = [f"ws_horz_{tid}" for tid in self.tid2idx_mapping] + [f"ws_vert_{tid}" for tid in self.tid2idx_mapping]
        # self.training_data_loaded = {output: False for output in self.outputs}
        self.training_data_shape = {output: None for output in self.outputs}
    
    # @property
    # def true_wind_field(self):
    #     return self.true_wind_field
    
    # @true_wind_field.setter
    # def true_wind_field(self, true_wind_field):
    #     self.true_wind_field = true_wind_field
    
    def set_true_wind_field(self, true_wind_field):
        self.true_wind_field = true_wind_field
    
    def _get_ws_cols(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame]):
        if isinstance(historic_measurements, pl.DataFrame):
            return historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        elif isinstance(historic_measurements, pd.DataFrame):
            return [col for col in historic_measurements.columns if (col.startswith("ws_horz") or col.startswith("ws_vert"))]
    
    def _compute_output_score(self, output, params, limit_train_val=None):
        # logging.info(f"Defining model for output {output}.")
        # model = self.create_model(**{re.search(f"\\w+(?=_{output})", k).group(0): v for k, v in params.items() if k.endswith(f"_{output}")})
        model = self.create_model(**params)
        
        # get training data for this output
        logging.info(f"Getting training data for output {output}.")
        # randomly sample from training data
        
        X_train, y_train = self._get_output_data(output=output, split="train", reload=False)
        X_val, y_val = self._get_output_data(output=output, split="val", reload=False)
        
        if limit_train_val:
            random_indices = np.random.choice(np.arange(X_train.shape[0]), size=int(limit_train_val * X_train.shape[0]))
            X_train, y_train = X_train[random_indices, :], y_train[random_indices]
            
            random_indices = np.random.choice(np.arange(X_val.shape[0]), size=int(limit_train_val * X_val.shape[0]))
            X_val, y_val = X_val[random_indices, :], y_val[random_indices]
        
        # evaluate with cross-validation
        logging.info(f"Fitting model for output {output} with {X_train.shape[0]} training data points.")
        model.fit(X_train, y_train)
        logging.info(f"Computing score for output {output} with {X_val.shape[0]} validation data points.")
        return mean_squared_error(y_true=y_val, y_pred=model.predict(X_val))
    
    def _tuning_objective(self, trial, multiprocessor, limit_train_val, max_cpus):
        """
        Objective function to be minimized in Optuna
        """
        # define hyperparameter search space 
        params = self.get_params(trial)
            
        # max_cpus = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
        if multiprocessor:
            if multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                logging.info(f"Starting MPICommExecutor in _tuning_objective with {comm_size} CPUs")
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif multiprocessor == "cf":
                # max_cpus = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
                logging.info(f"Starting ProcessPoolExecutor in _tuning_objective with {max_cpus} CPUs")
                executor = ProcessPoolExecutor(max_workers=max_cpus,
                                              mp_context=mp.get_context("spawn"))
            
            with executor as ex:
                futures = [ex.submit(self._compute_output_score, output=output, params=params, limit_train_val=limit_train_val) for output in self.outputs]
                scores = [fut.result() for fut in futures]
        else:
            logging.info(f"Starting Sequential Executor in _tuning_objective with {1} workers")
            scores = []
            for output in self.outputs:
                scores.append(self._compute_output_score(output=output, params=params, limit_train_val=limit_train_val))
        
        logging.info(f"Completed trial {trial.number}.")
        return sum(scores)
    
    def prepare_data(self, dataset_splits, scale=True, reload=True, multiprocessor=None):
        """
        Prepares the training/val data for tuning for each output based on the historic measurements.
        
        Args:
            historic_measurements (Union[pd.DataFrame, pl.DataFrame]): The historical measurements to use for training.
        
        Returns:
            None
        """
        
        if RUN_ONCE := ((multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (multiprocessor != "mpi") or (multiprocessor is None)):
            logging.info(f"Reloading data.")
            for ds_type, ds_list in dataset_splits.items():
                for ds in ds_list:
                    if ds.shape[0] < self.n_context + self.n_prediction:
                        logging.warning(f"{ds_type} dataset with continuity groups {list(ds["continuity_group"].unique())} have insufficient length!")
                        continue
                        
        # For each output, prepare the training data
        if multiprocessor is not None:
            if multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                max_cpus = comm_size
            elif multiprocessor == "cf":
                # max_cpus = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
                max_cpus = mp.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_cpus,
                                                mp_context=mp.get_context("spawn"))
            with executor as ex:
                # if multiprocessor == "mpi":
                #     ex.max_workers = comm_size
                
                logging.info(f"Running _get_output_data in parallel with {ex} max_cpus={max_cpus}.")
                for ds in ds_list:
                    dataset_splits[ds_type] = [ds for ds in ds_list if ds.shape[0] >= self.n_context + self.n_prediction]
                
                futures = []
                for split, ds_list in dataset_splits.items():
                    for output in self.outputs:
                        futures.append(ex.submit(self._get_output_data, 
                                        measurements=ds_list, 
                                        output=output, 
                                        split=split, 
                                        reload=reload, 
                                        scale=scale, return_data=False))
                
                
                for f, fut in enumerate(futures):
                    # logging.info(f"Calling result on {f}th future object in prepare_data.")
                    fut.result()
                    
                # logging.info("Calling result on future objects in prepare_data.")
                # [fut.result() for fut in futures]
                # logging.info("Finished calling result on future objects in prepare_data.")
        else: 
            for split, ds_list in dataset_splits.items(): 
                measurements = [ds for ds in ds_list if ds.shape[0] >= int((self.context_timedelta + self.prediction_timedelta) / self.measurements_timedelta)]
                for output in self.outputs:
                    # logging.info(f"Getting data for split {split} output {output}.")
                    self._get_output_data(measurements=measurements, output=output, split=split, reload=reload, scale=scale)

        if RUN_ONCE:
            logging.info(f"Finished loading data.")
        return
    
    # def tune_hyperparameters_single(self, historic_measurements, scaler, feat_type, tid, study_name, seed, restart_tuning, backend, storage_dir, n_trials=1):
    def tune_hyperparameters_single(self, seed, optuna_storage, 
                                    config,
                                    n_trials_per_worker=1,
                                    total_study_trials=100,
                                    worker_id=0,
                                    multiprocessor=None,
                                    limit_train_val=None,
                                    restart_tuning=False,
                                    max_cpus=None,
                                    optimize_callbacks=None):
        
        comm = MPI.COMM_WORLD
        RUN_ONCE = (multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (multiprocessor != "mpi") or (multiprocessor is None)
        
        # for case when argument is list of multiple continuous time series AND to only get the training inputs/outputs relevant to this model
        # Log safely without credentials if they were included (they aren't for socket trust)
        if RUN_ONCE:
            if hasattr(optuna_storage, "url"):
                log_storage_url = optuna_storage.url.split('@')[0] + '@...' if '@' in optuna_storage.url else optuna_storage.url
                logging.info(f"Using Optuna optuna_storage URL: {log_storage_url}")

            # Configure pruner based on settings
            pruner = None
            if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
                pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
                logging.info(f"Configuring pruner: type={pruning_type}")

                if pruning_type == "patient":
                    patience = config["optuna"]["pruning"].get("patience", 0)
                    min_delta = config["optuna"]["pruning"].get("min_delta", 0.0)

                    # Configure wrapped pruner if specified
                    wrapped_config = config["optuna"]["pruning"].get("wrapped_pruner")
                    wrapped_pruner_instance = None

                    if wrapped_config and isinstance(wrapped_config, dict):
                        wrapped_type = wrapped_config.get("type", "").lower()
                        logging.info(f"Configuring wrapped pruner of type: {wrapped_type}")

                        if wrapped_type == "percentile":
                            percentile = wrapped_config.get("percentile", 50.0)
                            n_startup_trials = wrapped_config.get("n_startup_trials", 4)
                            n_warmup_steps = wrapped_config.get("n_warmup_steps", 12)
                            interval_steps = wrapped_config.get("interval_steps", 1)
                            n_min_trials = wrapped_config.get("n_min_trials", 1)

                            wrapped_pruner_instance = PercentilePruner(
                                percentile=percentile,
                                n_startup_trials=n_startup_trials,
                                n_warmup_steps=n_warmup_steps,
                                interval_steps=interval_steps,
                                n_min_trials=n_min_trials
                            )
                            logging.info(f"Created wrapped PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")
                            
                        elif wrapped_type == "successivehalving":
                            min_resource = wrapped_config.get("min_resource", 2)
                            reduction_factor = wrapped_config.get("reduction_factor", 2)
                            min_early_stopping_rate = wrapped_config.get("min_early_stopping_rate", 0)
                            bootstrap_count = wrapped_config.get("bootstrap_count", 0)

                            wrapped_pruner_instance = SuccessiveHalvingPruner(
                                min_resource=min_resource,
                                reduction_factor=reduction_factor,
                                min_early_stopping_rate=min_early_stopping_rate,
                                bootstrap_count=bootstrap_count
                            )
                            logging.info(f"Created wrapped SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")
                        
                        else:
                            logging.warning(f"Unknown wrapped pruner type: {wrapped_type}. Defaulting to NopPruner.")
                            wrapped_pruner_instance = NopPruner()
                    else:
                        logging.warning("No wrapped pruner configuration found. Defaulting to NopPruner.")
                        wrapped_pruner_instance = NopPruner()
                    
                    # If no valid wrapped pruner is configured, use NopPruner
                    if wrapped_pruner_instance is None:
                        logging.warning("No valid wrapped pruner configuration found. PatientPruner will wrap NopPruner.")
                        wrapped_pruner_instance = NopPruner()

                    # Create PatientPruner wrapping the configured pruner
                    pruner = PatientPruner(
                        wrapped_pruner=wrapped_pruner_instance,
                        patience=patience,
                        min_delta=min_delta
                    )
                    logging.info(f"Created PatientPruner with patience={patience}, min_delta={min_delta} wrapping {type(wrapped_pruner_instance).__name__}")

                elif pruning_type == "hyperband":
                    min_resource = config["optuna"]["pruning"].get("min_resource", 2)
                    max_resource = config["optuna"]["pruning"].get("max_resource", 10)
                    reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
                    bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)
                    
                    pruner = HyperbandPruner(
                        min_resource=min_resource,
                        max_resource=max_resource,
                        reduction_factor=reduction_factor,
                        bootstrap_count=bootstrap_count
                    )
                    logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_resource}, reduction_factor={reduction_factor}, bootstrap_count={bootstrap_count}")

                elif pruning_type == "successivehalving":
                    min_resource = config["optuna"]["pruning"].get("min_resource", 2)
                    reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
                    min_early_stopping_rate = config["optuna"]["pruning"].get("min_early_stopping_rate", 0)
                    bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)

                    pruner = SuccessiveHalvingPruner(
                        min_resource=min_resource,
                        reduction_factor=reduction_factor,
                        min_early_stopping_rate=min_early_stopping_rate,
                        bootstrap_count=bootstrap_count
                    )
                    logging.info(f"Created SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")

                elif pruning_type == "percentile":
                    percentile = config["optuna"]["pruning"].get("percentile", 25)
                    n_startup_trials = config["optuna"]["pruning"].get("n_startup_trials", 5)
                    n_warmup_steps = config["optuna"]["pruning"].get("n_warmup_steps", 2)
                    interval_steps = config["optuna"]["pruning"].get("interval_steps", 1)
                    n_min_trials = config["optuna"]["pruning"].get("n_min_trials", 1)

                    pruner = PercentilePruner(
                        percentile=percentile,
                        n_startup_trials=n_startup_trials,
                        n_warmup_steps=n_warmup_steps,
                        interval_steps=interval_steps,
                        n_min_trials=n_min_trials
                    )
                    logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")

                else:
                    logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
                    pruner = NopPruner()
            else:
                logging.info("Pruning is disabled, using NopPruner")
                pruner = NopPruner()
        
        # Create study on Worker 1, load on other Worker
        study = None # Initialize study variable
        objective_fn = None
        direction = "minimize" # minimize mean_squared_error
        if RUN_ONCE:  
            try:
                if worker_id == 1:
                    logging.info(f"Rank 1: Creating/loading Optuna study '{self.study_name}' with pruner: {type(pruner).__name__}")
                    study = create_study(study_name=self.study_name,
                                            storage=optuna_storage,
                                            direction=direction,
                                            load_if_exists=not restart_tuning, # Only load if not restarting
                                            sampler=TPESampler(
                                                seed=seed,
                                                n_startup_trials=config["optuna"]["sampler_params"]["tpe"].get("n_startup_trials", 16),
                                                multivariate=config["optuna"]["sampler_params"]["tpe"].get("multivariate", True),
                                                constant_liar=config["optuna"]["sampler_params"]["tpe"].get("constant_liar", True),
                                                group=config["optuna"]["sampler_params"]["tpe"].get("group", False)
                                            ),
                                            pruner=pruner) # minimize mse ie minimize mse
                    logging.info(f"Rank 1: Study '{self.study_name}' created or loaded successfully.")
                    
                else:
                    # Non-rank-1 workers MUST load the study created by Rank 1
                    
                    logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{self.study_name}'")
                    # Add a small delay and retry mechanism for loading, in case Rank 1 is slightly delayed
                    max_retries = 6 # Increased retries slightly
                    retry_delay = 10 # Increased delay slightly
                    for attempt in range(max_retries):
                        try:
                            study = load_study(
                                study_name=self.study_name,
                                storage=optuna_storage,
                                sampler=TPESampler(seed=seed), # Sampler might be needed for load_study too
                                pruner=pruner
                            )
                            logging.info(f"Rank {worker_id}: Study '{self.study_name}' loaded successfully on attempt {attempt+1}.")
                            break # Exit loop on success
                        except KeyError as e: # Optuna <3.0 raises KeyError if study doesn't exist yet
                            if attempt < max_retries - 1:
                                logging.warning(f"Rank {worker_id}: Study '{self.study_name}' not found yet (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                                time.sleep(retry_delay)
                            else:
                                logging.error(f"Rank {worker_id}: Failed to load study '{self.study_name}' after {max_retries} attempts (KeyError). Aborting.")
                                raise
                        except Exception as e: # Catch other potential loading errors (e.g., DB connection issues)
                            logging.error(f"Rank {worker_id}: An unexpected error occurred while loading study '{self.study_name}' on attempt {attempt+1}: {e}", exc_info=True)
                            # Decide whether to retry on other errors or raise immediately
                            if attempt < max_retries - 1:
                                logging.warning(f"Retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                            else:
                                logging.error(f"Rank {worker_id}: Failed to load study '{self.study_name}' after {max_retries} attempts due to persistent errors. Aborting.")
                                raise # Re-raise other errors after retries
                    
                    # Check if study was successfully loaded after the loop
                    if study is None:
                        # This condition should ideally be caught by the error handling within the loop, but added for safety.
                        raise RuntimeError(f"Rank {worker_id}: Could not load study '{self.study_name}' after multiple retries.")
        
            except Exception as e:
                # Log error with rank information
                logging.error(f"Rank {worker_id}: Error creating/loading study '{self.study_name}': {str(e)}", exc_info=True)
                # Log optuna_storage URL safely
                if hasattr(optuna_storage, "url"):
                    log_storage_url_safe = str(optuna_storage.url).split('@')[0] + '@...' if '@' in str(optuna_storage.url) else str(optuna_storage.url)
                    logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
                else:
                    logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
                raise
                
            # max_workers = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
            max_cpus = max_cpus or mp.cpu_count() # TODO TEST might be more efficient to allow each rank to use all cores, even if they block eachother
            logging.info(f"Rank {worker_id}: Participating in Optuna study {self.study_name} with {max_cpus} CPUs")
            objective_fn = partial(self._tuning_objective, multiprocessor=multiprocessor, limit_train_val=limit_train_val, max_cpus=max_cpus)
        
        if multiprocessor == "mpi":
            study = comm.bcast(study, root=0)
            objective_fn = comm.bcast(objective_fn, root=0)
        
        if optimize_callbacks is None:
            optimize_callbacks = []
        elif not isinstance(optimize_callbacks, list):
            optimize_callbacks = [optimize_callbacks]

        try:
            n_trials_per_worker = config["optuna"].get("n_trials_per_worker", 10)
            total_study_trials_config = config["optuna"].get("total_study_trials")
            
            n_trials_setting_for_optimize = None
            
            # Determine number of trials to run
            if isinstance(total_study_trials_config, int) and total_study_trials_config > 0:
                total_study_trials = total_study_trials_config
                study.set_user_attr("total_study_trials", total_study_trials)
                logging.info(f"Set global trial limit to {total_study_trials} trials.")
                n_trials_setting_for_optimize = None
                
                max_trials_cb = MaxTrialsCallback(
                    n_trials=total_study_trials,
                    states=(TrialState.COMPLETE, TrialState.PRUNED) # INFO: Do not count failed trials
                )
                optimize_callbacks.append(max_trials_cb)
                logging.info(f"MaxTrialsCallback added for {total_study_trials} trials.")
            else:
                # Fall back to per-worker limit if no global limit is set
                n_trials_setting_for_optimize = n_trials_per_worker
                logging.info(f"No valid global trial limit found (value: {total_study_trials_config}). Using per-worker limit of {n_trials_per_worker}.")
                n_trials_setting_for_optimize = n_trials_per_worker
        
            # Let Optuna handle trial distribution - each worker will ask the storage for a trial
            # Show progress bar only on rank 0 to avoid cluttered logs
            study.optimize(objective_fn,
                           n_trials=n_trials_setting_for_optimize, 
                           callbacks=optimize_callbacks,
                           show_progress_bar=(worker_id==1))
        except KeyError as e:
            logging.error(f"Configuration key missing: {e}")
        except Exception as e:
            logging.error(f"Rank {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
            raise
        
        if RUN_ONCE and worker_id == 1 and study:
            # --- Launch Dashboard (Rank 1 only) ---
            # if hasattr(optuna_storage, "url"):
            #     launch_optuna_dashboard(config, optuna_storage.url) # Call imported function
            # --------------------------------------
            # logging.info("Rank 0: Starting W&B summary run creation.")

            # Wait for all expected trials to complete
            num_workers = int(os.environ.get('WORLD_SIZE', 1))
            
            if total_study_trials:
                expected_total_trials = total_study_trials
                logging.info(f"Rank 1: Expecting a maximum of {expected_total_trials} trials (global limit).")
            else:
                expected_total_trials = num_workers * n_trials_per_worker
                logging.info(f"Rank 1: Expecting a total of {expected_total_trials} trials ({num_workers} workers * {n_trials_per_worker} trials/worker).")

            logging.info("Rank 1: Waiting for all expected Optuna trials to reach a terminal state...")
            wait_interval_seconds = 30
            while True:
                # Refresh trials from optuna_storage
                all_trials_current = study.get_trials(deepcopy=False)
                finished_trials = [t for t in all_trials_current if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
                num_finished = len(finished_trials)
                num_total_in_db = len(all_trials_current) # Current count in DB

                logging.info(f"Rank 1: Trial status check: {num_finished} finished / {num_total_in_db} in DB (expected total: {expected_total_trials}).")

                if num_finished >= expected_total_trials:
                    logging.info(f"Rank 1: All {expected_total_trials} expected trials have reached a terminal state.")
                    break
                elif num_total_in_db > expected_total_trials and num_finished >= expected_total_trials:
                    logging.warning(f"Rank 1: Found {num_total_in_db} trials in DB (expected {expected_total_trials}), but {num_finished} finished trials meet the expectation.")
                    break

                logging.info(f"Rank 1: Still waiting for trials to finish ({num_finished}/{expected_total_trials}). Sleeping for {wait_interval_seconds} seconds...")
                time.sleep(wait_interval_seconds)

            # Fetch best trial *before* initializing summary run
            best_trial = None
            try:
                best_trial = study.best_trial
                logging.info(f"Rank 1: Fetched best trial: Number={best_trial.number}, Value={best_trial.value}")
            except ValueError:
                logging.warning("Rank 1: Could not retrieve best trial (likely no trials completed successfully).")
            except Exception as e_best_trial:
                logging.error(f"Rank 1: Error fetching best trial: {e_best_trial}", exc_info=True)
                    
            # Log best trial details (only rank 0)
            if len(study.trials) > 0:
                logging.info("Number of finished trials: {}".format(len(study.trials)))
                logging.info("Best trial:")
                trial = study.best_trial
                logging.info("  Value: {}".format(trial.value))
                logging.info("  Params: ")
                for key, value in trial.params.items():
                    logging.info("    {}: {}".format(key, value))
            else:
                logging.warning("No trials were completed")
        
        # for output in self.outputs:
        #     os.remove(os.path.join(self.temp_save_dir, f"Xy_train_{output}.dat"))
        
        return study.best_params
    
    def _get_output_data(self, output, reload, split, measurements=None, scale=None, return_scaler=False, return_data=True):
        assert split in ["train", "test", "val"]
        feat_type = re.search(f"\\w+(?=_{self.turbine_signature})", output).group()
        tid = re.search(self.turbine_signature, output).group()
        Xy_path = os.path.join(self.model_save_dir, f"Xy_{self.study_name}_{split}_{output}.dat")
        
        input_turbine_indices = self.cluster_turbines[self.tid2idx_mapping[tid]]
        output_idx = input_turbine_indices.index(self.tid2idx_mapping[tid])
            
        if reload or not os.path.exists(Xy_path): 
            assert measurements is not None and scale is not None, "Must provide measurements df and scale boolean to reload data in _get_output_data"
            input_select = [f"{feat_type}_{self.idx2tid_mapping[t]}" for t in input_turbine_indices]
            if isinstance(measurements, Iterable):
                X_all = []
                y_all = []
                for d, ds in enumerate(measurements):
                    # don't scale for single dataset, scale for all of them
                    ds = ds.gather_every(self.n_prediction_interval)
                    
                    training_inputs = ds.select(input_select).to_numpy()
                        
                    X, y = self._prepare_arrays(training_inputs, feat_type, tid, output_idx)
                    X_all.append(X)
                    y_all.append(y)
                    
                    logging.info(f"Generated {d}th {split} data.")
                
                X_all = np.vstack(X_all)
                y_all = np.concatenate(y_all)
                
                if scale:
                    X_all = self.scaler[output].fit_transform(X_all)
                    y_all = (y_all * self.scaler[output].scale_[output_idx]) + self.scaler[output].min_[output_idx]
                
            else:
                training_inputs = ds.select(input_select).to_numpy()
                if scale: 
                    training_inputs = self.scaler[output].fit_transform(training_inputs)
                X_all, y_all = self._prepare_arrays(training_inputs, feat_type, tid, output_idx)
            
            data_shape = (X_all.shape[0], X_all.shape[1] + 1)
            fp = np.memmap(Xy_path, dtype="float32", 
                           mode="w+", shape=data_shape)
            
            np.save(Xy_path.replace(".dat", "_shape.npy"), data_shape)
            fp[:, :-1] = X_all
            fp[:, -1] = y_all
            fp.flush()
            logging.info(f"Saved {split} data to {Xy_path} with input shape {X_all.shape}")
        
        else:
            # assert os.path.exists(Xy_path), "Must run prepare_training_data before tuning"
            # logging.info(f"Loading existing {split} data from {Xy_path}")
            data_shape = tuple(np.load(Xy_path.replace(".dat", "_shape.npy")))
            fp = np.memmap(Xy_path, dtype="float32", 
                           mode="r", shape=data_shape)
            X_all = fp[:, :-1]
            y_all = fp[:, -1]
            
            # logging.info(f"Loaded {split} data from {Xy_path} with input shape {X_all.shape}")
        
        # logging.info(f"Deleting filepointer to {Xy_path}")
        del fp
        
        if return_data:
            # logging.info(f"Returning data from _get_output_data for Xy_path {Xy_path}")
            if return_scaler:
                return X_all, y_all, self.scaler[output]
            else:
                return X_all, y_all
        else:
            # logging.info(f"Returning None from _get_output_data for Xy_path {Xy_path}")
            return None
    
    def set_tuned_params(self, optuna_storage, study_name):
        """_summary_

        Args:
            backend (_type_): journal, sqlite, or mysql
            study_name (_type_): _description_
            storage_dir (FilePath): required for sqlite or journal optuna_storage

        Raises:
            Exception: _description_
            Exception: _description_
        """
        logging.info(f"Setting tuned parameters from study {study_name}.")
        try:
            study_id = optuna_storage.get_study_id_from_name(study_name)
            study = optuna_storage.get_best_trial(study_id)
            logging.info(f"Best trial found, number: {study.number}, value: {study.value}, params: {study.params}")
            trials = sorted(study.get_trials(), key=lambda trial: trial.value, reverse=True)
            logging.info(f"Best trials: {trials}")
            for output in self.outputs:
                self.model[output] = self.create_model(**study.params)
        except KeyError:
            logging.error(f"Optuna study {study_name} not found. Please run tuning.py first. Using default parameters for now.")
            for output in self.outputs:
                self.model[output] = self.create_model(**{k: v for k, v in self.kwargs.items() if k in self.model[output].get_params()})
        # self.model[output].set_params(**optuna_storage.get_best_trial(study_id).params)
        # optuna_storage.get_all_studies()[0]._study_id
        
    def predict_sample(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], n_samples: int):
        """_summary_
        Predict a given number of samples for each time step in the horizon
        Args:
            n_samples (int): _description_
        """
        raise NotImplementedError()

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame]):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        raise NotImplementedError()

    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        """_summary_
        Generate the parameters of the forecasted distribution
        """
        raise NotImplementedError()

    def get_pred_interval(self, current_time):
        return pl.datetime_range(start=current_time, end=current_time + self.prediction_timedelta, interval=self.measurements_timedelta, eager=True, closed="right").rename("time")

    @staticmethod
    def compute_score(forecast_wf, true_wf, metric, feature_types, probabilistic=False, turbine_ids="all", plot=False, label=None, fig_dir="./"):
        if turbine_ids != "all":
            forecast_wf = forecast_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
            true_wf = true_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
        else:
            turbine_ids = sorted(true_wf.select(pl.col("turbine_id").unique()).to_numpy().flatten(),
                                 key=lambda tid: re.search("\\d+", tid).group(0))
        
        forecast_wf = forecast_wf.select("time", "turbine_id", "feature", "value", "data_type")\
                                 .filter(pl.col("feature").is_in(feature_types))\
                                 .sort("turbine_id", "feature")
                                 
        # first_timestamp = forecast_wf.select(pl.col("time").first()).item()
        # last_timestamp = forecast_wf.select(pl.col("time").last()).item()
        # .filter(pl.col("time").is_between(first_timestamp, 
        #                                                    last_timestamp,
        #                                                    closed="both"))\
        true_wf = true_wf.select("time", "turbine_id", "feature", "value", "data_type")\
                         .filter(pl.col("time").is_in(forecast_wf.select(pl.col("time")))) \
                         .filter(pl.col("feature").is_in(feature_types))\
                         .sort("turbine_id", "feature")
        
        metrics = {"feature": [], "score": [], "turbine_id": []}
        for feat_type in feature_types:
            for tid in turbine_ids:
                score = metric(y_true=true_wf.filter((pl.col("feature") == feat_type) & (pl.col("turbine_id") == tid)).select("value").to_numpy(),
                       y_pred=forecast_wf.filter((pl.col("feature") == feat_type) & (pl.col("turbine_id") == tid)).select("value").to_numpy())
                metrics["feature"].append(feat_type)
                metrics["turbine_id"].append(tid)
                metrics["score"].append(score)
        
        metrics = pl.DataFrame(data=metrics)
        
        if plot:
            fig, ax = plt.subplots(1, 1)
            # for f, feat_type in enumerate(feature_types):
            sns.barplot(data=metrics, ax=ax, y="score", x="feature", hue="turbine_id")
            for bars in ax.containers:
                ax.bar_label(bars, fmt="%.3f")
            plt.tight_layout()
            
            fig_path = os.path.join(fig_dir, f'scores{label}.png')
            logging.info(f"Saving compute_score to {fig_path}.")
            fig.savefig(fig_path)
            
        return metrics
        
    @staticmethod
    def plot_forecast(forecast_wf, true_wf, continuity_groups=None, feature_types=None, feature_labels=None, prediction_type="point", 
                      per_turbine_target=False, turbine_ids="all", turbine_labels=None, label="", fig_dir="./", include_turbine_legend=False, multiple_forecasters=True,
                      use_common_timedelta=True, dt=None):
        
        # hue command either differentiates forecasters or turbines. When turbine != all, the turbines are shown on different plots
        assert (multiple_forecasters and turbine_ids != "all") or (not multiple_forecasters and turbine_ids == "all")
        
        if isinstance(forecast_wf, pd.DataFrame):
            forecast_wf = pl.DataFrame(forecast_wf)
            
        if isinstance(true_wf, pd.DataFrame):
            true_wf = pl.DataFrame(true_wf)
        
        if feature_types is None:
            feature_types = ["ws_horz", "ws_vert"]
            feature_labels = ["$u$ Wind Speed (m/s)", "$v$ Wind Speed (m/s)"]
        
        if turbine_ids == "all":
            fig, axs = plt.subplots(1, len(feature_types), sharex=True)
            axs = axs[np.newaxis, :]
        else:
            fig, axs = plt.subplots(len(turbine_ids), len(feature_types), sharex=True, figsize=(15.12, 8.8))
                
        if continuity_groups is not None and "continuity_group" in true_wf.collect_schema().names():
            true_wf = true_wf.filter(pl.col("continuity_group").is_in(continuity_groups))
            forecast_wf = forecast_wf.filter(pl.col("time").is_in(true_wf.select(pl.col("time"))))
        
        if turbine_ids != "all":
            forecast_wf = forecast_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
            true_wf = true_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
        
        if isinstance(forecast_wf, pl.LazyFrame):
            forecast_wf = forecast_wf.collect()
        
        if use_common_timedelta:
            if dt is None:
                dt =  forecast_wf.sort("time").group_by(["continuity_group", "forecaster", "turbine_id", "feature"], maintain_order=True).agg(pl.col("time").diff().slice(1).max().alias("dt")).select("dt").max().item()
                if dt is not None:
                    dt = int(dt.total_seconds())
            # dt = 30
            # forecast_wf.sort("time").group_by(["continuity_group", "forecaster", "turbine_id", "feature"], maintain_order=True).agg(pl.col("time").diff().slice(1).max().alias("dt")).select("dt").max().item().total_seconds()
            # forecast_wf.sort("time").with_columns(dt=pl.col("time").diff()).sort("dt")
            # forecast_wf.filter((pl.col("test_idx") <= 0) & (pl.col("feature") == "loc_ws_horz")).sort("time").with_columns(dt=pl.col("time").diff()).sort("dt")
            logging.info(f"Found greatest forecaster sampling time {dt}s. Downsampling forecast data.")
            forecast_wf = forecast_wf.with_columns(pl.col("time").dt.round(f"{dt}s").alias("time").cast(pl.Datetime(time_unit="us")))\
                                     .group_by(["time", "test_idx", "feature", "turbine_id", "data_type", "forecaster"], maintain_order=True)\
                                     .agg(cs.numeric().first())
            
        assert forecast_wf.select(pl.col("time")).unique().select(pl.len()).item() > 1, "Need more than one data point to plot a time series, try adding more values to continuity_groups or setting it to None"
        forecast_wf = forecast_wf.sort("time")
        for f, feat in enumerate(feature_types):
            if turbine_ids == "all":
                sns.lineplot(data=true_wf.filter(
                                (pl.col("feature") == feat) & (pl.col("time").is_between(forecast_wf.select(pl.col("time").min()).item(), forecast_wf.select(pl.col("time").max()).item(), closed="both"))), 
                                    x="time", y="value", ax=axs[0, f], hue="turbine_id", alpha=0.25)
                true_line_handle = axs[f].lines
            else:
                for t, tid in enumerate(turbine_ids):
                    sns.lineplot(data=true_wf.filter(
                                    (pl.col("feature") == feat) & (pl.col("turbine_id") == tid) & (pl.col("time").is_between(forecast_wf.select(pl.col("time").min()).item(), forecast_wf.select(pl.col("time").max()).item(), closed="both"))), 
                                        x="time", y="value", ax=axs[t, f], color="black", alpha=0.25)
                    true_line_handle = axs[t, f].lines
            if prediction_type == "distribution":
                if per_turbine_target:
                    # TODO test
                    if turbine_ids == "all":
                        sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}")), 
                                        x="time", y="value", ax=axs[0, f], dashes=[[4, 4]], marker="o", linestyle="--",
                                        hue="forecaster" if (multiple_forecasters and "forecaster" in forecast_wf.columns) else None, err_style="bars")
                        
                        axs[0, f].fill_between(
                            forecast_wf.select("time"), 
                            forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) - forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                            forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) + forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                            alpha=0.2, 
                        )
                    else:
                        for t, tid in enumerate(turbine_ids):
                            sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}") & (pl.col("turbine_id") == tid)), 
                                        x="time", y="value", ax=axs[t, f], dashes=[[4, 4]], marker="o", linestyle="--",
                                        hue="forecaster" if (multiple_forecasters and "forecaster" in forecast_wf.columns) else None, err_style="bars")
                            # forecaster_df = forecaster_df.sort("time", "test_idx").group_by(["time", "feature", "turbine_id"], maintain_order=True).agg(pl.col("value").first())
                            axs[t, f].fill_between(
                                forecast_wf.select("time"), 
                                forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) - forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                                forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) + forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                                alpha=0.2, 
                            )
                else:
                    if turbine_ids == "all":
                        sns.lineplot(data=forecast_wf.filter(pl.col("feature") == f"loc_{feat}"), 
                                    x="time", y="value", hue="turbine_id", ax=axs[0, f], dashes=[[4, 4]], marker="o", linestyle="--", err_style="bars")
                    else:
                        for t, tid in enumerate(turbine_ids):
                            sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}") & (pl.col("turbine_id") == tid)), 
                                    x="time", y="value", ax=axs[t, f], dashes=[[4, 4]], marker="o", linestyle="--",
                                    hue="forecaster" if (multiple_forecasters and "forecaster" in forecast_wf.columns) else None, err_style="bars")
                    
                    for t, tid in enumerate(forecast_wf["turbine_id"].unique(maintain_order=True)):
                        # color = loc_ax.get_lines()[t].get_color()
                        tid_df = forecast_wf.filter((pl.col("feature").str.ends_with(feat)) & (pl.col("turbine_id") == tid))
                        
                        ax = axs[0, f] if turbine_ids == "all" else axs[t, f]
                        
                        if multiple_forecasters:
                            for ff, forecaster in enumerate(tid_df.select(pl.col("forecaster").unique(maintain_order=True)).to_numpy().flatten()):
                                color = sns.color_palette()[ff]
                                forecaster_df = tid_df.filter(pl.col("forecaster") == forecaster)
                                if forecaster_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.len()).item() == 0:
                                    continue
                                
                                # this gets most uncertain predictions ie from earliest test_idx that captured it
                                forecaster_df = forecaster_df.sort("time", "test_idx").group_by(["time", "feature", "turbine_id"], maintain_order=True).agg(pl.col("value").first())
                                ax.fill_between(
                                    forecaster_df.filter(pl.col("feature") == f"loc_{feat}").select("time").to_numpy().flatten(), 
                                    (forecaster_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                                    - forecaster_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                                    (forecaster_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                                    + forecaster_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                                    alpha=0.2, color=color
                                )
                        else:
                            tid_df = tid_df.sort("time", "test_idx").group_by(["time", "feature", "turbine_id"], maintain_order=True).agg(pl.col("value").first())
                            ax.fill_between(
                                tid_df.filter(pl.col("feature") == f"loc_{feat}").select("time").to_numpy().flatten(), 
                                (tid_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                                - tid_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                                (tid_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                                + tid_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                                alpha=0.2, 
                            )
            elif prediction_type == "point":
                if turbine_ids == "all":
                    sns.lineplot(data=forecast_wf.filter(pl.col("feature") == feat), x="time", y="value", 
                                 hue="turbine_id", dashes=[[4, 4]], marker="o", linestyle="--", ax=axs[f], err_style="bars")
                else:
                    for t, tid in enumerate(turbine_ids):
                        sns.lineplot(data=forecast_wf.filter((pl.col("feature") == feat) & (pl.col("turbine_id") == tid)), 
                                     x="time", y="value", dashes=[[4, 4]], marker="o", linestyle="--", ax=axs[t, f], 
                                     hue="forecaster" if (multiple_forecasters and "forecaster" in forecast_wf.columns) else None, err_style="bars")
                    
            elif prediction_type == "sample":
                raise NotImplementedError()
            
            x_start = true_wf.filter((pl.col("time") <= pl.lit(forecast_wf.select(pl.col("time").min()).item()))).select(pl.col("time").last()).item()
            x_end = forecast_wf.select(pl.col("time").max()).item()
            
            axs[-1, f].set(xlabel="Time (min)", xlim=(x_start, x_end))
            axs[0, f].set(title=feature_labels[f])
            
            # x1_delta = timedelta(seconds=int(forecast_wf.filter(pl.col("test_idx") == forecast_wf.select(pl.col("test_idx").first())).select(pl.col("time").diff().slice(1,1)).item().total_seconds()))
            x1_delta = timedelta(seconds=5)
            # x2_delta = timedelta(minutes=15)
            n_ticks = 5
            x2_delta = forecast_wf.select(pl.col("time").max().alias("last_time") - pl.col("time").min().alias("first_time")).item() / n_ticks
            # x2_delta = timedelta(seconds=int(np.round(x2_delta.total_seconds() / (15*60)) * (15*60)))
            
            x_time_vals = [x_start + i * x2_delta for i in range(1+int((x_end - x_start) / x2_delta))]
            xtick_labels = [int((x - x_start) / x1_delta) for x in x_time_vals]
            
            axs[-1, f].set_xticks(x_time_vals)
            axs[-1, f].set_xticklabels(xtick_labels)
            
            for t in range(axs.shape[0]):
                axs[t, f].set_ylabel("")
                axs[t, f].legend([], [], frameon=False)
        
        if turbine_ids != "all":
            for t, tid in enumerate(turbine_ids):
                if turbine_labels:
                    axs[t, 0].set_ylabel(turbine_labels[t])
                else:
                    axs[t, 0].set_ylabel(f"Turbine {tid}")
        
        
        axs[0, -1].legend([], [], frameon=False)
        h, l = axs[0, -1].get_legend_handles_labels()
        # labels_1 = ["True"] #, "Forecast"] # removing data type
        leg1 = axs[0, -1].legend(true_line_handle, ["True"], loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        
        if turbine_ids == "all" and include_turbine_legend:
            labels_2 = ["turbine_id"] + sorted(list(forecast_wf.select(pl.col("turbine_id").unique()).to_numpy().flatten()))
            labels_2 = [label for label in labels_2 if label in l]
            handles_2 = [h[l.index(label)] for label in labels_2]
            second_legend = True
        elif multiple_forecasters and "forecaster" in forecast_wf.columns:
            labels_2 = sorted(list(forecast_wf.select(pl.col("forecaster").unique()).to_numpy().flatten()))
            labels_2 = [label for label in labels_2 if label in l]
            handles_2 = [h[l.index(label)] for label in labels_2]
            labels_2 = [" ".join(re.findall("[A-Z][^A-Z]*", re.search("\\w+(?=Forecast)", label).group())) 
                  if ("Forecast" in label) else (label.capitalize() if not label[0].isupper() else label).replace("_", " ") for label in labels_2]
    
            labels_2 = ["".join(label.split(" ")) if all(l.isupper() or l.isspace() for l in label) else label for label in labels_2]
            second_legend = True
        else:
            second_legend = False
        
        # labels_1 = [label for label in labels_1 if label in l]
        # handles_1 = [h[l.index(label)] for label in labels_1]
        # leg1 = axs[0, -1].legend(handles_1, labels_1, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        
        if second_legend:
            leg2 = axs[0, -1].legend(handles_2, labels_2, loc='upper left', bbox_to_anchor=(1.01, 0.6), frameon=False)
            axs[0, -1].add_artist(leg1)
        
        # axs[-].set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(forecast_wf.select(pl.col("time").min()).item()], forecast_wf.select(pl.col("time").max()).item()))
        fig.subplots_adjust(right=0.75)
        # plt.tight_layout()
        fig_path = os.path.join(fig_dir, f'forecast_ts{label}.png')
        logging.info(f"Saving plot_forecast to {fig_path}")
        fig.savefig(fig_path)
        
        xlim_rng = axs[-1, -1].get_xlim()[1] - axs[-1, -1].get_xlim()[0]
        time_rng = x_end - x_start
        new_time_range = timedelta(minutes=15)
        new_time_lim = (x_start, x_start + new_time_range)
        new_xlim = (axs[-1, -1].get_xlim()[0], axs[-1, -1].get_xlim()[0] + (new_time_range/time_rng)*xlim_rng)
        n_ticks = 5
        xdelta = int(np.round((new_time_range/n_ticks).total_seconds() / 30) * 30) / 60
        new_xticks = np.linspace(new_xlim[0], new_xlim[1], n_ticks)
        new_xticklabels = [int(i * xdelta) for i in range(n_ticks)]
        
        for ax in axs[-1, :]:
            ax.set_xlim(new_xlim)
            ax.set_xticks(new_xticks)
            ax.set_xticklabels(new_xticklabels)
        # plt.autoscale(enable=True, axis='y', tight=True)
        fig_path = fig_path.replace(".png", "_reduced.png")
        logging.info(f"Saving reduced plot_forecast to {fig_path}")
        fig.savefig(fig_path)
        return fig

    @staticmethod
    def plot_turbine_data(long_df, fig_dir, label=""):
        fig_ts, ax_ts = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
        # fig_ts.set_size_inches(12, 6)
        if hasattr(ax_ts, '__len__'):
            ax_ts = ax_ts.flatten()
        else:
            ax_ts = [ax_ts]
        
        for cg in long_df.select(pl.col("continuity_group")).unique().to_numpy().flatten(): 
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="ws_horz", ax=ax_ts[0])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="ws_vert", ax=ax_ts[1])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="nd_cos", ax=ax_ts[2])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="nd_sin", ax=ax_ts[3])

        ax_ts[0].set(title='Downwind Freestream Wind Speed, U [m/s]', ylabel="")
        ax_ts[1].set(title='Crosswind Freestream Wind Speed, V [m/s]', ylabel="")
        ax_ts[2].set(title='Nacelle Direction Cosine [-]', ylabel="")
        ax_ts[3].set(title='Nacelle Direction Sine [-]', ylabel="")

        # handles, labels, kwargs = mlegend._parse_legend_args([ax_ts[0]], ncol=2, title="Wind Seed")
        # ax_ts[0].legend_ = mlegend.Legend(ax_ts[0], handles, labels, **kwargs)
        # ax_ts[0].legend_.set_ncols(2)
        for i in range(0, len(ax_ts)):
            ax_ts[i].legend([], [], frameon=False)
        
        # time = long_df.select("time").to_numpy().flatten()
        # for i in range(len(ax_ts) - 2, len(ax_ts)):
        #     ax_ts[i].set(xticks=time[0:-1:int(60 * 12 // (time[1] - time[0]))], xlabel='Time [s]')
            # xlim=(time.iloc[0], 3600.0)) 
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f'wind_field_ts{label}.png')
        logging.info(f"Saving plot_turbine_data to {fig_path}")
        fig_ts.savefig(fig_path)
    