"""Module for wind speed component forecasting and sf functionality."""

from typing import Optional, Union
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from itertools import chain, repeat, islice
import os
import datetime
from datetime import timedelta
import yaml
import time
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from torch.distributions import MultivariateNormal
from torch import Tensor
import gc
from memory_profiler import profile
import pickle
import glob

# from joblib import parallel_backend

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except:
    print("No MPI available on system.")

from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.util import period_index
from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.field_names import FieldName
from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.distributions import DistributionOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
from gluonts.model.forecast import SampleForecast
from gluonts.torch.model.forecast import DistributionForecast

from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule

from wind_forecasting.preprocessing.data_inspector import DataInspector
from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.postprocessing.probabilistic_metrics import continuous_ranked_probability_score_gaussian, reliability, resolution, uncertainty, sharpness, pi_coverage_probability, pi_normalized_average_width, coverage_width_criterion 
from wind_forecasting.run_scripts.testing import get_checkpoint
from wind_forecasting.run_scripts.tuning import get_tuned_params, generate_df_setup_params
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage
from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard, log_optuna_visualizations_to_wandb

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from mysql.connector import Error as SQLError, connect as sql_connect
from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, NopPruner
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState # Added for checking trial status

# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.utils.validation import check_is_fitted

from filterpy.kalman import KalmanFilter
from floris import FlorisModel
from scipy.signal import lfilter
from scipy.stats import multivariate_normal as mvn

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#ARIMA packages
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
from scipy.special import inv_boxcox




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
    model_config: Optional[dict]
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
    
    def _compute_output_score(self, output, params):
        # logging.info(f"Defining model for output {output}.")
        # model = self.create_model(**{re.search(f"\\w+(?=_{output})", k).group(0): v for k, v in params.items() if k.endswith(f"_{output}")})
        model = self.create_model(**params)
        
        # get training data for this output
        # logging.info(f"Getting training data for output {output}.")
        # TODO this is v inefficient, better to load all at once...
        X_train, y_train = self._get_output_data(output=output, split="train", reload=False)
        X_val, y_val = self._get_output_data(output=output, split="val", reload=False)
        
        # evaluate with cross-validation
        logging.info(f"Computing score for output {output}.")
        # train_split = np.random.choice(X_train.shape[0], replace=False, size=int(X_train.shape[0] * 0.75))
        # train_split = np.isin(range(X_train.shape[0]), train_split)
        # test_split = ~train_split
        
        model.fit(X_train, y_train)
        return (-mean_squared_error(y_true=y_val, y_pred=model.predict(X_val)))
    
    def _tuning_objective(self, trial):
        """
        Objective function to be minimized in Optuna
        """
        # define hyperparameter search space 
        params = self.get_params(trial)
         
        # train svr model
        # total_score = 0
        # for output in self.outputs:
        # if self.multiprocessor == "mpi" and mpi_exists:
        #     executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
        #     # logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
        # else:
        max_workers = mp.cpu_count()
        executor = ProcessPoolExecutor(max_workers=max_workers,
                                            mp_context=mp.get_context("spawn"))
        # logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
        with executor as ex:
            futures = [ex.submit(self._compute_output_score, output=output, params=params) for output in self.outputs]
            scores = [fut.result() for fut in futures]
        
        logging.info(f"Completed trial {trial.number}.")
        return sum(scores)
    
    def prepare_data(self, dataset_splits, scale=True, reload=True):
        """
        Prepares the training/val data for tuning for each output based on the historic measurements.
        
        Args:
            historic_measurements (Union[pd.DataFrame, pl.DataFrame]): The historical measurements to use for training.
        
        Returns:
            None
        """
        logging.info(f"Reloading {dataset_splits} data.")
        for ds_type, ds_list in dataset_splits.items():
            for ds in ds_list:
                if ds.shape[0] < self.n_context + self.n_prediction:
                    logging.warning(f"{ds_type} dataset with continuity groups {list(ds['continuity_group'].unique())} have insufficient length!")
                    continue
                    
        # For each output, prepare the training data
        for output in self.outputs:
            for split, ds_list in dataset_splits.items():
                measurements = [ds for ds in ds_list if ds.shape[0] >= self.n_context + self.n_prediction]
                self._get_output_data(measurements=measurements, output=output, split=split, reload=reload, scale=scale)
     
    # def tune_hyperparameters_single(self, historic_measurements, scaler, feat_type, tid, study_name, seed, restart_tuning, backend, storage_dir, n_trials=1):
    def tune_hyperparameters_single(self, seed, storage, 
                                    config,
                                    n_trials_per_worker=1):
        # for case when argument is list of multiple continuous time series AND to only get the training inputs/outputs relevant to this model
        # Log safely without credentials if they were included (they aren't for socket trust)
        if hasattr(storage, "url"):
            log_storage_url = storage.url.split('@')[0] + '@...' if '@' in storage.url else storage.url
            logging.info(f"Using Optuna storage URL: {log_storage_url}")

        # Configure pruner based on settings
        if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
            pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
            
            min_resource = config["optuna"]["pruning"]["min_resource"]
            max_resource = config["optuna"]["pruning"]["max_resource"]
                
            logging.info(f"Configuring pruner: type={pruning_type}, min_resource={config["optuna"]["pruning"]['min_resource']}")

            if pruning_type == "hyperband":
                reduction_factor = config["optuna"]["pruning"]["reduction_factor"]
                
                pruner = HyperbandPruner(
                    min_resource=min_resource,
                    max_resource=max_resource,
                    reduction_factor=reduction_factor
                )
                logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_resource}, reduction_factor={reduction_factor}")
            elif pruning_type == "median":
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=min_resource)
                logging.info(f"Created MedianPruner with n_startup_trials=5, n_warmup_steps={min_resource}")
            elif pruning_type == "percentile":
                percentile = config["optuna"][percentile]
                pruner = PercentilePruner(percentile=percentile, n_startup_trials=5, n_warmup_steps=min_resource)
                logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials=5, n_warmup_steps={min_resource}")
            else:
                logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
                pruner = NopPruner()
        else:
            logging.info("Pruning is disabled, using NopPruner")
            pruner = NopPruner()
        
        # Get worker ID for study creation/loading logic
        # Use WORKER_RANK consistent with run_model.py. Default to '0' if not set.
        
        worker_id = os.environ.get('WORKER_RANK', '0')

        # Create study on rank 0, load on other ranks
        study = None # Initialize study variable
        
        logging.info("line 275")
        try:
            if worker_id == '0':
                logging.info(f"Rank 0: Creating/loading Optuna study '{self.study_name}' with pruner: {type(pruner).__name__}")
                study = create_study(study_name=self.study_name,
                                        storage=storage,
                                        direction="maximize",
                                        load_if_exists=True,
                                        sampler=TPESampler(seed=seed),
                                        pruner=pruner) # maximize negative mse ie minimize mse
                logging.info(f"Rank 0: Study '{self.study_name}' created or loaded successfully.")
                
                # --- Launch Dashboard (Rank 0 only) ---
                if hasattr(storage, "url"):
                    launch_optuna_dashboard(config, storage.url) # Call imported function
                # --------------------------------------
            else:
                # Non-rank-0 workers MUST load the study created by rank 0
                logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{self.study_name}'")
                # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
                max_retries = 6 # Increased retries slightly
                retry_delay = 10 # Increased delay slightly
                for attempt in range(max_retries):
                    try:
                        study = load_study(
                            study_name=self.study_name,
                            storage=storage,
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
            # Log storage URL safely
            if hasattr(storage, "url"):
                log_storage_url_safe = str(storage.url).split('@')[0] + '@...' if '@' in str(storage.url) else str(storage.url)
                logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
            else:
                logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
            raise
        
        max_workers = mp.cpu_count()
        logging.info(f"Worker {worker_id}: Participating in Optuna study {self.study_name} with {max_workers} workers")
        objective_fn = self._tuning_objective
        
        try:
            study.optimize(objective_fn,
                           n_trials=n_trials_per_worker, 
                           show_progress_bar=(worker_id=='0'))
        except Exception as e:
            logging.error(f"Worker {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
            raise
        
        logging.info("line 351")
        if worker_id == '0' and study:
            # logging.info("Rank 0: Starting W&B summary run creation.")

            # Wait for all expected trials to complete
            num_workers = int(os.environ.get('WORLD_SIZE', 1))
            expected_total_trials = num_workers * n_trials_per_worker
            logging.info(f"Rank 0: Expecting a total of {expected_total_trials} trials ({num_workers} workers * {n_trials_per_worker} trials/worker).")

            logging.info("Rank 0: Waiting for all expected Optuna trials to reach a terminal state...")
            wait_interval_seconds = 30
            while True:
                # Refresh trials from storage
                all_trials_current = study.get_trials(deepcopy=False)
                finished_trials = [t for t in all_trials_current if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
                num_finished = len(finished_trials)
                num_total_in_db = len(all_trials_current) # Current count in DB

                logging.info(f"Rank 0: Trial status check: {num_finished} finished / {num_total_in_db} in DB (expected total: {expected_total_trials}).")

                if num_finished >= expected_total_trials:
                    logging.info(f"Rank 0: All {expected_total_trials} expected trials have reached a terminal state.")
                    break
                elif num_total_in_db > expected_total_trials and num_finished >= expected_total_trials:
                    logging.warning(f"Rank 0: Found {num_total_in_db} trials in DB (expected {expected_total_trials}), but {num_finished} finished trials meet the expectation.")
                    break

                logging.info(f"Rank 0: Still waiting for trials to finish ({num_finished}/{expected_total_trials}). Sleeping for {wait_interval_seconds} seconds...")
                time.sleep(wait_interval_seconds)

            # Fetch best trial *before* initializing summary run
            best_trial = None
            try:
                best_trial = study.best_trial
                logging.info(f"Rank 0: Fetched best trial: Number={best_trial.number}, Value={best_trial.value}")
            except ValueError:
                logging.warning("Rank 0: Could not retrieve best trial (likely no trials completed successfully).")
            except Exception as e_best_trial:
                logging.error(f"Rank 0: Error fetching best trial: {e_best_trial}", exc_info=True)
                
        # Log best trial details (only rank 0)
        logging.info("line 393")
        if worker_id == '0' and study: # Check if study object exists
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
    
    def _get_output_data(self, output, reload, split, measurements=None, scale=None, return_scaler=False):
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
                for ds in measurements:
                    # don't scale for single dataset, scale for all of them
                    ds = ds.gather_every(self.n_prediction_interval)
                    
                    training_inputs = ds.select(input_select).to_numpy()
                        
                    X, y = self._prepare_arrays(training_inputs, feat_type, tid, output_idx)
                    X_all.append(X)
                    y_all.append(y)
                
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
            logging.info(f"Saved {split} data to {Xy_path}")
        else:
            # assert os.path.exists(Xy_path), "Must run prepare_training_data before tuning"
            # logging.info(f"Loading existing {split} data from {Xy_path}")
            data_shape = tuple(np.load(Xy_path.replace(".dat", "_shape.npy")))
            fp = np.memmap(Xy_path, dtype="float32", 
                           mode="r", shape=data_shape)
            X_all = fp[:, :-1]
            y_all = fp[:, -1]
               
        del fp
        if return_scaler:
            return X_all, y_all, self.scaler[output]
        else:
            return X_all, y_all
    
    def set_tuned_params(self, storage, study_name):
        """_summary_

        Args:
            backend (_type_): journal, sqlite, or mysql
            study_name (_type_): _description_
            storage_dir (FilePath): required for sqlite or journal storage

        Raises:
            Exception: _description_
            Exception: _description_
        """
        try:
            study_id = storage.get_study_id_from_name(study_name)
            for output in self.outputs:
                self.model[output] = self.create_model(**storage.get_best_trial(study_id).params)
        except KeyError:
            logging.error(f"Optuna study {study_name} not found. Please run tuning.py first. Using default parameters for now.")
            for output in self.outputs:
                self.model[output] = self.create_model(**{k: v for k, v in self.kwargs.items() if k in self.model[output].get_params()})
        # self.model[output].set_params(**storage.get_best_trial(study_id).params)
        # storage.get_all_studies()[0]._study_id
        
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
            fig.savefig(os.path.join(fig_dir, f'scores{label}.png'))
            
        return metrics
        
    @staticmethod
    def plot_forecast(forecast_wf, true_wf, continuity_groups=None, feature_types=None, feature_labels=None, prediction_type="point", per_turbine_target=False, turbine_ids="all", label="", fig_dir="./",
                      include_turbine_legend=False):
        if isinstance(forecast_wf, pd.DataFrame):
            forecast_wf = pl.DataFrame(forecast_wf)
            
        if isinstance(true_wf, pd.DataFrame):
            true_wf = pl.DataFrame(true_wf)
        
        if feature_types is None:
            feature_types = ["ws_horz", "ws_vert"]
            feature_labels = ["Horizontal Wind Speed (m/s)", "Vertical Wind Speed (m/s)"]
        
        if turbine_ids == "all":
            fig, axs = plt.subplots(1, len(feature_types), sharex=True)
            axs = axs[np.newaxis, :]
        else:
            fig, axs = plt.subplots(len(turbine_ids), len(feature_types), sharex=True)
                
        if continuity_groups is not None and "continuity_group" in true_wf.collect_schema().names():
            true_wf = true_wf.filter(pl.col("continuity_group").is_in(continuity_groups))
            forecast_wf = forecast_wf.filter(pl.col("time").is_in(true_wf.select(pl.col("time"))))
        
        if turbine_ids != "all":
            forecast_wf = forecast_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
            true_wf = true_wf.filter(pl.col("turbine_id").is_in(turbine_ids))
        
        assert forecast_wf.select(pl.col("time")).unique().select(pl.len()).item() > 1, "Need more than one data point to plot a time series, try adding more values to continuity_groups or setting it to None"
        
        for f, feat in enumerate(feature_types):
            if turbine_ids == "all":
                sns.lineplot(data=true_wf.filter(
                                (pl.col("feature") == feat) & (pl.col("time").is_between(forecast_wf.select(pl.col("time").min()).item(), forecast_wf.select(pl.col("time").max()).item(), closed="both"))), 
                                    x="time", y="value", ax=axs[0, f], style="data_type", hue="turbine_id")
            else:
                for t, tid in enumerate(turbine_ids):
                    sns.lineplot(data=true_wf.filter(
                                    (pl.col("feature") == feat) & (pl.col("turbine_id") == tid) & (pl.col("time").is_between(forecast_wf.select(pl.col("time").min()).item(), forecast_wf.select(pl.col("time").max()).item(), closed="both"))), 
                                        x="time", y="value", ax=axs[t, f], style="data_type")
            
            if prediction_type == "distribution":
                if per_turbine_target:
                    # TODO test
                    if turbine_ids == "all":
                        sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}")), 
                                        x="time", y="value", ax=axs[0, f], style="data_type", dashes=[[4, 4]], marker="o")
                        
                        axs[0, f].fill_between(
                            forecast_wf.select("time"), 
                            forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) - forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                            forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) + forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                            alpha=0.2, 
                        )
                    else:
                        for t, tid in enumerate(turbine_ids):
                            sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}") & (pl.col("turbine_id") == tid)), 
                                        x="time", y="value", ax=axs[t, f], style="data_type", dashes=[[4, 4]], marker="o")
                        
                            axs[t, f].fill_between(
                                forecast_wf.select("time"), 
                                forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) - forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                                forecast_wf.filter((pl.col("feature") == f"loc_{feat}")) + forecast_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                                alpha=0.2, 
                            )
                else:
                    if turbine_ids == "all":
                        sns.lineplot(data=forecast_wf.filter(pl.col("feature") == f"loc_{feat}"), 
                                    x="time", y="value", hue="turbine_id", style="data_type", ax=axs[0, f], dashes=[[4, 4]], marker="o")
                    else:
                        for t, tid in enumerate(turbine_ids):
                            sns.lineplot(data=forecast_wf.filter((pl.col("feature") == f"loc_{feat}") & (pl.col("turbine_id") == tid)), 
                                    x="time", y="value",  style="data_type", ax=axs[t, f], dashes=[[4, 4]], marker="o")
                    
                    for t, tid in enumerate(forecast_wf["turbine_id"].unique(maintain_order=True)):
                        # color = loc_ax.get_lines()[t].get_color()
                        tid_df = forecast_wf.filter((pl.col("feature").str.ends_with(feat)) & (pl.col("turbine_id") == tid))
                        color = sns.color_palette()[t]
                        ax = axs[0, f] if turbine_ids == "all" else axs[t, f]
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
                    sns.lineplot(data=forecast_wf.filter(pl.col("feature") == feat), x="time", y="value", hue="turbine_id", style="data_type", dashes=[[4, 4]], marker="o", ax=axs[f])
                else:
                    for t, tid in enumerate(turbine_ids):
                        sns.lineplot(data=forecast_wf.filter((pl.col("feature") == feat) & (pl.col("turbine_id") == tid)), x="time", y="value", style="data_type", dashes=[[4, 4]], marker="o", ax=axs[t, f])
                    
            elif prediction_type == "sample":
                raise NotImplementedError()
            
            x_start = true_wf.filter((pl.col("time") <= pl.lit(forecast_wf.select(pl.col("time").min()).item()))).select(pl.col("time").last()).item()
            x_end = forecast_wf.select(pl.col("time").max()).item()
            
            axs[-1, f].set(xlabel="Time (min)", xlim=(x_start, x_end))
            axs[0, f].set(title=feature_labels[f])
            
            x1_delta = timedelta(seconds=int(forecast_wf.select(pl.col("time").diff().slice(1,1)).item().total_seconds()))
            x2_delta = timedelta(minutes=1)
            x_time_vals = [x_start + i * x2_delta for i in range(1+int((x_end - x_start) / x2_delta))]
            # forecast_wf.filter(pl.col("feature") == feat).select("time").to_pandas().values.flatten()
            # n_skips = int(timedelta(minutes=15) / x_delta)
            # x_time_vals = x_time_vals[::n_skips]
            # n_skips = int(len(x_time_vals) // 10)
            # x_time_vals = x_time_vals[::n_skips]
            xtick_labels = [int((x - x_start) / x1_delta) for x in x_time_vals]
            # xticks = xticks.astype("timedelta64[s]") / x_delta
            axs[-1, f].set_xticks(x_time_vals)
            axs[-1, f].set_xticklabels(xtick_labels)
            
            for t in range(axs.shape[0]):
                axs[t, f].set_ylabel("")
                axs[t, f].legend([], [], frameon=False)
        
        if turbine_ids != "all":
            for t, tid in enumerate(turbine_ids):
                axs[t, 0].set_ylabel(f"Turbine {tid}")
            
        axs[0, -1].legend([], [], frameon=False)
        h, l = axs[0, -1].get_legend_handles_labels()
        labels_1 = ["True", "Forecast"] # removing data type
        if turbine_ids == "all":
            labels_2 = ["turbine_id"] + sorted(list(forecast_wf.select(pl.col("turbine_id").unique()).to_numpy().flatten()))
            labels_2 = [label for label in labels_2 if label in l]
            handles_2 = [h[l.index(label)] for label in labels_2]
        
        labels_1 = [label for label in labels_1 if label in l]
        handles_1 = [h[l.index(label)] for label in labels_1]
        leg1 = axs[0, -1].legend(handles_1, labels_1, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        
        if turbine_ids == "all" and include_turbine_legend:
            leg2 = axs[0, -1].legend(handles_2, labels_2, loc='upper left', bbox_to_anchor=(1.01, 0.9), frameon=False)
            axs[0, -1].add_artist(leg1)
        # axs[-].set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(forecast_wf.select(pl.col("time").min()).item()], forecast_wf.select(pl.col("time").max()).item()))
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'forecast_ts{label}.png'))
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
        fig_ts.savefig(os.path.join(fig_dir, f'wind_field_ts{label}.png'))
    

@dataclass
class PerfectForecast(WindForecast):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    
    col_mapping: Optional[dict] = None
    is_probabilistic = False
    
    def reset(self):
        pass
    
    
    def __post_init__(self):
        # logging.info(f"id(wind_field_ts) in PerfectForecast __init__ is {id(self.true_wind_field)}")
        super().__post_init__()
        self.train_first = False
        if isinstance(self.true_wind_field, pd.DataFrame):
            self.true_wind_field = pl.from_pandas(self.true_wind_field)
        self.true_wind_field = self.true_wind_field.select(pl.col("time"), cs.starts_with("ws_"))
    
    # @profile
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        
        sub_df = (self.true_wind_field.rename(self.col_mapping) if self.col_mapping else self.true_wind_field)\
            .filter(pl.col("time").is_between(current_time, current_time + self.prediction_timedelta, closed="right"))
        
        # slice true_wind_field_ts to reduce memory reqs
        # self.true_wind_field = self.true_wind_field.filter(pl.col("time") > current_time)
        if isinstance(historic_measurements, pl.DataFrame):
            return sub_df
            # assert sub_df.select(pl.len()).item() == int(self.prediction_timedelta / self.measurements_timedelta)
        elif isinstance(historic_measurements, pd.DataFrame):
            return sub_df.to_pandas()
            
        return sub_df

@dataclass
class PersistenceForecast(WindForecast):
    """ Wind speed component forecasting using persistence model that assumes future values equal current value. """
    is_probabilistic = False
    
    def reset(self):
        pass
    
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        
        pred_slice = self.get_pred_interval(current_time)
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
         
        assert historic_measurements.select((pl.col("time") == current_time).any()).item()
        last_measurement = historic_measurements.filter(pl.col("time") == current_time)
        pred = {k: [v[0]] * len(pred_slice) for k, v in last_measurement.to_dict().items() if k != "time"}
        
        pred =  pl.concat([pred_slice.to_frame(), pl.DataFrame(pred)], how="horizontal")
        if return_pl:
            return pred
        else:
            return pred.to_pandas()
        
             
@dataclass
class SpatialFilterForecast(WindForecast):
    """
    Reads wind speed components from upstream turbines, and computes time of arrival based on taylor's frozen wake hypothesis.
    """
    is_probabilistic = False
    def __post_init__(self):
        super().__post_init__()
        self.train_first = False
        self.n_turbines = self.fmodel.n_turbines
        self.measurement_layout = np.vstack([self.fmodel.layout_x, self.fmodel.layout_y]).T
        
        # the turbines to consider in the spatial filtering for the estimation of the wind direction at each turbine
        self.n_neighboring_turbines = self.kwargs["n_neighboring_turbines"]
        if self.n_neighboring_turbines:
            self.cluster_turbines = [sorted(np.arange(self.n_turbines), 
                        key=lambda t: np.linalg.norm(self.measurement_layout[tid, :] - self.measurement_layout[t, :]))[:self.n_neighboring_turbines]
                                    for tid in range(self.n_turbines)]
        else:
            self.cluster_turbines = [np.arange(self.n_turbines)] * self.n_turbines
    
    def reset(self):
        pass
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] # pred_slice.filter(pred_slice == current_time + self.prediction_timedelta)
        outputs = self._get_ws_cols(historic_measurements)
        
        # multivariate with state matrix containing all horizontal and vertical wind speed measurements
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        
        new_measurements = historic_measurements.slice(-1, 1)
        
        pred = self.full_farm_directional_weighted_average(new_measurements)
        
        # self.last_measurement_time = historic_measurements.select(pl.col("time").last()).item()
        
        pred = {output: pred[:, o] for o, output in enumerate(outputs)}
        pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
        
        if return_pl:
            return pred
        else:
            return pred.to_pandas()
    
    
    def full_farm_directional_weighted_average(
        self,
        new_measurements: Union[pd.DataFrame, pl.DataFrame]
        # data_in,
        # wind_speeds,
        # wind_directions,
        # shift_distance,
        # is_circular=False,
        # is_bearing=False,
    ):
        """_summary_
        QUESTION
        Args:
            data_in (pd.DataFrame): num_columns = n_turbines, and indices corresponding to time
            measurement_layout (np.ndarray): 0th dim = n_turbines, 1st dim = x,y coords
            wind_directions (np.ndarray): array of wind direction estimations for each turbine
            shift_distance (float): distance from turbine at which to estimate wind direction

        Returns:
            _type_: _description_
        """

        # nTurbs = len(data_in.columns)
        # turbine_list = np.arange(0, self.n_turbines)

        # if is_bearing:  # Convert to RH CCW angle
        #     wd_mean = SpatialFilterForecast.bearing2angle(wd_mean)
        #     data_in = data_in.applymap(SpatialFilterForecast.bearing2angle)
        # re.search("(?<=ws_horz_)\\d+", "ws_horz_1")
        turbine_ids = sorted(set(re.search("(?<=\\w_)\\d+$", col).group(0) for col in new_measurements.select(cs.starts_with("ws_")).columns), key=lambda tid: int(re.search("\\d+", tid).group(0)))
        ws_horz = new_measurements.select(cs.starts_with("ws_horz_"))\
                                  .rename(lambda old_col: re.search("(?<=\\w_)\\d+$", old_col).group(0))
        ws_vert = new_measurements.select(cs.starts_with("ws_vert_"))\
                                  .rename(lambda old_col: re.search("(?<=\\w_)\\d+$", old_col).group(0))
        
        wm = new_measurements.select(**{tid: ((pl.col(f"ws_horz_{tid}")**2 + pl.col(f"ws_vert_{tid}")**2).sqrt()) for tid in turbine_ids})
        wd = new_measurements.select(**{tid: 180.0 + (pl.arctan2(pl.col(f"ws_horz_{tid}"), pl.col(f"ws_vert_{tid}")).degrees()) for tid in turbine_ids})
        
        ws_inputs = np.dstack([ws_horz.select(turbine_ids).to_numpy(), ws_vert.select(turbine_ids).to_numpy()])
        
        # farm_wind_direction = wd.select(pl.mean_horizontal(pl.all())).to_numpy().flatten()
        weights = self.neighbor_weights_directional_gaussian(cluster_turbines=self.cluster_turbines, 
                                                             wind_dirs=wd.to_numpy(), wind_speeds=wm.to_numpy() ) #, shift_distance)
        
        pred = np.zeros_like(ws_inputs)
         
        for tid in range(self.n_turbines):
            idx = self.cluster_turbines[tid]
            pred[:, tid, :] = np.einsum("ntd,nt->nd", ws_inputs[:, idx, :], weights[tid])
        
        return pred.T.reshape((2*self.n_turbines, -1)).T 

    def neighbor_weights_directional_gaussian(
        self, cluster_turbines, wind_dirs, wind_speeds, #shift_distance=0
    ):
        """
        wd_mean should be in radians, CCW
        mu = 0: no sf
        sigma = None: will default to using the standard deviation of turbine distances.
        """

        weights = dict()
        # n_turbines = np.shape(measurement_layout)[0]
       
        for i in range(self.n_turbines):
            idx = cluster_turbines[i]
            cluster_wind_direction = np.mean(np.deg2rad(wind_dirs[:, idx]), axis=1)
            cluster_layout = self.measurement_layout[idx, :]
            shift_distance = self.prediction_timedelta.total_seconds() * wind_speeds[:, i]
            
            center_point = (self.measurement_layout[i, :] + np.array(
                [
                    -shift_distance * np.sin(np.pi + cluster_wind_direction), # -cos=
                    -shift_distance * np.cos(np.pi + cluster_wind_direction),
                ]
            ).T)

            # sigma defaults to using the standard deviation of turbine distances.
            covariance = np.var(
                np.linalg.norm(cluster_layout[np.newaxis, :, :] - center_point[:, np.newaxis, :], axis=2), axis=1
            )

            f = []
            for t in range(center_point.shape[0]):
                f.append(mvn.pdf(cluster_layout, mean=center_point[t, :], cov=covariance[t] * np.identity(2)))
            f = np.array(f)
            
            fsum = f.sum(axis=1)[:, np.newaxis]
            weights[i] = np.divide(f, fsum, out=np.zeros_like(f), where=(fsum!=0))
            
            if fsum == 0:
                logging.warning(f"The center point, determined by prediction_timedelta, is too far from turbine {i}'s clusters to have any nonzero weights, assuming persistance for turbine {i}.")
                weights[i][:, np.where(idx == i)[0]] = 1

        return weights

    @staticmethod
    def bearing2angle(bearing):
        """
        Convert bearing angle in degrees to CCW positive angle in radians.
        """
        return SpatialFilterForecast.wrap_angle(np.pi / 180 * (90 - bearing), -np.pi, np.pi)
    
    @staticmethod
    def angle2bearing(angle):
        return SpatialFilterForecast.wrap_angle(90 - 180 / np.pi * angle, 0.0, 360.0)

    @staticmethod
    def wrap_angle(x, low, high):
        """
        Wrap to x to interval [low, high)
        """
        x = float(x)
        step = high - low

        while x >= high:
            x = x - step
        while x < low:
            x = x + step

        return x

    @staticmethod
    def weighted_circular_mean3(x, weights):
        """
        Find weighted mean value of x after shifting all values to
        [low, high) (method 3).

        Inputs:
            x - data for mean (list, Series, or 1D array)
            weights - weights for mean (list, Series, or 1D array) (same
                    size as x)
        Outputs:
            (x_mean) - scalar circular mean of x
        """

        # Convert list, series to array
        if type(x) == np.ndarray:
            x = np.e ** (1j * x)
            complex_mean = list(x @ np.array(weights))
        else:
            x = np.array([np.e ** (1j * x[i]) for i in range(len(x))])
            complex_mean = x @ np.array(weights)

        return np.angle(complex_mean)
 

@dataclass
class GaussianForecast(WindForecast):
    """ Wind speed component forecasting using Gaussian parameterization. """
    
    # def read_measurements(self):
    #     pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

@dataclass
class SVRForecast(WindForecast):
    """Wind speed component forecasting using Support Vector Regression."""
    is_probabilistic = False
    def __post_init__(self):
        super().__post_init__()
        self.train_first = True
        self.max_n_samples = self.kwargs["max_n_samples"] 

        self.n_turbines = self.fmodel.n_turbines
        self.n_outputs = self.n_turbines * 2
        self.measurement_layout = np.vstack([self.fmodel.layout_x, self.fmodel.layout_y]).T
        
        self.n_neighboring_turbines = self.kwargs["n_neighboring_turbines"] 
        if self.n_neighboring_turbines:
            self.cluster_turbines = [sorted(np.arange(self.n_turbines), 
                        key=lambda t: np.linalg.norm(self.measurement_layout[tid, :] - self.measurement_layout[t, :]))[:self.n_neighboring_turbines]
                                    for tid in range(self.n_turbines)]
        else:
            self.cluster_turbines = [np.arange(self.n_turbines)] * self.n_turbines
        
        # rescale this since SVR predicts a sample for every self.prediction_timedelta (not multistep)
        if (self.context_timedelta % self.prediction_timedelta).total_seconds() != 0:
            self.context_timedelta = ((self.context_timedelta // self.prediction_timedelta) + 1) * self.prediction_timedelta
            
        self.n_prediction_interval = self.n_prediction # for SVR, we consider measurments prediction_timedelta apart
        self.prediction_interval = self.n_prediction_interval * self.measurements_timedelta
        self.n_context = int(self.context_timedelta / self.prediction_interval)
        self.n_prediction = 1
        
        # if self.max_n_samples is None:
        #     self.max_n_samples = (self.n_context + self.n_prediction) * 1000
            
        self.last_measurement_time = None
        # study_name=f"tuning_{self.model_key}_{self.model_config['experiment']['run_name']}"
        self.study_name = f"svr_{self.model_config['experiment']['run_name']}"
        self.model_save_dir = os.path.join(self.model_config["experiment"]["log_dir"], self.study_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        self.scaler = defaultdict(self.create_scaler)
        self.model = defaultdict(self.create_model)
            
        self.use_trained_models = self.kwargs.get("use_trained_models", True)
        
        model_files = glob.glob(os.path.join(self.model_save_dir, f"{self.study_name}_model_*_{int(self.prediction_timedelta.total_seconds())}.pkl"))
        scaler_files = glob.glob(os.path.join(self.model_save_dir, f"{self.study_name}_scaler_*_{int(self.prediction_timedelta.total_seconds())}.pkl"))
        
        if self.use_trained_models and len(model_files) == 0:
            logging.error(f"No trained models found in {self.model_save_dir}. Please run tuning.py first for the correct prediction time {int(self.prediction_timedelta.total_seconds())}.")
            raise Exception()
        
        # no need to load optuna trained hyperparams if we are loading models anyway
        if (not self.use_trained_models or len(model_files) < self.n_outputs or len(scaler_files) < self.n_outputs) and self.use_tuned_params:
            self.set_tuned_params(storage=self.kwargs["optuna_storage"], 
                                    study_name=self.study_name)
            self.use_trained_models = False
        
        # if we want to use previously trained models fetch them, otherwise models will need to be trained
        if self.use_trained_models:
            for model_file in model_files:
                # if os.path.exists(os.path.join(self.model_save_dir, f"svr_model_{output}_{self.prediction_timedelta.total_seconds()}.pkl")):
                output = re.search(f"(?<={self.study_name}_model_)([\\w\\_]+\\d+)(?=\\_)", os.path.basename(model_file)).group()
                prediction_length = float(re.search(f"(?<={self.study_name}_model_{output}_)([\\d\\.]+)(?=.pkl)", os.path.basename(model_file)).group())
                if prediction_length != self.prediction_timedelta.total_seconds():
                    continue
                with open(os.path.join(self.model_save_dir, model_file), "rb") as fp:
                    self.model[output] = pickle.load(fp)
                assert self.model[output].n_features_in_ == self.n_neighboring_turbines * self.n_context, f"SVR must be tuned and trained for n_neighboring_turbines = {self.n_neighboring_turbines} and context_timedelta = {self.context_timedelta}."
                
            for scaler_file in scaler_files:
                # if os.path.exists(os.path.join(self.model_save_dir, f"svr_scaler_{output}_{self.prediction_timedelta.total_seconds()}.pkl")):
                output = re.search(f"(?<={self.study_name}_scaler_)([\\w\\_]+\\d+)(?=\\_)", os.path.basename(scaler_file)).group()
                prediction_length = float(re.search(f"(?<={self.study_name}_scaler_{output}_)([\\d\\.]+)(?=.pkl)", os.path.basename(scaler_file)).group())
                if prediction_length != self.prediction_timedelta.total_seconds():
                    continue
                with open(os.path.join(self.model_save_dir, scaler_file), "rb") as fp:
                    self.scaler[output] = pickle.load(fp)
    
    def reset(self):
        pass
    
    def create_scaler(self):
        return MinMaxScaler(feature_range=(-1, 1))
    
    def create_model(self, **kwargs):
        return SVR(**kwargs)
   
    def _prepare_arrays(self, training_inputs, feat_type, tid, output_idx):
        
        X_train = np.ascontiguousarray(np.vstack([
            training_inputs[i:i+self.n_context, :].flatten()
            for i in range(max(training_inputs.shape[0] - self.n_context, 1))
        ]))
        
        # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_timedelta][output])
        y_train = np.ascontiguousarray(training_inputs[self.n_context:, output_idx])
        
        # assert X_train.shape[0] == y_train.shape[0]
        
        return X_train, y_train
    
    def get_params(self, trial):
         
        # return {
        #     **{f"C_{output}": trial.suggest_float(f"C_{output}", 1e-6, 1e6, log=True) for output in self.outputs},
        #     **{f"epsilon_{output}": trial.suggest_float(f"epsilon_{output}", 1e-6, 1e-1, log=True) for output in self.outputs},
        #     **{f"gamma_{output}": trial.suggest_categorical(f"gamma_{output}", ["scale", "auto"]) for output in self.outputs}
        # }
        return {
            "C": trial.suggest_float(f"C", 1e-6, 1e6, log=True),
            "epsilon": trial.suggest_float(f"epsilon", 1e-6, 1e-1, log=True),
            "gamma": trial.suggest_categorical(f"gamma", ["scale", "auto"])
        }
    

    def predict_sample(self, n_samples: int):
        pass
    
    def train_single_output(self, training_measurements, output, retrain_models, scale, scaler_params=None):
        
        if not retrain_models \
            and os.path.exists(os.path.join(self.model_save_dir, f"{self.study_name}_model_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")) \
                and (not scale or scaler_params or os.path.exists(os.path.join(self.model_save_dir, f"svr_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"))):
            
            logging.info(f"Loading trained SVR model for output {output}.")
            with open(os.path.join(self.model_save_dir, f"{self.study_name}_model_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"), "rb") as fp:
                self.model[output] = pickle.load(fp)

            if scale and scaler_params is None:
                with open(os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"), "rb") as fp:
                    self.scaler[output] = pickle.load(fp)
        else:
            
            X_train, y_train, self.scaler[output] = self._get_output_data(measurements=training_measurements, output=output, split="train", reload=False, 
                                                                          scale=scale, return_scaler=True)
            logging.info(f"Training SVR model for output {output}.")
            self.model[output].fit(X_train, y_train)
            
            with open(os.path.join(self.model_save_dir, f"{self.study_name}_model_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"), "wb") as fp:
                pickle.dump(self.model[output], fp, protocol=5)
                
            if scale and scaler_params is None:
                with open(os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"), "wb") as fp:
                    pickle.dump(self.scaler[output], fp, protocol=5)
        
        feat_type = re.search(f"\\w+(?=_{self.turbine_signature})", output).group()
        tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
        if scaler_params:
            input_turbine_indices = self.cluster_turbines[self.tid2idx_mapping[tid]]
            self.scaler[output].n_features_in_ = len(input_turbine_indices)
            for k, v in scaler_params.items():
                setattr(self.scaler[output], k, np.ones_like(input_turbine_indices) * v[feat_type])
                
            with open(os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl"), "wb") as fp:
                pickle.dump(self.scaler[output], fp, protocol=5)
        return self.model[output], self.scaler[output]
    
    def train_all_outputs(self, historic_measurements, scale, multiprocessor, retrain_models=True,
                          scaler_params=None):
        # training_measurements = historic_measurements.gather_every(self.n_prediction_interval)
        # scale = (training_measurements.select(pl.len()).item() > 1) and scale
        outputs = self._get_ws_cols(historic_measurements)
        
        # if training_measurements.select(pl.len()).item() >= self.n_context + self.n_prediction:
        #     if self.max_n_samples:
        #         training_measurements = training_measurements.tail(self.max_n_samples)
            
        # pred = {}
        if multiprocessor is not None:
            if multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif multiprocessor == "cf":
                executor = ProcessPoolExecutor()
            with executor as ex:
                if multiprocessor == "mpi":
                    ex.max_workers = comm_size
                    
                futures = [ex.submit(self.train_single_output, 
                                        training_measurements=None, 
                                        output=output, 
                                        scale=scale, 
                                        scaler_params=scaler_params,
                                        retrain_models=retrain_models) for output in outputs]
                for output, fut in zip(outputs, futures):
                    m, s = fut.result()
                    self.model[output] = m
                    self.scaler[output] = s
        else:    
            for output in outputs:
                self.model[output], self.scaler[output] = self.train_single_output(
                    training_measurements=None, 
                    output=output, scale=scale, 
                    scaler_params=scaler_params, retrain_models=retrain_models)
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        # TODO LOW include yaw angles in inputs?
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] 
        outputs = self._get_ws_cols(historic_measurements)
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        
        # training_measurements = historic_measurements.filter(((current_time - pl.col("time")).mod(self.prediction_interval) == 0))
        training_measurements = historic_measurements.filter(((current_time - pl.col("time")).dt.total_microseconds().mod(self.prediction_interval.total_seconds() * 1e6) == 0))

        
        
        # scale = (training_measurements.select(pl.len()).item() > 1)
        scale = True
        
        if training_measurements.select(pl.len()).item() >= self.n_context:
            if self.max_n_samples:
                training_measurements = training_measurements.tail(self.max_n_samples)
            
            pred = {}
            for output in outputs:
                feat_type = re.search(f"^\\w+(?=_{self.turbine_signature}$)", output).group()
                tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
                if not (hasattr(self.scaler[output], "min_") and hasattr(self.scaler[output], "scale_")) \
                    or (check_is_fitted(self.model[output]) is not None):
                    raise Exception(f"scaler/model for {output} has not been trained!")
                training_inputs = self._get_inputs(training_measurements, self.scaler[output], feat_type, tid, scale)
                
                pred[output] = self._predict(model=self.model[output], 
                                             training_inputs=training_inputs)
                
            # rescale back
            if scale:
                pred = {output: self._inverse_scale(pred, output).flatten() for output in outputs}
            else:
                pred = {output: pred[output][np.newaxis, :].flatten() for output in outputs}
            
            pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
            
        else:
            # not enough data points to train SVR, assume persistance
            logging.info(f"Not enough data points at time {current_time} to train SVR, have {historic_measurements.select(pl.len()).item() * self.measurements_timedelta} but require {self.n_context * self.prediction_timedelta}, assuming persistance instead.")
            pred = pl.concat([pred_slice.to_frame(), historic_measurements.slice(-1, 1).select(outputs)], how="horizontal")
            
        if return_pl: 
            return pred
        else:
            return pred.to_pandas()  
            
    def _inverse_scale(self, pred, output):
        tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
        output_idx = self.cluster_turbines[self.tid2idx_mapping[tid]].index(self.tid2idx_mapping[tid]) 
        return (pred[output][np.newaxis, :] - self.scaler[output].min_[output_idx]) / self.scaler[output].scale_[output_idx]

    def _get_inputs(self, training_measurements, scaler, feat_type, tid, scale):
        input_turbine_indices = self.cluster_turbines[self.tid2idx_mapping[tid]] 
        training_inputs = training_measurements.select([f"{feat_type}_{self.idx2tid_mapping[t]}" for t in input_turbine_indices]).to_numpy()
        
        if scale: 
            training_inputs = scaler.transform(training_inputs)
        
        return training_inputs
    
    def _predict(self, model, training_inputs):
        X_pred = np.ascontiguousarray(training_inputs)[-self.n_context:, :].flatten()[np.newaxis, :]
        y_pred = model.predict(X_pred)
        
        pred = y_pred[-self.n_prediction:] 
        
        return pred
    
    def predict_distr(self):
        raise NotImplementedError("SVRForecast does not support predict_distr()")

class KalmanFilterWrapper(KalmanFilter):
    def __init__(self, dim_x=1, dim_z=1, **kwargs): # TODO need this st cross validation will work in Optuna objective function, ideally clone could capture dim_x, dim_z automatically
        super().__init__(dim_x=dim_x, dim_z=dim_z)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X=None, **predict_kwargs):
        if X is None:
            super().predict(**predict_kwargs)
        else:
            pred = []
            self.x = X[:1]
            # for s in range(X.shape[0]): # loop over samples
                # pred.append([])
            for i in range(X.shape[0]): # loop over history and update matrices
                z = X[i:i+1]
                super().predict()
                self.update(z)
                pred.append(self.x[0])
            return np.array(pred)
    

@dataclass
class KalmanFilterForecast(WindForecast):
    """Wind speed component forecasting using Kalman filtering."""
    # def read_measurements(self):
    #     pass
    is_probabilistic = True
    def __post_init__(self):
        super().__post_init__()
        self.n_prediction_interval = self.n_prediction
        self.prediction_interval = self.n_prediction_interval * self.measurements_timedelta
        self.n_turbines = self.fmodel.n_turbines
        self.dim_x = self.dim_z = self.n_targets_per_turbine * self.n_turbines
        
        self.scaler = self.create_scaler()
        self.reset()
        
        # equal to number of n_prediction intervals for the kalman filter
        self.n_context = int(self.context_timedelta / self.prediction_timedelta)
        assert self.n_context >= 2, "For KalmanFilterForecaster, context_timedelta must be at least 2 times prediction_timedelta, since prediction_timedelta is the time interval at which it makes new estimates"
      
    def reset(self):
        self.model = self.create_model(
            dim_x=self.dim_x, 
            dim_z=self.dim_z,
            F=np.eye(self.dim_x), # identity matrix predicts x_t = x_t-1 + Q_t
            H=np.eye(self.dim_z) # identity matrix predicts x_t = x_t-1 + Q_t
        )
        self.last_measurement_time = None
        # store last context of w_t = x_t - x_(t-1) and v_t = z_t - H_t x_t
        self.historic_w = np.zeros((0, self.dim_x))
        self.historic_v = np.zeros((0, self.dim_z))
        self.historic_times = []
        self.initialized = False
         
    def create_scaler(self):
        return None
    
    def create_model(self, dim_x, dim_z, **kwargs):
        model = KalmanFilterWrapper(dim_x=dim_x, dim_z=dim_z)
        if "F" in kwargs:
            model.F = kwargs["F"]
        if "R" in kwargs:
            model.R = kwargs["R"]
        if "H" in kwargs:
            model.H = kwargs["H"]
        if "Q" in kwargs:
            model.Q = kwargs["Q"]
        return model
    
    def _prepare_arrays(self, historic_measurements, scaler):
        if scaler:
            historic_measurements = scaler.fit_transform(historic_measurements.to_numpy())
            
        X_train = historic_measurements[0:-self.n_context, 0]
        y_train = historic_measurements[self.n_context:, 0]
        
        return X_train, y_train
    
    def get_params(self, trial):
        return {
            # "n_prediction": np.array([[trial.suggest_float("n_prediction", 1e-3, 1e2, log=True)]]),
            # "R": np.array([[trial.suggest_float("R", 1e-3, 1e2, log=True)]]),
            # # "H": np.array([[trial.suggest_float("H", 1e-3, 1e2, log=True)]]),
            # "Q": np.array([[trial.suggest_float("Q", 1e-3, 1e2, log=True)]])
        }
    
    def predict_sample(self, n_samples: int):
        pass
        
    def _init_covariance(self, historic_noise: np.ndarray):
        cov = np.diag((1 / (self.n_context - 1)) \
            * np.sum((historic_noise[-self.n_context:, :] 
                      - (1 / self.n_context) * historic_noise[-self.n_context:, :].sum(axis=0))**2, axis=0))
        return cov
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time, return_var=False):
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] # pred_slice.filter(pred_slice == current_time + self.prediction_timedelta)
        outputs = self._get_ws_cols(historic_measurements)
        
        # multivariate with state matrix containing all horizontal and vertical wind speed measurements
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        
        if historic_measurements.select(pl.len()).item() > 1:
            print(f'one')
        # batch predict and update based on all measurements collected since last control execution
        if self.last_measurement_time is None:
            # zs = historic_measurements.filter(pl.col("time") >= current_time)\
            #                           .gather_every(n=self.n_prediction_interval)
            zs = historic_measurements.filter(((current_time - pl.col("time")).dt.total_microseconds().mod(self.prediction_interval.total_seconds() * 1e6) == 0))
        else:
            # collect all the measurments, prediction_timedelta apart, taken in the last n_controller time steps since predict_point was last called
            # zs = historic_measurements.filter(pl.col("time") >= (self.last_measurement_time + self.prediction_interval))\
            #                           .gather_every(n=self.n_prediction_interval)
            zs = historic_measurements.filter(pl.col("time") >= (self.last_measurement_time + self.prediction_interval)) \
                                      .filter(((current_time - pl.col("time")).dt.total_microseconds().mod(self.prediction_interval.total_seconds() * 1e6) == 0))
            assert zs.select(pl.len()).item() == 0 or zs.select(pl.col("time").last()).item() == self.last_measurement_time + self.prediction_interval
        
        if zs.select(pl.len()).item() == 0:
            # forecaster is called every n_controller time steps
            # n_prediction time steps may not have passed since last controller step
            # in this case, no new measurements will be available, and we can return the last state estimate
            # logging.info(f"No new measurements  available for KalmanFilterForecaster at time {current_time}, waiting on time {(self.last_measurement_time + self.prediction_timedelta)} returning last estimated state.")
            self.last_pred = self.last_pred.with_columns(time=pred_slice)
            if return_var:
                self.last_var = self.last_var.with_columns(time=pred_slice)
        else:
            
            self.last_measurement_time = zs.select(pl.col("time").last()).item()
            measurement_times = zs.select(pl.col("time")).to_series() 
            zs = zs.select(outputs).to_numpy()
            
            # logging.info(f"Adding {zs.shape[0]} new measurements to Kalman filter at time {current_time}.")
            
            # initialize state
            if not self.initialized:
                self.model.x = np.zeros_like(zs[0, :])
                self.model.P = np.eye(self.model.dim_x)
                Qs = [np.eye(self.model.dim_x)*1e-2 for j in range(zs.shape[0])]
                Rs = [np.eye(self.model.dim_z)*1e-2 for j in range(zs.shape[0])]
                self.initialized = True
            else:
                # update Qt and Rt based on previous value s of process and measurement noise
                len_w = self.historic_w.shape[0]
                len_v = self.historic_v.shape[0]
                Qs = [self._init_covariance(
                    historic_noise=self.historic_w[len_w - j - self.n_context:len_w - j, :]) for j in range(zs.shape[0]-1, -1, -1)]
                Rs = [self._init_covariance(
                    historic_noise=self.historic_v[len_v - j - self.n_context:len_v - j, :]) for j in range(zs.shape[0]-1, -1, -1)]
                # for r in Rs:
                #     np.fill_diagonal(a=r, val=np.max([np.diag(r), np.ones(r.shape[0]) * 1e-2]))
                
            init_x = self.model.x.copy()
            # use batch_filter to, on each controller sampling time
            # mean estimates from Kalman Filter
            means_p = np.zeros((zs.shape[0], self.model.dim_x)) # after predict step (prior)
            means = np.zeros((zs.shape[0], self.model.dim_x)) # after update step (posterior)
            
            # state covariances from Kalman Filter
            covariances_p = np.zeros((zs.shape[0], self.model.dim_x, self.model.dim_x)) # (prior)
            covariances = np.zeros((zs.shape[0], self.model.dim_x, self.model.dim_x)) # (posterior)
            
            # (means, covariances, means_p, covariances_p) = self.model.batch_filter(zs=z, Qs=Qt, Rs=Rt)
            # use single longer prediction time; by performing predict/update steps at time intervals == prediction_timedelta
            
            # for each measurement z at time step t, collected since the last controller step, 
            # predict the prior state x(t) for that time step based on the previous state x(t-1), 
            # and update the posterior estimate xhat(t) with the measurement
            # then the prediction is the persistance of that measurment into the future
            for i, z in enumerate(zs):
                self.model.predict(Q=Qs[i]) # outputs new prior/prediction
                means_p[i, :] = self.model.x
                covariances_p[i, :, :] = self.model.P

                self.model.update(z, R=Rs[i]) # outputs new posterior
                means[i, :] = self.model.x
                covariances[i, :, :] = self.model.P
            
            # in historic process (w) and measurment (v) noise, we only need to retain enough vectors to cover all of the measurements (spaced n_prediction apart) found in this interval of n_controller measurments, as well as the context length for each of those 
            # self.historic_times = (self.historic_times + list(measurement_times))[-int(np.ceil(self.n_controller / self.n_prediction)) - self.n_context:]
            # for testing
            # self.means_p = means_p.copy()
            # self.means = means.copy()
            # self.covariances_p = covariances_p.copy()
            # self.covariances = covariances.copy()
            
            self.historic_v = np.vstack([self.historic_v, np.atleast_2d(zs - np.matmul(means, self.model.H))])[-int(np.ceil(self.n_controller / self.n_prediction_interval)) - self.n_context:, :]
            means = np.vstack([init_x, means]) # concatenate initial guess of state on top to compute differences
            self.historic_w = np.vstack([self.historic_w, np.atleast_2d(means[1:, :] - np.matmul(means[:-1, :], self.model.F))])[-int(np.ceil(self.n_controller / self.n_prediction_interval)) - self.n_context:, :]
            
            x = means[-1, :]
            P = covariances[-1, :, :]
            
            x = np.dot(self.model.F, x) # predict step outputs new prior (in this case same, due to identity F)
            P = self.model._alpha_sq * np.dot(np.dot(self.model.F, P), self.model.F.T) + Qs[-1]
        
            pred = x 
            pred = {output: pred[o:o+1] for o, output in enumerate(outputs)}
            self.last_pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
            
            if return_var:
                var = np.diag(P)
                var = {output: var[o:o+1] for o, output in enumerate(outputs)}
                self.last_var = pl.DataFrame({"time": pred_slice}).with_columns(**var)
                
        if return_var:
            if return_pl:
                return self.last_pred, self.last_var
            else:
                return self.last_pred.to_pandas(), self.last_var.to_pandas()
        else:
            if return_pl:
                return self.last_pred
            else:
                return self.last_pred.to_pandas()
        
    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        outputs = self._get_ws_cols(historic_measurements)
        
        pred, var = self.predict_point(historic_measurements, current_time, return_var=True)
        var = var.with_columns(**{output: pl.col(output).sqrt() for output in outputs})
        
        res = pred.rename({col: f"loc_{col}" for col in outputs}).join(var.rename({col: f"sd_{col}" for col in outputs}), on="time", how="inner")
        if return_pl:
            return res
        else:
            return res.to_pandas()
    
    def _train_model(self, historic_measurements, model):
        pred = []
        # batch predict and update based on all measurements collected since last control execution
        model.x = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(historic_measurements.select(pl.len()).item()):
            z = historic_measurements.slice(i, 1).to_numpy().flatten()
            model.predict()
            model.update(z)
        
        z = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(self.n_prediction):
            model.predict()
            model.update(z)
            z = model.x
            pred.append(z[0])
            
        return model, np.array(pred)


@dataclass
class MLForecast(WindForecast):
    """Wind speed component forecasting using machine learning models."""
    is_probabilistic = True
    def __post_init__(self):
        super().__post_init__()
        self.n_prediction_interval = 1
        self.model_key = self.kwargs["model_key"]
        
        # assert isinstance(self.kwargs["estimator_class"], PyTorchLightningEstimator)
        # assert isinstance(self.kwargs["distr_output"], DistributionOutput)
        # assert isinstance(self.kwargs["lightning_module"], L.LightningModule)

        if self.use_tuned_params:
            try:
                logging.info("Getting tuned parameters")
                # study_name, backend, storage_dir
                tuned_params = get_tuned_params(storage=self.kwargs["optuna_storage"],
                                                study_name=f"tuning_{self.model_key}_{self.model_config['experiment']['run_name']}")
                logging.info(f"Declaring estimator {self.model_key.capitalize()} with tuned parameters")
                self.model_config["dataset"].update({k: v for k, v in tuned_params.items() if k in self.model_config["dataset"]})
                self.model_config["model"][self.model_key].update({k: v for k, v in tuned_params.items() if k in self.model_config["model"][self.model_key]})
                self.model_config["trainer"].update({k: v for k, v in tuned_params.items() if k in self.model_config["trainer"]})
            except Exception as e:
                logging.warning(e)
                logging.info(f"Declaring estimator {self.model_key.capitalize()} with default parameters")
        else:
            logging.info(f"Declaring estimator {self.model_key.capitalize()} with default parameters")
            
        self.model_prediction_timedelta = pd.Timedelta(seconds=self.model_config["dataset"]["prediction_length"])
            
        assert self.model_prediction_timedelta >= self.prediction_timedelta, "model is tuned for shorter prediction timedelta!"
        # self.context_timedelta = self.model_config["dataset"]["context_length"] \
        #     * pd.Timedelta(self.model_config["dataset"]["resample_freq"]).to_pytimedelta()

        # NOTE if ml method is tuned for given context length, we use that context length for that model
        self.data_module = DataModule(data_path=self.model_config["dataset"]["data_path"], 
                                      n_splits=self.model_config["dataset"]["n_splits"],
                                      continuity_groups=None, 
                                      train_split=(1.0 - self.model_config["dataset"]["val_split"] - self.model_config["dataset"]["test_split"]),
                                      val_split=self.model_config["dataset"]["val_split"], 
                                      test_split=self.model_config["dataset"]["test_split"], 
                                      prediction_length=self.model_config["dataset"]["prediction_length"], 
                                      context_length=self.model_config["dataset"]["context_length"],
                                      target_prefixes=["ws_horz", "ws_vert"], 
                                      feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                      freq=self.model_config["dataset"]["resample_freq"], 
                                      normalized=True,
                                      target_suffixes=self.model_config["dataset"]["target_turbine_ids"],
                                      per_turbine_target=self.model_config["dataset"]["per_turbine_target"], dtype=None,
                                      normalization_consts_path=self.model_config["dataset"]["normalization_consts_path"])
        self.data_module.get_dataset_info()
        self.scaler_params = self.data_module.compute_scaler_params()
        
        estimator_class = globals()[f"{self.model_key.capitalize()}Estimator"]
        lightning_module = globals()[f"{self.model_key.capitalize()}LightningModule"]
        distr_output = globals()[self.model_config["model"]["distr_output"]["class"]]
        
        estimator = estimator_class(
            freq=self.data_module.freq, 
            prediction_length=self.data_module.prediction_length,
            context_length=self.data_module.context_length,
            num_feat_dynamic_real=self.data_module.num_feat_dynamic_real, 
            num_feat_static_cat=self.data_module.num_feat_static_cat,
            cardinality=self.data_module.cardinality,
            num_feat_static_real=self.data_module.num_feat_static_real,
            input_size=self.data_module.num_target_vars,
            scaling=False,
            batch_size=self.model_config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=self.model_config["trainer"].setdefault("limit_train_batches", 50), # TODO set this to be arbitrarily high st limit train_batches dominates
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=self.data_module.context_length, min_future=self.data_module.prediction_length), # TODO should be context_len + max(seq_len) to avoid padding..
            validation_sampler=ValidationSplitSampler(min_past=self.data_module.context_length, min_future=self.data_module.prediction_length),
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=distr_output(dim=self.data_module.num_target_vars, **self.model_config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=self.model_config["trainer"],
            **self.model_config["model"][self.model_key]
        )
        self.data_module.freq = pd.Timedelta(self.data_module.freq).to_pytimedelta()
        
        metric = "val_loss_epoch"
        mode = "min"
        # log_dir = os.path.join(self.model_config["trainer"]["default_root_dir"], "lightning_logs")
        # "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/informer_aoifemac_awaken/wind_forecasting/z55orlbf/checkpoints/epoch=7-step=8000.ckpt"
        checkpoint_path = get_checkpoint(checkpoint=self.kwargs["model_checkpoint"], metric=metric, 
                                         mode=mode, log_dir=self.model_config["experiment"]["log_dir"])
        logging.info("Found pretrained model, loading...")
        model = lightning_module.load_from_checkpoint(checkpoint_path)
        transformation = estimator.create_transformation(use_lazyframe=False)
        
        self.distr_predictor = estimator.create_predictor(transformation, model, 
                                                          forecast_generator=DistributionForecastGenerator(estimator.distr_output))
        # self.sample_predictor = estimator.create_predictor(transformation, model, 
        #                                                    forecast_generator=SampleForecastGenerator())
    
    def reset(self):
        pass
    
    def _generate_test_data(self, historic_measurements: pl.DataFrame):
        # resample data to frequency model was trained on
            
        if self.data_module.freq != self.measurements_timedelta:
            if self.measurements_timedelta < self.data_module.freq:
                historic_measurements = historic_measurements.with_columns(
                    time=pl.col("time").dt.round(self.data_module.freq)
                    + pl.duration(seconds=historic_measurements.select(pl.col("time").last().dt.second() % self.data_module.freq.seconds).item()))\
                    .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                historic_measurements = historic_measurements.upsample(time_column="time", every=self.data_module.freq).fill_null(strategy="forward")
        
        historic_measurements = historic_measurements.with_columns(cs.numeric().cast(pl.Float32))
        # test_data must be iterable where each item generated is a dict with keys start, target, item_id, and feat_dynamic_real
        # this should include measurements at all turbines
        # repeats last value of feat_dynamic_reals (ie. nd_cos, nd_sin) for future_prediction TODO change this for MPC
        if self.data_module.per_turbine_target:
            test_data = (
                {
                    "item_id": f"TURBINE{turbine_id}",
                    "start": pd.Period(historic_measurements.select(pl.col("time").first()).item(), freq=self.data_module.freq), 
                    "target": historic_measurements.select(self.data_module.target_cols).to_numpy().T, 
                    "feat_dynamic_real": pl.concat([
                        historic_measurements.select(self.data_module.feat_dynamic_real_cols),
                        historic_measurements.select([pl.col(col).last().repeat_by(int(self.model_prediction_timedelta / self.data_module.freq)).explode() 
                                                      for col in self.data_module.feat_dynamic_real_cols])], how="vertical")
                } for turbine_id in self.data_module.target_suffixes)
        else:
            test_data = [{
                "start": pd.Period(historic_measurements.select(pl.col("time").first()).item(), freq=self.data_module.freq), 
                    "target": historic_measurements.select(self.data_module.target_cols).to_numpy().T, 
                    "feat_dynamic_real": pl.concat([
                        historic_measurements.select(self.data_module.feat_dynamic_real_cols),
                        historic_measurements.select([pl.col(col).last().repeat_by(int(self.model_prediction_timedelta / self.data_module.freq)).explode() 
                                                      for col in self.data_module.feat_dynamic_real_cols])], how="vertical").to_numpy().T
            }]
        return test_data
    
    def predict_sample(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time, n_samples: int):
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        historic_measurements = historic_measurements.with_columns([
            (cs.starts_with(k) * self.scaler_params["scale_"][k]) + self.scaler_params["min_"][k] for k in self.scaler_params["min_"]]
        )

        test_data = self._generate_test_data(historic_measurements)
            
        pred = self.sample_predictor.predict(test_data, num_samples=n_samples, output_distr_params=False)
        
        if self.data_module.per_turbine_target:
            # TODO test
            pred_df = pl.concat([pl.DataFrame(
                data={
                    **{"time": np.tile(pred.index.to_timestamp(), (n_samples,)),
                        "sample": np.repeat(np.arange(n_samples), (turbine_pred.prediction_length,))},
                    **{col: turbine_pred.samples[:, :, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).rename(columns={output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                              for output_type in self.data_module.target_cols}).sort_values(["sample", "time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
        else:
            pred = next(pred)
            # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
            pred_df = pl.DataFrame(
                data={
                    # "turbine_id": pred_turbine_id,
                    **{"time": np.tile(pred.index.to_timestamp(), (n_samples,)),
                       "sample": np.repeat(np.arange(n_samples), (pred.prediction_length,))},
                    **{col: pred.samples[:, :, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).sort(by=["sample", "time"])
        
        # denormalize data 
        pred_df = pred_df.with_columns([(cs.starts_with(col) - self.norm_min[c]) 
                                                    / self.norm_scale[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        pred_df = pred_df.filter(pl.col("time") <= (current_time + self.prediction_timedelta))
        # check if the data that trained the model differs from the frequency of historic_measurments
        if self.data_module.freq != self.measurements_timedelta:
            # resample historic measurements to historic_measurements frequency and return as pandas dataframe
            if self.measurements_timedelta > self.data_module.freq:
                pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_timedelta)
                                               + pl.duration(seconds=pred_df.select(pl.col("time").last().dt.second() % self.data_module.freq.seconds).item()))\
                                                              .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                pred_df = pred_df.upsample(time_column="time", every=self.measurements_timedelta).fill_null(strategy="forward")
        
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time): 
        # TODO check not including current time
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        feature_types = list(self.scaler_params["min_"].keys())
        
        if historic_measurements.select(pl.len()).item() >= self.n_context:
            historic_measurements = historic_measurements.with_columns([
                (cs.starts_with(feat_type) * self.scaler_params["scale_"][feat_type]) + self.scaler_params["min_"][feat_type]
                                                        for feat_type in feature_types])
            test_data = self._generate_test_data(historic_measurements)
            
            pred = self.distr_predictor.predict(test_data, num_samples=1, 
                                                output_distr_params={"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"})
            pred = next(pred)
            # .cast(pl.Datetime(time_unit="us"))
            
            if self.data_module.per_turbine_target:
                # TODO test
                pred_df = pl.concat([pl.DataFrame(
                    data={
                        **{"time": pred.index.to_timestamp().as_unit("us")},
                        **{col: turbine_pred.distribution.loc[:, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                    }
                ).rename({output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                                for output_type in self.data_module.target_cols}).sort_values(["time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
            else:
                # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
                pred_df = pl.DataFrame(
                    data={
                        **{"time": pred.index.to_timestamp().as_unit("us")},
                        **{col: pred.distribution.mean[:, c].cpu().numpy() for c, col in enumerate(self.data_module.target_cols)}
                    }
                ).sort(by=["time"])
            
            # denormalize data
            pred_df = pred_df.with_columns([
                (cs.starts_with(feat_type) - self.scaler_params["min_"][feat_type]) / self.scaler_params["scale_"][feat_type]
                                                        for feat_type in feature_types])
            
            pred_df = pred_df.filter(pl.col("time") <= (current_time + self.prediction_timedelta))
            # check if the data that trained the model differs from the frequency of historic_measurments
            if self.data_module.freq != self.measurements_timedelta:
                # resample historic measurements to historic_measurements frequency and return as pandas dataframe
                if self.measurements_timedelta > self.data_module.freq:
                    pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_timedelta)
                                                + pl.duration(seconds=pred_df.select(pl.col("time").last().dt.second() % self.data_module.freq.seconds).item()))\
                                                                .group_by("time").agg(cs.numeric().mean()).sort("time")
                else:
                    pred_df = pred_df.upsample(time_column="time", every=self.measurements_timedelta).fill_null(strategy="forward")
        else:
            # not enough data points to train SVR, assume persistance
            logging.info(f"Not enough data points at time {current_time} to train ML, have {historic_measurements.select(pl.len()).item()} but require {self.n_context}, assuming persistance instead.")
            pred_slice = self.get_pred_interval(current_time)
            pred_df = pl.concat([pred_slice.to_frame(), historic_measurements.slice(-1, 1).select(self.data_module.target_cols)], how="horizontal")
            
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()


    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        feature_types = list(self.scaler_params["min_"].keys())
        
        if historic_measurements.select(pl.len()).item() >= self.n_context:
            historic_measurements = historic_measurements.with_columns([
                    (cs.starts_with(feat_type) * self.scaler_params["scale_"][feat_type]) + self.scaler_params["min_"][feat_type]
                                                            for feat_type in feature_types])
            test_data = self._generate_test_data(historic_measurements)
                
            pred = self.distr_predictor.predict(test_data, num_samples=1, 
                                                output_distr_params={"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"})
            
            if self.data_module.per_turbine_target:
                # TODO TEST
                pred_df = pl.concat([pl.DataFrame(
                    data={
                        **{"time": pred.index.to_timestamp()},
                        **{f"loc_{col}": turbine_pred.distribution.mean[:, c].flatten() for c, col in enumerate(self.data_module.target_cols)},
                        **{f"sd_{col}": turbine_pred.distribution.stddev[:, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                    }
                ).rename({output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                                for output_type in self.data_module.target_cols}).sort_values(["time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
            else:
                
                pred = next(pred)
                # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
                pred_df = pl.DataFrame(
                    data={
                        **{"time": pred.index.to_timestamp()},
                        **{f"loc_{col}": pred.distribution.mean[:, c].cpu().numpy() for c, col in enumerate(self.data_module.target_cols)},
                        **{f"sd_{col}": pred.distribution.stddev[:, c].cpu().numpy() for c, col in enumerate(self.data_module.target_cols)}
                    }
                ).sort(by=["time"])
            
            # denormalize data
            pred_df = pred_df.with_columns([
                    (cs.starts_with(f"loc_{feat_type}") - self.scaler_params["min_"][feat_type]) / self.scaler_params["scale_"][feat_type]
                                                            for feat_type in feature_types])\
                             .with_columns([
                    cs.starts_with(f"sd_{feat_type}") / self.scaler_params["scale_"][feat_type]
                                                            for feat_type in feature_types])                                   
            pred_df = pred_df.filter(pl.col("time") <= (current_time + self.prediction_timedelta)) 
            # check if the data that trained the model differs from the frequency of historic_measurments
            if self.data_module.freq != self.measurements_timedelta:
                # resample historic measurements to historic_measurements frequency and return as pandas dataframe
                if self.measurements_timedelta > self.data_module.freq:
                    pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_timedelta)
                                                + pl.duration(seconds=pred_df.select(pl.col("time").last().dt.second() % self.data_module.freq.seconds).item()))\
                                                                .group_by("time").agg(cs.numeric().mean()).sort("time")
                else:
                    pred_df = pred_df.upsample(time_column="time", every=self.measurements_timedelta).fill_null(strategy="forward")
        else:
            # not enough data points to train SVR, assume persistance
            logging.info(f"Not enough data points at time {current_time} to train ML, have {historic_measurements.select(pl.len()).item()} but require {self.n_context}, assuming persistance instead.")
            pred_slice = self.get_pred_interval(current_time)
            pred_df = pl.DataFrame(
                    data={
                        **{"time": pred_slice},
                        **{f"loc_{col}": historic_measurements.select(pl.col(col).last().repeat_by(len(pred_slice)).explode())
                                                      for col in self.data_module.target_cols},
                        **{f"sd_{col}": [0] * len(pred_slice) for c, col in enumerate(self.data_module.target_cols)}
                    }
            )
            # pred_df = pl.concat([pred_slice.to_frame(), 
            #                      historic_measurements.slice(-1, 1).select(self.data_module.target_cols)
            #                      ], how="horizontal")
            
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()

@dataclass
class ARIMAForecast(WindForecast):
    """Wind speed forecasting using ARIMA model"""
    is_probabilistic = False

    def __post_init__(self):
        super().__post_init__()
        self.models = {}
        self.fitted = False

    def train(self, historic_measurements: pl.DataFrame, turbine_ids=None):
        if turbine_ids is None:
            turbine_ids = [
                col.split("_")[-1]
                for col in historic_measurements.columns
                if col.startswith("ws_horz_")
            ]
        
        for turbine_id in turbine_ids:
            logging.info(f"Training ARIMA model for {turbine_id}.")
            turbine_df = historic_measurements.select(
                pl.col("time"),
                pl.col(f"ws_horz_{turbine_id}")
            ).sort("time").unique(subset=["time"])

            ts = turbine_df.to_pandas().set_index("time")[f"ws_horz_{turbine_id}"]
            model = sm.tsa.ARIMA(ts, order=(1, 0, 0)).fit()
            
            self.models[turbine_id] = model
            self.fitted = True
    
    def model_items(self):
            """Returns the list of turbine IDs for which models are trained."""
            return self.models.keys()
       
    def predict_point(self, historic_measurements, current_time=None):
        if not self.fitted:
            raise ValueError("ARIMA model not fitted. Call train() method first.")
        
        horizon = self.n_prediction
        prediction_freq = pd.Timedelta(self.measurements_timedelta)

        if current_time is None:
            current_time = historic_measurements.select(pl.col("time").max()).item()
            
        forecast_times = pd.date_range(start=current_time + prediction_freq, periods=horizon, freq=prediction_freq)
        forecast_df = pl.DataFrame({"time": forecast_times})

        for turbine_id in self.model_items():
            model = self.models[turbine_id]
            forecast = model.forecast(steps=horizon)
            turbine_forecast = pl.DataFrame({
                    "time": forecast_times,
                    f"ws_horz_{turbine_id}": forecast
                })
            forecast_df = forecast_df.join(turbine_forecast, on="time", how="left")
        
        return forecast_df.sort("time")


           #forecast_df = pd.DataFrame({"time": forecast_times, target_col: forecast})
           #forecast_frames.append(pl.from_pandas(forecast_df))
        
       #return pl.concat(forecast_frames, how="diagonal")

def plot_wind_ts(data_df, save_path, turbine_ids="all", include_filtered_wind_dir=True, controller_timedelta=None, legend_loc="best", single_plot=False, fig=None, ax=None, case_label=None):
    #TODO only plot some turbines, not ones with overlapping yaw offsets, eg single column on farm
    colors = sns.color_palette("Paired")
    colors = [colors[1], colors[3], colors[5]]

    if not single_plot:
        fig, ax = plt.subplots(1, 1)
    
    # ax = np.atleast_1d(ax)
    
    plot_seed = data_df["continuity_group"].unique()[0]
    
    for seed in data_df["continuity_group"].unique():
        if seed != plot_seed:
            continue
        seed_df = data_df.filter(data_df["continuity_group"] == seed, data_df["feature"].is_in(["wd", "wd_filt"]))\
                         .select("turbine_id", "time", "feature", "value")
        if turbine_ids != "all":
            seed_df = seed_df.filter(pl.col("turbine_id").is_in(turbine_ids))
        
        sns.lineplot(data=seed_df, x="time", y="value", style="feature", hue="turbine_id", ax=ax)
        # if include_filtered_wind_dir:
        #     sns.lineplot(data=seed_df, x="time", y="FilteredFreestreamWindDir", label="Filtered wind dir.", color="black", linestyle="--", ax=ax[ax_idx])
    
    
    h, l = ax.get_legend_handles_labels()
    h = [handle for handle, label in zip(h, l) if label in ["wd", "wd_filt"]]
    l = ["Raw", "LPF'd"]
    ax.legend(h, l)
    ax.set(title="Wind Direction [$^\\circ$]", ylabel="", xlabel="Time", 
           xlim=(seed_df["time"].min(), seed_df["time"].max()))
     
    results_dir = os.path.dirname(save_path)
    plt.tight_layout()
    fig.savefig(save_path)
    return fig, ax

def first_ord_filter(x, time_const=35, dt=60):
    lpf_alpha = np.exp(-(1 / time_const) * dt)
    b = [1 -lpf_alpha]
    a = [1, -lpf_alpha]
    return lfilter(b, a, x)

def transform_wind(inp_df, added_wm=None, added_wd=None):
    original_cols = np.array(inp_df.collect_schema().names())
    if added_wm:
        inp_df = inp_df.with_columns((cs.starts_with("wm_") + added_wm).name.keep())
    
    if added_wd:
        inp_df = inp_df.with_columns((cs.starts_with("wd_") + added_wd).mod(360.0).name.keep())
    
    ws_horz = inp_df.select(cs.starts_with("wm_")).rename(lambda old_col: re.search("(?<=wm_)\\d+", old_col).group()) * inp_df.select(((180.0 + cs.starts_with("wd_")).radians().sin()).name.keep()).rename(lambda old_col: re.search("(?<=wd_)\\d+", old_col).group())
    ws_vert = inp_df.select(cs.starts_with("wm_")).rename(lambda old_col: re.search("(?<=wm_)\\d+", old_col).group()) * inp_df.select(((180.0 + cs.starts_with("wd_")).radians().cos()).name.keep()).rename(lambda old_col: re.search("(?<=wd_)\\d+", old_col).group())  
    
    inp_df = inp_df.with_columns(ws_horz.select(pl.all().name.prefix("ws_horz_")))
    inp_df = inp_df.with_columns(ws_vert.select(pl.all().name.prefix("ws_vert_")))
    # inp_df = inp_df.select(**{col: -pl.col(f"wm_{re.search('\\d+', col).group()}") for col in inp_df.columns if col.startswith("ws_horz_")}) 
    
    return inp_df.select(original_cols)

def make_predictions(forecaster, test_data, prediction_type):
    
    forecasts = []
    controller_times = test_data.gather_every(forecaster.n_controller).select(pl.col("time"))
    n_splits = test_data.select(pl.col("continuity_group").n_unique()).item()
    
    # for kf testing
    # means_p = []
    # means = []
    # covariances_p = []
    # covariances = []
    
    # for i, (inp, label) in enumerate(iter(test_data)):
    for d, ds in enumerate(test_data.partition_by("continuity_group")):
        # start = inp[FieldName.START].to_timestamp()
         # end = (label[FieldName.START] + label['target'].shape[1]).to_timestamp()
         
        # start = ds[FieldName.START].to_timestamp()
        # end = (ds[FieldName.START] + ds['target'].shape[1]).to_timestamp()
        start = ds.select(pl.col("time").first()).item()
        end = ds.select(pl.col("time").last()).item()
        logging.info(f"Getting predictions for {d}th split starting at {start} and ending at {end} using {forecaster.__class__.__name__} with prediction_timedelta {forecaster.prediction_timedelta}.")
        forecasts.append([])
        # split_true_wf = true_wind_field.filter(pl.col("time").is_between(start, end, closed="both"))
        split_controller_times = controller_times.filter(pl.col("time").is_between(start, end, closed="both"))
        forecaster.reset()
        for current_row in split_controller_times.iter_rows(named=True):
            
            current_time = current_row["time"]
            
            if current_time - start >= forecaster.context_timedelta:
                logging.info(f"Predicting future wind field using {forecaster.__class__.__name__} at time {current_time}/{end} of split {d}/{n_splits}.")
                if not forecaster.fitted:
                    forecaster.train(ds.filter(pl.col("time") <= current_time))
                if prediction_type == "distribution" and forecaster.is_probabilistic:
                    pred = forecaster.predict_distr(
                        ds.filter(pl.col("time") <= current_time), current_time)
                elif prediction_type == "point" or not forecaster.is_probabilistic:
                    pred = forecaster.predict_point(
                        ds.filter(pl.col("time") <= current_time), current_time)
                elif prediction_type == "sample":
                    raise NotImplementedError()
                
                # for kf testing
                # means_p.append(forecaster.means_p)
                # means.append(forecaster.means)
                # covariances_p.append(forecaster.covariances_p)
                # covariances.append(forecaster.covariances)
            
                forecasts[-1].append(pred)
            
            # if current_time >= label[FieldName.START].to_timestamp():
            #     # fetch predictions from label part
            #     forecasts[-1].append(pred)
        
        # if save_last_pred:
        #     forecasts[-1] = [wf.filter(pl.col("time") < (
        #         pl.col("time").first() + max(forecaster.controller_timedelta, forecaster.prediction_timedelta))) 
        #                                     for wf in forecasts[-1]] 
        #     forecasts[-1] = pl.concat(forecasts[-1], how="vertical_relaxed").group_by("time", maintain_order=True).agg(pl.all().last())
    
    # if save_last_pred:
    #     true = [test_data.filter(pl.col("time").is_between(
    #                 wf.select(pl.col("time").first()).item(), wf.select(pl.col("time").last()).item(), closed="both"))
    #             .with_columns(data_type=pl.lit("True"), continuity_group=pl.lit(split_idx))
    #             for split_idx, wf in enumerate(forecasts)]
    #     forecasts = [wf.with_columns(data_type=pl.lit("Forecast"), continuity_group=pl.lit(split_idx)) for split_idx, wf in enumerate(forecasts)]
    # else:
    # forecasts = np.concatenate(forecasts)
    true = test_data
    test_idx = 0
    for f in range(len(forecasts)):
        for ff in range(len(forecasts[f])):
            forecasts[f][ff] = forecasts[f][ff].with_columns(test_idx=pl.lit(test_idx))
            # time_cond = pl.col("time").is_between(
            #                 forecasts[f][ff].select(pl.col("time").first()).item(), forecasts[f][ff].select(pl.col("time").last()).item(), 
            #                 closed="both")
            # true = true.with_columns([
            #     pl.when(time_cond)\
            #         .then(pl.lit(test_idx))\
            #         .otherwise(pl.col("test_idx"))\
            #         .alias("test_idx"), 
            #     pl.when(time_cond)\
            #         .then(pl.lit(f))\
            #         .otherwise(pl.col("continuity_group"))\
            #         .alias("continuity_group")])
            test_idx += 1
        forecasts[f] = pl.concat(forecasts[f], how="vertical_relaxed")
    forecasts = pl.concat(forecasts, how="vertical_relaxed").with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")))
    forecasts = forecasts.filter(pl.col("time").is_in(true.select(pl.col("time"))))
    # true = true.filter(pl.col("time").is_in(forecasts.select(pl.col("time"))))
    
    # true = true.filter(pl.col("time").is_between(
    #     forecasts.select(pl.col("time").first()).item(), forecasts.select(pl.col("time").last()).item(), closed="both"))

    if False:
        means_p = np.vstack(means_p)
        means = np.vstack(means)
        covariances_p = np.vstack([np.diag(c) for cov in covariances_p for c in cov])
        covariances = np.vstack([np.diag(c) for cov in covariances for c in cov])
        df = pd.concat([
            pd.DataFrame(data=means_p, columns=[f"mean_p_{i}" for i in range(means_p.shape[1])]),
            pd.DataFrame(data=means, columns=[f"mean_{i}" for i in range(means.shape[1])]),
            pd.DataFrame(data=covariances_p, columns=[f"covariance_p_{i}" for i in range(covariances_p.shape[1])]),
            pd.DataFrame(data=covariances, columns=[f"covariance_{i}" for i in range(covariances.shape[1])])
        ], axis=1)
        df["time"] = split_controller_times.to_pandas()

        fig, ax = plt.subplots(2, 1, sharex=True)
        sns.lineplot(df, x="time", y="mean_p_0", marker="o", ax=ax[0], linestyle=":", color=sns.color_palette()[0], label="Prior")
        sns.lineplot(df, x="time", y="mean_p_7", marker="o", ax=ax[1], linestyle=":", color=sns.color_palette()[0])
        sns.lineplot(df, x="time", y="mean_0", marker="o", ax=ax[0], linestyle="-", color=sns.color_palette()[1], label="Posterior")
        sns.lineplot(df, x="time", y="mean_7", marker="o", ax=ax[1], linestyle="-", color=sns.color_palette()[1])
        ax[0].legend()
        color = sns.color_palette()[0]
        ax[0].plot(df["time"], df["mean_p_0"] - 1*np.sqrt(df["covariance_p_0"]), 
            alpha=0.2, color=color
        )
        ax[0].plot(df["time"], df["mean_p_0"] + 1*np.sqrt(df["covariance_p_0"]), 
            alpha=0.2, color=color
        )
        
        ax[1].plot(
            df["time"], df["mean_p_7"] - 1*np.sqrt(df["covariance_p_7"]),
            alpha=0.2, color=color
        )
        ax[1].plot(
            df["time"], df["mean_p_7"] + 1*np.sqrt(df["covariance_p_7"]), 
            alpha=0.2, color=color
        )
        
        color = sns.color_palette()[1]
        ax[0].plot(
            df["time"], df["mean_0"] - 1*np.sqrt(df["covariance_0"]),
            alpha=0.2, color=color
        )
        ax[0].plot(
            df["time"], df["mean_0"] + 1*np.sqrt(df["covariance_0"]), 
            alpha=0.2, color=color
        )
        
        ax[1].plot(
            df["time"], df["mean_7"] - 1*np.sqrt(df["covariance_7"]),
            alpha=0.2, color=color
        )
        ax[1].plot(
            df["time"], df["mean_7"] + 1*np.sqrt(df["covariance_7"]), 
            alpha=0.2, color=color
        )
        ax[1].set_xlabel("Time (s)")
        ax[0].set_title("$u$ Wind Speed (m/s)")
        ax[1].set_title("$v$ Wind Speed (m/s)")
        ax[0].set_ylabel("")
        ax[1].set_ylabel("")
    
    
    return forecasts

def generate_wind_field_df(datasets, target_cols, feat_dynamic_real_cols):
    full_target = np.concatenate([ds[FieldName.TARGET] for ds in datasets], axis=-1)
    full_feat_dynamic_reals = np.concatenate([ds[FieldName.FEAT_DYNAMIC_REAL] for ds in datasets], axis=-1)[:, :full_target.shape[1]]
    full_splits = np.atleast_2d(np.hstack([np.repeat(int(re.search("(?<=SPLIT)\\d+", ds[FieldName.ITEM_ID]).group()), (ds[FieldName.TARGET].shape[1],)) for ds in datasets])).astype(int)
    index = pd.concat([period_index(
        {FieldName.START: ds[FieldName.START], FieldName.TARGET: ds[FieldName.TARGET]}
        ).to_series() for ds in datasets]).index

    wf = pd.DataFrame(np.concatenate([full_splits, full_target, full_feat_dynamic_reals], axis=0).transpose(),
                                    columns=["continuity_group"] + target_cols + feat_dynamic_real_cols,
                                    index=index)
    wf["continuity_group"] = wf["continuity_group"].astype(int)
     
    wf = wf.reset_index(names="time")
    wf["time"] = wf["time"].dt.to_timestamp()
    return pl.from_pandas(wf)

def unpivot_df(df, turbine_signature):
    return df.unpivot(index=["metric", "test_idx"], variable_name="feature", value_name="score")\
            .with_columns(turbine_id=pl.col("feature").str.extract(f"({turbine_signature})$"),
                          feature_type=pl.col("feature").str.extract(f"(.*)_{turbine_signature}$"))\
            .drop("feature")

def generate_forecaster_results(forecaster, data_module, evaluator, test_data, prediction_type):
    
    forecast_df = make_predictions(forecaster=forecaster, test_data=test_data, 
                                            prediction_type=prediction_type)
    
    forecast_df = forecast_df.partition_by("test_idx")
    if prediction_type == "distribution" and forecaster.is_probabilistic:
        value_vars = ["nd_cos", "nd_sin", "loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"]
        target_vars = ["loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"] 
        distributions = []
        for split_idx, wf in enumerate(forecast_df):
            loc = Tensor(wf.select([cs.starts_with(feat_type) & cs.contains("loc") for feat_type in target_vars]).to_numpy())
            cov = Tensor(np.apply_along_axis(
                        np.diag, axis=-1, 
                        arr=wf.select([cs.starts_with(feat_type) & cs.contains("sd_") for feat_type in target_vars]).to_numpy()**2))
            
            distr = DistributionForecast(
                distribution=MultivariateNormal(loc=loc, covariance_matrix=cov), 
                start_date=pd.Period(wf.select(pl.col("time").first()).item(), freq=data_module.freq), 
                item_id=f"SPLIT{split_idx}")
            distributions.append(distr)
        forecasts = distributions
    else:
        value_vars = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
        target_vars = ["ws_horz", "ws_vert"] 
        
        forecasts = [SampleForecast(
            samples=wf.select([cs.starts_with(feat_type) for feat_type in target_vars]).to_numpy()[np.newaxis, :, :], 
            start_date=pd.Period(wf.select(pl.col("time").first()).item(), freq=data_module.freq), 
            item_id=f"SPLIT{split_idx}") for split_idx, wf in enumerate(forecast_df)]
    
    true_df_pd = test_data.to_pandas()
    true_df_pd = true_df_pd.set_index(pd.PeriodIndex(true_df_pd["time"].dt.to_period(freq=data_module.freq)))[data_module.target_cols]\
                        .rename(columns={src: s for s, src in enumerate(data_module.target_cols)})
    # agg_metrics, ts_metrics = evaluator(
    #             islice(repeat(true_df_pd), len(forecast_df)),
    #             forecasts, 
    #             num_series=data_module.num_target_vars,
    #             include_metrics=[])
    
    # TODO test with sample forecast and distribution
    agg_metrics = []
    for fdf in forecast_df:
        test_idx = fdf.select(pl.col("test_idx").first()).item()
        mean_vars = [c for c in target_vars if "ws_" in c]
        
        tdf = test_data.filter(pl.col("time").is_in(fdf.select(pl.col("time"))))\
                       .select("time", cs.starts_with("ws_horz"), cs.starts_with("ws_vert"))
        
        fdf = fdf.select(pl.col("time"), *[cs.starts_with(tgt) for tgt in target_vars])
        
        
        mse = (tdf.select(data_module.target_cols) - fdf.select(cs.starts_with(mean_vars)))\
            .select(pl.all().pow(2).mean().sqrt()).with_columns(metric=pl.lit("RMSE"), test_idx=pl.lit(test_idx))
        mse = unpivot_df(mse, forecaster.turbine_signature)
            
        mae = (tdf.select(data_module.target_cols) - fdf.select(cs.starts_with(mean_vars)))\
            .select(pl.all().abs().mean()).with_columns(metric=pl.lit("MAE"), test_idx=pl.lit(test_idx))
        mae = unpivot_df(mae, forecaster.turbine_signature)
            
        agg_metrics += [mse, mae]
    
    fdf = forecast_df = pl.concat(forecast_df, how="vertical")
    tdf = test_data.filter(pl.col("time").is_in(forecast_df.select(pl.col("time"))))\
                       .select("time", cs.starts_with("ws_horz"), cs.starts_with("ws_vert"))
    if prediction_type == "distribution" and forecaster.is_probabilistic:
        
        picp = pl.DataFrame(
            data=np.atleast_2d(pi_coverage_probability(fdf.select(cs.starts_with("loc_")).to_numpy(), tdf.select(data_module.target_cols).to_numpy(), fdf.select(cs.starts_with("sd_")).to_numpy())),
            schema=data_module.target_cols)\
                .with_columns(metric=pl.lit("PICP"), test_idx=pl.lit(-1))
        picp = unpivot_df(picp, forecaster.turbine_signature)
        
        pinaw = pl.DataFrame(
            data=np.atleast_2d(pi_normalized_average_width(fdf.select(cs.starts_with("loc_")).to_numpy(), tdf.select(data_module.target_cols).to_numpy(), fdf.select(cs.starts_with("sd_")).to_numpy())),
            schema=data_module.target_cols)\
                    .with_columns(metric=pl.lit("PINAW"), test_idx=pl.lit(-1))
        pinaw = unpivot_df(pinaw, forecaster.turbine_signature)
        
        cwc = pl.DataFrame(
            data=np.atleast_2d(coverage_width_criterion(fdf.select(cs.starts_with("loc_")).to_numpy(), tdf.select(data_module.target_cols).to_numpy(), fdf.select(cs.starts_with("sd_")).to_numpy())),
            schema=data_module.target_cols)\
                .with_columns(metric=pl.lit("CWC"), test_idx=pl.lit(-1))
        cwc = unpivot_df(cwc, forecaster.turbine_signature)
        
        crps = pl.DataFrame(
            data=np.atleast_2d(continuous_ranked_probability_score_gaussian(fdf.select(cs.starts_with("loc_")).to_numpy(), tdf.select(data_module.target_cols).to_numpy(), fdf.select(cs.starts_with("sd_")).to_numpy())),
            schema=data_module.target_cols)\
                .with_columns(metric=pl.lit("CRPS"), test_idx=pl.lit(-1))
        crps = unpivot_df(crps, forecaster.turbine_signature)
        
        agg_metrics += [picp, pinaw, cwc, crps]
    
    # evaluator.get_metrics_per_ts(ts, forecast, include_metrics=include_metrics)
    # evaluator.get_aggregate_metrics(metrics_per_ts, include_metrics=include_metrics)
    agg_metrics = pl.concat(agg_metrics, how="vertical")
    agg_metrics = pl.concat([
        agg_metrics.select(["metric", "test_idx", "feature_type", "turbine_id", "score"]), 
        agg_metrics.group_by(["metric", "test_idx", "feature_type"], maintain_order=True).agg(pl.col("score").mean()).with_columns(turbine_id=pl.lit("all")).select(["metric", "test_idx", "feature_type", "turbine_id", "score"]),
        agg_metrics.group_by(["metric", "turbine_id", "feature_type"], maintain_order=True).agg(pl.col("score").mean()).with_columns(test_idx=pl.lit(-1)).select(["metric", "test_idx", "feature_type", "turbine_id", "score"])
        ], how="vertical").sort(["metric", "test_idx", "feature_type"])
    agg_metrics = pl.concat([
        agg_metrics,
        agg_metrics.group_by(["metric", "feature_type"], maintain_order=True).agg(pl.col("score").mean()).with_columns(test_idx=pl.lit(-1), turbine_id=pl.lit("all")).select(["metric", "test_idx", "feature_type", "turbine_id", "score"])
    ])
    return forecast_df, agg_metrics

def plot_score_vs_prediction_dt(agg_df, metrics, ax_indices):
    
    n_axes = len(ax_indices)
    left_metrics = [met for i, met in zip(ax_indices, metrics) if i == 0]
    right_metrics = [met for i, met in zip(ax_indices, metrics) if i == 1]
    
    # fig, ax1 = plt.subplots(1, 1)
    fig = plt.figure()
    if left_metrics and right_metrics:
        # TODO will have same color for different metrics over pos/neg
        ax1 = sns.scatterplot(agg_df.filter(pl.col("metric").is_in(left_metrics)).to_pandas(),
                        y="score", x="prediction_timedelta", style="forecaster", hue="metric", s=100)
        ax2 = ax1.twinx()
    
        sns.scatterplot(agg_df.filter(pl.col("metric").is_in(right_metrics)).to_pandas(),
                        y="score", x="prediction_timedelta", style="forecaster", hue="metric", ax=ax2, s=100)
        ax = ax2
        
    elif left_metrics or right_metrics:
        mets = left_metrics or right_metrics
        ax1 = sns.scatterplot(agg_df.filter(pl.col("metric").is_in(mets)).to_pandas(),
                        y="score", x="prediction_timedelta", style="forecaster", hue="metric", s=100)
        ax = ax1
        
    if left_metrics:
        ax1.set_ylabel(f"Score for {', '.join(left_metrics)} (-)")
        h1, l1 = ax1.get_legend_handles_labels()
        
    if right_metrics:
        ax2.set_ylabel(f"Score for {', '.join(right_metrics)} (-)")
        h2, l2 = ax2.get_legend_handles_labels()
        
    ax.set_xlabel("Prediction Length (s)")
    
    if left_metrics and right_metrics:
        l = l1[:l1.index("metric")] + l1[l1.index("metric"):] + l2[l2.index("metric")+1:]
        h = h1[:l1.index("metric")] + h1[l1.index("metric"):] + h2[l2.index("metric")+1:]
    elif left_metrics:
        l = l1
        h = h1
    else:
        l = l2
        h = h2
    
    new_labels = [" ".join(re.findall("[A-Z][^A-Z]*", re.search("\\w+(?=Forecast)", label).group())) 
                  if "Forecast" in label else (label.capitalize() if not label[0].isupper() else label).replace("_", " ") for label in l]
    
    l1, l2 = new_labels[:new_labels.index("Metric")], new_labels[new_labels.index("Metric"):]
    h1, h2 = h[:new_labels.index("Metric")], h[new_labels.index("Metric"):]
    leg1 = ax.legend(h1, l1, loc='upper right', bbox_to_anchor=(0.48, 1), frameon=False)
    leg2 = plt.legend(h2, l2, loc='upper left', bbox_to_anchor=(0.51, 1), frameon=False)
    ax.add_artist(leg1)
    plt.tight_layout()
    return fig

def plot_score_vs_forecaster(agg_df, metrics, good_directions):
    
    n_axes = len(np.unique(np.sign(good_directions)))
    pos_metrics = [met for sign, met in zip(good_directions, metrics) if sign > 0]
    neg_metrics = [met for sign, met in zip(good_directions, metrics) if sign < 0]
    
    # fig, ax = plt.subplots(1, 1)
    if pos_metrics and neg_metrics:
        # TODO will have same color for different metrics over pos/neg
        ax1 = sns.catplot(agg_df.loc[agg_df["metric"].isin(pos_metrics), :],
                    kind="bar",
                    hue="metric", x="forecaster", y="score")
        sub_ax1 = ax1.ax
        
        sub_ax2 = sub_ax1.twinx()
        ax2 = sns.catplot(agg_df.loc[agg_df["metric"].isin(neg_metrics), :],
                    kind="bar",
                    hue="metric", x="forecaster", y="score", ax=sub_ax2)
        ax = ax2
    elif pos_metrics:
        ax1 = sns.catplot(agg_df.loc[agg_df["metric"].isin(pos_metrics), :],
                    kind="bar",
                    hue="metric", x="forecaster", y="score")
        ax = ax1
    elif neg_metrics:
        ax2 = sns.catplot(agg_df.loc[agg_df["metric"].isin(neg_metrics), :],
                    kind="bar",
                    hue="metric", x="forecaster", y="score")
        ax = ax2
        
    if pos_metrics:
        ax1.ax.set_ylabel(f"Score for {', '.join(pos_metrics)} (-)")
        h1, l1 = ax1.ax.get_legend_handles_labels()
        
    if neg_metrics:
        ax2.ax.set_ylabel(f"Score for {', '.join(neg_metrics)} (-)")
        h2, l2 = ax2.ax.get_legend_handles_labels()
    
    if pos_metrics and neg_metrics:
        l = l1[:l1.index("metric")] + l1[l1.index("metric"):] + l2[l2.index("metric")+1:]
        h = h1[:l1.index("metric")] + h1[l1.index("metric"):] + h2[l2.index("metric")+1:]
    elif pos_metrics:
        l = l1
        h = h1
    else:
        l = l2
        h = h2
    
    ax.ax.set_xlabel("Forecaster")
    
    ax.legend.set_visible(False)
    # new_labels = [(re.search("\\w+(?=Forecast)", label).group() if "Forecast" in label else (label.capitalize() if not label[0].isupper() else label).replace("_", " ")) for label in l]
    ax.ax.set_xticklabels([" ".join(re.findall("[A-Z][^A-Z]*", re.search("\\w+(?=Forecast)", label._text).group())) for label in ax.ax.get_xticklabels()], rotation=45)
    new_labels = [(label.capitalize() if not label[0].isupper() else label).replace("_", " ") for label in l]
    ax.ax.legend(h, new_labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="ModelTuning")
    parser.add_argument("-mcnf", "--model_config", type=str, 
                        help="Filepath to model configuration with experiment, optuna, dataset, model, callbacks, trainer keys.")
    parser.add_argument("-dcnf", "--data_config", type=str, 
                        help="Filepath to data preprocessing configuration with filters, feature_mapping, turbine_signature, nacelle_calibration_turbine_pairs, dt, raw_data_directory, processed_data_path, raw_data_file_signature, turbine_input_path, farm_input_path keys.")
    parser.add_argument("-fd", "--fig_dir", type=str, 
                        help="Directory to save plots to.", default="./")
    parser.add_argument("-m", "--model", #type=str, 
                        # choices=["perfect", "persistence", "svr", "kf", "informer", "autoformer", "spacetimeformer", "sf"], 
                        required=True, nargs="+",
                        help="Which model(s) to simulate, compute score for, and plot.")
    parser.add_argument("-st", "--simulation_timestep", 
                        required=True, type=int,
                        help="Simulation time step to use (sec)")
    parser.add_argument("-rrv", "--rerun_validation",
                        action="store_true",
                        help="Whether to repeat validation for results that have already been stored.")
    parser.add_argument("-pi", "--prediction_interval", 
                        required=False, nargs="+", default=None,
                        help="Number of seconds to use as prediction_timedelta..")
    parser.add_argument("-mp", "--multiprocessor", type=str, choices=["mpi", "cf"], help="which multiprocessing backend to use, omit for sequential processing", 
                        required=False, default=None)
    parser.add_argument("-msp", "--max_splits", type=int, required=False, default=None,
                        help="Number of test splits to use.")
    parser.add_argument("-mst", "--max_steps", type=int, required=False, default=None,
                        help="Number of time steps to use.")
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default="latest", 
                        help="Which checkpoint to use: can be equal to 'latest', 'best', or an existing checkpoint path.")
    parser.add_argument("-pt", "--prediction_type", type=str, choices=["point", "sample", "distribution"], default="point",
                        help="Whether to make a point, sample, or distribution parameter prediction.")
    parser.add_argument("-awm", "--added_wind_mag", type=float, required=False, default=0.0,
                        help="Wind magnitude to add to all values (after transformation to wind magnitude and direction).")
    parser.add_argument("-awd", "--added_wind_dir", type=float, required=False, default=0.0,
                        help="Wind direction to add to all values (after transformation to wind magnitude and direction).")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot time series outputs.")
    parser.add_argument("-tp", "--use_tuned_params", action="store_true",
                        help="Use parameters tuned from Optuna optimization, otherwise use defaults set in Module class.")
    parser.add_argument("-tm", "--use_trained_models", action="store_true",
                        help="Use parameters trained and stored for models that require training, e.g. SVR, read existing trained models from file.")
    parser.add_argument("--plot_arima", action="store_true", help="Run ARIMA and plot historic wind speed data")
    args = parser.parse_args()
    
    assert all(model in ["perfect", "persistence", "svr", "kf", "informer", "autoformer", "spacetimeformer", "tactis", "sf", "arima"] for model in args.model)
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    
    TRANSFORM_WIND = {"added_wm": args.added_wind_mag, "added_wd": args.added_wind_dir}
    # args.fig_dir = os.path.join(os.path.dirname(whoc_file), "..", "examples", "wind_forecasting")
    
    os.makedirs(args.fig_dir, exist_ok=True)
     
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
    
    
    if args.prediction_interval is None:
        prediction_timedelta = [pd.Timedelta(seconds=model_config["dataset"]["prediction_length"])]
    else:
        prediction_timedelta = [pd.Timedelta(seconds=int(i)).to_pytimedelta() for i in args.prediction_interval]
        
    context_timedelta = pd.Timedelta(seconds=model_config["dataset"]["context_length"])
    measurements_timedelta = pd.Timedelta(seconds=args.simulation_timestep)
    # measurements_timedelta = pd.Timedelta(model_config["dataset"]["resample_freq"])
    
    controller_timedelta = max(pd.Timedelta(5, unit="s"), measurements_timedelta)

    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
        
    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
    
    id_var_selector = pl.exclude(
        f"^ws_horz_{turbine_signature}$", f"^ws_vert_{turbine_signature}$", 
                f"^nd_cos_{turbine_signature}$", f"^nd_sin_{turbine_signature}$",
                f"^loc_ws_horz_{turbine_signature}$", f"^loc_ws_vert_{turbine_signature}$",
                f"^sd_ws_horz_{turbine_signature}$", f"^sd_ws_vert_{turbine_signature}$")
    
    fmodel = FlorisModel(data_config["farm_input_path"])
    
    logging.info("Creating datasets")
    data_module = DataModule(data_path=model_config["dataset"]["data_path"], 
                             normalization_consts_path=model_config["dataset"]["normalization_consts_path"],
                             normalized=False, 
                             n_splits=1, #model_config["dataset"]["n_splits"],
                             continuity_groups=None, train_split=(1.0 - model_config["dataset"]["val_split"] - model_config["dataset"]["test_split"]),
                             val_split=model_config["dataset"]["val_split"], test_split=model_config["dataset"]["test_split"],
                             prediction_length=model_config["dataset"]["prediction_length"], context_length=model_config["dataset"]["context_length"],
                             target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                             freq=f"{int(measurements_timedelta.total_seconds())}s", 
                             target_suffixes=model_config["dataset"]["target_turbine_ids"],
                             per_turbine_target=model_config["dataset"]["per_turbine_target"], as_lazyframe=False, dtype=pl.Float32)
    
    if not os.path.exists(data_module.train_ready_data_path):
        data_module.generate_datasets()
    
    # true_wind_field = data_module.generate_splits(save=True, reload=False, splits=["test"])._df.collect()
    data_module.generate_splits(save=True, reload=False, splits=["test"])
    
    data_module.test_dataset = sorted(data_module.test_dataset, key=lambda ds: ds["target"].shape[1], reverse=True)
    if args.max_splits:
        test_data = data_module.test_dataset[:args.max_splits]
    else:
        test_data = data_module.test_dataset
    
    if args.max_steps:
        assert args.max_steps > int((context_timedelta + max(prediction_timedelta)) / measurements_timedelta), f"max_steps, if provided, must allow for context_timedelta + max(prediction_timedelta) = {int((context_timedelta + max(prediction_timedelta)) / measurements_timedelta)}"
        test_data = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in test_data]
    
    test_data = generate_wind_field_df(test_data, data_module.target_cols, data_module.feat_dynamic_real_cols)
    # window_length = model_config["dataset"]["prediction_length"] + model_config["dataset"].get("lead_time", 0)
    # window_length = int(test_data[0]["target"].shape[1] * (2/3))
    # _, test_template = split(test_data, offset=-window_length)
    # test_data = test_template.generate_instances(window_length, windows=1)
    delattr(data_module, "test_dataset")
    gc.collect()
    
    # assert pd.Timedelta(test_data[0]["start"].freq) == measurements_timedelta
    assert pd.Timedelta(test_data.select(pl.col("time").diff()).slice(1,1).item()) == measurements_timedelta
    # assert test_data.select(pl.col("time").slice(0, 2).diff()).slice(1,1).item() == measurements_timedelta
   
    custom_eval_fn = {
                "PICP": (pi_coverage_probability, "mean", "mean"),
                "PINAW": (pi_normalized_average_width, "mean", "mean"),
                "CWC": (coverage_width_criterion, "mean", "mean"),
                "CRPS": (continuous_ranked_probability_score_gaussian, "mean", "mean"),
    }
    evaluator = MultivariateEvaluator(
        custom_eval_fn=custom_eval_fn,
        num_workers=mp.cpu_count() if args.multiprocessor == "cf" else None,
    )
            
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    forecasters = []
    ## GENERATE PERFECT PREVIEW \
    if "perfect" in args.model:
        for td in prediction_timedelta:
            forecaster = PerfectForecast(
                measurements_timedelta=measurements_timedelta,
                controller_timedelta=controller_timedelta,
                prediction_timedelta=td,
                context_timedelta=context_timedelta,
                true_wind_field=test_data,
                fmodel=fmodel,
                tid2idx_mapping=tid2idx_mapping,
                turbine_signature=turbine_signature,
                use_tuned_params=False,
                model_config=None,
                kwargs={}
            )
                                
            forecasters.append(forecaster)
    
    ## GENERATE PERSISTENT PREVIEW
    if "persistence" in args.model:
        for td in prediction_timedelta:
            forecaster = PersistenceForecast(measurements_timedelta=measurements_timedelta,
                                                    controller_timedelta=controller_timedelta,
                                                    prediction_timedelta=td,
                                                    context_timedelta=context_timedelta,
                                                    fmodel=fmodel,
                                                    true_wind_field=None,
                                                    tid2idx_mapping=tid2idx_mapping,
                                                    turbine_signature=turbine_signature,
                                                    use_tuned_params=False,
                                                    model_config=None,
                                                    kwargs={})

            forecasters.append(forecaster)
        
    ## GENERATE SVR PREVIEW
    if "svr" in args.model:
        
        for td in prediction_timedelta:
            # TODO DEBUG
            db_setup_params = generate_df_setup_params("svr", model_config.update({"prediction_timedelta": td.total_seconds()}),)
            optuna_storage = setup_optuna_storage(
                db_setup_params=db_setup_params,
                restart_tuning=False,
                rank=rank
            )
            forecaster = SVRForecast(measurements_timedelta=measurements_timedelta,
                                    controller_timedelta=controller_timedelta,
                                    prediction_timedelta=td,
                                    context_timedelta=context_timedelta,
                                    fmodel=fmodel,
                                    true_wind_field=None,
                                    kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200,
                                                n_neighboring_turbines=3, max_n_samples=None, 
                                                study_name=f"svr_{model_config['experiment']['run_name']}",
                                                use_trained_models=args.use_trained_models,
                                                optuna_storage=optuna_storage,
                                                model_save_dir=data_config["model_save_dir"]),
                                    tid2idx_mapping=tid2idx_mapping,
                                    turbine_signature=turbine_signature,
                                    use_tuned_params=args.use_tuned_params,
                                    model_config=model_config)
            
            forecasters.append(forecaster)
        
        
    ## GENERATE KF PREVIEW 
    if "kf" in args.model:
        # tune this use single, longer, prediction time, since we have only identity state transition matrix, and must use final posterior only prediction
        for td in prediction_timedelta:
            forecaster = KalmanFilterForecast(measurements_timedelta=measurements_timedelta,
                                                controller_timedelta=controller_timedelta,
                                                prediction_timedelta=td, 
                                                context_timedelta=td*2,
                                                fmodel=fmodel,
                                                true_wind_field=None,
                                                tid2idx_mapping=tid2idx_mapping,
                                                turbine_signature=turbine_signature,
                                                use_tuned_params=False,
                                                model_config=model_config,
                                                kwargs={})
            forecasters.append(forecaster)
        
    ## GENERATE KF PREVIEW 
    if "sf" in args.model:
        # tune this use single, longer, prediction time, since we have only identity state transition matrix, and must use final posterior only prediction
        for td in prediction_timedelta:
            forecaster = SpatialFilterForecast(measurements_timedelta=measurements_timedelta,
                                                controller_timedelta=controller_timedelta,
                                                prediction_timedelta=td, 
                                                context_timedelta=context_timedelta,
                                                fmodel=fmodel,
                                                true_wind_field=None,
                                                tid2idx_mapping=tid2idx_mapping,
                                                turbine_signature=turbine_signature,
                                                use_tuned_params=False,
                                                model_config=model_config,
                                                kwargs=dict(n_neighboring_turbines=6))
            forecasters.append(forecaster)
        
    ## GENERATE ML PREVIEW
    elif any(ml_model in args.model for ml_model in ["informer", "autoformer", "spacetimeformer", "tactis"]):
        # "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/informer_aoifemac_awaken/wind_forecasting/z55orlbf/checkpoints/epoch=3-step=4000.ckpt"
            
        # TODO DEBUG
        db_setup_params = generate_df_setup_params(args.model, model_config)
        optuna_storage = setup_optuna_storage(
            db_setup_params=db_setup_params,
            restart_tuning=False,
            rank=rank
        )
        for model in [ml_model for ml_model in args.model if ml_model in ["informer", "autoformer", "spacetimeformer", "tactis"]]:
            # TODO DEBUG
            db_setup_params = generate_df_setup_params(model, model_config)
            optuna_storage = setup_optuna_storage(
                db_setup_params=db_setup_params,
                restart_tuning=False,
                rank=rank
            )
            forecaster = MLForecast(measurements_timedelta=measurements_timedelta,
                                    controller_timedelta=controller_timedelta,
                                    prediction_timedelta=pd.Timedelta(seconds=model_config["dataset"]["prediction_length"]),
                                    context_timedelta=pd.Timedelta(seconds=model_config["dataset"]["context_length"]),
                                    fmodel=fmodel,
                                    true_wind_field=None,
                                    tid2idx_mapping=tid2idx_mapping,
                                    turbine_signature=turbine_signature,
                                    use_tuned_params=True,
                                    model_config=model_config,
                                    kwargs=dict(model_key=model,
                                                model_checkpoint=args.checkpoint,
                                                optuna_storage=optuna_storage,
                                                study_name=db_setup_params["study_name"])
                                    )
        forecasters.append(forecaster)

    ## GENERATE ARIMA PREVIEW
    if "arima" in args.model:
        for td in prediction_timedelta:
            forecaster = ARIMAForecast(measurements_timedelta=measurements_timedelta,
                                        controller_timedelta=controller_timedelta,
                                        prediction_timedelta=td, 
                                        context_timedelta=context_timedelta,
                                        fmodel=fmodel,
                                        true_wind_field=None,
                                        tid2idx_mapping=tid2idx_mapping,
                                        turbine_signature=turbine_signature,
                                        use_tuned_params=False,
                                        model_config=model_config,
                                        kwargs={})
            forecasters.append(forecaster)
        
    # if args.multiprocessor is not None:
    #     if args.multiprocessor == "mpi":
    #         comm_size = MPI.COMM_WORLD.Get_size()
    #         executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
    #     elif args.multiprocessor == "cf":
    #         executor = ProcessPoolExecutor()
    #     with executor as ex:
    #         if args.multiprocessor == "mpi":
    #             ex.max_workers = comm_size
            
    #         test_futures = [ex.submit(generate_forecaster_results, forecaster=forecaster, 
    #                                 data_module=data_module, evaluator=evaluator, 
    #                                 test_data=test_data, 
    #                                 prediction_type=args.prediction_type) 
    #                    for forecaster in forecasters]
            
    #         results = [dict([(k, v) for k, v in chain(
    #             zip(["true_df", "forecast_df", "agg_metrics"], fut.result()), 
    #             zip(["forecaster_name", "prediction_timedelta"], [forecaster.__class__.__name__, forecaster.prediction_timedelta.total_seconds()]))]) 
    #                    for forecaster, fut in zip(forecasters, test_futures)] # agg_metrics, ts_metrics, forecast_fig
    # else:
    results = []
    for forecaster in forecasters:
        save_dir = os.path.join(os.path.dirname(model_config["dataset"]["data_path"]), "validation_results", 
                                forecaster.__class__.__name__,
                                str(forecaster.prediction_timedelta.total_seconds()))
        os.makedirs(save_dir, exist_ok=True)
        forecast_path = os.path.join(save_dir, "forecast.parquet")
        agg_metric_path = os.path.join(save_dir, "agg_metrics.parquet")
        if args.rerun_validation or not os.path.exists(forecast_path) or not os.path.exists(agg_metric_path):
            forecast_df, agg_metrics = generate_forecaster_results(
                forecaster=forecaster, data_module=data_module, 
                evaluator=evaluator, test_data=test_data,
                prediction_type=args.prediction_type)
            results.append({
                "forecaster_name": forecaster.__class__.__name__,
                "forecast_df": forecast_df,
                "agg_metrics": agg_metrics, 
                "prediction_timedelta": forecaster.prediction_timedelta.total_seconds()
                })
            
            results[-1]["forecast_df"].write_parquet(forecast_path)
            results[-1]["agg_metrics"].write_parquet(agg_metric_path)
        else:
            results.append({
                "forecaster_name": forecaster.__class__.__name__,
                "forecast_df": pl.read_parquet(forecast_path),
                "agg_metrics": pl.read_parquet(agg_metric_path), 
                "prediction_timedelta": forecaster.prediction_timedelta.total_seconds()
                })
    # results[0]["agg_metrics"].group_by(["test_idx", "feature_type"], maintain_order=True).agg(pl.col("score").mean()).with_columns(turbine_id=pl.lit("all"))
    
    for f, forecaster in enumerate(forecasters):
        save_dir = os.path.join(os.path.dirname(model_config["dataset"]["data_path"]), "validation_results", 
                                forecaster.__class__.__name__,
                                str(forecaster.prediction_timedelta.total_seconds()))
        if args.prediction_type == "distribution" and forecaster.is_probabilistic:
            value_vars = ["nd_cos", "nd_sin", "loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"]
            target_vars = ["loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"]
        else:
            value_vars = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
            target_vars = ["ws_horz", "ws_vert"]
    
        id_vars = results[f]["forecast_df"].select(id_var_selector).columns  
        forecasts_long = DataInspector.unpivot_dataframe(results[f]["forecast_df"], 
                                                    value_vars=value_vars, 
                                                    turbine_signature=forecaster.turbine_signature)\
                                            .unpivot(index=["turbine_id"] + id_vars, on=target_vars, 
                                                    variable_name="feature", value_name="value")\
                                            .with_columns(data_type=pl.lit("Forecast"))
        
        id_vars = test_data.select(id_var_selector).columns  
        true_long = DataInspector.unpivot_dataframe(test_data, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=forecaster.turbine_signature)\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], 
                                                    variable_name="feature", value_name="value")\
                                            .with_columns(data_type=pl.lit("True"))
        
        forecast_fig = WindForecast.plot_forecast(forecasts_long, true_long, 
                                                  continuity_groups=[0], turbine_ids=["5", "74", "75"], 
                                                  label=f"_{forecaster.__class__.__name__}_{data_config['config_label']}", 
                                                  fig_dir=save_dir, include_turbine_legend=True) 
    
    
    all_metrics = results[0]["agg_metrics"].select(pl.col("metric").unique()).to_numpy().flatten()
    # get the metrics we care about, there is also "MSE", "MAE", "abs_error", "QuantileLoss", 
    metrics = [metric for metric in all_metrics if any(m in metric for m in ["MAE", "RMSE", "PICP", "PINAW", "CWC", "CRPS"])]
    
    agg_df = pl.concat([
        res["agg_metrics"].with_columns(forecaster=pl.lit(res["forecaster_name"]), 
                                        prediction_timedelta=pl.lit(res["prediction_timedelta"]))
        for res in results], how="vertical")
    
    plotting_metrics_dirs = [(met, direc) for met, direc in 
                        zip(["MAE", "RMSE", "PICP", "PINAW", "CWC", "CRPS"], [0, 0, 1, 1, 1, 1]) 
                        if met in pd.unique(agg_df["metric"])]
    plotting_metrics = [v[0] for v in plotting_metrics_dirs]
    ax_indices = [v[1] for v in plotting_metrics_dirs]
    # generate scatterplot of metric vs prediction time for different models (different colors) and different metrics (different_styles) (crps, picp, pinaw, cwc, mse, mae)
    plot_score_vs_prediction_dt(agg_df, 
                                metrics=plotting_metrics,
                                ax_indices=ax_indices)

    # best_prediction_dt = agg_df.groupby(["metric", "prediction_timedelta"])["score"].mean().idxmax()
    # generate grouped barcharpt of metrics (crps, picp, pinaw, cwc, mse, mae) grouped together vs model on x axis for best prediction time
    best_prediction_dt = agg_df.groupby("prediction_timedelta")["score"].mean().idxmax()
    plot_score_vs_forecaster(agg_df.loc[agg_df["prediction_timedelta"] == best_prediction_dt, :], 
                             metrics=plotting_metrics,
                                good_directions=[-1, 1, -1, -1, -1])
    
    print("here")

