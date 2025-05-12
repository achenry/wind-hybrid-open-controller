from typing import Union
from dataclasses import dataclass

import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from typing import Optional, Union
from dataclasses import dataclass
import os
import joblib
import optuna
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
#from datetime import datetime, timedelta

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

# from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from functools import reduce


import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#ARIMA packages
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from whoc.wind_forecast.wind_forecast_base import WindForecast

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_palette("Paired")

class ARIMAForecast(WindForecast):
    """Wind speed forecasting using ARIMA model"""
    is_probabilistic = False
    study_name: Optional[str] = None

    def __post_init__(self, model_save_dir=None, study_name=None):
        print("ARIMAForecast initialized")
        super().__post_init__()
        self.models = {}
        self.data = None
        self.fitted = False
        #self.boxcox_params = {}
        self.boxcox_params = {"horz": {}, "vert": {}}  
        self.persistence_fallback = {}  
        self.model = {"horz": {}, "vert": {}}          # prepare for both
        #self.model = {}
        self.scaler = {}
        self.storage = 'sqlite:///C:/Users/20202629/Desktop/Internship/wind-forecasting/examples/optuna/tuning_arima_windfarm_debug.db'
        # if self.study_name is None:
        #     self.study_name = "default_study_name"
        if self.study_name is None:
            self.study_name = 'arima_LUT_prediction_timedelta_420'
        # if not hasattr(self, "study_name"):
        #     self.study_name = f"{args.model}_ws_vert_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            #self.study_name = "arima_ws_vert_all_20250429_123701" #"tuning_arima_windfarm_debug" 
        self.study = self.create_or_load_study(self.study_name)
        #self.model_config["experiment"]["log_dir"] = "C:/Users/20202629/Desktop/Internship/wind-forecasting/examples/optuna"
        base_log_dir = "C:/Users/20202629/Desktop/Internship/wind-forecasting/examples/optuna"
        self.model_save_dir_horz = os.path.join(base_log_dir, self.study_name, "horz", "models")
        self.model_save_dir_vert = os.path.join(base_log_dir, self.study_name, "vert", "models")
        #self.model_save_dir = self.model_save_dir_horz or self.model_save_dir_vert
        #self.model_save_dir = os.path.join(self.model_config["experiment"]["log_dir"], self.study_name, "models")
        #self.model = {} # trained models from local directory
        #if self.model_save_dir:
        #    self.load_models(self.model_save_dir)
        self.load_models(self.model_save_dir_horz, "horz")
        self.load_models(self.model_save_dir_vert, "vert")
        os.makedirs(self.model_save_dir_horz, exist_ok=True)
        os.makedirs(self.model_save_dir_vert, exist_ok=True)
        self.n_prediction_interval = 1

    def _prepare_arrays(self, training_inputs, feat_type, tid, output_idx):
        # ARIMA assumes training_inputs is already the target time series (X is empty and y is next value)
        y = training_inputs[:, output_idx]
        X = None  
        return X, y

    def load_models(self, model_save_dir, mode):
        """Load ARIMA models from the specified directory."""

        if not os.path.exists(model_save_dir):
            print(f"[INFO] No models found in {model_save_dir}, skipping loading for '{mode}'.")
            return  # No models to load yet

        if mode not in self.model:
            self.model[mode] = {}
        if mode not in self.boxcox_params:
            self.boxcox_params[mode] = {}

        for fname in os.listdir(model_save_dir):
            if fname.endswith(".pkl") and fname != "boxcox_params.pkl":
                key = fname.replace(".pkl", "")
                path = os.path.join(model_save_dir, fname)
                self.model[mode][key] = joblib.load(path)
        boxcox_path = os.path.join(model_save_dir, "boxcox_params.pkl")
        if os.path.exists(boxcox_path):
            with open(boxcox_path, "rb") as f:
                self.boxcox_params[mode] = pickle.load(f)
        else:
            self.boxcox_params = {}
        self.fitted = True
                
    def create_or_load_study(self, study_name):
        try:
            # Attempt to load the study with the provided study_name
            study = optuna.load_study(study_name=study_name, storage=self.storage)
            print(f"Study '{study_name}' loaded successfully.")
        except KeyError:
            # If study doesn't exist, create a new study with the same name
            print(f"Study '{study_name}' not found. Creating a new one.")
            study = optuna.create_study(study_name=study_name, storage=self.storage)
            print(f"Study '{study_name}' created successfully.")
        return study

    def get_params(self, trial):
        """Retrieve hyperparameters for the current trial."""
        
        p = trial.suggest_int('p', 1, 5)  # AR parameter
        d = trial.suggest_int('d', 0, 2)  # Differencing order
        q = trial.suggest_int('q', 1, 5)  # MA parameter

        # Return a dictionary of the parameters to be used by the ARIMA model
        return {"p": p, "d": d, "q": q}

    def _tuning_objective(self, trial, historic_measurements: Union[pl.DataFrame, pd.DataFrame], multiprocessor=None, limit_train_val=None, turbine_ids=None):
        """Objective function for tuning the ARIMA model."""

        # obtain the hyperparameters from get_params
        params = self.get_params(trial)
        p, d, q = params["p"], params["d"], params["q"]

        rmse_results = []

        for series_name, ts_vert in historic_measurements.items():
            try:
                turbine_id = series_name.split("_")[-1]

                if ts_vert is None or len(ts_vert) < 2:
                    logging.warning(f"Skipping turbine {turbine_id}: Not enough data points.")
                    continue
                ts_vert_transformed = self.boxcox_transform(ts_vert, series_name)
                #model_horz = sm.tsa.ARIMA(ts_horz_transformed, order=(p, d, q)).fit()
                model_vert = SARIMAX(ts_vert_transformed, order=(p, d, q)).fit()


                forecast_vert = model_vert.forecast(steps=30)
                forecast_vert_original = self.inverse_boxcox(forecast_vert, series_name)

                actual = ts_vert[-30:]

                rmse = np.sqrt(mean_squared_error(actual, forecast_vert_original))
                rmse_results.append(rmse)

            except Exception as e:
                logging.warning(f"Error fitting ARIMA for horizontal wind speed on turbine {turbine_id}: {e}")
                continue
        if rmse_results:
            return np.mean(rmse_results)  # Minimize RMSE
        else:
            return float("inf")  # Return a large value if no valid results


        # if turbine_ids is None:
        #     if isinstance(historic_measurements, (pd.Series, pl.Series)):
        #         turbine_ids = [historic_measurements.name.split("_")[-1]]
        #     else:
        #         raise ValueError("Cannot infer turbine_ids: historic_measurements must be a Series.")
        # store_params = [] 

        # for turbine_id in turbine_ids:
        #     # Prepare data for the turbine (horizontal and vertical wind speeds)
        #     ts_horz = historic_measurements  # historic_measurements is already a Series
        #     ts_horz_transformed = self.boxcox_transform(ts_horz, f"ws_horz_{turbine_id}")

        #     #if ts_horz.dropna().shape[0] < 2 or ts_vert.dropna().shape[0] < 2:
        #     if ts_horz.len() < 2:
        #         logging.warning(f"Skipping turbine {turbine_id}: Not enough data points.")
        #         continue

        #     try:
        #         model_horz = sm.tsa.ARIMA(ts_horz_transformed, order=(p, d, q)).fit()
        #         forecast_horz = model_horz.forecast(steps=30)  # Forecast for next 30 time steps
        #         forecast_horz_original = self.inverse_boxcox(forecast_horz, f"ws_horz_{turbine_id}")
        #     except Exception as e:
        #         logging.warning(f"Error fitting ARIMA for horizontal wind speed on turbine {turbine_id}: {e}")
        #         continue
            
        #     horz_data = ts_horz[-30:]

        #     horz_rmse = np.sqrt(mean_squared_error(horz_data, forecast_horz_original))

        #     store_params.append(horz_rmse)

        # if len(store_params) > 0:
        #     return np.mean(store_params)
        # else:
        #     return float("inf") # data is not valid
    
    def define_data(self, data: pl.DataFrame):
        """Store data as turbine-specific Pandas Series in a dictionary."""
        self.historic_measurements = data
        self.data = {}
        time_sorted = data.sort("time")
        for col in time_sorted.columns:
            if col.startswith("ws_horz_") or col.startswith("ws_vert_"):
                        self.data[col] = time_sorted.select(["time", col]).to_pandas().set_index("time")[col]


    def create_model(self, turbine_id, feature_type, p, d, q):
        """Create and return an ARIMA model based on hyperparameters."""
        key = f"{feature_type}_{turbine_id}"
        if self.data is None or key not in self.data:
            raise ValueError(f"Data for {key} not found. Make sure define_data() has been called with proper structure.")
        ts = self.data[key]
        ts_transformed = self.boxcox_transform(ts, key)
        #model = sm.tsa.ARIMA(ts_transformed, order=(p, d, q)).fit()
        model = SARIMAX(ts_transformed, order=(p, d, q)).fit()

        return model

    def boxcox_transform(self, ts, feature_key):
        """Apply Box-Cox transformation to the data."""
        shift_val = 0
        if isinstance(ts, pl.Series):
            ts = ts.filter(~ts.is_null())  # Pandas specific NaN handling
        elif isinstance(ts, np.ndarray):
            ts = ts[~np.isnan(ts)]  # NumPy specific NaN handling
       
        if len(ts) == 0 or len(ts.unique()) == 1:
            # Handle empty or constant array case
            logging.warning(f"Box-Cox skipped: constant or empty data for '{feature_key}'. Using persistence fallback.")
            
            # Check if ts is a NumPy array or Pandas Series and access the last element
            #last_value = ts[-1] if isinstance(ts, np.ndarray) else ts.iloc[-1]
            last_value = ts[-1] if isinstance(ts, np.ndarray) else ts[-1]
            
            #self.boxcox_params[feature_key] = {"lambda": None, "shift": 0, "persistence": last_value if ts.size > 0 else np.nan}
            self.boxcox_params[feature_key] = {"lambda": None, "shift": 0, "persistence": last_value if ts.len() > 0 else np.nan}

            return ts  # Still return something usable
        
        if len(ts) < 10:
            logging.warning(f"Too few samples ({len(ts)}), skipping Box-Cox for '{feature_key}'. Returning original series.")
            self.boxcox_params[feature_key] = {"lambda": None, "shift": 0, "persistence": ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]}
            return ts

        if ts.min() <= 0:
            shift_val = abs(ts.min()) + 1
            ts = ts + shift_val

        try:
            ts_transformed, lmbda = boxcox(ts)
            self.boxcox_params[feature_key] = {"lambda": lmbda, "shift": shift_val}
            return ts_transformed
        except ValueError as e:
            logging.warning(f"Box-Cox failed for '{feature_key}' with error: {e}. Using persistence fallback.")
            
            # Check if ts is a NumPy array or Pandas Series and access the last element
            #last_value = ts[-1] if isinstance(ts, np.ndarray) else ts.iloc[-1]
            last_value = ts[-1] if isinstance(ts, np.ndarray) else ts.iloc[-1]

            
            #self.boxcox_params[feature_key] = {"lambda": None, "shift": 0, "persistence": last_value if ts.size > 0 else np.nan}
            self.boxcox_params[feature_key] = {"lambda": None, "shift": 0, "persistence": last_value if len(ts) > 0 else np.nan}

            return ts

    def save_boxcox_params(self, path=None):
        """Save Box-Cox parameters to file."""
        if not hasattr(self, "boxcox_params"):
            raise AttributeError("Box-Cox parameters not found.")
        path = path or os.path.join(self.model_save_dir, "boxcox_params.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.boxcox_params, f)

    def load_boxcox_params(self, path=None):
        """Load Box-Cox parameters from file."""
        path = path or os.path.join(self.model_save_dir, "boxcox_params.pkl")
        with open(path, "rb") as f:
            self.boxcox_params = pickle.load(f)

    
    def inverse_boxcox(self, ts_transformed, feature_key):
        """Apply inverse Box-Cox transformation to the data."""
        params = self.boxcox_params[feature_key]
        lmbda = params.get("lambda")
        shift_val = params.get("shift", 0)

        if lmbda is None:
            # Box-Cox was skipped, so just return persistence fallback
            persistence_value = params.get("persistence", np.nan)
            return np.full_like(ts_transformed, persistence_value, dtype=np.float64)

        ts_original = inv_boxcox(ts_transformed, lmbda) - shift_val
        return ts_original

    def train(self, historic_measurements: pl.DataFrame, turbine_ids=None, p=1, d=0, q=0):
        print(">>> ARIMAForecast.train() called")
        if turbine_ids is None:
            turbine_ids = [
                col.split("_")[-1]
                for col in historic_measurements.columns
                if col.startswith("ws_vert_")
            ]
        
        for turbine_id in turbine_ids:
            logging.info(f"Training ARIMA model for {turbine_id}.")

            # prepare vertical and horizontal wind speed 
            turbine_df_horz = historic_measurements.select(pl.col("time"), pl.col(f"ws_horz_{turbine_id}")).sort("time").unique(subset=["time"])
            turbine_df_vert = historic_measurements.select(pl.col("time"), pl.col(f"ws_vert_{turbine_id}")).sort("time").unique(subset=["time"])
            ts_horz = turbine_df_horz.to_pandas().set_index("time")[f"ws_horz_{turbine_id}"]
            ts_vert = turbine_df_vert.to_pandas().set_index("time")[f"ws_vert_{turbine_id}"]
            # Check for enough data
            if ts_horz.dropna().shape[0] < 2 or ts_vert.dropna().shape[0] < 2:
                logging.warning(
                    f"Skipping turbine {turbine_id}: Not enough data points for ARIMA (horz: {ts_horz.dropna().shape[0]}, vert: {ts_vert.dropna().shape[0]})."
                )
                continue
            self.persistence_fallback[f"ws_horz_{turbine_id}"] = ts_horz.iloc[-1]
            self.persistence_fallback[f"ws_vert_{turbine_id}"] = ts_vert.iloc[-1]


            # ARIMA prediction for both horizontal and vertical
            #ts_horz = turbine_df_horz.to_pandas().set_index("time")[f"ws_horz_{turbine_id}"]
            print(ts_horz.min())
            ts_horz_transformed = self.boxcox_transform(ts_horz, f"ws_horz_{turbine_id}")

            #model_horz = sm.tsa.ARIMA(ts_horz_transformed, order=(1, 1, 1)).fit()
            model_horz = SARIMAX(ts_horz_transformed, order=(p, d, q)).fit()


            #ts_vert = turbine_df_vert.to_pandas().set_index("time")[f"ws_vert_{turbine_id}"]
            ts_vert_transformed = self.boxcox_transform(ts_vert, f"ws_vert_{turbine_id}")
            #model_vert = sm.tsa.ARIMA(ts_vert_transformed, order=(1, 0, 0)).fit()
            model_vert = SARIMAX(ts_vert_transformed, order=(p, d, q)).fit()
            self.models[turbine_id] = {"ws_horz": model_horz, "ws_vert": model_vert}
            self.fitted = True

    def train_all_outputs(self, outputs, scale, multiprocessor, retrain_models=True, scaler_params=None):
        if not hasattr(self, "historic_measurements") or self.historic_measurements is None:
            raise ValueError("data must be set on the instance before training.")

        turbine_ids = [col.split("_")[-1] for col in outputs if col.startswith("ws_horz_")]

        if multiprocessor is not None:
            if multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif multiprocessor == "cf":
                max_workers = mp.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)

            with executor as ex:
                if multiprocessor == "mpi":
                    ex.max_workers = comm_size

                ex.map(lambda tid: self.train(self.historic_measurements, turbine_ids=[tid]), turbine_ids)
        else:
            self.train(self.historic_measurements, turbine_ids=turbine_ids)

    
    def model_items(self):
            """Returns the list of turbine IDs for which models are trained."""
            #return self.models.keys() manual hyperparameter tuning
            return self.model.keys()
    
    def reset(self):
        pass
       
    def predict_point(self, historic_measurements, current_time=None, return_long_format=True):
        print(">>> ARIMAForecast.predict_point() called")
        if not self.fitted:
            raise ValueError("ARIMA model not fitted. Call train() method first.")
        
        horizon = self.n_prediction
        prediction_freq = pd.Timedelta(self.measurements_timedelta)

        if not hasattr(self, "n_context"):
            self.n_context = int(self.context_timedelta / self.prediction_interval)

        if current_time is None:
            current_time = historic_measurements.select(pl.col("time").max()).item()
            
        forecast_times = pd.date_range(start=current_time + prediction_freq, periods=horizon, freq=prediction_freq)
        forecast_df = pl.DataFrame({"time": forecast_times})
        turbine_forecasts = []
        long_forecasts = []
        turbine_ids = sorted(k for k in self.model["horz"] if k.startswith("ws_horz_"))

        for turbine_id in turbine_ids:
            # historic data
            key_horz = turbine_id
            key_vert = turbine_id.replace("ws_horz_", "ws_vert_")

            turbine_df_horz = historic_measurements.select(pl.col("time"), pl.col(key_horz)).sort("time").unique(subset=["time"])
            turbine_df_vert = historic_measurements.select(pl.col("time"), pl.col(key_vert)).sort("time").unique(subset=["time"])
            raw_series_horz = turbine_df_horz.select(key_horz).to_pandas()[key_horz]
            raw_series_vert = turbine_df_vert.select(key_vert).to_pandas()[key_vert]
            raw_series_horz.index = pd.to_datetime(turbine_df_horz.select("time").to_pandas()["time"])
            raw_series_vert.index = pd.to_datetime(turbine_df_vert.select("time").to_pandas()["time"])
            series_horz_transformed = self.boxcox_transform(raw_series_horz, key_horz)
            series_vert_transformed = self.boxcox_transform(raw_series_vert, key_vert)

            sufficient_data = turbine_df_horz.height >= self.n_context and turbine_df_vert.height >= self.n_context

            if not sufficient_data: #Persistence will be used
                logging.info(f"Not enough data for turbine {turbine_id} at time {current_time}, falling back to persistence.")
                value_horz = turbine_df_horz.select(pl.col(key_horz)).last().item()
                value_vert = turbine_df_vert.select(pl.col(key_vert)).last().item()

                forecast_horz_original = np.full(horizon, value_horz)
                forecast_vert_original = np.full(horizon, value_vert)
            else:  # ARIMA forecast will be used
                # key_horz = f"ws_horz_{turbine_id}"
                # model_horz = self.models[turbine_id]["ws_horz"] manually setting hyperparams
                model_horz = self.model["horz"].get(key_horz)
                updated_model_horz = model_horz.append(series_horz_transformed, refit=False)
            
                if key_horz not in self.boxcox_params["horz"]:
                    logging.warning(f"No Box-Cox params for {key_horz}, falling back to persistence.")
                    value_horz = self.persistence_fallback.get(key_horz, np.nan)
                    forecast_horz_original = np.full(horizon, value_horz)
                else:
                    forecast_horz = model_horz.forecast(steps=horizon)
                    forecast_horz_test = updated_model_horz.forecast(steps=horizon)

                    forecast_horz_original = self.inverse_boxcox(forecast_horz, key_horz)
                    forecast_horz_original_test = self.inverse_boxcox(forecast_horz_test, key_horz)

                    if np.isnan(forecast_horz_original).any():
                        logging.warning(f"NaNs in forecast for {key_horz}, falling back to persistence.")
                        value_horz = self.persistence_fallback.get(key_horz, np.nan)
                        forecast_horz_original = np.full(horizon, value_horz)

                # Vertical
                #key_vert = f"ws_vert_{turbine_id}"
                #model_vert = self.models[turbine_id]["ws_vert"]
                model_vert = self.model["vert"].get(key_vert)
                updated_model_vert = model_vert.append(series_vert_transformed, refit=False)

                if key_vert not in self.boxcox_params["vert"]:
                    logging.warning(f"No Box-Cox params for {key_vert}, falling back to persistence.")
                    value_vert = self.persistence_fallback.get(key_vert, np.nan)
                    forecast_vert_original = np.full(horizon, value_vert)
                else:
                    forecast_vert = model_vert.forecast(steps=horizon)
                    forecast_vert_test = updated_model_vert.forecast(steps=horizon)
                    forecast_vert_original = self.inverse_boxcox(forecast_vert, key_vert)
                    forecast_vert_original_test = self.inverse_boxcox(forecast_vert_test, key_vert)
                    if np.isnan(forecast_vert_original).any():
                        logging.warning(f"NaNs in forecast for {key_vert}, falling back to persistence.")
                        value_vert = self.persistence_fallback.get(key_vert, np.nan)
                        forecast_vert_original = np.full(horizon, value_vert)

            #turbine_df = pl.DataFrame({
            #"time": forecast_times,
            #f"ws_horz_{turbine_id}": forecast_horz_original,
            #f"ws_vert_{turbine_id}": forecast_vert_original
            #})

            turbine_df = pl.DataFrame({
            "time": forecast_times,
            f"ws_horz_{turbine_id}": forecast_horz_original_test,
            f"ws_vert_{turbine_id}": forecast_vert_original_test
            })
        
            turbine_forecasts.append(turbine_df)
        
            if return_long_format:
                df_horz = pl.DataFrame({
                    "turbine_id": [turbine_id] * horizon,
                    "time": forecast_times,
                    "feature": ["ws_horz"] * horizon,
                    "value": forecast_horz_original_test,
                    "data_type": ["Forecast"] * horizon
                })

                df_vert = pl.DataFrame({
                    "turbine_id": [turbine_id] * horizon,
                    "time": forecast_times,
                    "feature": ["ws_vert"] * horizon,
                    "value": forecast_vert_original_test,
                    "data_type": ["Forecast"] * horizon
                })

                long_forecasts.extend([df_horz, df_vert])

        forecast_df = reduce(lambda df1, df2: df1.join(df2, on="time", how="left"), turbine_forecasts, forecast_df)
        # naming issue
        forecast_df = forecast_df.rename({col: col.replace("ws_horz_", "", 1) if col != "time" else col for col in forecast_df.columns})


        if return_long_format:
            return forecast_df.sort("time"), pl.concat(long_forecasts).sort(["turbine_id", "time", "feature"])
        else:
            return forecast_df.sort("time")
