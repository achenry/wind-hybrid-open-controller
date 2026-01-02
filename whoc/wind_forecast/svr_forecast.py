
from typing import Union
import os
from collections import defaultdict
import glob
import re
import pickle
from dataclasses import dataclass
from memory_profiler import profile
import inspect

import pandas as pd
import polars as pl
import numpy as np


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from whoc.wind_forecast.wind_forecast_base import WindForecast

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class SVRForecast(WindForecast):
    """Wind speed component forecasting using Support Vector Regression."""
    is_probabilistic = False
    def __post_init__(self):
        super().__post_init__()
        
        self.max_n_samples = self.kwargs["max_n_samples"] 
        self.model_config = self.kwargs["model_config"]

        self.n_turbines = self.fmodel.n_turbines
        self.measurement_layout = np.vstack([self.fmodel.layout_x, self.fmodel.layout_y]).T
        
        self.n_neighboring_turbines = self.kwargs["n_neighboring_turbines"] 
        self.dataset_hparams = {"n_neighboring_turbines": self.n_neighboring_turbines}
        self.dataset_hparams_choices = {"n_neighboring_turbines": [1, 3, 5]}
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
        self.study_name = f"tuning_svr_{self.model_config['experiment']['run_name']}"
        self.model_save_dir = os.path.join(self.model_config["experiment"]["log_dir"], self.study_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        self.scaler = defaultdict(self.create_scaler)
        self.model = defaultdict(self.create_model)
            
        self.use_trained_models = self.kwargs.get("use_trained_models", True)
        
        model_files = glob.glob(os.path.join(self.model_save_dir, f"{self.study_name}_model_*_{int(self.prediction_timedelta.total_seconds())}.pkl"))
        scaler_files = glob.glob(os.path.join(self.model_save_dir, f"{self.study_name}_scaler_*_{int(self.prediction_timedelta.total_seconds())}.pkl"))
        
        if self.use_trained_models and len(model_files) == 0:
            logging.error(f"No trained models found in {self.model_save_dir}. Please run tuning.py first for the correct prediction time {int(self.prediction_timedelta.total_seconds())}.")
            raise Exception
        
        # no need to load optuna trained hyperparams if we are loading models anyway
        self.n_outputs = (self.n_turbines if self.target_turbine_indices is None else len(self.target_turbine_indices)) * 2
        if (not self.use_trained_models or len(model_files) < self.n_outputs or len(scaler_files) < self.n_outputs) and self.use_tuned_params:
            self.set_tuned_params(optuna_storage=self.kwargs["optuna_storage"], 
                                    study_name=self.study_name)
            self.use_trained_models = False
            logging.info("No available trained models.") # TODO train here
            # self.train_all_outputs(scale=True, 
            #                         multiprocessor=args.multiprocessor, 
            #                         retrain_models=True,
            #                         scaler_params=None)
        
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
    
    def reset(self, **kwargs):
        pass
    
    def create_scaler(self):
        # return MinMaxScaler(feature_range=(-1, 1))
        return StandardScaler()
    
    def create_model(self, **kwargs):
        return SVR(**{k: v for k, v in kwargs.items() if k in inspect.signature(SVR).parameters})
   
    def _prepare_arrays(self, training_inputs, output_idx):
        
        X_train = np.ascontiguousarray(np.vstack([
            training_inputs[i:i+self.n_context, :].flatten()
            for i in range(max(training_inputs.shape[0] - self.n_context, 1))
        ]))
        
        # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_timedelta][output])
        y_train = np.ascontiguousarray(training_inputs[self.n_context:, output_idx])
        
        assert X_train.shape[0] == y_train.shape[0]
        
        return X_train, y_train
    
    def get_params(self, trial):
         
        # return {
        #     **{f"C_{output}": trial.suggest_float(f"C_{output}", 1e-6, 1e6, log=True) for output in self.outputs},
        #     **{f"epsilon_{output}": trial.suggest_float(f"epsilon_{output}", 1e-6, 1e-1, log=True) for output in self.outputs},
        #     **{f"gamma_{output}": trial.suggest_categorical(f"gamma_{output}", ["scale", "auto"]) for output in self.outputs}
        # }
        return {
            "C": trial.suggest_float(f"C", 1e-3, 10, log=True),
            "epsilon": trial.suggest_float(f"epsilon", 1e-3, 10, log=True),
            "gamma": trial.suggest_categorical(f"gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "n_neighboring_turbines": trial.suggest_categorical("n_neighboring_turbines", self.dataset_hparams_choices["n_neighboring_turbines"]),
        }
    

    def predict_sample(self, n_samples: int):
        pass
    
    def train_single_output(self, training_measurements, output, retrain_models, scale, scaler_params=None):
        
        # feat_type = re.search(f"\\w+(?=_{self.turbine_signature})", output).group()
        tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
        model_save_path = os.path.join(self.model_save_dir, f"{self.study_name}_model_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")
        scaler_save_path = os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")
        if not retrain_models and os.path.exists(model_save_path)  and (not scale or scaler_params or os.path.exists(scaler_save_path)):
            logging.info(f"Loading trained SVR model for output {output}.")
            with open(model_save_path, "rb") as fp:
                self.model[output] = pickle.load(fp)

            if scale and scaler_params is None:
                with open(scaler_save_path, "rb") as fp:
                    self.scaler[output] = pickle.load(fp)
        else:
            
            X_train, y_train, self.scaler[output] = self._get_output_data(measurements=training_measurements, output=output, split="train", reload=False, 
                                                                          scale=scale, return_scaler=True, dataset_hparams=self.dataset_hparams)
            logging.info(f"Fitting SVR model for output {output} with {X_train.shape[0]} data points.")
            self.model[output].fit(X_train, y_train)
            
            model_save_path = os.path.join(self.model_save_dir, f"{self.study_name}_model_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")
            logging.info(f"Saving SVR model for output {output} to {model_save_path}.")
            with open(model_save_path, "wb") as fp:
                pickle.dump(self.model[output], fp, protocol=5)
            
            # if scale and scaler_params is None:
            #     # self.scaler[output].fit(X_train)
            #     scaler_save_path = os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")
            #     logging.info(f"Saving SVR scaler for output {output} to {scaler_save_path}.")
            #     with open(scaler_save_path, "wb") as fp:
            #         pickle.dump(self.scaler[output], fp, protocol=5)
        
        
        if scaler_params:
            scaler_save_path = os.path.join(self.model_save_dir, f"{self.study_name}_scaler_{output}_{int(self.prediction_timedelta.total_seconds())}.pkl")
            logging.info(f"Setting SVR scaler for output {output} to given values.")
            input_turbine_indices = self.cluster_turbines[self.tid2idx_mapping[tid]]
            self.scaler[output].n_features_in_ = len(input_turbine_indices)
            for k, v in scaler_params.items():
                setattr(self.scaler[output], k, np.ones_like(input_turbine_indices) * v[output])
            
            logging.info(f"Saving SVR scaler for output {output} to {scaler_save_path}.")
            with open(scaler_save_path, "wb") as fp:
                pickle.dump(self.scaler[output], fp, protocol=5)
                
        return self.model[output], self.scaler[output]
    
    def train_all_outputs(self, scale, multiprocessor, retrain_models=True,
                          scaler_params=None):
        if multiprocessor is not None:
            if multiprocessor == "mpi":
                mpi_exists = False
                try:
                    from mpi4py import MPI
                    from mpi4py.futures import MPICommExecutor
                    mpi_exists = True
                except ImportError as e:
                    import traceback
                    print(f"ERROR: Failed to import mpi4py. MPI will not be available. Error: {e}")
                    print(traceback.format_exc())
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif multiprocessor == "cf":
                # max_workers = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
                max_workers = mp.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers,
                                                mp_context=mp.get_context("spawn"))
            with executor as ex:
                if multiprocessor == "mpi":
                    ex.max_workers = comm_size
                    
                futures = [ex.submit(self.train_single_output, 
                                        training_measurements=None, 
                                        output=output, 
                                        scale=scale, 
                                        scaler_params=scaler_params,
                                        retrain_models=retrain_models) for output in self.outputs]
                for output, fut in zip(self.outputs, futures):
                    m, s = fut.result()
                    self.model[output] = m
                    self.scaler[output] = s
        else:    
            for output in self.outputs:
                self.model[output], self.scaler[output] = self.train_single_output(
                    training_measurements=None, 
                    output=output, scale=scale, 
                    scaler_params=scaler_params, retrain_models=retrain_models)
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        # TODO LOW include yaw angles in inputs?
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] 
        # outputs = self._get_ws_cols(historic_measurements)
        
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
            for output in self.outputs:
                feat_type = re.search(f"^\\w+(?=_{self.turbine_signature}$)", output).group()
                tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
                if not (hasattr(self.scaler[output], "mean_") and hasattr(self.scaler[output], "scale_")) \
                    or (check_is_fitted(self.model[output]) is not None):
                    raise Exception(f"scaler/model for {output} has not been trained! Try using the --use_trained_models flag.")
                training_inputs = self._get_inputs(training_measurements, self.scaler[output], feat_type, tid, scale)
                
                pred[output] = self._predict(model=self.model[output], 
                                             training_inputs=training_inputs)
                
            # rescale back TODO
            if scale:
                pred = {output: self._inverse_scale(pred, output).flatten() for output in self.outputs}
            else:
                pred = {output: pred[output][np.newaxis, :].flatten() for output in self.outputs}
            
            pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
            
        else:
            # not enough data points to train SVR, assume persistence
            logging.info(f"Not enough data points at time {current_time} to train SVR, have {historic_measurements.select(pl.len()).item() * self.measurements_timedelta} but require {self.n_context * self.prediction_timedelta}, assuming persistence instead.")
            pred = pl.concat([pred_slice.to_frame(), historic_measurements.slice(-1, 1).select(self.outputs)], how="horizontal")
            
        if return_pl: 
            return pred
        else:
            return pred.to_pandas()  
            
    def _inverse_scale(self, pred, output):
        tid = re.search(f"(?<=_){self.turbine_signature}$", output).group()
        output_idx = self.cluster_turbines[self.tid2idx_mapping[tid]].index(self.tid2idx_mapping[tid]) 
        # return (pred[output][np.newaxis, :] - self.scaler[output].min_[output_idx]) / self.scaler[output].scale_[output_idx]
        return (pred[output][np.newaxis, :] * self.scaler[output].scale_[output_idx]) + self.scaler[output].mean_[output_idx]

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


    
