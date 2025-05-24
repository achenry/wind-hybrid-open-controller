import copy
from pathlib import Path
import warnings

import lightning.pytorch as pylt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TimeSeriesDataSet
#from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

from dataclasses import dataclass
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Union

import optuna
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except ImportError as e:
    import traceback
    print(f"ERROR: Failed to import mpi4py. MPI will not be available. Error: {e}")
    print(traceback.format_exc())

from sklearn.metrics import mean_squared_error
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from whoc.wind_forecast.wind_forecast_base import WindForecast
from lightning.pytorch import Trainer
from  lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer



class TemporalFusionTransformerForecast(WindForecast):
    is_probabilistic = True
    study_name: Optional[str] = None

    def __init__(self, study_name=None, model_save_dir=None, **kwargs):
        print("Initializing TemporalFusionTransformerForecast")
        super().__init__(**kwargs)
        self.model = None
        self.data = None
        self.fitted = False
        self.study_name = study_name or 'tft_default'
        self.storage = 'sqlite:///C:/Users/20202629/Desktop/Internship/wind-forecasting/examples/optuna/optuna_tft_study.db'

        self.study = self.create_or_load_study(self.study_name)
        self.model_save_dir = model_save_dir or f"./models/{self.study_name}/"
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.cluster_turbines = {i: [i] for i in range(len(self.tid2idx_mapping))}
        self.n_prediction_interval = int(self.prediction_timedelta / self.measurements_timedelta)

    def define_data(self, data: pl.DataFrame):
        """Define TimeSeriesDataSet for tft"""
        df = data.to_pandas()
        df["time_idx"] = ((df["time"] - df["time"].min()).dt.total_seconds() // 60).astype(int) # data is 1 minute resolution

        target_col = "ws_horz_6"
        max_encoder_length = 60
        max_prediction_length = 7 # for greedy we determine the prediction length to be 7 minutes

        self.data = TimeSeriesDataSet(df, time_idx="time_idx", target=target_col, group_ids=["continuity_group"], max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length, time_varying_unknown_reals=[target_col], time_varying_known_reals=["time_idx"], target_normalizer=GroupNormalizer(groups=["continuity_group"]), allow_missing_timesteps=True)

    def create_model(self, **params):
        """
        Temporal Fusion Transformer model creation.
        """
        if not hasattr(self, "data") or self.data is None:
            self.define_data()
            raise ValueError("Training dataset not defined. Please define training dataset using the 'define_data' method.")
        self.model = TemporalFusionTransformer.from_dataset(
             dataset=self.data,
             **params)
        return self.model

    
    def _prepare_arrays(self, training_inputs, feat_type, tid, output_idx):
        """
        Prepare the input arrays for the model.
        """
        if isinstance(training_inputs, pd.DataFrame):
            y_col = training_inputs.columns[output_idx]
            y = training_inputs[y_col].values
            X = training_inputs.drop(columns=[y_col]).values
        else:
            y = training_inputs[:, output_idx]
            X = np.delete(training_inputs, output_idx, axis=1)
        
        return X, y

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
    
    def find_optimal_learning_rate(self, trainer=None, max_epochs=10, batch_size=64):
        """
        Find the optimal learning rate for the model.
        """
        if not hasattr(self, "data") or self.data is None:
            raise ValueError("Data not defined. Please define data using the 'define_data' method.")
        if self.model is None:
            self.model = TemporalFusionTransformer.from_dataset(self.data, learning_rate=0.01, hidden_size=32, attention_head_size=1, dropout=0.1, hidden_continuous_size=32, loss=QuantileLoss(), optimizer="adam")
            print(f"Number of parameters in model: {self.model.size() / 1e3:.1f}k")

        if trainer is None:
             trainer = pylt.Trainer(max_epochs=max_epochs, accelerator="auto", logger=False, enable_checkpointing=False, enable_progress_bar=False)


        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(self.model, train_dataloaders=self.data.to_dataloader(train=True, batch_size=batch_size), min_lr=1e-6, max_lr=0.1e-1)
        suggested_lr = lr_finder.suggestion()

        fig = lr_finder.plot()
        plt.axvline(x=suggested_lr, color='red', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
        plt.title("Learning Rate Finder for Temporal Fusion Transformer")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Suggested learning rate: {suggested_lr}")

        self.model.hparams.learning_rate = suggested_lr	

        return suggested_lr
    
    def train_model(self, max_epochs=10, batch_size=64, trainer=None):
        """
        Train the Temporal Fusion Transformer model.
        """
        if not hasattr(self, "data") or self.data is None:
            raise ValueError("Data not defined. Please define data using the 'define_data' method.")
        if self.model is None:
            self.model = TemporalFusionTransformer.from_dataset(self.data, learning_rate=0.0011, hidden_size=8, attention_head_size=1, dropout=0.1, hidden_continuous_size=8, loss=QuantileLoss(), optimizer="ranger")
            print(f"Number of parameters in model: {self.model.size() / 1e3:.1f}k")

        if trainer is None:
            trainer = pylt.Trainer(max_epochs=max_epochs, accelerator="auto", logger=False, enable_checkpointing=False, enable_progress_bar=False)

        train_dataloader = self.data.to_dataloader(train=True, batch_size=batch_size)
        val_dataloader = self.data.to_dataloader(train=False, batch_size=batch_size)
        trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    def get_params(self, trial):
        """
        Get the parameters for the model from the trial.
        """
        hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32, 64])
        attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [4, 8, 16])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam"])

        return {
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
        }
    