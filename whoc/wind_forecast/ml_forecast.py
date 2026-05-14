from typing import Union
from dataclasses import dataclass
import os
import re
import types

import inspect
from memory_profiler import profile
import torch


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
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator as TactisEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import (
    TACTiS2LightningModule as TactisLightningModule,
)

from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.run_scripts.testing import get_checkpoint, load_estimator_from_checkpoint

import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from whoc.wind_forecast.wind_forecast_base import WindForecast

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sns.set_palette("Paired")


@dataclass
class MLForecast(WindForecast):
    """Wind speed component forecasting using machine learning models."""

    is_probabilistic = True

    def __post_init__(self):
        super().__post_init__()
        self.n_prediction_interval = 1
        self.model_key = self.kwargs["model_key"]
        self.model_config = self.kwargs["model_config"]
        self.device = None
        self.resample = self.kwargs.get("resample", True)  # Default to True if not specified

        # don't need this, can load hyperparamas from checkpoint
        # if self.use_tuned_params:
        #     try:
        #         logging.info("Getting tuned parameters")
        #         # study_name, backend, storage_dir
        #         tuned_params = get_tuned_params(storage=self.kwargs["optuna_storage"],
        #                                         study_name=f"tuning_{self.model_key}_{self.model_config['experiment']['run_name']}")
        #         logging.info(f"Declaring estimator {self.model_key.capitalize()} with tuned parameters")
        #         self.model_config["dataset"].update({k: v for k, v in tuned_params.items() if k in self.model_config["dataset"]})
        #         self.model_config["model"][self.model_key].update({k: v for k, v in tuned_params.items() if k in self.model_config["model"][self.model_key]})
        #         self.model_config["trainer"].update({k: v for k, v in tuned_params.items() if k in self.model_config["trainer"]})
        #     except Exception as e:
        #         logging.warning(e)
        #         logging.info(f"Declaring estimator {self.model_key.capitalize()} with default parameters")
        # else:
        #     logging.info(f"Declaring estimator {self.model_key.capitalize()} with default parameters")

        # self.context_timedelta = self.model_config["dataset"]["context_length"] \
        #     * pd.Timedelta(self.model_config["dataset"]["resample_freq"]).to_pytimedelta()

        estimator_class = globals()[f"{self.model_key.capitalize()}Estimator"]
        lightning_module_class = globals()[f"{self.model_key.capitalize()}LightningModule"]
        distr_output_class = globals()[self.model_config["model"]["distr_output"]["class"]]

        metric = "val_loss_epoch"
        mode = "min"
        # log_dir = os.path.join(self.model_config["trainer"]["default_root_dir"], "lightning_logs")
        # "/Users/ahenry/Documents/toolboxes/wind_forecasting/logging/informer_aoifemac_awaken/wind_forecasting/z55orlbf/checkpoints/epoch=7-step=8000.ckpt"

        # "/Users/ahenry/Documents/toolboxes/wind_forecasting/logging/wind_forecasting_awaken_pred60_informer/20250506_152133_0_0/epoch=0-step=100-val_loss=0.15.ckpt"
        log_dir = os.path.join(
            self.model_config["experiment"]["log_dir"],
            f"{self.model_config['experiment']['project_name']}_{self.model_key}",
        )
        checkpoint_path = get_checkpoint(
            checkpoint=self.kwargs["model_checkpoint"], metric=metric, mode=mode, log_dir=log_dir
        )

        if checkpoint_path is not None:
            checkpoint_hparams = load_estimator_from_checkpoint(
                checkpoint_path,
                lightning_module_class,
                self.model_config,
                self.model_key,
                train=False,
            )
            logging.info(
                f"Loaded checkpoint from {checkpoint_path}"
            )  # with hparams: {checkpoint_hparams}")
            self.data_module = DataModule(
                normalized_data_path=self.model_config["dataset"]["data_path"],
                n_splits=self.model_config["dataset"]["n_splits"],
                continuity_groups=None,
                train_split=(
                    1.0
                    - self.model_config["dataset"]["val_split"]
                    - self.model_config["dataset"]["test_split"]
                ),
                val_split=self.model_config["dataset"]["val_split"],
                test_split=self.model_config["dataset"]["test_split"],
                batch_size=self.model_config["dataset"]["batch_size"],
                as_lazyframe=True,
                # Use lengths determined above, converted to seconds
                prediction_length=(
                    checkpoint_hparams["prediction_length_int"] * checkpoint_hparams["freq"]
                ).total_seconds(),
                context_length=(
                    checkpoint_hparams["context_length_int"] * checkpoint_hparams["freq"]
                ).total_seconds(),
                target_prefixes=["ws_horz", "ws_vert"],
                feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                freq=checkpoint_hparams["freq_str"],  # Use original freq string
                use_normalization=self.model_config["dataset"].get("normalize", True),
                target_suffixes=self.model_config["dataset"]["target_turbine_ids"],
                per_turbine_target=self.model_config["dataset"]["per_turbine_target"],
                dtype=None,
                normalization_consts_path=self.model_config["dataset"]["normalization_consts_path"],
            )

            self.data_module.get_dataset_info()
            self.scaler_params = self.data_module.compute_scaler_params()
            logging.info(
                "Re-initialized DataModule and recomputed scaler_params based on checkpoint/config."
            )

            # Determine correct stage based on checkpoint epoch
            if self.model_key == "tactis":
                checkpoint_epoch = checkpoint_hparams["checkpoint"].get("epoch")
                # Ensure stage2_start_epoch is retrieved from hparams within init_args now
                stage2_start_epoch = checkpoint_hparams["init_args"].get("stage2_start_epoch")

                if checkpoint_epoch is None or stage2_start_epoch is None:
                    logging.warning(
                        "Could not determine stage from checkpoint epoch or hparams. Defaulting to Stage 2 for TACTiS loading."
                    )
                    correct_stage = 2  # Default assumption if info missing
                elif checkpoint_epoch >= stage2_start_epoch:
                    correct_stage = 2
                    logging.info(
                        f"Checkpoint epoch ({checkpoint_epoch}) >= stage2_start_epoch ({stage2_start_epoch}). Setting TACTiS stage to 2 for loading."
                    )
                else:
                    correct_stage = 1
                    logging.info(
                        f"Checkpoint epoch ({checkpoint_epoch}) < stage2_start_epoch ({stage2_start_epoch}). Setting TACTiS stage to 1 for loading."
                    )

                checkpoint_hparams["init_args"]["stage"] = (
                    correct_stage  # Set the stage in init_args BEFORE loading
                )

            # Instantiate the model using load_from_checkpoint, passing the correctly determined stage
            try:
                # Pass the init_args (which includes the correct stage) to load_from_checkpoint
                # Use strict=False to ignore the save_hyperparameters error internally,
                # as we've already ensured the model is configured correctly via init_args.
                model = lightning_module_class.load_from_checkpoint(
                    checkpoint_path,
                    strict=False,  # Allow loading even if save_hyperparameters fails internally
                    **checkpoint_hparams["init_args"],
                )
                logging.info(
                    f"Successfully loaded model from checkpoint {checkpoint_path} using init_args including stage {correct_stage if self.model_key == 'tactis' else 'N/A'}."
                )

            except Exception as e:
                logging.error(f"Error during LightningModule re-instantiation: {e}", exc_info=True)
                raise Exception(e)

            # self.data_module.context_length = init_args["model_config"]["context_length"]
            self.context_timedelta = self.data_module.context_length * pd.Timedelta(
                self.data_module.freq
            )
            self.model_prediction_timedelta = self.data_module.prediction_length * pd.Timedelta(
                self.data_module.freq
            )
            assert self.model_prediction_timedelta >= self.prediction_timedelta, (
                f"model fetched from checkpoint {checkpoint_path} is tuned for shorter prediction timedelta {self.model_prediction_timedelta} than the given one {self.prediction_timedelta}!"
            )

            self.n_context = int(
                self.context_timedelta / self.measurements_timedelta
            )  # number of simulation time steps in a context horizon
            # self.n_prediction = int(self.prediction_timedelta / self.measurements_timedelta) # number of simulation time steps in a prediction horizon

            # Prepare all arguments in a dictionary # TODO HIGH add limit_train_batches and batch_size to hparams, and also set in data_module above
            estimator_kwargs = {
                "freq": self.data_module.freq,
                "prediction_length": self.data_module.prediction_length,
                "num_feat_dynamic_real": self.data_module.num_feat_dynamic_real,
                "num_feat_static_cat": self.data_module.num_feat_static_cat,
                "cardinality": self.data_module.cardinality,
                "num_feat_static_real": self.data_module.num_feat_static_real,
                "input_size": self.data_module.num_target_vars,
                "scaling": "std"
                if checkpoint_hparams["init_args"]["model_config"]["scaling"] in ["True", "std"]
                else False,  # Scaling handled externally or internally by TACTiS
                "lags_seq": checkpoint_hparams["init_args"]["model_config"][
                    "lags_seq"
                ],  # TACTiS doesn't typically use lags
                "time_features": [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
                "batch_size": len(self.data_module.target_suffixes)
                if self.data_module.per_turbine_target
                else 1,  # self.data_module.batch_size, #self.model_config["dataset"].setdefault("batch_size", 128),
                "num_batches_per_epoch": self.model_config["trainer"].setdefault(
                    "limit_train_batches", 1000
                ),
                "context_length": self.data_module.context_length,
                "train_sampler": ExpectedNumInstanceSampler(
                    num_instances=1.0,
                    min_past=self.data_module.context_length,
                    min_future=self.data_module.prediction_length,
                ),
                "validation_sampler": ValidationSplitSampler(
                    min_past=self.data_module.context_length,
                    min_future=self.data_module.prediction_length,
                ),
                "trainer_kwargs": self.model_config["trainer"],
                # Include distr_output initially, will be removed conditionally
                #             "distr_output": distr_output_class(dim=self.data_module.num_target_vars, **self.model_config["model"]["distr_output"]["kwargs"]),
                "num_parallel_samples": checkpoint_hparams["init_args"]["model_config"][
                    "num_parallel_samples"
                ]
                if self.model_key == "tactis"
                else 100,  # Default 100 if not specified
            }
            estimator_sig = inspect.signature(estimator_class.__init__)
            estimator_params = [param.name for param in estimator_sig.parameters.values()]

            # Add model-specific arguments. Note that some params, such as num_feat_dynamic_real, are changed within Model, and so can't be used for estimator class
            model_config_source = checkpoint_hparams["init_args"]["model_config"]
            if model_config_source:
                estimator_kwargs.update(
                    {
                        k: v
                        for k, v in model_config_source.items()
                        if k in estimator_params and not hasattr(self.data_module, k)
                    }
                )
            else:
                logging.warning(
                    f"Could not find 'model_config' in checkpoint hparams or instance config for model {self.model_key}."
                )

            # Add distr_output only if the model is NOT tactis
            if self.model_key != "tactis":
                estimator_kwargs["distr_output"] = distr_output_class(
                    dim=self.data_module.num_target_vars,
                    **self.model_config["model"]["distr_output"]["kwargs"],
                )

            # Add use_pytorch_dataloader flag if specified in dataset config
            if "use_pytorch_dataloader" in self.model_config["dataset"]:
                estimator_kwargs["use_pytorch_dataloader"] = self.model_config["dataset"][
                    "use_pytorch_dataloader"
                ]
                logging.info(
                    f"Setting use_pytorch_dataloader={self.model_config['dataset']['use_pytorch_dataloader']} from config"
                )

            logging.info(f"Using final estimator_kwargs:\n {estimator_kwargs}")
            estimator = estimator_class(**estimator_kwargs)
            self.use_internal_scaling = estimator_kwargs["scaling"] and (
                estimator_kwargs["scaling"] != "False"
            )

            # TODO replace this with pytorch_dataloader?
            transformation = estimator.create_transformation(use_lazyframe=False)

            # Conditionally Create Forecast Generator
            if self.model_key == "tactis":
                # TACTiS uses SampleForecastGenerator internally for prediction
                # because its foweard pass returns samples not distribution parameters
                logging.info(f"Using SampleForecastGenerator for TACTiS model.")
                forecast_generator = SampleForecastGenerator()
            else:
                # Other models use DistributionForecastGenerator based on their distr_output
                logging.info(f"Using DistributionForecastGenerator for {self.model_key} model.")
                # Ensure estimator has distr_output before accessing
                if not hasattr(estimator, "distr_output"):
                    raise AttributeError(
                        f"Estimator for model '{self.model_key}' is missing 'distr_output' attribute needed for DistributionForecastGenerator."
                    )
                forecast_generator = DistributionForecastGenerator(estimator.distr_output)

            self.predictor = estimator.create_predictor(
                transformation, model, forecast_generator=forecast_generator
            )
            # self.data_module.freq = pd.Timedelta(self.data_module.freq).to_pytimedelta()
            self.sample_predictor = estimator.create_predictor(
                transformation, model, forecast_generator=SampleForecastGenerator()
            )
        else:
            raise FileNotFoundError(f"Cannot find checkpoint file in {log_dir}")

    def reset(self, **kwargs):
        if "assigned_gpu" in kwargs and kwargs["assigned_gpu"]:
            # os.environ["CUDA_VISIBLE_DEVICES"] = self.kwargs["assigned_gpu"]
            self.assigned_gpu = kwargs["assigned_gpu"]
            logging.info(
                f"Using assigned_gpu = {self.assigned_gpu} in MLForecast for {self.model_key} and self.prediction_timedelta = {self.prediction_timedelta}."
            )
            # torch.cuda.set_device(self.assigned_gpu)
            self.device = f"cuda:{self.assigned_gpu}"
            # Clear GPU memory before starting
            torch.cuda.empty_cache()
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            self.assigned_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
            logging.info(
                f"Using assigned_gpu = {os.environ['CUDA_VISIBLE_DEVICES']} in MLForecast for {self.model_key} and self.prediction_timedelta = {self.prediction_timedelta}."
            )
            # torch.cuda.set_device(self.assigned_gpu)
            self.device = "cuda"
            # Clear GPU memory before starting
            torch.cuda.empty_cache()
        else:
            self.assigned_gpu = None
            self.device = "cpu"

        self.predictor = self.predictor.to(self.device)
        if (
            hasattr(self, "sample_predictor") and self.sample_predictor is not None
        ):  # Check if it exists and is initialized
            self.sample_predictor = self.sample_predictor.to(self.device)

    def _generate_test_data(self, historic_measurements: pl.DataFrame):
        # resample data to frequency model was trained on
        current_time = historic_measurements.select(pl.col("time").last()).item()
        # Convert freq string to Timedelta for comparison and calculations
        # Ensure self.data_module.freq is treated as a string before conversion
        data_module_freq_td = pd.Timedelta(str(self.data_module.freq))
        if data_module_freq_td != self.measurements_timedelta:
            if self.measurements_timedelta < data_module_freq_td:
                # historic_measurements = historic_measurements.rolling(index_column="time", closed="both",
                #                                                       period=data_module_freq_td,
                #                                                       offset=(historic_measurements.select(pl.col("time").last() - pl.col("time").first()).item() % (data_module_freq_td))+(data_module_freq_td))\
                #                                                           .agg(cs.numeric().mean())
                historic_measurements = historic_measurements.with_columns(
                    pl.col("time").dt.round(data_module_freq_td)
                )
                last_rounded_time = historic_measurements.select(pl.col("time").last()).item()
                historic_measurements = (
                    historic_measurements.with_columns(
                        time=pl.col("time") + (current_time - last_rounded_time)
                    )
                    .group_by("time", maintain_order=True)
                    .agg(cs.numeric().mean())
                )
            else:  # TODO this needs to be time shifted as above
                historic_measurements = historic_measurements.upsample(
                    time_column="time", every=data_module_freq_td
                ).fill_null(strategy="forward")  # Use Timedelta here

        assert historic_measurements.select(pl.col("time").last()).item() == current_time
        historic_measurements = historic_measurements.with_columns(cs.numeric().cast(pl.Float32))
        # test_data must be iterable where each item generated is a dict with keys start, target, item_id, and feat_dynamic_real
        # this should include measurements at all turbines
        # repeats last value of feat_dynamic_reals (ie. nd_cos, nd_sin) for future_prediction TODO change this for MPC
        if self.data_module.per_turbine_target:
            test_data = (
                {
                    FieldName.ITEM_ID: f"TURBINE{turbine_id}",
                    FieldName.START: pd.Period(
                        historic_measurements.select(pl.col("time").first()).item(),
                        freq=self.data_module.freq,
                    ),
                    FieldName.TARGET: historic_measurements.select(
                        [f"{pfx}_{turbine_id}" for pfx in self.data_module.target_prefixes]
                    )
                    .to_numpy()
                    .T,
                    FieldName.FEAT_STATIC_CAT: np.array([t]),
                    FieldName.FEAT_DYNAMIC_REAL: pl.concat(
                        [
                            historic_measurements.select(
                                [
                                    f"{pfx}_{turbine_id}"
                                    for pfx in self.data_module.feat_dynamic_real_prefixes
                                ]
                            ),
                            historic_measurements.select(
                                [
                                    pl.col(f"{pfx}_{turbine_id}")
                                    .last()
                                    .repeat_by(
                                        int(
                                            self.model_prediction_timedelta.total_seconds()
                                            / data_module_freq_td.total_seconds()
                                        )
                                    )
                                    .explode()  # Use Timedelta seconds
                                    for pfx in self.data_module.feat_dynamic_real_prefixes
                                ]
                            ),
                        ],
                        how="vertical",
                    )
                    .to_numpy()
                    .T,
                }
                for t, turbine_id in enumerate(self.data_module.target_suffixes)
            )
        else:
            test_data = [
                {  # Make this is an iterable (list of one dict)
                    FieldName.ITEM_ID: "AGGREGATED_TIMESERIES",
                    FieldName.START: pd.Period(
                        historic_measurements.select(pl.col("time").first()).item(),
                        freq=self.data_module.freq,
                    ),
                    FieldName.TARGET: historic_measurements.select(self.data_module.target_cols)
                    .to_numpy()
                    .T,
                    FieldName.FEAT_DYNAMIC_REAL: pl.concat(
                        [
                            historic_measurements.select(self.data_module.feat_dynamic_real_cols),
                            historic_measurements.select(
                                [
                                    pl.col(col)
                                    .last()
                                    .repeat_by(
                                        int(
                                            self.model_prediction_timedelta.total_seconds()
                                            / data_module_freq_td.total_seconds()
                                        )
                                    )
                                    .explode()  # Use Timedelta seconds
                                    for col in self.data_module.feat_dynamic_real_cols
                                ]
                            ),
                        ],
                        how="vertical",
                    )
                    .to_numpy()
                    .T,
                }
            ]
        return test_data

    def predict_sample(
        self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time, n_samples: int
    ):
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True

        # normalize historic measurements
        if not self.use_internal_scaling:
            features = list(self.scaler_params["offset_"].keys())
            historic_measurements = historic_measurements.with_columns(
                [
                    (pl.col(feat) - self.scaler_params["offset_"][feat])
                    / self.scaler_params["scale_"][feat]
                    for feat in features
                ]
            )

        test_data = self._generate_test_data(historic_measurements)

        pred = self.sample_predictor.predict(
            test_data, num_samples=n_samples, output_distr_params=False
        )

        if self.data_module.per_turbine_target:
            # Convert generator to list so we can access the first forecast's index
            pred_list = list(pred)
            if not pred_list:
                pred_df = pl.DataFrame()
            else:
                # Use index from first forecast for time values
                time_index = pred_list[0].index.to_timestamp()
                # Assuming all turbine_pred objects have the same prediction_length
                prediction_len = pred_list[0].prediction_length

                turbine_feature_dfs = []
                for t, turbine_pred in enumerate(pred_list):
                    # Create a DataFrame for the current turbine's features
                    df_turbine_features = pl.DataFrame(
                        {
                            # Keys are generic prefixes, values are flattened samples for those features
                            prefix: turbine_pred.samples[:, :, c_prefix].flatten()
                            for c_prefix, prefix in enumerate(self.data_module.target_prefixes)
                        }
                    ).rename(
                        {
                            # Rename generic prefixes to turbine-specific column names
                            prefix: f"{prefix}_{self.data_module.target_suffixes[t]}"
                            for prefix in self.data_module.target_prefixes
                        }
                    )
                    turbine_feature_dfs.append(df_turbine_features)

                # Concatenate all per-turbine feature DataFrames horizontally
                if not turbine_feature_dfs:  # Should not happen if pred_list is not empty
                    features_df = pl.DataFrame()
                elif len(turbine_feature_dfs) == 1:
                    features_df = turbine_feature_dfs[0]
                else:
                    features_df = pl.concat(turbine_feature_dfs, how="horizontal")

                # Create the common time and sample DataFrame
                # Number of rows must match features_df (n_samples * prediction_len)
                df_time_sample = pl.DataFrame(
                    {
                        "time": np.tile(
                            time_index, n_samples
                        ),  # time_index has length prediction_len
                        "sample": np.repeat(np.arange(n_samples), prediction_len),
                    }
                )

                # Combine time/sample DataFrame with the features DataFrame
                if features_df.is_empty():
                    pred_df = df_time_sample  # Should only contain time and sample if no features
                else:
                    pred_df = pl.concat([df_time_sample, features_df], how="horizontal")

                pred_df = pred_df.sort(["sample", "time"])  # Sort at the end
        else:
            pred = next(pred)
            pred_df = pl.DataFrame(
                data={
                    **{
                        "time": np.tile(pred.index.to_timestamp(), (n_samples,)),
                        "sample": np.repeat(np.arange(n_samples), (pred.prediction_length,)),
                    },
                    **{
                        col: pred.samples[:, :, c].flatten()
                        for c, col in enumerate(self.data_module.target_cols)
                    },
                }
            ).sort(by=["sample", "time"])

        # denormalize data using scaler_params
        if not pred_df.is_empty() and not self.use_internal_scaling:
            pred_df = pred_df.with_columns(
                [
                    (pl.col(feat) * self.scaler_params["scale_"][feat])
                    + self.scaler_params["offset_"][feat]
                    for feat in features
                ]
            )

        pred_df = pred_df.filter(pl.col("time") <= (current_time + self.prediction_timedelta))
        # check if the data that trained the model differs from the frequency of historic_measurments
        # Convert freq string to Timedelta for comparison and calculations
        data_module_freq_td = pd.Timedelta(str(self.data_module.freq))
        if data_module_freq_td != self.measurements_timedelta:
            # resample historic measurements to historic_measurements frequency and return as pandas dataframe
            if self.measurements_timedelta > data_module_freq_td:  # Use Timedelta here
                pred_df = (
                    pred_df.with_columns(
                        time=pl.col("time").dt.round(data_module_freq_td)  # Use Timedelta here
                        + pl.duration(
                            seconds=pred_df.select(
                                pl.col("time").last().dt.second()
                                % data_module_freq_td.total_seconds()
                            ).item()
                        )
                    )
                    .group_by("time")
                    .agg(cs.numeric().mean())
                    .sort("time")
                )
            else:
                pred_df = pred_df.upsample(time_column="time", every=data_module_freq_td).fill_null(
                    strategy="forward"
                )  # Use Timedelta here

        if return_pl:
            return pred_df
        else:
            return pred_df.to_pandas()

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):

        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True

        # select mean features and rename
        pred_df = self.predict_distr(historic_measurements, current_time)
        pred_df = pred_df.select(
            pl.col("time"),
            cs.starts_with("loc_").name.map(
                lambda original_col: (
                    re.search("(?<=loc_)(.*)", original_col).group()
                    if "loc" in original_col
                    else original_col
                )
            ),
        )

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

        if historic_measurements.select(pl.len()).item() >= self.n_context:
            # normalize historic measurements ONLY IF NOT using internal scaling like tactis
            features = list(self.scaler_params["offset_"].keys())
            if not self.use_internal_scaling:
                historic_measurements = historic_measurements.with_columns(
                    [
                        (pl.col(feat) - self.scaler_params["offset_"][feat])
                        / self.scaler_params["scale_"][feat]
                        for feat in features
                    ]
                )
            else:
                pass
            test_data = self._generate_test_data(historic_measurements)
            logging.info(
                f"Using {torch.cuda.device_count()} GPU devices: {self.device} at {current_time} to make predictions for {self.model_key} with prediction_timedelta {self.prediction_timedelta}."
            )

            pred_iter = self.predictor.predict(
                test_data,
                num_samples=1
                if self.model_key != "tactis"
                else self.predictor.network.model.num_parallel_samples,
                output_distr_params={
                    "loc": "mean",
                    "cov_factor": "cov_factor",
                    "cov_diag": "cov_diag",
                },
            )

            if self.data_module.per_turbine_target:
                # Handle multiple forecast objects if per_turbine_target is True

                pred_list = list(pred_iter)

                # logging.info(f"pred_list[0] is on device {pred_list[0].samples.device}")

                if self.model_key == "tactis":
                    for p in range(len(pred_list)):
                        pred_list[p].distribution = types.SimpleNamespace()
                        # logging.info(f"TACTiS samples are stored on device {pred_list[p].samples.get_device()}")
                        # TODO is there any advantage to loading this onto GPU before computing mean,stddev?
                        samples_tensor = torch.from_numpy(pred_list[p].samples).to(
                            self.predictor.device
                        )  # .to(self.predictor.device)
                        pred_list[p].distribution.mean = samples_tensor.mean(dim=0)
                        pred_list[p].distribution.stddev = samples_tensor.std(dim=0)

                pred_df = pl.concat(
                    [
                        pl.DataFrame(
                            data={
                                **{"time": turbine_pred.index.to_timestamp().as_unit("us")},
                                **{
                                    f"loc_{col}": turbine_pred.distribution.mean[:, c].cpu().numpy()
                                    for c, col in enumerate(self.data_module.target_prefixes)
                                },
                                **{
                                    f"sd_{col}": turbine_pred.distribution.stddev[:, c]
                                    .cpu()
                                    .numpy()
                                    for c, col in enumerate(self.data_module.target_prefixes)
                                },
                            }
                        )
                        .rename(
                            {
                                f"{param}_{col}": f"{param}_{col}_{self.data_module.target_suffixes[t]}"
                                for param in ["loc", "sd"]
                                for col in self.data_module.target_prefixes
                            }
                        )
                        .sort(by=["time"])
                        for t, turbine_pred in enumerate(pred_list)
                    ],
                    how="align",
                )
            else:
                # single forecast object
                pred = next(pred_iter)  # Get the single forecast object

                if self.model_key == "tactis":
                    pred.distribution = types.SimpleNamespace()
                    samples_tensor = torch.from_numpy(pred.samples)  # .to(self.predictor.device)
                    pred.distribution.mean = samples_tensor.to(self.predictor.device).mean(dim=0)
                    pred.distribution.stddev = samples_tensor.std(dim=0)

                pred_df = pl.DataFrame(
                    data={
                        **{"time": pred.index.to_timestamp().as_unit("us")},
                        **{
                            f"loc_{col}": pred.distribution.mean[:, c].cpu().numpy()
                            for c, col in enumerate(self.data_module.target_cols)
                        },
                        **{
                            f"sd_{col}": pred.distribution.stddev[:, c].cpu().numpy()
                            for c, col in enumerate(self.data_module.target_cols)
                        },
                    }
                ).sort(by=["time"])

            # denormalize data ONLY IF NOT using internal scaling like tactis
            if not self.use_internal_scaling:
                pred_df = pred_df.with_columns(
                    [
                        (pl.col(f"loc_{col}") * self.scaler_params["scale_"][col])
                        + self.scaler_params["offset_"][col]
                        for col in self.data_module.target_cols
                    ]
                ).with_columns(
                    [
                        pl.col(f"sd_{col}") * self.scaler_params["scale_"][col]
                        for col in self.data_module.target_cols
                    ]
                )
            else:
                pass

            pred_df = pred_df.filter(pl.col("time") <= (current_time + self.prediction_timedelta))
            # check if the data that trained the model differs from the frequency of historic_measurments
            # Convert freq string to Timedelta for comparison and calculations
            if self.resample:
                data_module_freq_td = pd.Timedelta(str(self.data_module.freq))
                if data_module_freq_td != self.measurements_timedelta:
                    # resample historic measurements to historic_measurements frequency and return as pandas dataframe
                    if self.measurements_timedelta > data_module_freq_td:  # Use Timedelta here
                        pred_df = (
                            pred_df.with_columns(
                                time=pl.col("time").dt.round(
                                    data_module_freq_td
                                )  # Use Timedelta here
                                + pl.duration(
                                    seconds=pred_df.select(
                                        pl.col("time").last().dt.second()
                                        % data_module_freq_td.total_seconds()
                                    ).item()
                                )
                            )
                            .group_by("time")
                            .agg(cs.numeric().mean())
                            .sort("time")
                        )
                    else:
                        pred_df = pred_df.upsample(
                            time_column="time", every=self.measurements_timedelta
                        ).fill_null(strategy="forward")  # Use Timedelta here
        else:
            # not enough data points to train SVR, assume persistence
            logging.info(
                f"Not enough data points at time {current_time} to train ML, have {historic_measurements.select(pl.len()).item()} but require {self.n_context}, assuming persistence instead."
            )
            pred_slice = self.get_pred_interval(current_time)
            pred_df = pl.DataFrame(
                data={
                    **{"time": pred_slice},
                    **{
                        f"loc_{col}": historic_measurements.select(
                            pl.col(col).last().repeat_by(len(pred_slice)).explode()
                        )
                        for col in self.data_module.target_cols
                    },
                    **{
                        f"sd_{col}": [0] * len(pred_slice)
                        for c, col in enumerate(self.data_module.target_cols)
                    },
                }
            )

        if return_pl:
            return pred_df
        else:
            return pred_df.to_pandas()
