from whoc.wind_forecast.WindForecast import SVRForecast, generate_wind_field_df, ARIMAForecast
from whoc.wind_forecast.run_forecaster_validation import generate_wind_field_df
from whoc.wind_forecast.svr_forecast import SVRForecast
#from whoc.wind_forecast.arima_forecast import ARIMAForecast
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.dataset.split import slice_data_entry
import numpy as np
import polars as pl
import pandas as pd
import argparse
import yaml
import os
import sqlite3
import logging 
from floris import FlorisModel

import re
import random
import optuna
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage
from wind_forecasting.run_scripts.tuning import generate_df_setup_params
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from mpi4py import MPI
except Exception as e:
    logging.warning("Could not import MPI.")

def replace_env_vars(dirpath):
    env_vars = re.findall(r"(?:^|\/)\$(\w+)(?:\/|$)", dirpath)
    for env_var in env_vars:
        if env_var in os.environ:
            dirpath = dirpath.replace(f"${env_var}", os.environ[env_var])
    return dirpath

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-md", "--model", type=str, choices=["svr", "kf", "preview", "informer", "autoformer", "spacetimeformer", "arima"], required=True)
    parser.add_argument("-mcnf", "--model_config", type=str)
    parser.add_argument("-dcnf", "--data_config", type=str)
    parser.add_argument("-mp", "--multiprocessor", choices=["mpi", "cf", None], default=None)
    parser.add_argument("-msp", "--max_splits", type=int, required=False, default=None,
                        help="Number of test splits to use.")
    parser.add_argument("-ltv", "--limit_train_val", type=float, required=False, default=1,
                        help="Proportion of total training/validation data to randomly sample from during tuning.")
    parser.add_argument("-mst", "--max_steps", type=int, required=False, default=None,
                        help="Number of time steps to use.")
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    parser.add_argument("-m", "--mode", choices=["tune", "train"])
    parser.add_argument("-rd", "--reload_data", action="store_true", help="Whether to reload the train/validation data from the source, or to use existing .dat files.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")

    # parser.add_argument('--cores', required=False, default=None, help='Comma-separated list or range of core IDs (e.g., "0-9" or "10,11,12")')
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    
    if RUN_ONCE:
        logging.info("Parsing arguments and configuration yaml.")
    
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
        
    assert model_config["optuna"]["storage"]["backend"] in ["sqlite", "mysql", "journal"]
    
    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
        
    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
     
    fmodel = FlorisModel(data_config["farm_input_path"])

    storage_url = f"sqlite:///{model_config['optuna']['storage']['sqlite_path']}"
    study_name = f"{args.model}_ws_vert_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    study = optuna.create_study(
        study_name = study_name,
        storage=storage_url,
        direction=model_config['optuna']['direction'],
        load_if_exists=False,
    )
    
    if RUN_ONCE:
        logging.info("Creating datasets")
        
    data_module = DataModule(data_path=model_config["dataset"]["data_path"], 
                            normalization_consts_path=model_config["dataset"]["normalization_consts_path"],
                            normalized=True, 
                            n_splits=1, #model_config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - model_config["dataset"]["val_split"] - model_config["dataset"]["test_split"]),
                                val_split=model_config["dataset"]["val_split"], test_split=model_config["dataset"]["test_split"],
                                prediction_length=model_config["dataset"]["prediction_length"], context_length=model_config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=model_config["dataset"]["resample_freq"], target_suffixes=model_config["dataset"]["target_turbine_ids"],
                                    per_turbine_target=False, as_lazyframe=False, dtype=pl.Float32)
        
    # %% SETUP SEED
    if RUN_ONCE:
        logging.info(f"Setting random seed to {args.seed}")
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% INSTANTIATING MODEL
    if RUN_ONCE:
        logging.info("Instantiating model.")
          
    if args.model == "svr": 
        # NOTE: n_neighboring_turbines must be the same as in herculesinput_001.yaml
        forecaster = SVRForecast(measurements_timedelta=pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            controller_timedelta=None,
                            prediction_timedelta=data_module.prediction_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            context_timedelta=data_module.context_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            fmodel=fmodel,
                            true_wind_field=None,
                            kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200,
                                        n_neighboring_turbines=5, max_n_samples=None, 
                                        use_trained_models=False,
                                        model_config=model_config), # TODO move n_neighboring_turbines to cnofig
                            tid2idx_mapping=tid2idx_mapping,
                            turbine_signature=turbine_signature,
                            use_tuned_params=False)
        
    elif args.model == "arima":
        forecaster = ARIMAForecast(measurements_timedelta=pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            controller_timedelta=None,
                            study_name=study_name,
                            prediction_timedelta=data_module.prediction_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            context_timedelta=data_module.context_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            fmodel=fmodel,
                            true_wind_field=None,
                            #model_config=model_config,
                            kwargs=dict(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12), use_trained_models=False),
                            tid2idx_mapping=tid2idx_mapping,
                            turbine_signature=turbine_signature,
                            use_tuned_params=False)





        # original_save_dir = forecaster.model_save_dir
        # forecaster.model_save_dir = os.environ["TMPDIR"]
    # Use the WORKER_RANK variable set explicitly in the Slurm script's nohup block
    worker_id = int(os.environ.get('WORKER_RANK', 0))
    if RUN_ONCE:
        if "WORKER_RANK" in os.environ:
            logging.info(f"Determined worker rank from WORKER_RANK: {worker_id}")
        else:
            logging.info(f"Couldn't find WORKER_RANK env var, setting rank to {worker_id}.")
    
    # %% PREPARING DATA FOR TUNING
    if worker_id == 0 and RUN_ONCE:
        logging.info("Preparing data for tuning")
        if not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
            
        data_module.generate_splits(save=True, reload=reload, splits=["train", "val"])

    # get max_splits longest datasets
    if worker_id == 0 and (args.reload_data or reload):
        data_module.train_dataset = sorted(data_module.train_dataset, key=lambda ds: ds["target"].shape[1], reverse=True)
        data_module.val_dataset = sorted(data_module.val_dataset, key=lambda ds: ds["target"].shape[1], reverse=True)
        if args.max_splits:
            train_dataset = data_module.train_dataset[:args.max_splits]
            val_dataset = data_module.val_dataset[:args.max_splits]
        else:
            train_dataset = data_module.train_dataset
            val_dataset = data_module.val_dataset
        
        if args.max_steps:
            train_dataset = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in train_dataset]
            val_dataset = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in val_dataset]
            
        train_dataset = generate_wind_field_df(datasets=train_dataset, target_cols=data_module.target_cols, feat_dynamic_real_cols=data_module.feat_dynamic_real_cols)
        val_dataset = generate_wind_field_df(datasets=val_dataset, target_cols=data_module.target_cols, feat_dynamic_real_cols=data_module.feat_dynamic_real_cols)
        delattr(data_module, "train_dataset")
        delattr(data_module, "val_dataset")

        forecaster.prepare_data(dataset_splits={"train": train_dataset.partition_by("continuity_group"), "val": val_dataset.partition_by("continuity_group")}, 
                                scale=False, multiprocessor=args.multiprocessor, reload=args.reload_data)

        if RUN_ONCE:
            logging.info("Finished preparing data for tuning.")

    # %% TUNING MODEL
    
    optuna_storage = None
    if RUN_ONCE:
        logging.info(f"Initializing storage with restart_tuning={args.restart_tuning} on worker {worker_id}")
        
        db_setup_params = generate_df_setup_params(args.model, model_config)
        optuna_storage, _ = setup_optuna_storage(
            db_setup_params=db_setup_params,
            restart_tuning=args.restart_tuning,
            rank=0 if (worker_id == 0) else worker_id
        )
    
        logging.info("Running tune_hyperparameters_single")
    
    
    elif args.multiprocessor == "mpi":
        optuna_storage = comm.bcast(optuna_storage, root=0)
        
    if args.multiprocessor == "mpi":
        comm.Barrier()
    
    scaler_params = data_module.compute_scaler_params()
    
    if args.mode == "tune" and worker_id >= 0:
        
        if args.multiprocessor:
            logging.info(f"Using multiprocessor {args.multiprocessor}")

        #        historic_measurements = train_dataset["ws_horz_1"]
    
        historic_measurements = {
            col: train_dataset[col]
            for col in train_dataset.columns
            if col.startswith("ws_horz") 
        }

        logging.info(f"Tuning hyperparameters for all horizontal wind speeds: {list(historic_measurements.keys())}")

        ## Manually hyperparameter tuning for ARIMA

        df = pd.DataFrame(historic_measurements)
        avg_series = df.mean(axis=1)
       
        # plot the average series
        plt.figure(figsize=(12, 6))
        plt.plot(avg_series, label='Average Horizontal Wind Speed')
        plt.title('Average Vertical Wind Speed Across 7 Turbines', fontsize=22)
        plt.xlabel('Time (minutes)', fontsize=16)
        plt.ylabel('Average Wind Speed (m/s)', fontsize=16)
        plt.grid(True)
        plt.show()

        current_series = avg_series.copy()
        d = 0
        
        while True:
            print(f"nADF Test for d={d}")
            result = adfuller(current_series.dropna(), autolag='AIC')
            labels = ['ADF Statistic', 'p-value', 'Used Lag', 'Number of Observations Used']
            for value, label in zip(result[:4], labels):
                print(f"{label}: {value}")

            if result[1] < 0.05:
                print(f"Series is stationary at d={d}")
                break
            else:
                print(f"Series is non-stationary at d={d}, differencing the series.")
                current_series = current_series.diff().dropna()
                d += 1

                plt.figure(figsize=(12, 6))
                plt.plot(current_series.dropna())
                plt.title('First-Order Differenced Average Vertical Wind Speed Across 7 Turbines', fontsize=22)
                plt.xlabel('Time (minutes)', fontsize=16)
                plt.ylabel('Differenced Wind Speed (m/s)', fontsize=16)
                plt.grid(True)
                plt.show()
        print(f"Optimal d: {d}")

        # HYPERPARAMETER TUNING FOR P AND Q
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=80)
        sampled_series = current_series.iloc[:1000]# Downsample the series for ACF and PACF plots
        plot_acf(sampled_series.dropna(), lags=20, ax=ax1)
        ax1.set_title("Autocorrelation Function (ACF)", fontsize=17)
        ax1.set_xlabel("Lag (minutes)", fontsize=17)
        ax1.set_ylabel("Autocorrelation (-)", fontsize=17)
        ax1.tick_params(axis='both', labelsize=12)

        plot_pacf(sampled_series.dropna(), lags=20, ax=ax2, method='ywm')

        ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=17)
        ax2.set_xlabel("Lag (minutes)", fontsize=17)
        ax2.set_ylabel("Partial Autocorrelation (-)", fontsize=17)
        ax2.tick_params(axis='both', labelsize=12)

        fig.suptitle(
            "ACF and PACF of First-Order Differenced Average Horizontal Wind Speed Across 7 Turbines",
            fontsize=24
        )

        plt.tight_layout()
        plt.show()
            
        ## Here the manually tuning of ARIMA hyperparameters ends



        if args.tune:
            forecaster.tune_hyperparameters_single(historic_measurements=historic_measurements, 
                                                    storage=optuna_storage,
                                                    n_trials_per_worker=model_config["optuna"]["n_trials_per_worker"], 
                                                    seed=args.seed,
                                                    config=model_config,
                                                    worker_id=0 if RUN_ONCE and (worker_id == 0) else worker_id,
                                                    multiprocessor=args.multiprocessor,
                                                    limit_train_val=args.limit_train_val)
                                            #  trial_protection_callback=handle_trial_with_oom_protection)

        # %% After tuning completes
        logging.info("Optuna hyperparameter tuning completed.")
        
    elif args.mode == "train":
        # %% TRAINING MODEL
        if args.model == "svr":
            logging.info("Training model using best hyperparameters.")
            forecaster.set_tuned_params(storage=optuna_storage, study_name=forecaster.study_name)
            forecaster.train_all_outputs(outputs=data_module.target_cols, scale=False, 
                                        multiprocessor=args.multiprocessor, 
                                        retrain_models=True,
                                        scaler_params=scaler_params)
        
        # %% ARIMA TRAINING
        if args.model == "arima":
            logging.info("Training ARIMA model using best hyperparameters.")
            forecaster.set_tuned_params(storage=optuna_storage, study_name=study_name, data=train_dataset)
            forecaster.train_all_outputs(outputs=data_module.target_cols, 
                scale=False,
                multiprocessor=args.multiprocessor, 
                retrain_models=True,
                scaler_params=scaler_params)
        
        boxcox_path = os.path.join(forecaster.model_save_dir, "boxcox_params.pkl")
        forecaster.save_boxcox_params(boxcox_path)
        logging.info(f"Saved Box-Cox parameters to {boxcox_path}")
        # %% After training completes
        logging.info("Training completed.")
        