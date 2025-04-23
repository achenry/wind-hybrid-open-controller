from whoc.wind_forecast.WindForecast import SVRForecast, generate_wind_field_df
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.dataset.split import slice_data_entry
import numpy as np
import polars as pl
import pandas as pd
import argparse
import yaml
import os
import logging 
from floris import FlorisModel
import shutil
import re
import random
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage
from wind_forecasting.run_scripts.tuning import generate_df_setup_params

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
    parser.add_argument("-md", "--model", type=str, choices=["svr", "kf", "preview", "informer", "autoformer", "spacetimeformer"], required=True)
    parser.add_argument("-mcnf", "--model_config", type=str)
    parser.add_argument("-dcnf", "--data_config", type=str)
    parser.add_argument("-m", "--multiprocessor", choices=["mpi", "cf", None], default=None)
    parser.add_argument("-msp", "--max_splits", type=int, required=False, default=None,
                        help="Number of test splits to use.")
    parser.add_argument("-mst", "--max_steps", type=int, required=False, default=None,
                        help="Number of time steps to use.")
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
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
                            model_config=model_config,
                            kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200,
                                        n_neighboring_turbines=5, max_n_samples=None, 
                                        use_trained_models=False),
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
    if worker_id == 0:
        if RUN_ONCE:
            logging.info("Preparing data for tuning")
            if not os.path.exists(data_module.train_ready_data_path):
                data_module.generate_datasets()
                reload = True
            else:
                reload = False
            
            data_module.generate_splits(save=True, reload=reload, splits=["train", "val"])
            
        # get max_splits longest datasets
        data_module.generate_splits(save=True, reload=False, splits=["train", "val"])
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
                                scale=False, multiprocessor=args.multiprocessor)

        if RUN_ONCE:
            # logging.info(f"Moving prepared data from {forecaster.model_save_dir} to {original_save_dir}.")
            # filenames = os.listdir(forecaster.model_save_dir)
            # for fn in filenames:
            #     shutil.move(os.path.join(forecaster.model_save_dir, fn), original_save_dir)
            logging.info("Finished preparing data for tuning.")

    # %% TUNING MODEL
    if worker_id > 0:
        
        scaler_params = data_module.compute_scaler_params()
        
        logging.info("Initializing storage")
        db_setup_params = generate_df_setup_params(args.model, model_config)
        optuna_storage = setup_optuna_storage(
            db_setup_params=db_setup_params,
            restart_tuning=args.restart_tuning,
            rank=0 if RUN_ONCE and (worker_id == 0) else worker_id
        )
        
        logging.info("Running tune_hyperparameters_multi")
        #{"type": "hyperband", "min_resource": 2, "max_resource": 5, "reduction_factor": 3, "percentile": 25}
        
        forecaster.tune_hyperparameters_single(storage=optuna_storage,
                                                n_trials_per_worker=model_config["optuna"]["n_trials_per_worker"], 
                                                seed=args.seed,
                                                config=model_config,
                                                worker_id=0 if RUN_ONCE and (worker_id == 0) else worker_id,
                                                multiprocessor=args.multiprocessor)
                                        #  trial_protection_callback=handle_trial_with_oom_protection)

        # %% TRAINING MODEL
        logging.info("Training model using best hyperparameters.")
        forecaster.set_tuned_params(storage=optuna_storage, study_name=forecaster.study_name)
        forecaster.train_all_outputs(outputs=data_module.target_cols, scale=False, 
                                     multiprocessor=args.multiprocessor, 
                                     retrain_models=True,
                                     scaler_params=scaler_params)
        
        # %% After training completes
        logging.info("Optuna hyperparameter tuning completed.")