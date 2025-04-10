from whoc.wind_forecast.WindForecast import SVRForecast, generate_wind_field_df
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.dataset.split import split, slice_data_entry
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import argparse
import yaml
import os
import logging 
from floris import FlorisModel
import gc
import re
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    
    logging.info("Parsing arguments and configuration yaml.")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-md", "--model", type=str, choices=["svr", "kf", "preview", "informer", "autoformer", "spacetimeformer"], required=True)
    parser.add_argument("-mcnf", "--model_config", type=str)
    parser.add_argument("-dcnf", "--data_config", type=str)
    parser.add_argument("-sn", "--study_name", type=str)
    parser.add_argument("-m", "--multiprocessor", choices=["mpi", "cf"], default="cf")
    parser.add_argument("-msp", "--max_splits", type=int, required=False, default=None,
                        help="Number of test splits to use.")
    parser.add_argument("-mst", "--max_steps", type=int, required=False, default=None,
                        help="Number of time steps to use.")
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("-i", "--initialize", action="store_true")
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
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
                                    per_turbine_target=model_config["dataset"]["per_turbine_target"], as_lazyframe=False, dtype=pl.Float32)
        
    # %% SETUP SEED
    logging.info(f"Setting random seed to {args.seed}")
    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% PREPARING DIRECTORIES
    for direc in [model_config["optuna"]["storage_dir"], data_config["temp_storage_dir"]]:
        env_vars = re.findall(r"(?:^|\/)\$(\w+)(?:\/|$)", direc)
        for env_var in env_vars:
            if env_var in os.environ:
                direc = direc.replace(f"${env_var}", os.environ[env_var])

        logging.info(f"Making directory {direc}.")
        os.makedirs(direc, exist_ok=True)
    # if not os.path.exists(data_config["temp_storage_dir"]): # get permission denied for /tmp/scratch dirs otherwise
    # os.makedirs(data_config["temp_storage_dir"], exist_ok=True)
    
    # %% INSTANTIATING MODEL
    logging.info("Instantiating model.")  
    if args.model == "svr": 
        model = SVRForecast(measurements_timedelta=pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            controller_timedelta=None,
                            prediction_timedelta=data_module.prediction_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            context_timedelta=data_module.context_length*pd.Timedelta(model_config["dataset"]["resample_freq"]),
                            fmodel=fmodel,
                            true_wind_field=None,
                            model_config=model_config,
                            kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200,
                                        n_neighboring_turbines=3, max_n_samples=None),
                            tid2idx_mapping=tid2idx_mapping,
                            turbine_signature=turbine_signature,
                            use_tuned_params=False,
                            temp_save_dir=data_config["temp_storage_dir"],
                            multiprocessor=args.multiprocessor)
    
    
    # %% PREPARING DATA FOR TUNING
    if args.initialize:
        # %% READING WIND FIELD TRAINING DATA # TODO fetch training and test data here

    
        logging.info("Preparing data for tuning")
        if not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
            
        true_wind_field = data_module.generate_splits(save=True, reload=reload, splits=["train", "val"])._df.collect()
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
        
        model.prepare_data(train_dataset=train_dataset.partition_by("continuity_group"), val_dataset=val_dataset.partition_by("continuity_group"), scale=False)
        
        logging.info("Reinitializing storage") 
        if args.restart_tuning:
            storage = model.get_storage(
                backend=model_config["optuna"]["storage"]["backend"], 
                    study_name=args.study_name, 
                    storage_dir=model_config["optuna"]["storage_dir"])
            for s in storage.get_all_studies():
                storage.delete_study(s._study_id)
    else: 
        # %% TUNING MODEL
        logging.info("Running tune_hyperparameters_multi")
        pruning_kwargs = model_config["optuna"]["pruning"] 
        
        #{"type": "hyperband", "min_resource": 2, "max_resource": 5, "reduction_factor": 3, "percentile": 25}
        model.tune_hyperparameters_single(study_name=args.study_name,
                                        backend=model_config["optuna"]["storage"]["backend"],
                                        n_trials=model_config["optuna"]["n_trials"], 
                                        storage_dir=model_config["optuna"]["storage_dir"],
                                        seed=args.seed,
                                        pruning_kwargs=pruning_kwargs)
                                        #  trial_protection_callback=handle_trial_with_oom_protection)
    
        # %% TESTING LOADING HYPERPARAMETERS
        # Test setting parameters
        # model.set_tuned_params(backend=model_config["optuna"]["backend"], study_name_root=args.study_name, 
        #                        storage_dir=model_config["optuna"]["storage_dir"]) 
    
        # %% After training completes
        # torch.cuda.empty_cache()
        gc.collect()
        logging.info("Optuna hyperparameter tuning completed.")