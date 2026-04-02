from whoc.wind_forecast.run_forecaster_validation import generate_wind_field_df
from whoc.wind_forecast.svr_forecast import SVRForecast
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.dataset.split import slice_data_entry
import numpy as np
import polars as pl
import pandas as pd
import argparse
import yaml
import os
import logging 
import glob
from floris import FlorisModel
import multiprocessing as mp
from datetime import datetime
import re
import random
from wind_forecasting.utils.optuna_storage import setup_optuna_storage
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params
from itertools import product
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    parser.add_argument("-utp", "--use_tuned_params", action="store_true")
    parser.add_argument("-rm", "--retrain_models", action="store_true")
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
    parser.add_argument("-tti", "--target_turbine_indices", metavar="C", nargs="+", required=False, default=None, type=int)
    
    # parser.add_argument('--cores', required=False, default=None, help='Comma-separated list or range of core IDs (e.g., "0-9" or "10,11,12")')
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
    if args.multiprocessor == "mpi":
        try:
            from mpi4py import MPI
        except Exception as e:
            logging.warning("Could not import MPI.")
        comm = MPI.COMM_WORLD
        RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    else:
        RUN_ONCE = True
        comm_rank = 0
        
    if RUN_ONCE:
        logging.info("Parsing arguments and configuration yaml.")
    
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
        
    assert model_config["optuna"]["storage"]["backend"] in ["sqlite", "mysql", "journal"]
    
    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
    
    if args.target_turbine_indices:
        args.target_turbine_indices = sorted(args.target_turbine_indices)
    
    if len(data_config["turbine_signature"]) == 1:
        # if args.target_turbine_indices:
        #     tid2idx_mapping = dict(sorted(data_config["turbine_mapping"][0].items(), key=lambda tup: tup[1])[idx] for idx in args.target_turbine_indices)
        #     tid2idx_mapping = {str(k): i for i, k in enumerate(tid2idx_mapping.keys())}
        # else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        # if args.target_turbine_indices:
        #     tid2idx_mapping = dict(sorted(data_config["turbine_mapping"][0].items(), key=lambda tup: tup[1])[idx] for idx in args.target_turbine_indices)
        #     tid2idx_mapping = {str(k): i for i, k in enumerate(tid2idx_mapping.values())}
        # else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
     
    fmodel = FlorisModel(data_config["farm_input_path"])
    # if args.target_turbine_indices:
    #     # TODO only tune/train for target_turbine_indices
    #     fmodel._reinitialize(layout_x=fmodel.layout_x[args.target_turbine_indices], 
    #                          layout_y=fmodel.layout_y[args.target_turbine_indices])
        # fmodel.n_turbines = fmodel.core.farm.n_turbines
    
    if RUN_ONCE:
        logging.info("Creating datasets")
    
    # TODO don't use normalized path if not necessary
    data_module = DataModule(data_path=model_config["dataset"]["data_path"], 
                            normalization_consts_path=model_config["dataset"]["normalization_consts_path"],
                            use_normalization=False, 
                            n_splits=1, #model_config["dataset"]["n_splits"],
                            continuity_groups=None, 
                            train_split=(1.0 - model_config["dataset"]["val_split"] - model_config["dataset"]["test_split"]),
                            val_split=model_config["dataset"]["val_split"], 
                            test_split=model_config["dataset"]["test_split"],
                            prediction_length=model_config["dataset"]["prediction_length"], 
                            context_length=model_config["dataset"]["context_length"],
                            target_prefixes=["ws_horz", "ws_vert"], 
                            feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                            freq=model_config["dataset"]["resample_freq"], 
                            target_suffixes=model_config["dataset"]["target_turbine_ids"],
                            per_turbine_target=False, 
                            as_lazyframe=False, 
                            dtype=pl.Float32)
        
    # %% SETUP SEED
    if RUN_ONCE:
        logging.info(f"Setting random seed to {args.seed}")
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% GET WORKER ID
    # Use the WORKER_RANK variable set explicitly in the Slurm script's nohup block
    worker_id = int(os.environ.get('WORKER_RANK', 0))
    if RUN_ONCE:
        if "WORKER_RANK" in os.environ:
            logging.info(f"Determined worker rank from WORKER_RANK: {worker_id}")
        else:
            logging.info(f"Couldn't find WORKER_RANK env var, setting rank to {worker_id}.")
    
    # %% INSTANTIATING OPTUNA STORAGE
    optuna_storage = None
    if RUN_ONCE:
        logging.info(f"Initializing storage with restart_tuning={args.restart_tuning} on worker {worker_id}")
        # base study prefix should not include postfix to fetch storage
        fetch_existing_study = (args.mode == "tune" and not args.restart_tuning) or (args.mode == "train" and args.use_tuned_params)
        
        if fetch_existing_study:
            logging.info(f"Continue previous tuning.")
            get_most_recent_study = True
            job_id = os.environ.get('SLURM_JOB_ID')
            # try to get specific study based on model_config['experiment']['run_name'], but if only the base prefix is used, find the mos recent
            
            # split experiment/run_name into base prefix and suffix
            
            if job_id:
                # If running in SLURM, remove the job ID from the given study name to continue
                
                if (base_match := re.search(".*(?=_\\d{8})", model_config['experiment']['run_name'])) is not None:
                    # there is no job id or datetime suffix in the given study name
                    # use base prefix and get most recent study name for full name
                    final_study_name = f"tuning_{args.model}_{model_config['experiment']['run_name']}"
                    model_config['experiment']['run_name'] = base_match.group()
                    get_most_recent_study = False
                    logging.info(f"Use given full study name {final_study_name}.")
                
            else:
                # Otherwise use a timestamp
                # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                if (base_match := re.search(".*(?=_\\d{14})", model_config['experiment']['run_name'])) is not None:
                    final_study_name = f"tuning_{args.model}_{model_config['experiment']['run_name']}"
                    model_config['experiment']['run_name'] = base_match.group()
                    get_most_recent_study = False
                    logging.info(f"Use given full study name {final_study_name}.")
                
            
            logging.info(f"Set model_config['experiment']['run_name'] to {model_config['experiment']['run_name']} to fetch optuna db object. Retaining final_study_name {final_study_name}.")

        db_setup_params = generate_db_setup_params(args.model, model_config)
            
        optuna_storage, _ = setup_optuna_storage(
            db_setup_params=db_setup_params,
            restart_tuning=args.restart_tuning, # Use the potentially overridden flag
            rank=0 if (worker_id == 0) else worker_id
            # No force_sqlite_path argument anymore
        )
        if fetch_existing_study and get_most_recent_study:
            logging.info(f"Fetching most recent study name with base prefix {db_setup_params['base_study_prefix']} for tuning.")
            if job_id:
                suffix_pattern = f"(?<={db_setup_params['base_study_prefix']}_)(\\d{{8}})"
            else:
                suffix_pattern = f"(?<={db_setup_params['base_study_prefix']}_)(\\d{{14}})"
            
            all_studies = optuna_storage.get_all_studies()
            
            # loop through and drop invalid studies
            for study in all_studies:
                if re.search(suffix_pattern, study.study_name) is None:
                    optuna.delete_study(study_name=study.study_name, storage=optuna_storage)
            
            if job_id:
                final_study_name = sorted(all_studies, 
                    key=lambda study: int(re.search(suffix_pattern, study.study_name).group()))[-1].study_name
            else:
                final_study_name = sorted(all_studies, 
                    key=lambda study: datetime.strptime(
                        re.search(suffix_pattern, study.study_name).group(),
                        "%Y%m%d%H%M%S"))[-1].study_name
                
        logging.info("Running tune_hyperparameters_single")
    
    elif args.multiprocessor == "mpi":
        optuna_storage = comm.bcast(optuna_storage, root=0)
    
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
                            kwargs=dict(kernel=model_config["model"]["svr"]["kernel"], 
                                        C=model_config["model"]["svr"]["C"], 
                                        degree=model_config["model"]["svr"]["degree"], 
                                        gamma=model_config["model"]["svr"]["gamma"], 
                                        epsilon=model_config["model"]["svr"]["epsilon"], 
                                        cache_size=model_config["model"]["svr"]["cache_size"],
                                        n_neighboring_turbines=model_config["model"]["svr"]["n_neighboring_turbines"], 
                                        max_n_samples=None, 
                                        use_trained_models=False,
                                        model_config=model_config),
                            tid2idx_mapping=tid2idx_mapping,
                            turbine_signature=turbine_signature,
                            use_tuned_params=False,
                            target_turbine_indices=args.target_turbine_indices)
        # original_save_dir = forecaster.model_save_dir
        # forecaster.model_save_dir = os.environ["TMPDIR"]
    
    # %% PREPARING DATA FOR TUNING
    if worker_id == 0 and RUN_ONCE:
        logging.info("Generating train/val datasets.")
        if args.reload_data or not os.path.exists(data_module.train_ready_data_path):
            logging.info("Generating new train/val datasets.")
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
        
        logging.info(msg="Generating train/val splits.")
        data_module.generate_splits(save=True, reload=reload, splits=["train", "val"])
    else:
        data_module.get_dataset_info()

    if worker_id == 0:
        # logging.info(f"Number of Xy paths: {num_Xy_paths} out of required {required_num_Xy_paths}")
        logging.info("Preparing data for tuning")
        data_module.datasets["train"] = sorted(data_module.datasets["train"], key=lambda ds: ds["target"].shape[1], reverse=True)
        data_module.datasets["val"] = sorted(data_module.datasets["val"], key=lambda ds: ds["target"].shape[1], reverse=True)
        if args.max_splits:
            train_dataset = data_module.datasets["train"][:args.max_splits]
            val_dataset = data_module.datasets["val"][:args.max_splits]
        else:
            train_dataset = data_module.datasets["train"]
            val_dataset = data_module.datasets["val"]

        if args.max_steps:
            train_dataset = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in train_dataset]
            val_dataset = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in val_dataset]
            
        train_dataset = generate_wind_field_df(datasets=train_dataset, target_cols=data_module.target_cols, feat_dynamic_real_cols=data_module.feat_dynamic_real_cols)
        val_dataset = generate_wind_field_df(datasets=val_dataset, target_cols=data_module.target_cols, feat_dynamic_real_cols=data_module.feat_dynamic_real_cols)
        del data_module.datasets["train"]
        del data_module.datasets["val"]
        delattr(data_module, "datasets")
        
    if args.multiprocessor == "mpi":
        comm.Barrier()

    if args.mode == "tune":

        # check that all data corresponding to forecaster dataset_hparams is saved
        dataset_hparams = list(forecaster.dataset_hparams_choices.keys())
        for hparam_set in product(*forecaster.dataset_hparams_choices.values()):
            
            suffix = ("_" + "_".join([f"{k}{v}" for k, v in zip(dataset_hparams, hparam_set)])) if len(forecaster.dataset_hparams) else ""
            num_Xy_paths = glob.glob(os.path.join(forecaster.model_save_dir, f"Xy_{forecaster.study_name}_*_*{suffix}.dat"))
            num_Xy_paths = [os.path.basename(fp) for fp in num_Xy_paths]
            if args.target_turbine_indices is not None:
                num_Xy_paths = [fp for fp in num_Xy_paths 
                                if forecaster.tid2idx_mapping[re.findall(f"Xy_{forecaster.study_name}_.*_(.*){suffix}.dat", fp)[0]] in args.target_turbine_indices]
            num_Xy_paths = len(num_Xy_paths)
            
            required_num_Xy_paths = (data_module.num_target_vars if args.target_turbine_indices is None else (len(args.target_turbine_indices) * len(data_module.target_prefixes))) * 2 # val and train
            
            # logging.info(f"worker_id = {worker_id}, reload = {args.reload_data or reload}, num_Xy_paths = {num_Xy_paths}, required_num_Xy_paths = {required_num_Xy_paths}")
            if worker_id == 0 and (args.reload_data or reload or num_Xy_paths < required_num_Xy_paths):
                logging.info(f"Preparing data with suffix {suffix} for tuning")
                
                forecaster.prepare_data(
                    dataset_splits={"train": train_dataset.partition_by("continuity_group"), "val": val_dataset.partition_by("continuity_group")}, 
                    scale=True, 
                    multiprocessor=args.multiprocessor, 
                    reload=args.reload_data or reload,
                    dataset_hparams={k: v for k, v in zip(dataset_hparams, hparam_set)})
            
                if RUN_ONCE:
                    logging.info(f"Finished preparing data with suffix {suffix} for tuning.")
    elif args.mode == "train":
        if args.use_tuned_params:
            logging.info("Using tuned hyperparameters.")
            forecaster.set_tuned_params(optuna_storage=optuna_storage, study_name=final_study_name)
        elif len(model_config["model"][args.model]):
            logging.info("Using model config hyperparameters.")
            forecaster.set_tuned_params(config_params=model_config["model"][args.model])
        else:
            logging.info("Using default hyperparameters.")
            forecaster.set_tuned_params()

        logging.info(f"CHECK THIS MATCHES TUNING. Preparing data for training with dataset hyperparameters {forecaster.dataset_hparams}.")
        suffix = ("_" + "_".join([f"{k}{v}" for k, v in forecaster.dataset_hparams.items()])) if len(forecaster.dataset_hparams) else ""
        num_Xy_paths = glob.glob(os.path.join(forecaster.model_save_dir, f"Xy_{forecaster.study_name}_*_*{suffix}.dat"))
        num_Xy_paths = [os.path.basename(fp) for fp in num_Xy_paths]
        if args.target_turbine_indices is not None:
            num_Xy_paths = [fp for fp in num_Xy_paths 
                            if forecaster.tid2idx_mapping[re.findall(f"Xy_{forecaster.study_name}_.*_(.*){suffix}.dat", fp)[0]] in args.target_turbine_indices]
        num_Xy_paths = len(num_Xy_paths)
       
        required_num_Xy_paths = (data_module.num_target_vars if (args.target_turbine_indices is None) else (len(args.target_turbine_indices) * len(data_module.target_prefixes))) * 2 # val and train
        logging.info(f"Found {num_Xy_paths} existing Xy paths for training with format Xy_{forecaster.study_name}_*_*{suffix}.dat. Require {required_num_Xy_paths}.")
            
        if worker_id == 0 and (args.reload_data or reload or num_Xy_paths < required_num_Xy_paths):
            forecaster.prepare_data(
                dataset_splits={"train": train_dataset.partition_by("continuity_group"), "val": val_dataset.partition_by("continuity_group")}, 
                scale=True, 
                multiprocessor=args.multiprocessor, 
                reload=args.reload_data or reload)

        if RUN_ONCE:
            logging.info("Finished preparing data for training.")

    # %% TUNING MODEL
    

    # scaler_params = data_module.compute_scaler_params()
    
    worker_id = int(os.environ.get('WORKER_RANK', 1))
    if args.mode == "tune" and worker_id > 0:
        
        if args.multiprocessor:
            logging.info(f"Using multiprocessor {args.multiprocessor}")
        
        try:
            allowed_cores = os.sched_getaffinity(0)
            num_allowed_cores = len(allowed_cores)
            logging.info(f"os.sched_getaffinity(0) reports: {num_allowed_cores} allowed cores.")
            logging.info(f"Allowed core list: {sorted(list(allowed_cores))}")
        except AttributeError:
            # os.sched_getaffinity is not available on all OSes (e.g., Windows)
            logging.warning("os.sched_getaffinity not available. Falling back to NTASKS_PER_TUNER.")
            num_allowed_cores = int(os.environ.get("NTASKS_PER_TUNER", mp.cpu_count()))
            
        forecaster.tune_hyperparameters_single(optuna_storage=optuna_storage,
                                                n_trials_per_worker=model_config["optuna"]["n_trials_per_worker"], 
                                                seed=args.seed,
                                                config=model_config,
                                                worker_id=1 if RUN_ONCE and (worker_id == 1) else worker_id,
                                                multiprocessor=args.multiprocessor,
                                                limit_train_val=args.limit_train_val,
                                                restart_tuning=args.restart_tuning,
                                                max_cpus=num_allowed_cores,
                                                study_name=None if args.restart_tuning else final_study_name)
                                                # max_cpus=int(os.environ.get("NTASKS_PER_TUNER", None)))
                                        #  trial_protection_callback=handle_trial_with_oom_protection)
        # %% After tuning completes
        logging.info("Optuna hyperparameter tuning completed.")
        
    elif args.mode == "train":
        # %% TRAINING MODEL
        logging.info("Training model.")
            
        forecaster.train_all_outputs(scale=True, 
                                    multiprocessor=args.multiprocessor,
                                    retrain_models=args.retrain_models,
                                    scaler_params=None,
                                    )
        # %% After training completes
        logging.info("Training completed.")
        
        # from datetime import timedelta
        # forecaster.predict_point(
        #     train_dataset.filter(pl.col("continuity_group") == 0).select(pl.all().slice(0, pl.len()-1)),
        #     current_time=train_dataset.filter(pl.col("continuity_group") == 0).select(pl.col("time").slice(-1, 1)).item())
        # print("Test prediction complete.")
        