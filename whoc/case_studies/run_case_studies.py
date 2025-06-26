import os
from concurrent.futures import ProcessPoolExecutor
import warnings
import re
import argparse
import csv
from itertools import cycle
import polars as pl

import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml
import pickle
from glob import glob
from memory_profiler import profile

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import whoc
try:
    from whoc.controllers.mpc_wake_steering_controller import MPC
except Exception:
    logging.warning("Cannot import MPC controller in current environment.")
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import (read_time_series_data, write_case_family_time_series_data, read_case_family_time_series_data, 
                                                    aggregate_time_series_data, read_case_family_agg_data, write_case_family_agg_data, 
                                                    generate_outputs, plot_simulations, plot_wind_farm, plot_breakdown_robustness, plot_horizon_length,
                                                    plot_cost_function_pareto_curve, plot_yaw_offset_wind_direction, plot_parameter_sweep, plot_power_increase_vs_prediction_time,
                                                    plot_power_vs_prediction_time, plot_agg_metrics_vs_forecaster)
try:
    from whoc.wind_forecast.perfect_forecast import PerfectForecast
    from whoc.wind_forecast.persistence_forecast import PersistenceForecast
    from whoc.wind_forecast.ml_forecast import MLForecast
    from whoc.wind_forecast.svr_forecast import SVRForecast
    from whoc.wind_forecast.kalman_filter_forecast import KalmanFilterForecast
    from whoc.wind_forecast.spatial_filter_forecast import SpatialFilterForecast
except ModuleNotFoundError:
    logging.warning("Cannot import wind forecast classes in current environment.")
# np.seterr("raise")

warnings.simplefilter('error', pd.errors.DtypeWarning)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_case_studies.py", description="Run FLORIS case studies for WHOC module.")
    parser.add_argument("case_ids", metavar="C", nargs="+", choices=[str(i) for i in range(len(case_families))])
    parser.add_argument("-gwf", "--generate_wind_field", action="store_true")
    parser.add_argument("-glut", "--generate_lut", action="store_true")
    parser.add_argument("-rs", "--run_simulations", action="store_true")
    parser.add_argument("-rrs", "--rerun_simulations", action="store_true")
    parser.add_argument("-stmp", "--skip_temps", action="store_true")
    parser.add_argument("-ps", "--postprocess_simulations", action="store_true")
    parser.add_argument("-rps", "--reprocess_simulations", action="store_true")
    parser.add_argument("-ras", "--reaggregate_simulations", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-st", "--stoptime", default="auto")
    parser.add_argument("-ns", "--n_seeds", default="auto")
    parser.add_argument("-ip", "--include_prediction", action="store_true")
    parser.add_argument("-ics", "--include_controller_signals", action="store_true")
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"], help="which multiprocessing backend to use, omit for sequential processing", default=None)
    parser.add_argument("-sd", "--save_dir", type=str, default=os.path.join(os.getcwd(), "simulation_results"))
    parser.add_argument("-wf", "--wf_source", type=str, choices=["floris", "scada"], required=True)
    parser.add_argument("-mcnf", "--model_config", type=str, required=False, default="")
    parser.add_argument("-dcnf", "--data_config", type=str, required=False, default="")
    parser.add_argument("-wcnf", "--whoc_config", type=str, required=True)
    parser.add_argument("-rl", "--ram_limit", type=int, required=False, default=75)
     
    args = parser.parse_args()
    args.case_ids = [int(i) for i in args.case_ids]

    # os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "false"
    if args.multiprocessor == "mpi":
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        comm = MPI.COMM_WORLD
        
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := comm.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    PLOT = True #sys.platform != "linux"
    # run simulations
    
    if RUN_ONCE:
        # os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml")
        
        logging.info(f"Reading WHOC config file {args.whoc_config}")
        with open(args.whoc_config, 'r') as file:
            whoc_config  = yaml.safe_load(file)
            
        if args.wf_source == "scada":
            # NOTE make sure this is the model config with the highest prediction length required, for splitting
            logging.info(f"Reading model config file {args.model_config}")
            with open(args.model_config, 'r') as file:
                model_config  = yaml.safe_load(file)
                
            logging.info(f"Reading preprocessing config file {args.data_config}")
            with open(args.data_config, 'r') as file:
                data_config  = yaml.safe_load(file)
            
            if len(data_config["turbine_signature"]) == 1:
                tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
            else:
                tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
            
            turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
            
        else:
            model_config = None
            data_config = None
            turbine_signature = None
            tid2idx_mapping = None
            # temp_storage_dir = None
            
        logging.info(f"running initialize_simulations for case_ids {[case_families[i] for i in args.case_ids]}")
        input_dicts, wind_field_config, wind_field_ts \
            = initialize_simulations(case_study_keys=[case_families[i] for i in args.case_ids], 
                                        regenerate_wind_field=args.generate_wind_field, 
                                        regenerate_lut=args.generate_lut, 
                                        run_simulations=args.run_simulations,
                                        rerun_simulations=args.rerun_simulations,
                                        reprocess_simulations=args.reprocess_simulations,
                                        n_seeds=args.n_seeds, 
                                        stoptime=args.stoptime, 
                                        save_dir=args.save_dir, 
                                        wf_source=args.wf_source,
                                        multiprocessor=args.multiprocessor, 
                                        whoc_config=whoc_config, base_model_config=model_config)
        logging.info(f"Resetting args.n_seeds to {len(wind_field_ts)}")
        args.n_seeds = len(wind_field_ts)
        
    if args.multiprocessor == "mpi":
        comm.Barrier()
        
    # if GPUs are available, use one CPU and one GPU per task
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"] # Note: must 'export' variable within nohup to find on Kestrel
        logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
        try:
            # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
            visible_gpus = [idx for idx in cuda_devices.split(',') if idx.strip()]
            num_visible_gpus = len(visible_gpus)
            if num_visible_gpus > 0:
                # max_workers = num_visible_gpus
                # TODO TESTING see if many cores can share less number of GPUs
                # max_workers = comm.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
                max_workers = int(os.environ.get("SLURM_NTASKS_PER_NODE", num_visible_gpus))
                logging.info(f"Found {num_visible_gpus} GPUs. Setting max_workers to num_visible_gpus={max_workers}.")
            else:
                max_workers = comm.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
                logging.warning(f"CUDA_VISIBLE_DEVICES is set but no valid GPU indices found. Setting max_workers to mp.cpu_count()={max_workers}.")
        except Exception as e:
            logging.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
        
        # Create an iterator that cycles through the available GPU IDs
        gpu_cycler = cycle(visible_gpus)
        
    else:
        max_workers = comm.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
        gpu_cycler = None
        
    if args.run_simulations: 
        if args.multiprocessor is not None:
                    
            if args.multiprocessor == "mpi":
                comm_size = comm.Get_size()
                executor = MPICommExecutor(comm, root=0, max_workers=max_workers)
            elif args.multiprocessor == "cf":
                executor = ProcessPoolExecutor(max_workers=max_workers,
                                               mp_context=mp.get_context("spawn"))
            with executor as run_simulations_exec:
                # if args.multiprocessor == "mpi":
                #     run_simulations_exec.max_workers = max_workers
                
                logging.info(f"Submitting simulate_controller calls to pool executor with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[d["controller"]["controller_class"]], 
                                                wind_forecast_class=globals()[d["controller"]["wind_forecast_class"]] if d["controller"]["wind_forecast_class"] else None,
                                                simulation_input_dict=d,
                                                wf_source=args.wf_source, 
                                                wind_case_idx=input_dicts[c]["wind_case_idx"], 
                                                wind_field_ts=wind_field_ts[input_dicts[c]["wind_case_idx"]],
                                                case_name=input_dicts[c]["case_name"],
                                                case_family=input_dicts[c]["case_family"], 
                                                verbose=args.verbose, 
                                                save_dir=args.save_dir, 
                                                rerun_simulations=args.rerun_simulations,
                                                skip_temps=args.skip_temps,
                                                multiprocessor=False, 
                                                turbine_signature=turbine_signature, 
                                                tid2idx_mapping=tid2idx_mapping,
                                                use_tuned_params=True, 
                                                model_config=model_config, wind_field_config=wind_field_config, 
                                                ram_limit=args.ram_limit,
                                                include_prediction=args.include_prediction,
                                                include_controller_signals=args.include_controller_signals,
                                                assigned_gpu=next(gpu_cycler) if gpu_cycler else None)

                        for c, d in enumerate(input_dicts)]
                
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[d["controller"]["controller_class"]], 
                                    wind_forecast_class=globals()[d["controller"]["wind_forecast_class"]] if d["controller"]["wind_forecast_class"] else None, 
                                    simulation_input_dict=d, 
                                    wf_source=args.wf_source,
                                    wind_case_idx=input_dicts[c]["wind_case_idx"], 
                                    wind_field_ts=wind_field_ts[input_dicts[c]["wind_case_idx"]],
                                    case_name=input_dicts[c]["case_name"],
                                    case_family=input_dicts[c]["case_family"],
                                    multiprocessor=False, 
                                    wind_field_config=wind_field_config, verbose=args.verbose, save_dir=args.save_dir, 
                                    rerun_simulations=args.rerun_simulations,
                                    skip_temps=args.skip_temps,
                                    turbine_signature=turbine_signature, tid2idx_mapping=tid2idx_mapping,
                                    use_tuned_params=True, model_config=model_config, ram_limit=args.ram_limit,
                                    include_prediction=args.include_prediction,
                                    include_controller_signals=args.include_controller_signals,
                                    assigned_gpu=next(gpu_cycler) if gpu_cycler else None)
    
    if args.postprocess_simulations:
        # if (not os.path.exists(os.path.join(args.save_dir, f"time_series_results.csv"))) or (not os.path.exists(os.path.join(args.save_dir, f"agg_results.csv"))):
        # regenerate some or all of the time_series_results_all and agg_results_all .csv files for each case family in case ids
        if args.reprocess_simulations \
            or not all(os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")) for i in args.case_ids) \
                or not all(os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")) for i in args.case_ids):
            if RUN_ONCE:
                # make a list of the time series csv files for all case_names and seeds in each case family directory
                case_family_case_names = {}
                for i in args.case_ids:
                    case_family_case_names[case_families[i]] = [fn for fn in glob(os.path.join(args.save_dir, case_families[i], "time_series_results_case_*_seed_*.csv")) if "_temp.csv" not in fn]
                    # case_family_case_names[case_families[i]] = [fn for fn in glob(os.path.join(args.save_dir, case_families[i], r"time_series_results_case_*_seed_[0-9]+.csv"))]

                # case_family_case_names["slsqp_solver_sweep"] = [f"time_series_results_case_alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval_seed_{s}" for s in range(6)]
            # if using multiprocessing
            if args.multiprocessor is not None:
                if args.multiprocessor == "mpi":
                    comm_size = comm.Get_size()
                    executor = MPICommExecutor(comm, root=0, max_workers=comm_size)
                elif args.multiprocessor == "cf":
                    executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
                with executor as run_simulations_exec:
                    # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)

                    # if args.reaggregate_simulations is true, or for any case family where doesn't time_series_results_all.csv exist, 
                    # read the time-series csv files for all case families, case names, and wind seeds
                    input_regex = "(?<=time_series_results_).+(?=_seed_\\d+.csv)"
                    read_futures = [run_simulations_exec.submit(
                                                    read_time_series_data, 
                                                    results_path=os.path.join(args.save_dir, case_families[i], fn),
                                                    input_dict_path=os.path.join(
                                                        args.save_dir, case_families[i], 
                                                            f"input_config_{re.search(input_regex, fn).group()}.pkl"))
                        for i in args.case_ids 
                        for fn in case_family_case_names[case_families[i]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))
                    ]

                    new_time_series_df = [fut.result() for fut in read_futures]
                    # if there are new resulting dataframes, concatenate them from a list into a dataframe
                    if new_time_series_df:
                        new_time_series_df = [pd.concat(new_time_series_df)]

                    read_futures = [run_simulations_exec.submit(read_case_family_time_series_data, 
                                                                case_family=case_families[i], save_dir=args.save_dir)
                                    for i in args.case_ids
                                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                    existing_time_series_df = [fut.result() for fut in read_futures]

                    # write time_series_all for each case
                    if len(new_time_series_df):
                        write_futures = [run_simulations_exec.submit(write_case_family_time_series_data, 
                                                                     case_family=case_families[i], 
                                                                     new_time_series_df=new_time_series_df[0],
                                                                     save_dir=args.save_dir)
                                        for i in args.case_ids
                                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                        _ = [fut.result() for fut in write_futures]
                    
            # else, run sequentially
            else:
                new_time_series_df = []
                existing_time_series_df = []
                for i in args.case_ids:
                    # all_ts_df_path = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        existing_time_series_df.append(read_case_family_time_series_data(case_families[i], save_dir=args.save_dir))
                
                new_case_family_time_series_df = [] 
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        for fn in case_family_case_names[case_families[i]]:
                            input_regex = "(?<=time_series_results_).+(?=_seed_\\d+.csv)"
                            new_case_family_time_series_df.append(
                                read_time_series_data(results_path=os.path.join(args.save_dir, case_families[i], fn),
                                                      input_dict_path=os.path.join(args.save_dir, case_families[i], 
                                                                                   f"input_config_{re.search(input_regex, fn).group()}.pkl")))

                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if new_case_family_time_series_df:
                        new_time_series_df.append(pd.concat(new_case_family_time_series_df))
                        write_case_family_time_series_data(case_families[i], new_time_series_df[-1], args.save_dir)
            
            if RUN_ONCE:    
                time_series_df = pd.concat(existing_time_series_df + new_time_series_df)
                    
                unique_seeds = time_series_df.groupby(["CaseFamily", "CaseName"], level=0)["WindSeed"].unique().values
                common_seeds = set(unique_seeds[0])
                for sds in unique_seeds[1:]:
                    common_seeds.intersection_update(sds)
                logging.info(f"Found {common_seeds} wind seeds common to all time series.")
                
                # common_seeds = pd.unique(time_series_df["WindSeed"])
                # time_series_df = time_series_df.loc[time_series_df.index.get_level_values("CaseName").isin([str(i) for i in range(0, 20)]) | time_series_df.index.get_level_values("CaseName").isin([str(i) for i in range(20, 35)])]
                
                if args.reaggregate_simulations or not all(os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")) for i in args.case_ids):
                    max_ctx_steps = []
                    min_stop_time = np.inf
                    for i in args.case_ids:
                        # for case_name in set([re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]):
                        case_family_df = time_series_df.iloc[time_series_df.index.get_level_values("CaseFamily") == case_families[i], :]
                        case_desc = pd.read_csv(os.path.join(args.save_dir, case_families[i], "case_descriptions.csv"), index_col=0)
                        
                        for _, row in case_desc.iterrows():
                            sim_dt = pd.to_timedelta(row["simulation_dt"], unit="s")
                            mncf_path = row["model_config_path"]
                            lpf_start_time = row["lpf_start_time"]
                            
                            if isinstance(row["model_config_path"], os.PathLike):
                                with open(mncf_path, mode='r') as fp:
                                    mcnf = yaml.safe_load(fp)
                            
                                # longest_ctx_steps.append(int((pd.Timedelta(mcnf["dataset"]["context_length"], unit="s") / pd.Timedelta(row["simulation_dt"], unit="s"))))
                                max_ctx_steps.append(int(mcnf["dataset"]["context_length"])) # in seconds
                                
                            max_ctx_steps.append(int(lpf_start_time))
                            
                        min_stop_time = min(min_stop_time, case_family_df.groupby(["CaseFamily", "CaseName", "WindSeed"], group_keys=False)["Time"].max().min())
                    
                    max_ctx_steps = max(max_ctx_steps)
                        
                # truncate greatest context length at beginning
                trunc_time_series_df = time_series_df.groupby(["CaseFamily", "CaseName"], group_keys=False).apply(lambda sub_df: sub_df.loc[sub_df["Time"] <= min_stop_time, :].iloc[max_ctx_steps:])
                trunc_time_series_df = trunc_time_series_df.loc[trunc_time_series_df["WindSeed"].isin(common_seeds), :]
            
                new_agg_df = aggregate_time_series_data(
                                                time_series_df=trunc_time_series_df,
                                                # input_dict_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.pkl"),
                                                # results_path=os.path.join(args.save_dir, case_families[i], f"agg_results_{case_name}.csv"),
                                                n_seeds=len(common_seeds))
                
                if new_agg_df is None:
                    new_agg_df = pd.DataFrame()
                    
                existing_agg_df = []
                for i in args.case_ids:
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        existing_agg_df.append(read_case_family_agg_data(case_families[i], save_dir=args.save_dir))
                
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        write_case_family_agg_data(case_families[i], new_agg_df, args.save_dir)
            
                all_agg_dfs = [df for df in existing_agg_df + [new_agg_df] if df.shape[0]]
                agg_df = pd.concat(all_agg_dfs)

        elif RUN_ONCE:
            time_series_df = []
            for i in args.case_ids:
                # warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                if os.path.exists(filepath):
                    logging.info(f"Reading combined time series file: {filepath}.")
                    with open(filepath, 'r', newline='') as fp:
                        csv_reader = csv.reader(fp)
                        columns = next(csv_reader)
                    numeric_columns = [col for col in columns if 
                                       any(c in col for c in ["TurbineYawAngle_", "TurbineYawAngleChange_", "TurbinePower_", "TurbineWindDir_", "TurbineWindMag_"])]
                    bool_columns = [col for col in columns if 
                                       any(c in col for c in ["TurbineOfflineStatus"])]
                    
                    # df = pd.read_csv(filepath, index_col=[0, 1], low_memory=False)
                    df = pl.read_csv(filepath, null_values="null",
                                     schema_overrides={**{"CaseName": str}, 
                                                    **{col: pl.Float32 for col in numeric_columns},
                                                    **{col: pl.Boolean for col in bool_columns}})
                    # df = df.with_columns(pl.when(pl.col(pl.String) == "null").then(pl.lit(None)).otherwise(pl.col(pl.String)).name.keep())
                    time_series_df.append(df)
            time_series_df = pl.concat(time_series_df, how="vertical").to_pandas().set_index(["CaseFamily", "CaseName"])
            
            agg_df = []
            for i in args.case_ids:
                # warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")
                if os.path.exists(filepath):
                    # try:
                    logging.info(f"Reading combined aggregate metrics file: {filepath}.")
                    
                    with open(filepath, 'r', newline='') as fp:
                        csv_reader = csv.reader(fp)
                        columns = next(csv_reader)
                    numeric_columns = [col for col in columns if 
                                       any(c in col for c in ["TurbineYawAngle_", "TurbineYawAngleChange_", "TurbinePower_", "TurbineWindDir_", "TurbineWindMag_"])]
                    bool_columns = [col for col in columns if 
                                       any(c in col for c in ["TurbineOfflineStatus"])]
                    
                    df = pd.read_csv(filepath, header=[0,1], index_col=[0, 1], skipinitialspace=True, 
                                     dtype={**{"CaseFamily": str, "CaseName": str}, 
                                            **{col: float for col in numeric_columns},
                                            **{col: bool for col in bool_columns}})
                    df.index = pd.MultiIndex.from_frame(df.index.to_frame().astype({"CaseFamily": str, "CaseName": str}))
                    
                    # df = pl.read_csv(filepath, skip_rows=2,
                    #                  schema_overrides={**{"CaseName": str}, 
                    #                                              **{col: pl.Float32 for col in numeric_columns},
                    #                                               **{col: pl.Boolean for col in bool_columns}})
                    agg_df.append(df)

            agg_df = pd.concat(agg_df)
            # agg_df = pl.concat(agg_df, how="vertical").to_pandas().set_index(["CaseFamily", "CaseName"])

        if RUN_ONCE and PLOT:
            
            if any(case_families.index(cf) in args.case_ids for cf in 
                   ["baseline_controllers_informer_forecasters_awaken", "baseline_controllers_autoformer_forecasters_awaken",
                    "baseline_controllers_spacetimeformer_forecasters_awaken", "baseline_controllers_tactis_forecasters_awaken",
                    "baseline_controllers_baseline_det_forecasters_awaken", "baseline_controllers_baseline_prob_forecasters_awaken"]) \
                        and (case_families.index("baseline_controllers_baseline_det_forecasters_awaken") in args.case_ids) \
                            and (case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids):
                from whoc.wind_forecast.run_forecaster_validation import WindForecast
                from wind_forecasting.preprocessing.data_inspector import DataInspector
                
                # if case_families.index("baseline_controllers_ml_forecasters_awaken") in args.case_ids:
                #     forecaster_case_fam = "baseline_controllers_ml_forecasters_awaken"
                # elif case_families.index("baseline_controllers_baseline_det_forecasters_awaken") in args.case_ids:
                #     forecaster_case_fam = "baseline_controllers_baseline_det_forecasters_awaken"
                # elif case_families.index("baseline_controllers_baseline_det_forecasters_awaken") in args.case_ids:
                #     forecaster_case_fam = "baseline_controllers_baseline_det_forecasters_awaken"
                
                cfs = ["baseline_controllers_informer_forecasters_awaken", "baseline_controllers_autoformer_forecasters_awaken",
                    "baseline_controllers_spacetimeformer_forecasters_awaken", "baseline_controllers_tactis_forecasters_awaken", 
                    "baseline_controllers_baseline_det_forecasters_awaken", "baseline_controllers_baseline_prob_forecasters_awaken",
                    "baseline_controllers_perfect_forecaster_awaken"]
                
                baseline_time_df = time_series_df.loc[time_series_df.index.get_level_values("CaseFamily").isin(cfs), :] #.reset_index(level="CaseFamily", drop=True)
                baseline_agg_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily").isin(cfs), :] #.reset_index(level="CaseFamily", drop=True)
                
                config_cols = ["controller_class", "wind_forecast_class", "prediction_timedelta", "uncertain", "model_key"]
                
                perfect_case_desc = pd.read_csv(os.path.join(args.save_dir, "baseline_controllers_perfect_forecaster_awaken", "case_descriptions.csv"), index_col=0)
                baseline_det_case_desc = pd.read_csv(os.path.join(args.save_dir, "baseline_controllers_baseline_det_forecasters_awaken", "case_descriptions.csv"), index_col=0)
                
                # list(pd.unique(baseline_det_case_desc["prediction_timedelta"]))
                base_val_idx = perfect_case_desc.loc[pd.to_timedelta(perfect_case_desc["prediction_timedelta"]).isin([pd.Timedelta(seconds=0)]), :].index.astype(str)
                base_cond = (((baseline_time_df.index.get_level_values("CaseFamily") == "baseline_controllers_perfect_forecaster_awaken") &
                     baseline_time_df.index.get_level_values("CaseName").isin(base_val_idx)) |
                    ((baseline_time_df.index.get_level_values("CaseFamily") != "baseline_controllers_perfect_forecaster_awaken")))
                baseline_time_df = baseline_time_df.loc[base_cond, :]
                
                base_cond = (((baseline_agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_perfect_forecaster_awaken") &
                     baseline_agg_df.index.get_level_values("CaseName").isin(base_val_idx)) |
                    ((baseline_agg_df.index.get_level_values("CaseFamily") != "baseline_controllers_perfect_forecaster_awaken")))
                baseline_agg_df = baseline_agg_df.loc[base_cond, :]
                
                for (case_family, case_name), _ in baseline_agg_df.iterrows():
                # for case_name, _ in baseline_time_df.iterrows():    
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.pkl"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='rb') as fp:
                        input_config = pickle.load(fp)
                        
                    full_config = {**input_config["controller"], **input_config["wind_forecast"]}
                    for col in config_cols:
                        baseline_time_df.loc[(baseline_time_df.index.get_level_values("CaseFamily") == case_family) & 
                                            (baseline_time_df.index.get_level_values("CaseName") == case_name), col] = full_config[col]
                        baseline_agg_df.loc[(baseline_agg_df.index.get_level_values("CaseFamily") == case_family) & 
                                            (baseline_agg_df.index.get_level_values("CaseName") == case_name), col] = full_config[col]
                
                ml_cond = baseline_agg_df["wind_forecast_class"] == "MLForecast"
                baseline_agg_df.loc[ml_cond, "wind_forecast_class"] = (baseline_agg_df.loc[ml_cond, "model_key"].str.capitalize() + "Forecast").values
               
                # Filter data for the two forecast types
                forecasters_agg_df = baseline_agg_df.loc[baseline_agg_df["wind_forecast_class"] != "PerfectForecast", :]
                perfect_agg_df = baseline_agg_df.loc[(baseline_agg_df["wind_forecast_class"] == "PerfectForecast") & baseline_agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers_baseline_det_forecasters"), :]
                controllers = pd.unique(perfect_agg_df["controller_class"])
                
                # PLOT 0) Farm power of perfect forecaster vs prediction timedela for different controllers
                # controller_labels = {"GreedyController": "Greedy", "LookupBasedWakeSteeringController": "LUT"}
                controller_labels = {
                    "GreedyControllerFalse": "Greedy",
                    "LookupBasedWakeSteeringControllerFalse": "Static LUT",
                    "LookupBasedWakeSteeringControllerTrue": "Dynamic LUT"
                }
                # ml_baseline_agg_df = baseline_agg_df.loc[(~baseline_agg_df["model_key"].isnull()) | (baseline_agg_df["wind_forecast_class"] == "PersistenceForecast"), :]
                ml_baseline_agg_df = baseline_agg_df.loc[(~baseline_agg_df["model_key"].isnull()) 
                                                         | ((baseline_agg_df["wind_forecast_class"] == "PerfectForecast") 
                                                            & (baseline_agg_df["prediction_timedelta"] == pd.Timedelta(seconds=0))), :]
                # if ml_baseline_agg_df.shape[0]:
                ml_baseline_agg_df["controller_class"] = ml_baseline_agg_df["controller_class"] + ml_baseline_agg_df["uncertain"].astype(str)
                ml_baseline_agg_df = ml_baseline_agg_df.sort_values("controller_class")
                if (ml_baseline_agg_df["model_key"].apply(type) == np.str_).any():
                    plot_agg_metrics_vs_forecaster(ml_baseline_agg_df,
                                                save_dir=args.save_dir, label="ml_forecasters_",
                                                controller_labels=controller_labels)
                
                other_baseline_agg_df = baseline_agg_df.loc[baseline_agg_df["model_key"].isnull(), :]
                other_baseline_agg_df["controller_class"] = other_baseline_agg_df["controller_class"] + other_baseline_agg_df["uncertain"].astype(str)
                other_baseline_agg_df = other_baseline_agg_df.sort_values("controller_class")
                # other_baseline_agg_df[["controller_class", "prediction_timedelta", "wind_forecast_class"]].sort_values(["controller_class", "prediction_timedelta", "wind_forecast_class"])
                if other_baseline_agg_df.shape[0]:
                    plot_agg_metrics_vs_forecaster(other_baseline_agg_df,
                                                save_dir=args.save_dir, label="baseline_forecasters_",
                                                controller_labels=controller_labels)
                
                # PLOT 1) Farm power of perfect forecaster vs prediction timedela for different controllers
                # plot_power_vs_prediction_time(baseline_agg_df, args.save_dir, "all_forecasters_")
                
                # PLOT 2) Yaw angles/power for persistent vs. other forecasters for best lead times
                best_forecaster_prediction_delta = forecasters_agg_df.groupby("wind_forecast_class", group_keys=False).apply(lambda x: x.sort_values(by=("FarmPower", "mean"), ascending=False).head(10)) #[("FarmPower", "mean")] 
                best_perfect_prediction_delta = perfect_agg_df.groupby("wind_forecast_class", group_keys=False).apply(lambda x: x.sort_values(by=("FarmPower", "mean"), ascending=False).head(10))
                
                # find best performing forecasters
                forecasters_agg_df.groupby("prediction_timedelta", group_keys=False).apply(lambda x: x.sort_values(by=("FarmPower", "mean"), ascending=False).head(10))
                
                # plot forecasters, persistent, perfect for 60/300sec predictions
                perfect_case_names = perfect_agg_df.loc[perfect_agg_df["prediction_timedelta"].isin(pd.unique(forecasters_agg_df["prediction_timedelta"]))].index.get_level_values("CaseName")
                persistence_case_names = baseline_agg_df.loc[(baseline_agg_df["wind_forecast_class"] == "PerfectForecast") & baseline_agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers_perfect_forecaster_awaken"), :]
                persistence_case_names = persistence_case_names.loc[persistence_case_names["prediction_timedelta"] == pd.Timedelta(seconds=0), :].index.get_level_values("CaseName")
                plotting_cases = [(df[1]._name[0], df[1]._name[1]) for df in forecasters_agg_df.iterrows()] \
                                 + [("baseline_controllers_baseline_det_forecasters_awaken", cn) for cn in perfect_case_names] \
                                    + [("baseline_controllers_perfect_forecaster_awaken", cn) for cn in persistence_case_names]
                label_mapping = {"74": "LUT Ds", "75": "LUT Us", "5": "Greedy"}
                fig_label_features = ["controller_class", "wind_forecast_class"]
                # plotting_cases = [(df[1]._name[0], str(df[1]._name[1])) for df in forecasters_agg_df.loc[(forecasters_agg_df["wind_forecast_class"] == "SVRForecast"), :].iterrows()]
                # plotting_cases = [(df[1]._name[0], str(df[1]._name[1])) for df in forecasters_agg_df.loc[(forecasters_agg_df["wind_forecast_class"] == "PersistenceForecast"), :].iterrows()]
                plot_simulations(
                        time_series_df, plotting_cases, args.save_dir, include_power=False, 
                        legend_loc="outer", single_plot=False, label_mapping=label_mapping,
                        fig_label_features=fig_label_features) 
            
            
            if (case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids
                or case_families.index("baseline_controllers_perfect_forecaster_flasc") in args.case_ids
                or case_families.index("baseline_controllers_forecasters_test_awaken") in args.case_ids
                or case_families.index("baseline_controllers_informer_forecaster_test_awaken") in args.case_ids
                or case_families.index("baseline_controllers_svr_forecaster_test_awaken") in args.case_ids):
                # if case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:
                #     forecaster_case_fam = "baseline_controllers_perfect_forecaster_awaken"
                # elif case_families.index("baseline_controllers_perfect_forecaster_flasc") in args.case_ids:
                #     forecaster_case_fam = "baseline_controllers_perfect_forecaster_flasc"
                # else:
                #     forecaster_case_fam = 
                    
                baseline_agg_df = agg_df #.loc[agg_df.index.get_level_values("CaseFamily") == forecaster_case_fam, :] #.reset_index(level="CaseFamily", drop=True)
                
                config_cols = ["controller_class", "wind_forecast_class", "prediction_timedelta", "uncertain", "use_upstream_wind", "filter_floris_wind", "use_lut_filtered_wind_mag", "interpolation_method"]
                
                for (case_family, case_name), _ in baseline_agg_df.iterrows():
                # for case_name, _ in baseline_time_df.iterrows():    
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.pkl"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='rb') as fp:
                        input_config = pickle.load(fp)
                        
                    full_config = {**input_config["controller"], **input_config["wind_forecast"]}
                    for col in config_cols:
                        if col not in full_config:
                            continue
                        baseline_agg_df.loc[(baseline_agg_df.index.get_level_values("CaseFamily") == case_family) & 
                                            (baseline_agg_df.index.get_level_values("CaseName") == case_name), col] = full_config[col]

                x = baseline_agg_df[[("FarmPower", "mean"), ("FarmPower", "std"), ("controller_class", ""), ("use_upstream_wind", ""), 
                                     ("filter_floris_wind", ""), ("use_lut_filtered_wind_mag", ""), ("interpolation_method", "")]].reset_index(drop=True).sort_values(("FarmPower", "mean"), ascending=False)
                x[[("FarmPower", "mean"), ("use_upstream_wind", ""), 
                   ("filter_floris_wind", ""), ("use_lut_filtered_wind_mag", ""), ("interpolation_method", "")]]
                
                # Find best farm power per wind seed
                extra_args = baseline_agg_df[config_cols]
                extra_args.columns = extra_args.columns.droplevel(1)
                time_series_df = pd.merge(time_series_df, extra_args, on=["CaseFamily", "CaseName"])
                x = time_series_df.reset_index(drop=True)[["controller_class", "prediction_timedelta", "FarmPower", "Time", "WindSeed"]].set_index(["Time", "controller_class", "WindSeed"]).sort_values(["controller_class", "Time"])
                
                # get lowest end time available
                end_times_per_seed = x.groupby(["WindSeed", "prediction_timedelta"]).apply(lambda x: x.sort_values("Time", ascending=True).tail(1)).reset_index(level=[0,1], drop=True).reset_index(0, drop=False)[["Time", "prediction_timedelta"]].groupby("WindSeed").agg("min")["Time"]
                x = x.groupby("WindSeed", group_keys=False).apply(func=(lambda x: x.loc[x.index.get_level_values("Time") <= end_times_per_seed[x.index.get_level_values("WindSeed")[0]]]))
                x = x[["prediction_timedelta", "FarmPower"]].groupby(["controller_class", "prediction_timedelta", "WindSeed"]).agg("mean").reset_index("prediction_timedelta")
                zero_case = x.loc[x["prediction_timedelta"] == pd.Timedelta(seconds=0), :]
                x = pd.merge(x, zero_case, on=["controller_class", "WindSeed"])
                x["FarmPower_x"] = 100 * ((x["FarmPower_x"] / x["FarmPower_y"]) - 1)
                
                # find % increase in farm power for each controller class, prediction_timedelta, and WindSeed, averaged over all prediction_timedelta_values
                x.loc[x["FarmPower_x"] > 0.5, :].groupby("controller_class", group_keys=False).apply(lambda x: x.sort_values("FarmPower_x", ascending=False))[["prediction_timedelta_x", "FarmPower_x"]] #.to_csv("/Users/ahenry/Desktop/perfect.csv")
                
                # find % increase in farm power for each controller class and Wind Seed, averaged over all prediction_timedelta_values
                x.groupby(["controller_class", "WindSeed"])["FarmPower_x"].agg("mean").groupby("controller_class", group_keys=False).apply(lambda x: x.sort_values(ascending=False))
                
                # find % increase in farm power for each controller class and prediction_timedelta values, averaged over all Wind Seeds
                x.groupby(["controller_class", "prediction_timedelta_x"])["FarmPower_x"].agg("mean").groupby("controller_class", group_keys=False).apply(lambda x: x.sort_values(ascending=False))
                
                
                perfect_agg_df = baseline_agg_df.loc[baseline_agg_df["wind_forecast_class"] == "PerfectForecast", :]
                controllers = pd.unique(perfect_agg_df["controller_class"])
                # 4, 8, 2 
                
                # x.loc[x.index.get_level_values("WindSeed").isin([8, 4, 2]), :].reset_index(drop=False)[["controller_class", "prediction_timedelta_x", "FarmPower_x"]].groupby(["controller_class", "prediction_timedelta_x"]).agg("mean")
                 
                # x.assign(prediction_timedelta_x=x["prediction_timedelta_x"].dt.total_seconds())\
                #     .loc[x.index.get_level_values("WindSeed").isin([4, 8, 2]), :]\
                #         .reset_index(drop=False)[["controller_class", "prediction_timedelta_x", "FarmPower_x"]]\
                #             .groupby(["controller_class", "prediction_timedelta_x"]).agg("mean")\
                #                 .reset_index(drop=False)\
                #                     .groupby("controller_class").apply(lambda x: x.sort_values("FarmPower_x", ascending=False)).to_csv("/Users/ahenry/Desktop/perfect.csv")
                # import seaborn as sns
                # sns.lineplot(data=
                # x.assign(prediction_timedelta_x=x["prediction_timedelta_x"].dt.total_seconds()).loc[x.index.get_level_values("WindSeed").isin([4, 8, 0, 2, 5, 1]) & (x.index.get_level_values("controller_class") == "GreedyController"), :].reset_index(drop=False)[["controller_class", "prediction_timedelta_x", "FarmPower_x"]].groupby(["controller_class", "prediction_timedelta_x"]).agg("mean").reset_index(drop=False), 
                # x="prediction_timedelta_x", y="FarmPower_x")
                
                # compare influence of filtering wind passed to floris and using upstream wind measurement
                if "use_upstream_wind" in perfect_agg_df.columns and "filter_floris_wind" in perfect_agg_df.columns:
                    perfect_agg_df.groupby(["use_upstream_wind", "filter_floris_wind", "controller_class"])["FarmPower"].agg("mean")[("FarmPower", "mean")]\
                                .groupby("controller_class", group_keys=False)\
                                .apply(lambda x: x.sort_values(ascending=False))
                
                # perfect_agg_df.sort_values(("FarmPower", "mean"))[[("prediction_timedelta", ""), ("controller_class", ""), ("FarmPower", "mean"), ("YawAngleChangeAbs", "mean")]].reset_index(drop=True)
                # PLOT 1) Farm power of perfect forecaster vs prediction timedelta for different controllers
                plot_power_vs_prediction_time(perfect_agg_df, args.save_dir, "perfect_forecaster_")
                
                plotting_cases = [("baseline_controllers_svr_forecaster_test_awaken", str(11))]
                # TODO can't have duplicate keys
                label_mapping = {"5": "Greedy", "74": "LUT Ds", "75": "LUT Us"}
                # "6,", "6,4"
                # label_mapping = {"7": "Greedy", "5": "LUT Ds", "7": "LUT Us"}
                plot_simulations(
                    time_series_df, plotting_cases, args.save_dir, include_power=True, 
                    legend_loc="outer", single_plot=False, label_mapping=label_mapping, seed_idx=0)
                
                # PLOT 2) Farm power ratio of other forecasters relative to perfect forecaster vs prediction timedela for different controllers (diff plots)
                # plot_df = plot_df.set_index(["controller_class", "prediction_timedelta"])
                # plot_df["power_ratio"] = (plot_df[("FarmPower", "mean")] / perfect_agg_df.set_index(["controller_class", "prediction_timedelta"])[("FarmPower", "mean")]) * 100
                # plot_df = plot_df.reset_index()
                # plot_power_increase_vs_prediction_time(plot_df, args.save_dir)    
            
            if ((case_families.index("baseline_controllers") in args.case_ids)):
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") != "baseline_controllers"]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0]) 
                                                & (mpc_df[("YawAngleChangeAbs", "mean")] < lut_df[("YawAngleChangeAbs", "mean")].iloc[0]), 
                                                [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]]\
                                                    .sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True)\
                                                        .reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPower", "mean")] > greedy_df[("FarmPower", "mean")].iloc[0]), 
                                                   [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]]\
                                                    .sort_values(by=("YawAngleChangeAbs", "mean"), ascending=True)\
                                                        .reset_index(level="CaseFamily", drop=True)

                100 * (better_than_lut_df.iloc[0]["FarmPower"] - lut_df.iloc[0]["FarmPower"]) / lut_df.iloc[0]["FarmPower"]
                100 * (better_than_lut_df.iloc[0]["FarmPower"] - greedy_df.iloc[0]["FarmPower"]) / greedy_df.iloc[0]["FarmPower"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - lut_df.iloc[0]["YawAngleChangeAbs"]) / lut_df.iloc[0]["YawAngleChangeAbs"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - greedy_df.iloc[0]["YawAngleChangeAbs"]) / greedy_df.iloc[0]["YawAngleChangeAbs"]
                
                if True:
                    plotting_cases = [("wind_preview_type", better_than_lut_df.iloc[0]._name),   
                                        ("baseline_controllers", "LUT"),
                                        ("baseline_controllers", "Greedy")
                        ]
                    # plotting_cases = [("baseline_controllers_forecasters_test_awaken", 2),
                    #                   ("baseline_controllers_forecasters_test_awaken", 3)]
                    plot_simulations(
                        time_series_df, plotting_cases, args.save_dir, include_power=True, 
                        legend_loc="outer", single_plot=False) 

            if ((case_families.index("baseline_controllers") in args.case_ids)) and (case_families.index("cost_func_tuning") in args.case_ids):
                
                mpc_alpha_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning"]

                if case_families.index("baseline_controllers") in args.case_ids:
                    lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                elif case_families.index("baseline_controllers_3") in args.case_ids:
                    lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName") == "Greedy")]

                mpc_alpha_df[[("RelativeTotalRunningOptimizationCostMean", "mean"), ("RelativeRunningOptimizationCostTerm_0", "mean"), ("RelativeRunningOptimizationCostTerm_1", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]]\
                        .sort_values(by=("FarmPower", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True) 
            

                # better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0]) & (mpc_alpha_df[("YawAngleChangeAbs", "mean")] < lut_df[("YawAngleChangeAbs", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPower", "mean")] > greedy_df[("FarmPower", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("FarmPower", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)

                # plot_simulations(time_series_df=time_series_df, 
                #                  plotting_cases=[("cost_func_tuning", "alpha_0.001"),
                #                                   ("cost_func_tuning", "alpha_0.999")], save_dir=args.save_dir)

                
                # x = agg_df.loc[(agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning") 
                #            & ((agg_df.index.get_level_values("CaseName") == "alpha_0.001") 
                #               | (agg_df.index.get_level_values("CaseName") == "alpha_0.999")), 
                #            [('YawAngleChangeAbs', 'mean'), ('FarmPower', 'mean'), 
                #             ('RelativeRunningOptimizationCostTerm_0', 'mean'), ('RelativeRunningOptimizationCostTerm_1', 'mean')]
                #             ].sort_values(by=('FarmPower', 'mean'), ascending=False).reset_index(level="CaseFamily", drop=True)
                # x.columns = x.columns.droplevel(1)
                better_than_lut_df = better_than_lut_df.sort_values(by=("FarmPower", "mean"), ascending=False)
                100 * (better_than_lut_df.iloc[0]["FarmPower"] - lut_df.iloc[0]["FarmPower"]) / lut_df.iloc[0]["FarmPower"]
                100 * (better_than_lut_df.iloc[0]["FarmPower"] - greedy_df.iloc[0]["FarmPower"]) / greedy_df.iloc[0]["FarmPower"]
                
                better_than_lut_df = better_than_lut_df.sort_values(by=("YawAngleChangeAbs", "mean"), ascending=True)
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - lut_df.iloc[0]["YawAngleChangeAbs"]) / lut_df.iloc[0]["YawAngleChangeAbs"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - greedy_df.iloc[0]["YawAngleChangeAbs"]) / greedy_df.iloc[0]["YawAngleChangeAbs"]

                100 * (lut_df.iloc[0]["FarmPower"] - greedy_df.iloc[0]["FarmPower"]) / greedy_df.iloc[0]["FarmPower"]
                100 * (lut_df.iloc[0]["YawAngleChangeAbs"] - greedy_df.iloc[0]["YawAngleChangeAbs"]) / greedy_df.iloc[0]["YawAngleChangeAbs"]

                plot_cost_function_pareto_curve(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("scalability") in args.case_ids):
                floris_input_files = case_studies["scalability"]["floris_input_file"]["vals"]
                lut_paths = case_studies["scalability"]["lut_path"]["vals"]
                plot_wind_farm(floris_input_files, lut_paths, args.save_dir)
            
            if case_families.index("breakdown_robustness") in args.case_ids:
                plot_breakdown_robustness(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("horizon_length") in args.case_ids):
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "horizon_length"][
                    [("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]
                    ].sort_values(by=("FarmPower", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)

                config_cols = ["controller_dt", "n_horizon"]
                for (case_family, case_name), _ in mpc_df.iterrows():
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.pkl"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='r') as fp:
                        input_config = pickle.load(fp)
                    
                    for col in config_cols:
                        mpc_df.loc[(mpc_df.index.get_level_values("CaseFamily") == case_family) & (mpc_df.index.get_level_values("CaseName") == case_name), col] = input_config["controller"][col]

                lut_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                greedy_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]]
                 
                plot_horizon_length(pd.concat([mpc_df, lut_df, greedy_df]), args.save_dir)

            if case_families.index("yaw_offset_study") in args.case_ids:
                
                mpc_alpha_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (~agg_df.index.get_level_values("CaseName").str.contains("LUT"))]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (agg_df.index.get_level_values("CaseName").str.contains("LUT"))]
                
                if "baseline_controllers_3" in agg_df.index.get_level_values("CaseFamily"):
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName").str.contains("Greedy"))]  
                    
                    better_than_lut_df = mpc_alpha_df.loc[((mpc_alpha_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0])
                                                        & (mpc_alpha_df[("YawAngleChangeAbs", "mean")] < lut_df[("YawAngleChangeAbs", "mean")].iloc[0])), 
                                                        [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]]\
                                                            .sort_values(by=("FarmPower", "mean"), ascending=False)\
                                                                .reset_index(level="CaseFamily", drop=True)
                    plotting_cases = [("yaw_offset_study", better_than_lut_df.iloc[0]._name),   
                                                    ("baseline_controllers_3", "LUT_3turb"),
                                                    ("baseline_controllers_3", "Greedy")
                    ]
                    # NOTE USE THIS CALL TO GENERATE TIME SERIES PLOTS
                    plot_simulations(
                        time_series_df, plotting_cases, args.save_dir, include_power=True, legend_loc="outer", single_plot=False) 


                # plot yaw vs wind dir
                # set(time_series_df.loc[time_series_df.index.get_level_values("CaseFamily") == "yaw_offset_study", :].index.get_level_values("CaseName").values)
                case_names = ["LUT_3turb", "StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_11_3turb", 
                              "StochasticIntervalElliptical_11_3turb", "StochasticSample_25_3turb", "StochasticSample_100_3turb"]
                case_labels = ["LUT", "MPC\n1 RI Samples", "MPC\n11 RI Samples", "MPC\n11 EI Samples", "MPC\n25 * S Samples", "MPC\n100 S Samples"]
                plot_yaw_offset_wind_direction(time_series_df, case_names, case_labels,
                                            os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                            os.path.join(args.save_dir, "yaw_offset_study", "yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], 
                                            include_yaw=True, include_power=True, scatter=False)
                
                for sub_case_names, sub_case_labels, filename in zip([["LUT_3turb"], 
                                                                      ["StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_11_3turb", "StochasticIntervalElliptical_11_3turb"],
                                                                      ["StochasticSample_25_3turb", "StochasticSample_100_3turb"]], 
                                                           [["LUT"], ["MPC\n1 * RI Samples", "MPC\n11 * RI Samples", "MPC\n11 * EI Samples"], 
                                                            ["MPC\n25 * Stochastic Samples", "MPC\n100 * Stochastic Samples"]],
                                                           ["lut", "stochastic_interval", "stochastic_sample"]):
                    plot_yaw_offset_wind_direction(time_series_df, sub_case_names, sub_case_labels,
                                                os.path.join(os.path.dirname(whoc.__file__), 
                                                             f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"),
                                                os.path.join(args.save_dir, "yaw_offset_study", 
                                                             f"yawoffset_winddir_{filename}_ts.png"), 
                                                             plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True, scatter=False)

            if (case_families.index("baseline_controllers_3") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids or case_families.index("n_wind_preview_samples") in args.case_ids):
                # find best diff_type, nu, and decay for each sampling type
                 
                if case_families.index("gradient_type") in args.case_ids:
                    MPC_TYPE = "gradient_type"
                elif case_families.index("n_wind_preview_samples") in args.case_ids:
                    MPC_TYPE = "n_wind_preview_samples"

                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == MPC_TYPE][
                    [("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]
                    ].sort_values(by=("FarmPower", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)

                config_cols = ["n_wind_preview_samples", "wind_preview_type", "diff_type", "nu", "decay_type", "max_std_dev", "n_horizon"]
                for (case_family, case_name), _ in mpc_df.iterrows():
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.yaml"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='r') as fp:
                        input_config = yaml.safe_load(fp)
                    
                    for col in config_cols:
                        mpc_df.loc[(mpc_df.index.get_level_values("CaseFamily") == case_family) & (mpc_df.index.get_level_values("CaseName") == case_name), col] = input_config["controller"][col]
            
                mpc_df["diff_direction"] = mpc_df["diff_type"].apply(lambda s: s.split("_")[1] if s != "none" else None)
                mpc_df["diff_steps"] = mpc_df["diff_type"].apply(lambda s: s.split("_")[0] if s != "none" else None)
                mpc_df["n_wind_preview_samples_index"] = None

                unique_sir_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples"]))
                unique_sie_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples"]))
                unique_ss_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples"]))
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples"].apply(lambda n: np.where(unique_sir_n_samples == n)[0][0]).astype("int").values
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples"].apply(lambda n: np.where(unique_sie_n_samples == n)[0][0]).astype("int").values
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples"].apply(lambda n: np.where(unique_ss_n_samples == n)[0][0]).astype("int").values
                mpc_df.columns = mpc_df.columns.droplevel(1)

                # read params from input configs rather than CaseName
                lut_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                lut_df.columns = lut_df.columns.droplevel(1)
                greedy_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]]
                greedy_df.columns = greedy_df.columns.droplevel(1)

                # better_than_lut_df = mpc_df.loc[(mpc_df["FarmPower"] > lut_df["FarmPower"].iloc[0]), ["YawAngleChangeAbs", "OptimizationConvergenceTime", "FarmPower"] + config_cols].sort_values(by="FarmPower", ascending=False).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df["FarmPower"] > lut_df["FarmPower"].iloc[0]) 
                                                & (mpc_df["YawAngleChangeAbs"] < lut_df["YawAngleChangeAbs"].iloc[0]), 
                                                ["YawAngleChangeAbs", "OptimizationConvergenceTime", "FarmPower"] + config_cols]\
                                                    .sort_values(by="FarmPower", ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut_df.groupby("wind_preview_type").head(3)[["n_wind_preview_samples", "wind_preview_type", "diff_type", "nu", "decay_type", "max_std_dev"]]
                   # better_than_lut_df = better_than_lut_df.reset_index(level="CaseName", drop=True)

                # better_than_lut_df = better_than_lut_df.sort_values("FarmPower", ascending=False)
                # 100 * (better_than_lut_df.iloc[0]["FarmPower"] - lut_df.iloc[0]["FarmPower"]) / lut_df.iloc[0]["FarmPower"]
                # 100 * (better_than_lut_df.iloc[0]["FarmPower"] - greedy_df.iloc[0]["FarmPower"]) / greedy_df.iloc[0]["FarmPower"]

                # better_than_lut_df = better_than_lut_df.sort_values("YawAngleChangeAbs", ascending=True)
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - lut_df.iloc[0]["YawAngleChangeAbs"]) / lut_df.iloc[0]["YawAngleChangeAbs"]
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbs"] - greedy_df.iloc[0]["YawAngleChangeAbs"]) / greedy_df.iloc[0]["YawAngleChangeAbs"]             

                # best_case_names = better_than_lut_df.groupby(["wind_preview_type"])["diff_type"].idxmax()
                # better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)
                better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)\
                                  .loc[better_than_lut_df.index.get_level_values("CaseName").isin(
                                      better_than_lut_df.groupby(["wind_preview_type"])["FarmPower"].idxmax()),
                                      ["wind_preview_type", "diff_type", "decay_type", "max_std_dev", "nu"]]

                for param in ["diff_type", "decay_type", "max_std_dev", "nu"]:
                    for agg_type in ["mean", "max"]: 
                        logging.info(f"\nFor parameter {param}, taking the {agg_type} of FarmPowerMean over all other parameters, the best parameter for each wind_preview_type is:")
                        logging.info(better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)\
                                        .groupby(["wind_preview_type", param])["FarmPower"].agg(agg_type)\
                                        .groupby("wind_preview_type").idxmax().values)
                
                                #   .loc[better_than_lut_df.index.get_level_values("CaseName").isin(
                                #       better_than_lut_df.groupby(["wind_preview_type"])["FarmPower"].idxmax()),
                                #       ["wind_preview_type", "diff_type", "decay_type", "max_std_dev", "nu"]]

                if True:
                    plot_parameter_sweep(pd.concat([mpc_df, lut_df, greedy_df]), MPC_TYPE, args.save_dir, 
                                         plot_columns=["FarmPower", "diff_type", "decay_type", "max_std_dev", "n_wind_preview_samples", "wind_preview_type", "nu"],
                                         merge_wind_preview_types=False, estimator="mean")
                
                plotting_cases = [(MPC_TYPE, better_than_lut_df.sort_values(by="FarmPower", ascending=False).iloc[0]._name),   
                                                ("baseline_controllers_3", "LUT"),
                                                ("baseline_controllers_3", "Greedy")
                ]

                plot_simulations(
                    time_series_df, plotting_cases, args.save_dir, include_power=True, legend_loc="outer", single_plot=False) 


                # find best power decay type
                # power_decay_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "power_decay_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("FarmPower", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)

            if case_families.index("wind_preview_type") in args.case_ids:
                # TODO get best parameters from each sweep and add to other sweeps, then rerun to compare with LUT
                # find best wind_preview_type and number of samples, if best is on the upper end, increase n_wind_preview_samples in wind_preview_type sweep
                wind_preview_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "wind_preview_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("FarmPower", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)


            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids):
               
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "gradient_type", :]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                
                # get mpc configurations for which the generated farm power is greater than lut, and the resulting yaw actuation lesser than lut
                # better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbs", "mean")] < lut_df[("YawAngleChangeAbs", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPower", "mean")] > lut_df[("FarmPower", "mean")].iloc[0]), [("YawAngleChangeAbs", "mean"), ("OptimizationConvergenceTime", "mean"), ("FarmPower", "mean")]].sort_values(by=("FarmPower", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut = pd.read_csv(os.path.join(args.save_dir, "better_than_lut.csv"), header=[0,1], index_col=[0], skipinitialspace=True)
                better_than_lut_df.to_csv(os.path.join(args.save_dir, "better_than_lut.csv"))
                # better_than_lut_df = mpc_df.loc[(mpc_df[("YawAngleChangeAbs", "mean")] < lut_df[("YawAngleChangeAbs", "mean")].iloc[0]), [("YawAngleChangeAbs", "mean"), ("RelativeTotalRunningOptimizationCostMean", "mean"), ("FarmPower", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                
                # get mpc configurations for which the generated farm power is greater than greedy
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPower", "mean")] > greedy_df[("FarmPower", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].sort_values(by=("YawAngleChangeAbs", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                # better_than_greedy_df = better_than_greedy_df.loc[better_than_greedy_df.index.isin(better_than_lut_df.index)]
                # better_than_lut_df.loc[better_than_lut_df.index.isin(better_than_greedy_df.index)]
                # greedy warm start better,
                
                # lut_df[[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].iloc[0]
                # greedy_df[[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean")]].iloc[0]
                # mpc_df.sort_values(by=("FarmPower", "mean"), ascending=False)[[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]].reset_index(level="CaseFamily", drop=True)
                # mpc_df.sort_values(by=("YawAngleChangeAbs", "mean"), ascending=True)[[("YawAngleChangeAbs", "mean"), ("FarmPower", "mean"), ("OptimizationConvergenceTime", "mean")]].iloc[0]
                # print(better_than_lut_df.iloc[0]._name)
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPower", "mean")] - lut_df.iloc[0][("FarmPower", "mean")]) / lut_df.iloc[0][("FarmPower", "mean")]
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPower", "mean")] - greedy_df.iloc[0][("FarmPower", "mean")]) / greedy_df.iloc[0][("FarmPower", "mean")]
                
                # 100 * (better_than_lut_df.iloc[0][("FarmPower", "mean")] - lut_df.iloc[0][("FarmPower", "mean")]) / lut_df.iloc[0][("FarmPower", "mean")]
                # 100 * (better_than_lut_df.iloc[0][("FarmPower", "mean")] - greedy_df.iloc[0][("FarmPower", "mean")]) / greedy_df.iloc[0][("FarmPower", "mean")]
                
                # plot multibar of farm power vs. stochastic interval n_wind_preview_samples, stochastic sample n_wind_preview_samples
                # 

                # alpha_1.0_controller_class_MPC_diff_type_chain_cd_dt_15_n_horizon_24_n_wind_preview_samples_7_nu_0.001_


            if all(case_families.index(cf) in args.case_ids for cf in ["baseline_controllers", "solver_type",
             "wind_preview_type", "warm_start"]):
                generate_outputs(agg_df, args.save_dir)       

            if case_families.index("baseline_controllers_perfect_forecaster_flasc") in args.case_ids \
                or case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:

                if case_families.index("baseline_controllers_perfect_forecaster_flasc") in args.case_ids:
                    mpc_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_perfect_forecaster_flasc", :]
                elif case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:
                    mpc_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_perfect_forecaster_awaken", :]

                config_cols = ["wind_forecast_class", "prediction_timedelta"]

                for (case_family, case_name), _ in mpc_df.iterrows():
                    input_fn = f"input_config_case_{case_name}.pkl"
                    input_path = os.path.join(args.save_dir, case_family, input_fn)

                    with open(input_path, mode='rb') as fp:
                        input_config = pickle.load(fp)

                    controller_config = input_config.get("controller", {})
                    wind_forecast_config = input_config.get("wind_forecast", {})
                    full_config = {**controller_config, **wind_forecast_config}
                    
                    for col in config_cols:
                        mpc_df.loc[
                            (mpc_df.index.get_level_values("CaseFamily") == case_family) & 
                            (mpc_df.index.get_level_values("CaseName") == case_name), 
                            col
                        ] = full_config[col]  

                # Filter data for the two forecast types
                forecasters_df = mpc_df.loc[mpc_df["wind_forecast_class"] != "PerfectForecast", :]
                perfect_df = mpc_df.loc[mpc_df["wind_forecast_class"] == "PerfectForecast", :]

                # Only perform forecaster comparison analysis if non-perfect forecasters exist
                if len(forecasters_df) > 0 and "prediction_timedelta" in forecasters_df.columns and "prediction_timedelta" in perfect_df.columns:
                    merged_df = forecasters_df.merge(
                        perfect_df,
                        on=["CaseFamily", "prediction_timedelta"],
                        suffixes=("_kalman", "_perfect")
                    )

                    merged_df["power_ratio"] = (merged_df["FarmPowerMean_kalman", "mean"] / merged_df["FarmPowerMean_perfect", "mean"]) * 100

                    plot_df = merged_df[["prediction_timedelta", "power_ratio"]]

                    # Display the prepared data (for debugging)
                    print(plot_df.head())
                    plot_power_increase_vs_prediction_time(plot_df, args.save_dir)
                else:
                    print(f"Skipping forecaster comparison analysis - only {len(forecasters_df)} non-perfect forecasters found")