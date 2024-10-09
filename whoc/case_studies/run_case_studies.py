import os
from concurrent.futures import ProcessPoolExecutor
import warnings
import re
import argparse

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import numpy as np
import pandas as pd
import yaml

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import (read_time_series_data, write_case_family_time_series_data, read_case_family_time_series_data, 
                                                    aggregate_time_series_data, read_case_family_agg_data, write_case_family_agg_data, 
                                                    generate_outputs, plot_simulations, plot_wind_farm, plot_breakdown_robustness, 
                                                    plot_cost_function_pareto_curve, plot_yaw_offset_wind_direction, plot_parameter_sweep)

# np.seterr("raise")

warnings.simplefilter('error', pd.errors.DtypeWarning)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_case_studies.py", description="Run FLORIS case studies for WHOC module.")
    parser.add_argument("case_ids", metavar="C", nargs="+", choices=[str(i) for i in range(len(case_families))])
    parser.add_argument("-gwf", "--generate_wind_field", action="store_true")
    parser.add_argument("-glut", "--generate_lut", action="store_true")
    parser.add_argument("-rs", "--run_simulations", action="store_true")
    parser.add_argument("-rrs", "--rerun_simulations", action="store_true")
    parser.add_argument("-ps", "--postprocess_simulations", action="store_true")
    parser.add_argument("-rps", "--reprocess_simulations", action="store_true")
    parser.add_argument("-ras", "--reaggregate_simulations", action="store_true")
    parser.add_argument("-st", "--stoptime", type=float, default=3600)
    parser.add_argument("-ns", "--n_seeds", type=int, default=6)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"])
    parser.add_argument("-sd", "--save_dir", type=str)
   
    # "/projects/ssc/ahenry/whoc/floris_case_studies" on kestrel
    # "/projects/aohe7145/whoc/floris_case_studies" on curc
    # "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies" on mac
    # python run_case_studies.py 0 1 2 3 4 5 6 7 -rs -p -st 480 -ns 1 -m cf -sd "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"
    args = parser.parse_args()
    args.case_ids = [int(i) for i in args.case_ids]

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": max(d["group"] for d in case_studies[case_family].values()) + 1, "vals": [i for i in range(args.n_seeds)]}

    # os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "false"
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    PLOT = True #sys.platform != "linux"
    if args.run_simulations:
        # run simulations
        
        if RUN_ONCE:
            print(f"running initialize_simulations for case_ids {[case_families[i] for i in args.case_ids]}")
            
            case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in args.case_ids], regenerate_wind_field=args.generate_wind_field, regenerate_lut=args.generate_lut, n_seeds=args.n_seeds, stoptime=args.stoptime, save_dir=args.save_dir)
        
        if args.multiprocessor is not None:
            if args.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                # comm_rank = MPI.COMM_WORLD.Get_rank()
                # node_name = MPI.Get_processor_name()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                # executor = MPIPoolExecutor(max_workers=mp.cpu_count(), root=0)
            elif args.multiprocessor == "cf":
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                if args.multiprocessor == "mpi":
                    run_simulations_exec.max_workers = comm_size
                  
                # print(f"run_simulations line 64 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]],
                                                case_name="_".join([f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
                        for c, d in enumerate(input_dicts)]
                
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                                wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
    
    if args.postprocess_simulations:
        # if (not os.path.exists(os.path.join(args.save_dir, f"time_series_results.csv"))) or (not os.path.exists(os.path.join(args.save_dir, f"agg_results.csv"))):
        # regenerate some or all of the time_series_results_all and agg_results_all .csv files for each case family in case ids
        if args.reprocess_simulations:
            if RUN_ONCE:
                # make a list of the time series csv files for all case_names and seeds in each case family directory
                case_family_case_names = {}
                for i in args.case_ids:
                    case_family_case_names[case_families[i]] = [fn for fn in os.listdir(os.path.join(args.save_dir, case_families[i])) if ".csv" in fn and "time_series_results_case" in fn]

                # case_family_case_names["slsqp_solver_sweep"] = [f"time_series_results_case_alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval_seed_{s}" for s in range(6)]

            # if using multiprocessing
            if args.multiprocessor is not None:
                if args.multiprocessor == "mpi":
                    comm_size = MPI.COMM_WORLD.Get_size()
                    executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                elif args.multiprocessor == "cf":
                    executor = ProcessPoolExecutor()
                with executor as run_simulations_exec:
                    if args.multiprocessor == "mpi":
                        run_simulations_exec.max_workers = comm_size
                    
                    print(f"run_simulations line 107 with {run_simulations_exec._max_workers} workers")
                    # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)

                    # if args.reaggregate_simulations is true, or for any case family where doesn't time_series_results_all.csv exist, 
                    # read the time-series csv files for all case families, case names, and wind seeds
                    read_futures = [run_simulations_exec.submit(
                                                    read_time_series_data, 
                                                    results_path=os.path.join(args.save_dir, case_families[i], fn))
                        for i in args.case_ids 
                        for fn in case_family_case_names[case_families[i]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))
                    ]

                    new_time_series_df = [fut.result() for fut in read_futures]
                    # if there are new resulting dataframes, concatenate them from a list into a dataframe
                    if new_time_series_df:
                        new_time_series_df = pd.concat(new_time_series_df)

                    read_futures = [run_simulations_exec.submit(read_case_family_time_series_data, 
                                                                case_family=case_families[i], save_dir=args.save_dir)
                                    for i in args.case_ids
                                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                    existing_time_series_df = [fut.result() for fut in read_futures]

                    if len(new_time_series_df):
                        write_futures = [run_simulations_exec.submit(write_case_family_time_series_data, 
                                                                     case_family=case_families[i], 
                                                                     new_time_series_df=new_time_series_df, 
                                                                     save_dir=args.save_dir)
                                        for i in args.case_ids
                                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                        _ = [fut.result() for fut in write_futures]
                    
                    time_series_df = pd.concat(existing_time_series_df + [new_time_series_df])

                    # if args.reaggregate_simulations is true, or for any case family where doesn't agg_results_all.csv exist, compute the aggregate stats for each case families and case name, over all wind seeds
                    futures = [run_simulations_exec.submit(aggregate_time_series_data,
                                                             time_series_df=time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i]) & (time_series_df.index.get_level_values("CaseName") == case_name), :],
                                                                yaml_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.yaml"),
                                                                n_seeds=args.n_seeds)
                        for i in args.case_ids
                        for case_name in pd.unique(time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i])].index.get_level_values("CaseName"))
                        # for case_name in [re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], 
                                                                                           "agg_results_all.csv"))
                    ]

                    new_agg_df = [fut.result() for fut in futures]
                    new_agg_df = [df for df in new_agg_df if df is not None]
                    if len(new_agg_df):
                        new_agg_df = pd.concat(new_agg_df)
                    else:
                        new_agg_df = pd.DataFrame()
                    # if args.reaggregate_simulations is false, read the remaining aggregate data from each agg_results_all csv file
                    read_futures = [run_simulations_exec.submit(read_case_family_agg_data, 
                                                                case_family=case_families[i], save_dir=args.save_dir)
                                    for i in args.case_ids 
                                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv"))]
                    existing_agg_df = [fut.result() for fut in read_futures]

                    if len(new_agg_df):
                        write_futures = [run_simulations_exec.submit(write_case_family_agg_data,
                                                                     case_family=case_families[i],
                                                                     new_agg_df=new_agg_df,
                                                                     save_dir=args.save_dir)
                                        for i in args.case_ids
                                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv"))]
                        _ = [fut.result() for fut in write_futures]

                agg_df = pd.concat(existing_agg_df + [new_agg_df])
                    
            # else, run sequentially
            else:
                new_time_series_df = []
                existing_time_series_df = []
                for i in args.case_ids:
                    # all_ts_df_path = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        existing_time_series_df.append(read_case_family_time_series_data(case_families[i], save_dir=args.save_dir))
                
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        new_case_family_time_series_df = []
                        for fn in case_family_case_names[case_families[i]]:
                            new_case_family_time_series_df.append(read_time_series_data(results_path=os.path.join(args.save_dir, case_families[i], fn)))

                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if new_case_family_time_series_df:
                        new_time_series_df.append(pd.concat(new_case_family_time_series_df))
                        write_case_family_time_series_data(case_families[i], new_time_series_df[-1], args.save_dir)
                
                time_series_df = pd.concat(existing_time_series_df + new_time_series_df)
                
                new_agg_df = []
                for i in args.case_ids:
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        # for case_name in set([re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]):
                        case_family_df = time_series_df.iloc[time_series_df.index.get_level_values("CaseFamily") == case_families[i], :]
                        for case_name in pd.unique(case_family_df.index.get_level_values("CaseName")):
                            case_name_df = case_family_df.iloc[case_family_df.index.get_level_values("CaseName") == case_name, :]
                            res = aggregate_time_series_data(
                                                            time_series_df=case_name_df,
                                                             yaml_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.yaml"),
                                                            # results_path=os.path.join(args.save_dir, case_families[i], f"agg_results_{case_name}.csv"),
                                                            n_seeds=args.n_seeds)
                            if res is not None:
                                new_agg_df.append(res)

                new_agg_df = pd.concat(new_agg_df)

                existing_agg_df = []
                for i in args.case_ids:
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        existing_agg_df.append(read_case_family_agg_data(case_families[i], save_dir=args.save_dir))
                
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        write_case_family_agg_data(case_families[i], new_agg_df, args.save_dir)
                
                agg_df = pd.concat(existing_agg_df + [new_agg_df])

        elif RUN_ONCE:
            time_series_df = []
            for i in args.case_ids:
                warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                if os.path.exists(filepath):
                    try:
                        time_series_df.append(pd.read_csv(filepath, index_col=[0, 1]))
                    except pd.errors.DtypeWarning as w:
                        print(f"DtypeWarning with combined time series file {filepath}: {w}")
                        warnings.simplefilter('ignore', pd.errors.DtypeWarning)
                        bad_df = pd.read_csv(filepath, index_col=[0, 1])
                        bad_cols = [bad_df.columns[int(s) - len(bad_df.index.names)] for s in re.findall(r"(?<=Columns \()(.*)(?=\))", w.args[0])[0].split(",")]
                        bad_df.loc[bad_df[bad_cols].isna().any(axis=1)]
            time_series_df = pd.concat(time_series_df)
            
            agg_df = []
            for i in args.case_ids:
                warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")
                if os.path.exists(filepath):
                    try:
                        agg_df.append(pd.read_csv(filepath, header=[0,1], index_col=[0, 1], skipinitialspace=True))
                    except pd.errors.DtypeWarning as w:
                        print(f"DtypeWarning with combined time series file {filepath}: {w}")
                        warnings.simplefilter('ignore', pd.errors.DtypeWarning)
                        bad_df = pd.read_csv(filepath, header=[0,1], index_col=[0, 1], skipinitialspace=True)
                        bad_cols = [bad_df.columns[int(s) - len(bad_df.index.names)] for s in re.findall(r"(?<=Columns \()(.*)(?=\))", w.args[0])[0].split(",")]
                        bad_df.loc[bad_df[bad_cols].isna().any(axis=1)]

            agg_df = pd.concat(agg_df)

        if RUN_ONCE and PLOT:
            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("cost_func_tuning") in args.case_ids):
                # TODO HIGH find out why lower alpha is resulting in higher power, and why higher alpha is not resulting in significantly lower yaw actuation 
                mpc_alpha_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning"]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_alpha_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)

                plot_simulations(time_series_df, [("cost_func_tuning", "alpha_0.001"),
                                                  ("cost_func_tuning", "alpha_0.999")], args.save_dir)
                
                x = agg_df.loc[(agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning") 
                           & ((agg_df.index.get_level_values("CaseName") == "alpha_0.001") 
                              | (agg_df.index.get_level_values("CaseName") == "alpha_0.999")), 
                           [('YawAngleChangeAbsMean', 'mean'), ('FarmPowerMean', 'mean'), 
                            ('RelativeRunningOptimizationCostTerm_0', 'mean'), ('RelativeRunningOptimizationCostTerm_1', 'mean')]
                            ].sort_values(by=('FarmPowerMean', 'mean'), ascending=False).reset_index(level="CaseFamily", drop=True)
                x.columns = x.columns.droplevel(1)

                plot_cost_function_pareto_curve(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("scalability") in args.case_ids):
                floris_input_files = case_studies["scalability"]["floris_input_file"]["vals"]
                lut_paths = case_studies["scalability"]["lut_path"]["vals"]
                plot_wind_farm(floris_input_files, lut_paths, args.save_dir)
            
            if case_families.index("breakdown_robustness") in args.case_ids:
                plot_breakdown_robustness(agg_df, args.save_dir)

            if case_families.index("yaw_offset_study") in args.case_ids:
                
                mpc_alpha_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (~agg_df.index.get_level_values("CaseName").str.contains("LUT"))]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (agg_df.index.get_level_values("CaseName").str.contains("LUT"))] 
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), 
                                                      [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]\
                                                        .sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True)\
                                                            .reset_index(level="CaseFamily", drop=True)

                # plot yaw vs wind dir
                case_names = ["LUT_3turb", "StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_9_3turb", 
                              "StochasticIntervalElliptical_9_3turb", "StochasticSample_50_3turb", "StochasticSample_500_3turb"]
                case_labels = ["LUT", "MPC\n1 RI Samples", "MPC\n5 RI Samples", "MPC\n9 EI Samples", "MPC\n50 * S Samples", "MPC\n500 S Samples"]
                plot_yaw_offset_wind_direction(time_series_df, case_names, case_labels,
                                            os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                            os.path.join(args.save_dir, "yaw_offset_study", "yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)
                
                for sub_case_names, sub_case_labels, filename in zip([["LUT_3turb"], 
                                                                      ["StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_9_3turb"],
                                                                      ["StochasticIntervalElliptical_9_3turb"],  
                                                                      ["StochasticSample_50_3turb", "StochasticSample_500_3turb"]], 
                                                           [["LUT"], ["MPC\n1 * RI Samples", "MPC\n9 * RI Samples"], ["MPC\n9 * EI Samples"], 
                                                            ["MPC\n50 * Stochastic Samples", "MPC\n500 * Stochastic Samples"]],
                                                           ["lut", "stochastic_interval_rectangular", "stochastic_interval_elliptical", "stochastic_sample"]):
                    plot_yaw_offset_wind_direction(time_series_df, sub_case_names, sub_case_labels,
                                                os.path.join(os.path.dirname(whoc.__file__), 
                                                             f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"),
                                                os.path.join(args.save_dir, "yaw_offset_study", 
                                                             f"yawoffset_winddir_{filename}_ts.png"), 
                                                             plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)

            if (case_families.index("baseline_controllers") in args.case_ids or case_families.index("baseline_controllers_3") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids or case_families.index("n_wind_preview_samples") in args.case_ids):
                # find best diff_type, nu, and decay for each sampling type
                 
                if case_families.index("gradient_type") in args.case_ids:
                    MPC_TYPE = "gradient_type"
                elif case_families.index("n_wind_preview_samples") in args.case_ids:
                    MPC_TYPE = "n_wind_preview_samples"

                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == MPC_TYPE][
                    [("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]
                    ].sort_values(by=("FarmPowerMean", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)

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
                lut_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                lut_df.columns = lut_df.columns.droplevel(1)
                greedy_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]]
                greedy_df.columns = greedy_df.columns.droplevel(1)

                # better_than_lut_df = mpc_df.loc[(mpc_df["FarmPowerMean"] > lut_df["FarmPowerMean"].iloc[0]), ["YawAngleChangeAbsMean", "OptimizationConvergenceTime", "FarmPowerMean"] + config_cols].sort_values(by="FarmPowerMean", ascending=False).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df["FarmPowerMean"] > lut_df["FarmPowerMean"].iloc[0]) 
                                                & (mpc_df["YawAngleChangeAbsMean"] < lut_df["YawAngleChangeAbsMean"].iloc[0]), 
                                                ["YawAngleChangeAbsMean", "OptimizationConvergenceTime", "FarmPowerMean"] + config_cols]\
                                                    .sort_values(by="FarmPowerMean", ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut_df.groupby("wind_preview_type").head(3)[["n_wind_preview_samples", "wind_preview_type", "diff_type", "nu", "decay_type", "max_std_dev"]]
                   # better_than_lut_df = better_than_lut_df.reset_index(level="CaseName", drop=True)

                # better_than_lut_df = better_than_lut_df.sort_values("FarmPowerMean", ascending=False)
                # 100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - lut_df.iloc[0]["FarmPowerMean"]) / lut_df.iloc[0]["FarmPowerMean"]
                # 100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - greedy_df.iloc[0]["FarmPowerMean"]) / greedy_df.iloc[0]["FarmPowerMean"]

                # better_than_lut_df = better_than_lut_df.sort_values("YawAngleChangeAbsMean", ascending=True)
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - lut_df.iloc[0]["YawAngleChangeAbsMean"]) / lut_df.iloc[0]["YawAngleChangeAbsMean"]
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - greedy_df.iloc[0]["YawAngleChangeAbsMean"]) / greedy_df.iloc[0]["YawAngleChangeAbsMean"]             

                if True:
                    plot_parameter_sweep(pd.concat([mpc_df, lut_df, greedy_df]), MPC_TYPE, args.save_dir, 
                                         plot_columns=["FarmPowerMean", "diff_type", "decay_type", "max_std_dev", "n_wind_preview_samples", "wind_preview_type", "nu"],
                                         merge_wind_preview_types=False, estimator="max")
                
                plotting_cases = [(MPC_TYPE, better_than_lut_df.sort_values(by="FarmPowerMean", ascending=False).iloc[0]._name),   
                                                ("baseline_controllers_3", "LUT"),
                                                ("baseline_controllers_3", "Greedy")
                ]

                plot_simulations(
                    time_series_df, plotting_cases, args.save_dir, include_power=False, legend_loc="outer", single_plot=False) 


                # find best power decay type
                # power_decay_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "power_decay_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)

            if case_families.index("wind_preview_type") in args.case_ids:
                # TODO get best parameters from each sweep and add to other sweeps, then rerun to compare with LUT
                # find best wind_preview_type and number of samples, if best is on the upper end, increase n_wind_preview_samples in wind_preview_type sweep
                wind_preview_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "wind_preview_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)


            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids):
               
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "gradient_type", :]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                
                # get mpc configurations for which the generated farm power is greater than lut, and the resulting yaw actuation lesser than lut
                # better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), [("YawAngleChangeAbsMean", "mean"), ("OptimizationConvergenceTime", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut = pd.read_csv(os.path.join(args.save_dir, "better_than_lut.csv"), header=[0,1], index_col=[0], skipinitialspace=True)
                better_than_lut_df.to_csv(os.path.join(args.save_dir, "better_than_lut.csv"))
                # better_than_lut_df = mpc_df.loc[(mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("YawAngleChangeAbsMean", "mean"), ("RelativeTotalRunningOptimizationCostMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                
                # get mpc configurations for which the generated farm power is greater than greedy
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                # better_than_greedy_df = better_than_greedy_df.loc[better_than_greedy_df.index.isin(better_than_lut_df.index)]
                # better_than_lut_df.loc[better_than_lut_df.index.isin(better_than_greedy_df.index)]
                # greedy warm start better,
                
                # lut_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                # greedy_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                # mpc_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].reset_index(level="CaseFamily", drop=True)
                # mpc_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].iloc[0]
                # print(better_than_lut_df.iloc[0]._name)
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                # 100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                # 100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                # plot multibar of farm power vs. stochastic interval n_wind_preview_samples, stochastic sample n_wind_preview_samples
                # 

                # alpha_1.0_controller_class_MPC_diff_type_chain_cd_dt_15_n_horizon_24_n_wind_preview_samples_7_nu_0.001_


            if all(case_families.index(cf) in args.case_ids for cf in ["baseline_controllers", "solver_type",
             "wind_preview_type", "warm_start", 
              "horizon_length", "scalability"]):
                generate_outputs(agg_df, args.save_dir)