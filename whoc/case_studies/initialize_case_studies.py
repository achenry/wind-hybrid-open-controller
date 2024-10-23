import os
import pickle
import yaml
import copy
import io
import sys
from glob import glob
from itertools import product
from functools import partial

import pandas as pd
import numpy as np

from whoc import __file__ as whoc_file
from whoc.wind_field.WindField import plot_ts
from whoc.wind_field.WindField import generate_multi_wind_ts, WindField, write_abl_velocity_timetable, first_ord_filter
from whoc.case_studies.process_case_studies import plot_wind_field_ts
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel

from hercules.utilities import load_yaml

if sys.platform == "linux":
    N_COST_FUNC_TUNINGS = 21
    # if os.getlogin() == "ahenry":
    #     # Kestrel
    #     STORAGE_DIR = "/projects/ssc/ahenry/whoc/floris_case_studies"
    # elif os.getlogin() == "aohe7145":
    #     STORAGE_DIR = "/projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies"
elif sys.platform == "darwin":
    N_COST_FUNC_TUNINGS = 21
    # STORAGE_DIR = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"

# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers": { "dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          },
    "solver_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    # "alpha": {"group": 0, "vals": [1.0]},
                    # "max_std_dev": {"group": 0, "vals": [2]},
                    #  "warm_start": {"group": 0, "vals": ["lut"]},
                    #     "dt": {"group": 0, "vals": [15]},
                    #      "decay_type": {"group": 0, "vals": ["exp"]},
                    #     "wind_preview_type": {"group": 0, "vals": ["stochastic_sample"]},
                    #     "n_wind_preview_samples": {"group": 0, "vals": [9]},
                    #     "n_horizon": {"group": 0, "vals": [12]},
                    #     "diff_type": {"group": 0, "vals": ["direct_cd"]},
                        # "nu": {"group": 0, "vals": [0.0001]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                         "case_names": {"group": 1, "vals": ["Sequential SLSQP", "SLSQP", "Sequential Refine"]},
                        "solver": {"group": 1, "vals": ["sequential_slsqp", "slsqp", "serial_refine"]}
    },
    "wind_preview_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          "case_names": {"group": 1, "vals": [
                                                            "Stochastic Interval Elliptical 3", "Stochastic Interval Elliptical 5", "Stochastic Interval Elliptical 7", "Stochastic Interval Elliptical 9", "Stochastic Interval Elliptical 11", 
                                                            "Stochastic Interval Rectangular 3", "Stochastic Interval Rectangular 5", "Stochastic Interval Rectangular 7", "Stochastic Interval Rectangular 9", "Stochastic Interval Rectangular 11",
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100", "Stochastic Sample 250", "Stochastic Sample 500",
                                                            "Perfect", "Persistent"]},
                         "n_wind_preview_samples": {"group": 1, "vals": [3, 5, 7, 9, 11] * 2 + [25, 50, 100, 250, 500] + [1, 1]},
                       
                         "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_elliptical"] * 5 + ["stochastic_interval_rectangular"] * 5 + ["stochastic_sample"] * 5 + ["perfect", "persistent"]}
                          },
    "warm_start": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                   "case_names": {"group": 1, "vals": ["Greedy", "LUT", "Previous"]},
                   "warm_start": {"group": 1, "vals": ["greedy", "lut", "previous"]}
                   },
    "horizon_length": {"controller_class": {"group": 0, "vals": ["MPC"]},
                        "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                    f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                    f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                       "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [12, 24, 36]]},
                       "n_horizon": {"group": 1, "vals": [12, 24, 36]}
                    },
    "breakdown_robustness":  # case_families[5]
        {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
         "dt": {"group": 1, "vals": [15, 5, 5]},
         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{25}.yaml")]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{25}.csv")]},
        #   "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))]},
          "offline_probability": {"group": 2, "vals": list(np.linspace(0, 0.1, N_COST_FUNC_TUNINGS))}
        },
    "scalability": {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
                    "dt": {"group": 1, "vals": [15, 5, 5]},
                    # "case_names": {"group": 2, "vals": ["3 Turbines", "9 Turbines", "25 Turbines"]},
                    "num_turbines": {"group": 2, "vals": [3, 9, 25]},
                    "floris_input_file": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3, 9, 25]]},
                    "lut_path": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                    f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{nturb}.csv") for nturb in [3, 9, 25]]},
    },
    "cost_func_tuning": {"controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 3)}" for f in list(np.concatenate([np.linspace(0, 0.8, int(N_COST_FUNC_TUNINGS//2)), 0.801 + (1-np.logspace(-3, 0, N_COST_FUNC_TUNINGS - int(N_COST_FUNC_TUNINGS//2)))*0.199]))]},
                         "alpha": {"group": 1, "vals": list(np.concatenate([np.linspace(0, 0.8, int(N_COST_FUNC_TUNINGS//2)), 0.801 + (1-np.logspace(-3, 0, N_COST_FUNC_TUNINGS - int(N_COST_FUNC_TUNINGS//2)))*0.199]))},
                         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
    },
    "yaw_offset_study": {"controller_class": {"group": 1, "vals": ["MPC", "MPC", "MPC", "LookupBasedWakeSteeringController", "MPC", "MPC"]},
                          "case_names": {"group": 1, "vals":[f"StochasticIntervalRectangular_1_3turb", f"StochasticIntervalRectangular_9_3turb", f"StochasticIntervalElliptical_9_3turb", 
                                                             f"LUT_3turb", f"StochasticSample_50_3turb", f"StochasticSample_500_3turb"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular"] * 2 + ["stochastic_interval_elliptical"] + ["none"] + ["stochastic_sample"] * 2},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 9, 9, 1, 50, 500]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]}
    },
    "baseline_plus_controllers": {"dt": {"group": 1, "vals": [5, 5, 60.0, 60.0, 60.0, 60.0]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy", "MPC_with_Filter", "MPC_without_Filter", "MPC_without_state_cons", "MPC_without_dyn_state_cons"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController", "MPC", "MPC", "MPC", "MPC"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True, True, False, False, False]},
    },
    "baseline_controllers_3": { "dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
    },
    "gradient_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                    "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular"] * 4 + ["stochastic_interval_elliptical"] * 4 + ["stochastic_sample"] * 6},
                    "n_wind_preview_samples": {"group": 1, "vals": [10] * 4 + [33] * 4 + [100] * 6},
                    "diff_type": {"group": 1, "vals": ["direct_cd", "direct_fd", "chain_cd", "chain_fd"] * 2 + ["direct_cd", "direct_fd", "direct_zscg", "chain_cd", "chain_fd", "chain_zscg"]},
                    "nu": {"group": 2, "vals": [0.0001, 0.001, 0.01]},
                    "decay_type": {"group": 3, "vals": ["none", "exp", "cosine", "linear", "zero"]},
                    # "decay_const": {"group": 2, "vals": [31, 45, 60, 90] * 3 + [90, 90]},
                    # "decay_all": {"group": 3, "vals": ["True", "False"]},
                    # "clip_value": {"group": 4, "vals": [30, 44]},
                    "max_std_dev": {"group": 4, "vals": [1, 1.5, 2]}
    },
    "n_wind_preview_samples": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                          "case_names": {"group": 1, "vals": [
                                                            "Stochastic Interval Elliptical 11", "Stochastic Interval Elliptical 21", "Stochastic Interval Elliptical 33", 
                                                            "Stochastic Interval Rectangular 5", "Stochastic Interval Rectangular 7", "Stochastic Interval Rectangular 11", 
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100",
                                                            "Perfect", "Persistent"]},
                        "nu": {"group": 1, "vals": [0.001] * 4 + [0.0001] * 4 + [0.001] * 4},
                        "max_std_dev": {"group": 1, "vals": [1.5] * 8 + [2] * 4 + [2, 2]},
                        "decay_type": {"group": 1, "vals": ["exp"] * 4 + ["cosine"] * 4 + ["exp"] * 4 + ["none", "none"]},
                        "n_wind_preview_samples": {"group": 1, "vals": [11, 21, 33] + [5, 7, 11] + [25, 50, 100] + [1, 1]},
                        "diff_type": {"group": 1, "vals": ["direct_cd"] * 3 + ["chain_cd"] * 3 + ["chain_cd"] * 3 + ["chain_cd", "chain_cd"]},
                         "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_elliptical"] * 3 + ["stochastic_interval_rectangular"] * 3 + ["stochastic_sample"] * 3 + ["perfect", "persistent"]}
     },
    "generate_sample_figures": {
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "n_horizon": {"group": 0, "vals": [3]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular", "stochastic_interval_elliptical", "stochastic_sample"]},
                             "n_wind_preview_samples": {"group": 1, "vals": [5, 8, 25]},
                             "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{1}.yaml")]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{1}.csv")]}
    },
    "cost_func_tuning_small": {
        "controller_class": {"group": 0, "vals": ["MPC"]},
        "n_horizon": {"group": 0, "vals": [6]},
        # "wind_preview_type": {"group": 2, "vals": ["stochastic_sample", "stochastic_interval_rectangular", "stochastic_interval_elliptical"]},
        # "n_wind_preview_samples": {"group": 2, "vals": [100, 10, 10]},
        "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 3)}" for f in [0.0, 0.001, 0.5, 0.999, 1]]},
        "alpha": {"group": 1, "vals": [0.0, 0.001, 0.5, 0.999, 1]},
        "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                    f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                       f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
    }
}

def convert_str(val):
    def try_type(val, data_type):
        try:
            data_type(val)
            return True
        except:
            return False
#        return isinstance(val, data_type)  ### this doesn't work b/c of numpy data types; they're not instances of base types
    def try_list(val):
        try:
            val[0]
            return True
        except:
            return False

    if try_type(val, int) and int(val) == float(val):
        return int(val)
    elif try_type(val, float):
        return float(val)
    elif val=='True':
        return True
    elif val=='False':
        return False
    # elif type(val)!=str and try_list(val):
    #     return ", ".join(['{:}'.format(i) for i in val])
    else:
        return val

def case_naming(n_cases, namebase=None):
    # case naming
    case_name = [('%d'%i).zfill(len('%d'%(n_cases-1))) for i in range(n_cases)]
    if namebase:
        case_name = [namebase+'_'+caseid for caseid in case_name]

    return case_name

def CaseGen_General(case_inputs, namebase=''):
    """ Cartesian product to enumerate over all combinations of set of variables that are changed together"""

    # put case dict into lists
    change_vars = sorted(case_inputs.keys())
    change_vals = [case_inputs[var]['vals'] for var in change_vars]
    change_group = [case_inputs[var]['group'] for var in change_vars]

    # find number of groups and length of groups
    group_set = list(set(change_group))
    group_len = [len(change_vals[change_group.index(i)]) for i in group_set]

    # case matrix, as indices
    group_idx = [range(n) for n in group_len]
    matrix_idx = list(product(*group_idx))

    # index of each group
    matrix_group_idx = [np.where([group_i == group_j for group_j in change_group])[0].tolist() for group_i in group_set]

    # build final matrix of variable values
    matrix_out = []
    for i, row in enumerate(matrix_idx):
        row_out = [None]*len(change_vars)
        for j, val in enumerate(row):
            for g in matrix_group_idx[j]:
                row_out[g] = change_vals[g][val]
        matrix_out.append(row_out)
    try:
        matrix_out = np.asarray(matrix_out, dtype=str)
    except:
        matrix_out = np.asarray(matrix_out)
    n_cases = np.shape(matrix_out)[0]

    # case naming
    case_name = case_naming(n_cases, namebase=namebase)

    case_list = []
    for i in range(n_cases):
        case_list_i = {}
        for j, var in enumerate(change_vars):
            case_list_i[var] = convert_str(matrix_out[i,j])
        case_list.append(case_list_i)

    return case_list, case_name

def initialize_simulations(case_study_keys, regenerate_lut, regenerate_wind_field, n_seeds, stoptime, save_dir):
    """_summary_

    Args:
        case_study_keys (_type_): _description_
        regenerate_lut (_type_): _description_
        regenerate_wind_field (_type_): _description_
        n_seeds (_type_): _description_
        stoptime (_type_): _description_
        save_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_dict = load_yaml(os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml"))

    with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(save_dir, 'wind_field_data/raw_data')
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime

    if "slsqp_solver_sweep" not in case_studies or "dt" not in case_studies["slsqp_solver_sweep"]:
        max_controller_dt = input_dict["controller"]["dt"]
    else:
        max_controller_dt = max(case_studies["slsqp_solver_sweep"]["dt"]["vals"])
    
    if "horizon_length" not in case_studies or "n_horizon" not in case_studies["horizon_length"]:
        max_n_horizon = input_dict["controller"]["n_horizon"]
    else:
        max_n_horizon = max(case_studies["horizon_length"]["n_horizon"]["vals"])

    # wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
    wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
    wind_field_config["preview_dt"] = int(max_controller_dt / input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = input_dict["dt"]
    
    # wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / input_dict["dt"]) \
        + max_n_horizon * int(max_controller_dt/ input_dict["dt"])
    wind_field_config["n_samples_per_init_seed"] = 1
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["distribution_params_path"] = os.path.join(save_dir, "wind_field_data", "wind_preview_distribution_params.pkl")  
    wind_field_config["time_series_dt"] = 1
    
    # TODO check that wind field has same dt or interpolate...
    seed = 0
    if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
        n_seeds = 6
        print("regenerating wind fields")
        wind_field_config["regenerate_distribution_params"] = True # set to True to regenerate from constructed mean and covaraicne
        full_wf = WindField(**wind_field_config)
        if not os.path.exists(wind_field_dir):
            os.makedirs(wind_field_dir)
        wind_field_data = generate_multi_wind_ts(full_wf, wind_field_dir, init_seeds=[seed + i for i in range(n_seeds)])
        write_abl_velocity_timetable([wfd.df for wfd in wind_field_data], wind_field_dir) # then use these timetables in amr precursor
        # write_abl_velocity_timetable(wind_field_data, wind_field_dir) # then use these timetables in amr precursor
        lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["dt"])
        plot_wind_field_ts(wind_field_data[0].df, wind_field_dir, filter_func=partial(first_ord_filter, alpha=lpf_alpha))
        plot_ts(pd.concat([wfd.df for wfd in wind_field_data]), wind_field_dir)
        wind_field_filenames = [os.path.join(wind_field_dir, f"case_{i}.csv") for i in range(n_seeds)]
        regenerate_wind_field = True
    
    # if wind field data exists, get it
    WIND_TYPE = "stochastic"
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for f, fn in enumerate(wind_field_filenames):
            wind_field_data.append(pd.read_csv(fn, index_col=0))
            
            if WIND_TYPE == "step":
                # n_rows = len(wind_field_data[-1].index)
                wind_field_data[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                wind_field_data[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                wind_field_data[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                wind_field_data[-1].loc[45:, f"FreestreamWindDir"] = 270.0
    
    # write_abl_velocity_timetable(wind_field_data, wind_field_dir)
    
    # true wind disturbance time-series
    #plot_wind_field_ts(pd.concat(wind_field_data), os.path.join(wind_field_fig_dir, "seeds.png"))
    wind_mag_ts = [wind_field_data[case_idx]["FreestreamWindMag"].to_numpy() for case_idx in range(n_seeds)]
    wind_dir_ts = [wind_field_data[case_idx]["FreestreamWindDir"].to_numpy() for case_idx in range(n_seeds)]

    # regenerate floris lookup tables for all wind farms included
    if regenerate_lut:
        lut_input_dict = dict(input_dict)
        for lut_path, floris_input_file in zip(case_studies["scalability"]["lut_path"]["vals"], 
                                                        case_studies["scalability"]["floris_input_file"]["vals"]):
            fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                            offline_probability=input_dict["controller"]["offline_probability"],
                                            dt=input_dict["dt"],
                                            yaw_rate=input_dict["controller"]["yaw_rate"],
                                            config_path=floris_input_file)
            lut_input_dict["controller"]["lut_path"] = lut_path
            lut_input_dict["controller"]["generate_lut"] = True
            ctrl_lut = LookupBasedWakeSteeringController(fi, lut_input_dict, wind_mag_ts=wind_mag_ts[0], wind_dir_ts=wind_dir_ts[0])

        input_dict["controller"]["generate_lut"] = False

    assert np.all([np.isclose(wind_field_data[case_idx]["Time"].iloc[1] - wind_field_data[case_idx]["Time"].iloc[0], input_dict["dt"]) for case_idx in range(n_seeds)]), "sampling time of wind field should be equal to simulation sampling time"

    input_dicts = []
    case_lists = []
    case_name_lists = []
    n_cases_list = []

    for case_study_key in case_study_keys:
        case_list, case_names = CaseGen_General(case_studies[case_study_key], namebase=case_study_key)
        case_lists = case_lists + case_list
        case_name_lists = case_name_lists + case_names
        n_cases_list.append(len(case_list))

        # make save directory
        results_dir = os.path.join(save_dir, case_study_key)
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Load default settings and make copies
        start_case_idx = len(input_dicts)
        input_dicts = input_dicts + [copy.deepcopy(input_dict) for i in range(len(case_list))]

        # make adjustements based on case study
        for c, case in enumerate(case_list):
            for property_name, property_value in case.items():
                if isinstance(property_value, np.str_):
                    input_dicts[start_case_idx + c]["controller"][property_name] = str(property_value)
                else:
                    input_dicts[start_case_idx + c]["controller"][property_name] = property_value
                    
            fn = f'input_config_case_{"_".join([f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case.items() if key not in ["wind_case_idx", "seed", "floris_input_file", "lut_path"]]) if "case_names" not in case else case["case_names"]}.yaml'.replace("/", "_")
            
            with io.open(os.path.join(results_dir, fn), 'w', encoding='utf8') as fp:
                yaml.dump(input_dicts[start_case_idx + c], fp, default_flow_style=False, allow_unicode=True)

    # instantiate controller and run_simulations simulation
    wind_field_config["regenerate_distribution_params"] = False

    with open(os.path.join(save_dir, "init_simulations.pkl"), "wb") as fp:
        pickle.dump({"case_lists": case_lists, "case_name_lists": case_name_lists, "input_dicts": input_dicts, "wind_field_config": wind_field_config,
                     "wind_mag_ts": wind_mag_ts, "wind_dir_ts": wind_dir_ts}, fp)

    return case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts

# 0, 1, 2, 3, 6
case_families = ["baseline_controllers", "solver_type", # 0, 1
                    "wind_preview_type", "warm_start", # 2, 3
                     "horizon_length", "cost_func_tuning",  # 4, 5
                    "yaw_offset_study", "scalability", # 6, 7
                    "breakdown_robustness", # 8
                    "gradient_type", "n_wind_preview_samples", # 9, 10
                    "generate_sample_figures", "baseline_controllers_3", # 11, 12
                    "cost_func_tuning_small"] # 13