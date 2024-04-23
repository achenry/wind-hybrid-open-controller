# from concurrent.futures import ProcessPoolExecutor, wait
from mpi4py.futures import MPIPoolExecutor
# from mpi4py.futures import wait as mpi_wait
from dask_mpi import initialize
from dask.distributed import Client
from dask.distributed import wait as dask_wait
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as cf_wait
import multiprocessing as mp

import pandas as pd
import numpy as np
import os
from glob import glob
import shutil
import yaml
from time import perf_counter
from itertools import product
import copy
import io
import re
import sys

from whoc import __file__ as whoc_file
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.wind_field.WindField import generate_multi_wind_ts, WindField
from whoc.postprocess_case_studies import plot_wind_field_ts, plot_opt_var_ts, plot_opt_cost_ts, plot_power_ts, barplot_opt_cost, compare_simulations, plot_cost_function_pareto_curve, plot_breakdown_robustness

from hercules.utilities import load_yaml

# from warnings import simplefilter
# simplefilter('error')
N_COST_FUNC_TUNINGS = 21

if sys.platform == "linux":
    STORAGE_DIR = "/projects/ssc/ahenry/whoc/floris_case_studies"
elif sys.platform == "darwin":
    STORAGE_DIR = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)


# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers": {"seed": {"group": 0, "vals": [0]},
                            #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                                "generate_lut": {"group": 0, "vals": [False]},
                                "num_turbines": {"group": 0, "vals": [9]}, 
                                "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "greedy": {"seed": {"group": 0, "vals": [0]},
            #    "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                                "case_names": {"group": 1, "vals": ["Greedy"]},
                                "controller_class": {"group": 1, "vals": ["GreedyController"]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                                "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "slsqp_solver": {"seed": {"group": 0, "vals": [0]},
                    #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                    "generate_lut": {"group": 0, "vals": [False]},
                    "num_turbines": {"group": 0, "vals": [9]}, 
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "case_names": {"group": 1, "vals": ['SLSQP']},
                           "solver": {"group": 1, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "sequential_slsqp_solver": {"seed": {"group": 0, "vals": [0]},
                                # "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                    "num_turbines": {"group": 0, "vals": [9]}, 
                          "n_horizon": {"group": 0, "vals": [5]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "case_names": {"group": 1, "vals": ["Sequential SLSQP"]},
                           "solver": {"group": 1, "vals": ["sequential_slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "serial_refine_solver": {"seed": {"group": 0, "vals": [0]},
                            #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                    "num_turbines": {"group": 0, "vals": [9]}, 
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "case_names": {"group": 1, "vals": ["Sequential Refine"]},
                           "solver": {"group": 1, "vals": ["serial_refine"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "solver_type": {"seed": {"group": 0, "vals": [0]},
                    # "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                    "num_turbines": {"group": 0, "vals": [9]}, 
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "case_names": {"group": 1, "vals": ['SLSQP', "Sequential SLSQP", "Sequential Refine"]},
                           "solver": {"group": 1, "vals": ["slsqp", "sequential_slsqp", "serial_refine"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "stochastic_preview_type": {"seed": {"group": 0, "vals": [0]},
                        #   "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "case_names": {"group": 1, "vals": ["Stochastic"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "persistent_preview_type": {"seed": {"group": 0, "vals": [0]},
                        #   "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "case_names": {"group": 1, "vals": ["Persistent"]},
                          "wind_preview_type": {"group": 1, "vals": ["persistent"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "perfect_preview_type": {"seed": {"group": 0, "vals": [0]},
                        #   "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "case_names": {"group": 1, "vals": ["Perfect"]},
                          "wind_preview_type": {"group": 1, "vals": ["perfect"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "wind_preview_type": {"seed": {"group": 0, "vals": [0]},
                        #   "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                          "n_horizon": {"group": 0, "vals": [10]}, 
                          "alpha": {"group": 0, "vals": [0.5]}, 
                          "case_names": {"group": 1, "vals": ["Perfect", "Persistent", "Stochastic"]},
                          "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent", "stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "lut_warm_start": {"seed": {"group": 0, "vals": [0]},
                    #    "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                   "generate_lut": {"group": 0, "vals": [False]},
                   "n_horizon": {"group": 0, "vals": [10]}, 
                   "alpha": {"group": 0, "vals": [0.5]}, 
                   "wind_preview_type": {"group": 0, "vals": ["stochastic"]},
                   "case_names": {"group": 0, "vals": ["LUT"]},
                   "warm_start": {"group": 0, "vals": ["lut"]},
                   "solver": {"group": 0, "vals": ["slsqp"]},
                   "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                   },
    "warm_start": {"seed": {"group": 0, "vals": [0]},
                #    "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "num_turbines": {"group": 0, "vals": [9]}, 
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                   "generate_lut": {"group": 0, "vals": [False]},
                   "n_horizon": {"group": 0, "vals": [10]}, 
                   "alpha": {"group": 0, "vals": [0.5]}, 
                   "wind_preview_type": {"group": 0, "vals": ["stochastic"]},
                   "case_names": {"group": 1, "vals": ["Greedy", "LUT", "Previous"]},
                   "warm_start": {"group": 1, "vals": ["greedy", "lut", "previous"]},
                   "solver": {"group": 0, "vals": ["slsqp"]},
                   "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                   },
    "cost_func_tuning": {"seed": {"group": 0, "vals": [0]},
                        #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                         "num_turbines": {"group": 0, "vals": [9]}, 
                         "controller_class": {"group": 0, "vals": ["MPC"]},
                         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                         "generate_lut": {"group": 0, "vals": [False]},
                         "n_horizon": {"group": 0, "vals": [10]}, 
                         "case_names": {"group": 1, "vals": [f"alpha_{f}" for f in list(np.linspace(0, 1.0, N_COST_FUNC_TUNINGS))]},
                         "alpha": {"group": 1, "vals": list(np.linspace(0, 1.0, N_COST_FUNC_TUNINGS))}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "scalability": {"seed": {"group": 0, "vals": [0]},
                    # "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                    "num_turbines": {"group": 1, "vals": [3, 9, 25, 100]},
                    "controller_class": {"group": 0, "vals": ["MPC"]},
                    "lut_path": {"group": 1, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{nturb}.csv") for nturb in [3, 9, 25, 100]]},
                    "generate_lut": {"group": 0, "vals": [False]},
                    "n_horizon": {"group": 0, "vals": [10]}, 
                    "alpha": {"group": 0, "vals": [0.5]}, 
                    "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                    "warm_start": {"group": 0, "vals": ["greedy"]}, 
                    "solver": {"group": 0, "vals": ["slsqp"]},
                    "case_names": {"group": 1, "vals": ["3 Turbines", "9 Turbines", "25 Turbines", "100 Turbines"]},
                    "floris_input_file": {"group": 1, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3, 9, 25, 100]]},
    },
    "horizon_length": {"seed": {"group": 0, "vals": [0]},
                             "num_turbines": {"group": 0, "vals": [9]},
                            #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]}, 
                       "controller_class": {"group": 0, "vals": ["MPC"]},
                       "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{9}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
                             "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [6, 8, 10, 12, 14]]},
                   "n_horizon": {"group": 1, "vals": [6, 8, 10, 12, 14]}, 
                   "alpha": {"group": 0, "vals": [0.5]}, 
                          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                          "warm_start": {"group": 0, "vals": ["greedy"]}, 
                          "solver": {"group": 0, "vals": ["slsqp"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_9.yaml")]}
                          },
    "breakdown_robustness": 
        {"seed": {"group": 0, "vals": [0]},
        #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
         "num_turbines": {"group": 0, "vals": [25]}, 
         "controller_class": {"group": 0, "vals": ["MPC"]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv")]},
                             "generate_lut": {"group": 0, "vals": [False]},
          "n_horizon": {"group": 0, "vals": [10]}, 
          "alpha": {"group": 0, "vals": [0.5]}, 
          "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
          "warm_start": {"group": 0, "vals": ["greedy"]}, 
          "solver": {"group": 0, "vals": ["slsqp"]},
          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_25.yaml")]},
          "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in [0, 0.025, 0.05, 0.5, 0.2]]},
          "offline_probability": {"group": 1, "vals": [0, 0.025, 0.05, 0.5, 0.2]}
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

def simulate_controller(controller_class, input_dict, **kwargs):
    print(f"Running instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    # Load a FLORIS object for AEP calculations
    greedy_fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                          offline_probability=input_dict["controller"]["offline_probability"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
    fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                        offline_probability=input_dict["controller"]["offline_probability"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
    
    kwargs["wind_field_config"]["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])

    ctrl = controller_class(fi, input_dict=input_dict, **kwargs)
    # TODO use coroutines or threading for hercules interfaces
    # optionally warm-start with LUT solution
    
    yaw_angles_ts = []
    yaw_angles_change_ts = []
    turbine_powers_ts = []
    turbine_wind_mag_ts = []
    turbine_wind_dir_ts = []
    turbine_offline_status_ts = []

    greedy_turbine_powers_ts = []

    convergence_time_ts = []
    # opt_codes_ts = []

    opt_cost_ts = []
    opt_cost_terms_ts = []

    n_time_steps = int(kwargs["wind_field_config"]["simulation_max_time"] / input_dict["dt"])
    greedy_fi.env.set(wind_speeds=kwargs["wind_mag_ts"][:n_time_steps], 
                    wind_directions=kwargs["wind_dir_ts"][:n_time_steps],
                    turbulence_intensities=[greedy_fi.env.core.flow_field.turbulence_intensities[0]] * n_time_steps,
                    yaw_angles=np.zeros((1, ctrl.n_turbines)))
    greedy_fi.env.run()
    greedy_turbine_powers_ts = np.max(greedy_fi.env.get_turbine_powers(), axis=1)

    n_future_steps = int(ctrl.dt // input_dict["dt"]) - 1
    
    # for k, t in enumerate(np.arange(0, kwargs["wind_field_config"]["simulation_max_time"], input_dict["dt"])):
    t = 0
    k = 0
    
    while t < kwargs["wind_field_config"]["simulation_max_time"]:

        # recompute controls and step floris forward by ctrl.dt
        # reiniitialize FLORIS interface with for current disturbances and disturbance up to (and excluding) next controls computation
        
        # consider yaw angles as most recently sent from last time-step
        fi.step(disturbances={"wind_speeds": kwargs["wind_mag_ts"][k:k + n_future_steps + 1],
                            "wind_directions": kwargs["wind_dir_ts"][k:k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=None if t > 0 else {"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines},
                            seed=k)
        
        ctrl.current_freestream_measurements = [
                kwargs["wind_mag_ts"][k] * np.sin((kwargs["wind_dir_ts"][k] - 180.) * (np.pi / 180.)),
                kwargs["wind_mag_ts"][k] * np.cos((kwargs["wind_dir_ts"][k] - 180.) * (np.pi / 180.))
        ]
        
        start_time = perf_counter()
        # get measurements from FLORIS int, then compute controls in controller class, set controls_dict, then send controls to FLORIS interface (calling calculate_wake)
        fi.time = np.arange(t, t + ctrl.dt, input_dict["dt"])
        # only step yaw angles by up to yaw_rate * input_dict["dt"] for each time-step
        ctrl.step()
        end_time = perf_counter()

        # convergence_time_ts.append((end_time - start_time) if ((t % ctrl.dt) == 0.0) else np.nan)
        convergence_time_ts += ([end_time - start_time] + [np.nan] * n_future_steps)

        # opt_codes_ts.append(ctrl.opt_code)
        if hasattr(ctrl, "opt_cost"):
            opt_cost_terms_ts += ([ctrl.opt_cost_terms] + [[np.nan] * 2] * n_future_steps)
            opt_cost_ts += ([ctrl.opt_cost] + [np.nan] * n_future_steps)
        else:
            opt_cost_terms_ts += [[np.nan] * 2] * (n_future_steps + 1)
            opt_cost_ts += [np.nan] * (n_future_steps + 1)
        
        if hasattr(ctrl, "init_sol"):
            init_states = np.array(ctrl.init_sol["states"]) * ctrl.yaw_norm_const
            init_ctrl_inputs = ctrl.init_sol["control_inputs"]
        else:
            init_states = [np.nan] * ctrl.n_turbines

            init_ctrl_inputs = [np.nan] * ctrl.n_turbines

        # assert np.all(ctrl.controls_dict['yaw_angles'] == ctrl.measurements_dict["wind_directions"] - fi.env.floris.farm.yaw_angles)
        # TODO add freestream wind mags/dirs provided to controller, yaw angles computed at this time-step, resulting turbine powers, wind mags, wind dirs
        # yaw_angles_ts += list(fi.env.floris.flow_field.wind_directions[:, np.newaxis] - fi.env.floris.farm.yaw_angles)
         # Note these are results from previous time step
        yaw_angles_ts += list(ctrl.measurements_dict["yaw_angles"])

        # Note these are results from previous time step
        turbine_powers_ts += list(ctrl.measurements_dict["turbine_powers"])

        # Note these are results from previous time step
        turbine_wind_mag_ts += list(ctrl.measurements_dict["wind_speeds"])
        # turbine_wind_mag_ts += list(fi.env.floris.flow_field.wind_speeds)

        # Note these are results from previous time step
        turbine_wind_dir_ts += list(ctrl.measurements_dict["wind_directions"])
        # turbine_wind_dir_ts += list(fi.env.floris.flow_field.wind_directions)

        # turbine_offline_status_ts.append(fi.offline_status) # TODO this should be included in measurements?
        turbine_offline_status_ts += list(fi.offline_status)

        print(f"\nTime = {t}",
            f"Measured Freestream Wind Direction = {kwargs['wind_dir_ts'][k]}",
            f"Measured Freestream Wind Magnitude = {kwargs['wind_mag_ts'][k]}",
            f"Measured Turbine Wind Directions = {ctrl.measurements_dict['wind_directions'][0, :] if ctrl.measurements_dict['wind_directions'].ndim == 2 else ctrl.measurements_dict['wind_directions']}",
            f"Measured Turbine Wind Magnitudes = {ctrl.measurements_dict['wind_speeds'][0, :] if ctrl.measurements_dict['wind_speeds'].ndim == 2 else ctrl.measurements_dict['wind_speeds']}",
            f"Measured Yaw Angles = {ctrl.measurements_dict['yaw_angles'][0, :] if ctrl.measurements_dict['yaw_angles'].ndim == 2 else ctrl.measurements_dict['yaw_angles']}",
            f"Measured Turbine Powers = {ctrl.measurements_dict['turbine_powers'][0, :] if ctrl.measurements_dict['turbine_powers'].ndim == 2 else ctrl.measurements_dict['turbine_powers']}",
            f"Distance from Initial Yaw Angle Solution = {np.linalg.norm(ctrl.controls_dict['yaw_angles'] - init_states[:ctrl.n_turbines])}",
            f"Distance from Initial Yaw Angle Change Solution = {np.linalg.norm((ctrl.controls_dict['yaw_angles'] - yaw_angles_ts[-(n_future_steps + 1)]) - init_ctrl_inputs[:ctrl.n_turbines])}",
            # f"Optimizer Output = {ctrl.opt_code['text']}",
            # f"Optimized Yaw Angle Solution = {ctrl.opt_sol['states'] * ctrl.yaw_norm_const}",
            # f"Optimized Yaw Angle Change Solution = {ctrl.opt_sol['control_inputs']}",
            f"Optimized Yaw Angles = {ctrl.controls_dict['yaw_angles']}",
            f"Optimized Yaw Angle Changes = {ctrl.controls_dict['yaw_angles'] - yaw_angles_ts[-(n_future_steps + 1)]}",
            # f"Optimized Power Cost = {opt_cost_terms_ts[-1][0]}",
            # f"Optimized Yaw Change Cost = {opt_cost_terms_ts[-1][1]}",
            f"Convergence Time = {convergence_time_ts[-(n_future_steps + 1)]}",
            sep='\n')
        
        t += ctrl.dt
        k += int(ctrl.dt / input_dict["dt"])
    else:
        last_measurements = fi.get_measurements()
        yaw_angles_ts = np.vstack([yaw_angles_ts, last_measurements["yaw_angles"]])
        yaw_angles_change_ts = np.diff(yaw_angles_ts, axis=0)
        yaw_angles_change_ts = yaw_angles_change_ts[n_future_steps:, :]
        yaw_angles_ts = yaw_angles_ts[n_future_steps + 1:, :]

        turbine_powers_ts = np.vstack([turbine_powers_ts, last_measurements["turbine_powers"]])
        turbine_powers_ts = turbine_powers_ts[n_future_steps + 1:, :]

        turbine_wind_mag_ts = np.vstack([turbine_wind_mag_ts, last_measurements["wind_speeds"]])
        turbine_wind_mag_ts = turbine_wind_mag_ts[n_future_steps + 1:, :]

        turbine_wind_dir_ts = np.vstack([turbine_wind_dir_ts, last_measurements["wind_directions"]])
        turbine_wind_dir_ts = turbine_wind_dir_ts[n_future_steps + 1:, :]
    
    # yaw_angles_change_ts = np.vstack(yaw_angles_change_ts)
    
    turbine_offline_status_ts = np.vstack(turbine_offline_status_ts)

    # greedy_turbine_powers_ts = np.vstack(greedy_turbine_powers_ts)
    # opt_cost_terms_ts = np.vstack(opt_cost_terms_ts)
    # running_opt_cost_terms_ts = np.vstack(running_opt_cost_terms_ts)

    running_opt_cost_terms_ts = np.zeros_like(opt_cost_terms_ts)
    Q = input_dict["controller"]["alpha"]
    R = (1 - input_dict["controller"]["alpha"])

    norm_turbine_powers = np.divide(turbine_powers_ts, greedy_turbine_powers_ts[:, np.newaxis],
                                    where=greedy_turbine_powers_ts[:, np.newaxis]!=0,
                                    out=np.zeros_like(turbine_powers_ts))
    norm_yaw_angle_changes = yaw_angles_change_ts / (ctrl.dt * ctrl.yaw_rate)
    
    running_opt_cost_terms_ts[:, 0] = np.sum(np.stack([-0.5 * (norm_turbine_powers[:, i])**2 * Q for i in range(ctrl.n_turbines)], axis=1), axis=1)
    running_opt_cost_terms_ts[:, 1] = np.sum(np.stack([0.5 * (norm_yaw_angle_changes[:, i])**2 * R for i in range(ctrl.n_turbines)], axis=1), axis=1)

    results_df = pd.DataFrame(data={
        "CaseName": [kwargs["case_name"]] *  int(kwargs["wind_field_config"]["simulation_max_time"] // input_dict["dt"]),
        "WindSeed": [kwargs["wind_case_idx"]] * int(kwargs["wind_field_config"]["simulation_max_time"] // input_dict["dt"]),
        "Time": np.arange(0, kwargs["wind_field_config"]["simulation_max_time"], input_dict["dt"]),
        "FreestreamWindMag": kwargs["wind_mag_ts"][:yaw_angles_ts.shape[0]],
        "FreestreamWindDir": kwargs["wind_dir_ts"][:yaw_angles_ts.shape[0]],
        **{
            f"TurbineYawAngle_{i}": yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        **{
            f"TurbineYawAngleChange_{i}": yaw_angles_change_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbinePower_{i}": turbine_powers_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindMag_{i}": turbine_wind_mag_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindDir_{i}": turbine_wind_dir_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineOfflineStatus_{i}": turbine_offline_status_ts[:, i] for i in range(ctrl.n_turbines)
        },
        "FarmYawAngleChangeAbsSum": np.sum(np.abs(yaw_angles_change_ts), axis=1),
        "RelativeFarmYawAngleChangeAbsSum": (np.sum(np.abs(yaw_angles_change_ts) * ~turbine_offline_status_ts, axis=1)) / (np.sum(~turbine_offline_status_ts, axis=1)),
        "FarmPower": np.sum(turbine_powers_ts, axis=1),
        "RelativeFarmPower": np.sum(turbine_powers_ts, axis=1) / (np.sum(~turbine_offline_status_ts * np.array([max(fi.env.core.farm.turbine_definitions[i]["power_thrust_table"]["power"]) for i in range(ctrl.n_turbines)]), axis=1)),
        # **{
        #     f"OptimizationCostTerm_{i}": opt_cost_terms_ts[:, i] for i in range(opt_cost_terms_ts.shape[1])
        # },
        # "TotalOptimizationCost": np.sum(opt_cost_terms_ts, axis=1),
        "OptimizationConvergenceTime": convergence_time_ts,
        **{
            f"RunningOptimizationCostTerm_{i}": running_opt_cost_terms_ts[:, i] for i in range(running_opt_cost_terms_ts.shape[1])
        },
        "TotalRunningOptimizationCost": np.sum(running_opt_cost_terms_ts, axis=1),
    })

    return results_df

def run_simulations(case_study_keys, regenerate_wind_field, n_seeds, run_parallel, multi):

    input_dict = load_yaml(os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml"))

    with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(os.path.dirname(whoc_file), '../examples/wind_field_data/raw_data')        
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    if DEBUG:
        input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = 300
    else:
        input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = 3600

    wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
    wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
    wind_field_config["preview_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = input_dict["dt"]
    
    # wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_preview_steps"] = int(input_dict["hercules_comms"]["helics"]["config"]["stoptime"] / input_dict["dt"]) \
        + max(case_studies["horizon_length"]["n_horizon"]["vals"]) * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_samples_per_init_seed"] = 1
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["distribution_params_path"] = os.path.join(os.path.dirname(whoc_file), "..", "examples", "wind_field_data", "wind_preview_distribution_params.pkl")  
    wind_field_config["time_series_dt"] = 1

    print("run_simulations line 546")
    # TODO check that wind field has same dt or interpolate...
    seed = 0
    if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
        wind_field_config["regenerate_distribution_params"] = True
        full_wf = WindField(**wind_field_config)
        generate_multi_wind_ts(full_wf, init_seeds=[seed + i for i in range(n_seeds)])
        wind_field_filenames = [os.path.join(wind_field_dir, f"case_{i}.csv") for i in range(n_seeds)]
        regenerate_wind_field = True
    
    print("run_simulations line 555")
    # if wind field data exists, get it
    WIND_TYPE = "stochastic"
    wind_field_fig_dir = os.path.join(os.path.dirname(whoc_file), '../examples/wind_field_data/figs') 
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(fn))
            plot_wind_field_ts(wind_field_data[-1], os.path.join(wind_field_fig_dir, fn.split(".")[0]))

            if WIND_TYPE == "step":
                # n_rows = len(wind_field_data[-1].index)
                wind_field_data[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                wind_field_data[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                wind_field_data[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                wind_field_data[-1].loc[45:, f"FreestreamWindDir"] = 270.0
    
    # true wind disturbance time-series
    wind_mag_ts = [wind_field_data[case_idx]["FreestreamWindMag"].to_numpy() for case_idx in range(n_seeds)]
    wind_dir_ts = [wind_field_data[case_idx]["FreestreamWindDir"].to_numpy() for case_idx in range(n_seeds)]

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
        results_dir = os.path.join(STORAGE_DIR, case_study_key)
        
        # if os.path.exists(results_dir):
        #     shutil.rmtree(results_dir)
        
        os.makedirs(results_dir)

        # Load default settings and make copies
        start_case_idx = len(input_dicts)
        input_dicts = input_dicts + [copy.deepcopy(input_dict) for i in range(len(case_list))]

        # make adjustements based on case study
        for c, case in enumerate(case_list):
            for property_name, property_value in case.items():
                if type(property_value) is np.str_:
                    input_dicts[start_case_idx + c]["controller"][property_name] = str(property_value)
                else:
                    input_dicts[start_case_idx + c]["controller"][property_name] = property_value

            with io.open(os.path.join(results_dir, f"input_config_{c}.yaml"), 'w', encoding='utf8') as fp:
                yaml.dump(input_dicts[start_case_idx + c], fp, default_flow_style=False, allow_unicode=True)

    # instantiate controller and run_simulations simulation
    print("run_simulations line 613")
    wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_samples_per_init_seed"] = input_dict["controller"]["n_wind_preview_samples"]
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["time_series_dt"] = int(input_dict["controller"]["dt"] // input_dict["dt"])

    print(f"about to submit calls to simulate_controller")
    if run_parallel:
        # if platform == "linux":
        
        if multi == "dask":
            futures = [client.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name=case_lists[c]["case_names"],
                                                lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
                        for c, d in enumerate(input_dicts)]
            # dask_wait(futures)
            results = client.gather(futures)
            # results = [fut.result() for fut in futures]
        # elif platform == "darwin":
        else:
            if multi == "mpi":
                executor = MPIPoolExecutor(max_workers=mp.cpu_count())
            else:
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                print(f"run_simulations line 618 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name=case_lists[c]["case_names"],
                                                lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
                        for c, d in enumerate(input_dicts)]
                # cf_wait(futures)
                results = [fut.result() for fut in futures]

        print("run_simulations line 626")

    else:
        results = []
        for c, d in enumerate(input_dicts):
            results.append(simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                               wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                               case_name=case_lists[c]["case_names"],
                                               lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"],
                                               wind_field_config=wind_field_config, verbose=False))

    # unpack rsults for each case
    results_dfs = {}
    
    for r, res in enumerate(results):
        results_dfs[case_name_lists[r]] = res

        results_dir = os.path.join(os.path.dirname(whoc_file), "case_studies", "_".join(case_name_lists[r].split('_')[:-1]))

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print(f"744 about to save df {r} of {len(results)}")
        results_dfs[case_name_lists[r]].to_csv(os.path.join(results_dir, f"time_series_results_case_{case_lists[r]['case_names']}_seed_{case_lists[r]['wind_case_idx']}.csv"))
        print(f"746 finished saving df {r} of {len(results)}")

        # fi.calculate_wake(wind_dir_ts[:-2, np.newaxis] - results_dfs[case_name_lists[f]][[f"TurbineYawAngle_{i}" for i in range(9)]].to_numpy())
        # turbine_powers = fi.get_turbine_powers()
        # assert np.all(fi.get_turbine_powers() == results_dfs[case_name_lists[f]][[f"TurbinePower_{i}" for i in range(9)]].to_numpy())

def get_results_data(results_dirs):
    results_dfs = {}
    for results_dir in results_dirs:
        for f, fn in enumerate([fn for fn in os.listdir(results_dir) if ".csv" in fn]):
            seed = int(re.findall(r"(?<=seed\_)(\d*)", fn)[0])
            case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0]

            df = pd.read_csv(os.path.join(results_dir, fn), index_col=0)

            # df["WindSeed"] = [0] * len(df.index)
            # df.to_csv(os.path.join(results_dir, f"time_series_results_case_{r}.csv"))
            case_tag = f"{os.path.split(os.path.basename(results_dir))[-1]}_{case_name}"
            if case_tag not in results_dfs:
                results_dfs[case_tag] = df
            else:
                results_dfs[case_tag] = pd.concat([results_dfs[case_tag], df])
    return results_dfs

def process_simulations(results_dirs):
    results_dfs = get_results_data(results_dirs)
    compare_results_df = compare_simulations(results_dfs)
    
    plot_breakdown_robustness(compare_results_df, case_studies, os.path.join(os.path.dirname(whoc_file), "case_studies"))
    plot_cost_function_pareto_curve(compare_results_df, case_studies, os.path.join(os.path.dirname(whoc_file), "case_studies"))

    # TODO generate results table in tex
    # solver_type_df = compare_results_df.loc[compare_results_df.index.get_level_values("CaseFamily") == "solver_type", :].reset_index("CaseName")
    # solver_type_df.loc[solver_type_df.CaseName == 'SLSQP', ("RelativeYawAngleChangeAbsMean", "mean")]

    x = compare_results_df.loc[(compare_results_df.index.get_level_values("CaseFamily") != "scalability") & (compare_results_df.index.get_level_values("CaseFamily") != "breakdown_robustness"), :]
    # x = x.loc[:, x.columns.get_level_values(1) == "mean"]
    x = x.loc[:, ("TotalRunningOptimizationCostMean", "mean")]
    x = x.groupby("CaseFamily", group_keys=False).nsmallest(3)
    # Set alpha to 0.1, n_horizon to 12, solver to SLSQP, warm-start to LUT

    get_result = lambda case_family, case_name, parameter: compare_results_df.loc[(compare_results_df.index.get_level_values("CaseFamily") == case_family) & (compare_results_df.index.get_level_values("CaseName") == case_name), (parameter, "mean")].iloc[0]
    # get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean')
    # get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean')
    # get_result('solver_type', 'SLSQP', 'TotalRunningOptimizationCostMean')
    # get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTimeMean')
    compare_results_latex = (
    f"\\begin{{tabular}}{{l|llll}}\n"
    f"\\textbf{{Case Family}} & \\textbf{{Case Name}} & \\thead{{\\textbf{{Relative Mean}} \\\\ \\textbf{{Farm Power [MW]}}}} & \\thead{{\\textbf{{Relative Mean Absolute}} \\\\ \\textbf{{Yaw Angle Change [deg]}}}} & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Convergence Time [s]}}}} \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Solver}}}} & SLSQP                                  & ${get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                          Sequential SLSQP                       & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('solver_type', 'Sequential SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                          Serial Refine                          & ${get_result('solver_type', 'Sequential Refine', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('solver_type', 'Sequential Refine', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('solver_type', 'Sequential Refine', 'OptimizationConvergenceTimeMean')):d}$  \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Wind Preview Model}}}} & Perfect                    & ${get_result('wind_preview_type', 'Perfect', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('wind_preview_type', 'Perfect', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('wind_preview_type', 'Perfect', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                      Persistent                 & ${get_result('wind_preview_type', 'Preview', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('wind_preview_type', 'Preview', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('wind_preview_type', 'Preview', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                      Stochastic                 & ${get_result('wind_preview_type', 'Stochastic', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('wind_preview_type', 'Stochastic', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('wind_preview_type', 'Stochastic', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Warm-Starting Method}}}} & Greedy                   & ${get_result('warm_start', 'Greedy', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('warm_start', 'Greedy', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('warm_start', 'Greedy', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        LUT                      & ${get_result('warm_start', 'LUT', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('warm_start', 'LUT', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('warm_start', 'LUT', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        Previous Solution        & ${get_result('warm_start', 'Previous', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('warm_start', 'Previous', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('warm_start', 'Previous', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{4}}{{*}}{{\\textbf{{Wind Farm Size}}}}       & $3 \\times 1$             & ${get_result('scalability', '3 Turbines', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('scalability', '3 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('scalability', '3 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $3 \\times 3$             & ${get_result('scalability', '9 Turbines', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('scalability', '9 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('scalability', '9 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $5 \\times 5$             & ${get_result('scalability', '25 Turbines', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('scalability', '25 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('scalability', '25 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $10 \\times 10$           & ${get_result('scalability', '100 Turbines', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('scalability', '100 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('scalability', '100 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{5}}{{*}}{{\\textbf{{Horizon Length}}}}       & $6$                      & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('horizon_length_N', 'N_p = 6', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $8$                      & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('horizon_length_N', 'N_p = 8', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $10$                     & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('horizon_length_N', 'N_p = 10', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $12$                     & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('horizon_length_N', 'N_p = 12', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $14$                     & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('horizon_length_N', 'N_p = 14', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{5}}{{*}}{{\\textbf{{Probability of Turbine Failure}}}} & $0\%$          & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                                  $1\%$          & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                                  $5\%$          & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                                  $10\%$         & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                                  $20\%$         & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeFarmPowerMean'):.3e}$ & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${int(get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \n"
    f"\end{{tabular}}"
    )
    with open(os.path.join(os.path.dirname(whoc_file), "case_studies", "comparison_time_series_results_table.tex"), "w") as fp:
        fp.write(compare_results_latex)
    

def plot_simulations(results_dirs):
    results_dfs = get_results_data(results_dirs)
    for r, results_dir in enumerate(results_dirs):
        
        for f, (input_fn, data_fn) in enumerate(zip(sorted([fn for fn in os.listdir(results_dir) if "input_config" in fn]), 
                                   sorted([fn for fn in os.listdir(results_dir) if "time_series_results" in fn]))):
            with open(os.path.join(results_dir, input_fn), 'r') as fp:
                input_config = yaml.safe_load(fp)

            df = results_dfs[f"{os.path.basename(results_dir)}_{f}"]

            # if "Time" not in df.columns:
            #     df["Time"] = np.arange(0, 3600.0 - 60.0, 60.0)

            plot_wind_field_ts(df, os.path.join(results_dir, "wind_ts.png"))

            plot_opt_var_ts(results_dfs[r], input_config["controller"]["yaw_limits"], results_dir)
            # plot_opt_var_ts(df, (-30.0, 30.0), os.path.join(results_dir, f"opt_vars_ts_{f}.png"))
            
            plot_opt_cost_ts(df, os.path.join(results_dir, f"opt_costs_ts_{f}.png"))
        
            plot_power_ts(df, os.path.join(results_dir, f"yaw_power_ts_{f}.png"))
    
    summary_df = pd.read_csv(os.path.join(os.path.dirname(whoc_file), "case_studies", f"comparison_time_series_results.csv"), index_col=0)
    barplot_opt_cost(summary_df, os.path.join(os.path.dirname(whoc_file), "case_studies"), relative=True)

if __name__ == '__main__':

    REGENERATE_WIND_FIELD = False

    case_families = ["baseline_controllers", "solver_type",
                        "wind_preview_type", "warm_start", 
                        "horizon_length", "breakdown_robustness",
                        "scalability", "cost_func_tuning"]

    DEBUG = sys.argv[1].lower() == "debug"
    if sys.argv[2].lower() == "dask":
        MULTI = "dask"
        initialize()
        client = Client()
    elif sys.argv[2].lower() == "mpi":
        MULTI = "mpi"
    else:
        MULTI = "cf"

    PARALLEL = sys.argv[3].lower() == "parallel"
    if len(sys.argv) > 4:
        CASE_FAMILY_IDX = [int(i) for i in sys.argv[4:]]
    else:
        CASE_FAMILY_IDX = list(range(len(case_families)))

    if DEBUG:
        N_SEEDS = 1
    else:
        N_SEEDS = 6

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": 2, "vals": [i for i in range(N_SEEDS)]}

    # MISHA QUESTION how to make AMR-Wind wait for control solution?
    # run_simulations(["baseline_controllers"], REGENERATE_WIND_FIELD)
    # mp.set_start_method('fork')
    os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "true"
    # run_simulations(["perfect_preview_type"], REGENERATE_WIND_FIELD)
    print([case_families[i] for i in CASE_FAMILY_IDX])
    run_simulations([case_families[i] for i in CASE_FAMILY_IDX], regenerate_wind_field=REGENERATE_WIND_FIELD, n_seeds=N_SEEDS, run_parallel=PARALLEL, multi=MULTI)
    # results_dirs = [os.path.join(os.path.dirname(whoc_file), "case_studies", case_key) 
    #                 for case_key in ["baseline_controllers", "solver_type",
    #                                  "wind_preview_type", "warm_start", 
    #                                  "horizon_length", "breakdown_robustness",
    #                                  "scalability", "cost_func_tuning"]]
    # results_dirs = [os.path.join(os.path.dirname(whoc_file), "case_studies", case_key) for case_key in ["baseline_controllers", "solver_type",
    #                                                                                                         "wind_preview_type", "warm_start", "scalability", "cost_func_tuning",
    #                                                                                                         "horizon_length", "breakdown_robustness"]]
    # compute stats over all seeds
    # process_simulations(results_dirs)
    
    # plot_simulations(results_dirs)