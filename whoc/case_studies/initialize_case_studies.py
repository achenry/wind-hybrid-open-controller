import os
import pickle
import yaml
import copy
import sys
import shutil
from glob import glob
from itertools import product
from functools import partial
from memory_profiler import profile
from wind_forecasting.preprocessing.data_module import DataModule
from whoc.wind_forecast.WindForecast import generate_wind_field_df
import gc
import re
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage
from wind_forecasting.run_scripts.tuning import generate_df_setup_params
#from line_profiler import profile
# from datetime import timedelta

import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np

from whoc import __file__ as whoc_file
from whoc.case_studies.process_case_studies import plot_wind_field_ts
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
elif sys.platform == "win32" or sys.platform == "cygwin":  # Add Windows check
    N_COST_FUNC_TUNINGS = 21

# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers_forecasters_test_flasc": {
                                    "target_turbine_indices": {"group": 1, "vals": ["6,4", "6,"]},
                                    "controller_class": {"group": 1, "vals": ["GreedyController", "LookupBasedWakeSteeringController"]},
                                    "controller_dt": {"group": 0, "vals": [60]},
                                    "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    "simulation_dt": {"group": 0, "vals": [60]},
                                    "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm.yaml"]},
                                    "uncertain": {"group": 2, "vals": [True]},
                                    "wind_forecast_class": {"group": 3, "vals": ["KalmanFilterForecast"]}, #, ", "KalmanFilterForecast", "SpatialFilterForecast"]},
                                    "study_name": {"group": 3, "vals": ["svr_aoifemac_flasc"]},
                                    "prediction_timedelta": {"group": 4, "vals": [120]}, #, 120, 180]},
                                    "yaw_limits": {"group": 0, "vals": ["-15,15"]}
                                    },
    "baseline_controllers_forecasters_test_awaken": {
                                    "controller_dt": {"group": 0, "vals": [5]},
                                    "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    "simulation_dt": {"group": 0, "vals": [1]},
                                    "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/gch_KP_v4.yaml"]},
                                    "yaw_limits": {"group": 0, "vals": ["-15,15"]},
                                    "target_turbine_indices": {"group": 1, "vals": ["74,73", "4,"]},
                                    "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                    "prediction_timedelta": {"group": 1, "vals": [300, 60]},
                                    # "target_turbine_indices": {"group": 1, "vals": ["74,73"]},
                                    # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController"]},
                                    "uncertain": {"group": 3, "vals": [True, False, False, False]},
                                    "wind_forecast_class": {"group": 3, "vals": ["KalmanFilterForecast", "KalmanFilterForecast", "PersistenceForecast", "SpatialFilterForecast"]}, # "MLForecast"
                                    # "model_key": {"group": 3, "vals": ["informer"]},
                                    # "wind_forecast_class": {"group": 3, "vals": ["MLForecast"]},
    },
    "baseline_controllers_forecasters_flasc": {"controller_dt": {"group": 0, "vals": [5]},
                                               "simulation_dt": {"group": 0, "vals": [1]},
                                               "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm.yaml"]},
                                                # "lut_path": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm_lut.csv"]},
                                               "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                                "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
                                                "yaw_limits": {"group": 0, "vals": ["-15,15"],
                                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                                "target_turbine_indices": {"group": 1, "vals": ["4,6", "4,"]},
                                                "uncertain": {"group": 2, "vals": [False, False, False, True, False,
                                                                                   True, False,
                                                                                   True, False,
                                                                                   True, False,
                                                                                   True, False]},
                                                "wind_forecast_class": {"group": 2, "vals": ["PerfectForecast", "PersistenceForecast", "SpatialFilterForecast", "KalmanFilterForecast", "SVRForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast"]},
                                                "model_key": {"group": 2, "vals": [None, None, None, None, None,
                                                                                   "informer", "informer", 
                                                                                   "autoformer", "autoformer", 
                                                                                   "spacetimeformer", "spacetimeformer", 
                                                                                   "tactis", "tactis"]},
                                                "prediction_timedelta": {"group": 3, "vals": [60, 120, 180]},
                                                }
                                    },
    "baseline_controllers_perfect_forecaster_awaken": {
        "controller_dt": {"group": 0, "vals": [5]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [1]},
        "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/gch_KP_v4.yaml"]},
        # "lut_path": {"group": 0, "vals": ["../../examples/inputs/gch_KP_v4_lut.csv"]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        "controller_class": {"group": 1, "vals": ["GreedyController", "LookupBasedWakeSteeringController"]},
        "target_turbine_indices": {"group": 1, "vals": ["4,", "74,73"]},
        "uncertain": {"group": 1, "vals": [False, False]},
        "wind_forecast_class": {"group": 0, "vals": ["PerfectForecast"]},
        # "controller_class": {"group": 1, "vals": ["GreedyController"]},
        # "target_turbine_indices": {"group": 1, "vals": ["4,"]},
        # "uncertain": {"group": 1, "vals": [False]},
        # "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast"]},
        # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController"]},
        # "target_turbine_indices": {"group": 1, "vals": ["74,73", "74,73"]},
        # "uncertain": {"group": 1, "vals": [False, True]},
        # "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast", "PerfectForecast"]},
        "prediction_timedelta": {"group": 2, "vals": [60, 0, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080]},
        },
    "baseline_controllers_perfect_forecaster_flasc": {
        "controller_dt": {"group": 0, "vals": [5]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [1]},
        "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm.yaml"]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        "controller_class": {"group": 1, "vals": ["GreedyController", "LookupBasedWakeSteeringController"]},
        "target_turbine_indices": {"group": 1, "vals": ["6,", "6,4"]},
        "uncertain": {"group": 1, "vals": [False, False]},
        "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast", "PerfectForecast"]},
        "prediction_timedelta": {"group": 2, "vals": [60, 120, 180]} #240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080]},
        },
    "baseline_controllers_ml_forecasters_awaken": {
        "controller_dt": {"group": 0, "vals": [5]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [1]},
        "floris_input_file": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4.yaml"
                                                ]},
        "lut_path": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4_lut.csv",
                                        ]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        "wind_forecast_class": {"group": 0, "vals": ["MLForecast"]},
        "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController", "GreedyController"]},
        "model_config_path": {"group": 1, "vals": [
            "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred300.yaml", 
            "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred300.yaml", 
            "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml"]},
        # "model_config_path": {"group": 1, "vals": [
        #     "/Users/ahenry/Documents/toolboxes/wind_forecasting/config/training/training_inputs_aoifemac_awaken_pred300.yaml", 
        #     "/Users/ahenry/Documents/toolboxes/wind_forecasting/config/training/training_inputs_aoifemac_awaken_pred300.yaml", 
        #     "/Users/ahenry/Documents/toolboxes/wind_forecasting/config/training/training_inputs_aoifemac_awaken_pred60.yaml"]},
        "prediction_timedelta": {"group": 1, "vals": [300, 300, 60]},
        "uncertain": {"group": 1, "vals": [True, False, False]},
        "target_turbine_indices": {"group": 1, "vals": ["74,73", "74,73", "4,"]},
        "model_key": {"group": 2, "vals": ["informer", "autoformer", "spacetimeformer", "tactis"]}
    },
    "baseline_controllers_baseline_det_forecasters_awaken": {
        "controller_dt": {"group": 0, "vals": [5]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [1]},
        "floris_input_file": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4.yaml"
                                                ]},
        "lut_path": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4_lut.csv",
                                        ]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        "uncertain": {"group": 0, "vals": [False]},
        "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
        "prediction_timedelta": {"group": 1, "vals": [300, 60]},
        "target_turbine_indices": {"group": 1, "vals": ["74,73", "4,"]},
        "wind_forecast_class": {"group": 2, "vals": ["SVRForecast", "SpatialFilterForecast", "PersistentForecast"]},
    },
    "baseline_controllers_baseline_prob_forecasters_awaken": {
        "controller_dt": {"group": 0, "vals": [5]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [1]},
        "floris_input_file": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4.yaml"
                                                ]},
        "lut_path": {"group": 0, "vals": [
            "../../examples/inputs/gch_KP_v4_lut.csv",
                                        ]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        "uncertain": {"group": 0, "vals": [False]},
        "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController", "GreedyController"]},
        "prediction_timedelta": {"group": 1, "vals": [300, 300, 60]},
        "target_turbine_indices": {"group": 1, "vals": ["74,73", "74,73", "4,"]},
        "wind_forecast_class": {"group": 0, "vals": ["KalmanFilterForecast"]}
    },
    "baseline_controllers": { "controller_dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "use_lut_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          },
    "solver_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    # "alpha": {"group": 0, "vals": [1.0]},
                    # "max_std_dev": {"group": 0, "vals": [2]},
                    #  "warm_start": {"group": 0, "vals": ["lut"]},
                    #     "controller_dt": {"group": 0, "vals": [15]},
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
                                                            "Perfect", "Persistent",
                                                            "Stochastic Interval Elliptical 3", "Stochastic Interval Elliptical 5", "Stochastic Interval Elliptical 11", 
                                                            "Stochastic Interval Rectangular 3", "Stochastic Interval Rectangular 5", "Stochastic Interval Rectangular 11",
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100"
                                                            ]},
                         "n_wind_preview_samples": {"group": 1, "vals": [1, 1] + [3, 5, 11] * 2 + [25, 50, 100]},
                         "decay_type": {"group": 1, "vals": [None] * 2 + ["exp"] * 3 + ["none"] * 3 + ["cosine"] * 3},
                         "max_std_dev": {"group": 1, "vals": [None] * 2 + [2] * 3 + [2] * 3 + [1] * 3},  
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent"] + ["stochastic_interval_elliptical"] * 3 + ["stochastic_interval_rectangular"] * 3 + ["stochastic_sample"] * 3}
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
                    #    "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [6, 12, 24, 36]]},
                        "controller_dt": {"group": 1, "vals": [15, 30, 45, 60]},
                       "n_horizon": {"group": 2, "vals": [6, 12, 18, 24]}
                    },
    "breakdown_robustness":  # case_families[5]
        {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
         "controller_dt": {"group": 1, "vals": [15, 5, 5]},
         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{25}.yaml")]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{25}.csv")]},
        #   "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))]},
          "offline_probability": {"group": 2, "vals": list(np.linspace(0, 0.1, N_COST_FUNC_TUNINGS))}
        },
    "scalability": {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
                    "controller_dt": {"group": 1, "vals": [15, 5, 5]},
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
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
    },
    "yaw_offset_study": {"controller_class": {"group": 1, "vals": ["MPC", "MPC", "MPC", "LookupBasedWakeSteeringController", "MPC", "MPC"]},
                          "case_names": {"group": 1, "vals":[f"StochasticIntervalRectangular_1_3turb", f"StochasticIntervalRectangular_11_3turb", f"StochasticIntervalElliptical_11_3turb", 
                                                             f"LUT_3turb", f"StochasticSample_25_3turb", f"StochasticSample_100_3turb"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular"] * 2 + ["stochastic_interval_elliptical"] + ["none"] + ["stochastic_sample"] * 2},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 11, 11, 1, 25, 100]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]}
    },
    "baseline_plus_controllers": {"controller_dt": {"group": 1, "vals": [5, 5, 60.0, 60.0, 60.0, 60.0]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy", "MPC_with_Filter", "MPC_without_Filter", "MPC_without_state_cons", "MPC_without_dyn_state_cons"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController", "MPC", "MPC", "MPC", "MPC"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True, True, False, False, False]},
    },
    "baseline_controllers_3": { "controller_dt": {"group": 1, "vals": [5, 5]},
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
                             "n_horizon": {"group": 0, "vals": [24]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular", "stochastic_interval_elliptical", "stochastic_sample"]},
                             "n_wind_preview_samples": {"group": 1, "vals": [5, 8, 500]},
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
    },
    "sr_solve": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                         "case_names": {"group": 0, "vals": ["Serial Refine"]},
                        "solver": {"group": 0, "vals": ["serial_refine"]}
    },
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

# @profile
def initialize_simulations(case_study_keys, regenerate_lut, regenerate_wind_field, 
                           n_seeds, stoptime, save_dir, wf_source, multiprocessor,
                           whoc_config, base_model_config=None):
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

    os.makedirs(save_dir, exist_ok=True)
    
    simulation_dt = set(np.concatenate([case_studies[k]["simulation_dt"]["vals"] for k in case_study_keys]))
    assert len(simulation_dt) == 1, "There may only be a single value of 'simulation_dt'."
    simulation_dt = list(simulation_dt)[0]
    # simulation_timedelta = pd.Timedelta(seconds=list(simulation_dt)[0])
    
    if stoptime != "auto": 
        whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = int(stoptime)
    
    if "slsqp_solver_sweep" not in case_studies or "controller_dt" not in case_studies["slsqp_solver_sweep"]:
        max_controller_dt = whoc_config["controller"]["controller_dt"]
    else:
        max_controller_dt = max(case_studies["slsqp_solver_sweep"]["controller_dt"]["vals"])
    
    if "horizon_length" not in case_studies or "n_horizon" not in case_studies["horizon_length"]:
        max_n_horizon = whoc_config["controller"]["n_horizon"]
    else:
        max_n_horizon = max(case_studies["horizon_length"]["n_horizon"]["vals"])
    
    if (whoc_config["controller"]["target_turbine_indices"] is not None) and ((num_target_turbines := len(whoc_config["controller"]["target_turbine_indices"])) < whoc_config["controller"]["num_turbines"]):
        # need to change num_turbines, floris_input_file, lut_path
        whoc_config["controller"]["num_turbines"] = num_target_turbines
        lut_path = os.path.abspath(whoc_config["controller"]["lut_path"])
        floris_input_file = os.path.splitext(os.path.basename(whoc_config["controller"]["floris_input_file"]))[0]
        yaw_limits = (whoc_config["controller"]["yaw_limits"])[1]
        target_turbine_indices = tuple(int(i) for i in whoc_config["controller"]["target_turbine_indices"])
        uncertain_flag = whoc_config["controller"]["uncertain"]
        whoc_config["controller"]["lut_path"] = os.path.join(os.path.dirname(lut_path), 
                                                            f"lut_{floris_input_file}_{target_turbine_indices}_uncertain{uncertain_flag}_yawlimits{yaw_limits}.csv")
        whoc_config["controller"]["target_turbine_indices"] = tuple(whoc_config["controller"]["target_turbine_indices"])
        
    if whoc_config["controller"]["target_turbine_indices"] is None:
         whoc_config["controller"]["target_turbine_indices"] = "all"
         
    if n_seeds != "auto":
        n_seeds = int(n_seeds)

    if wf_source == "floris":
        from whoc.wind_field.WindField import plot_ts
        from whoc.wind_field.WindField import generate_multi_wind_ts, WindField, write_abl_velocity_timetable, first_ord_filter
    
        with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
            wind_field_config = yaml.safe_load(fp)

        # instantiate wind field if files don't already exist
        wind_field_dir = os.path.join(save_dir, 'wind_field_data/raw_data')
        wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
        os.makedirs(wind_field_dir, exist_ok=True)

        # wind_field_config["simulation_max_time"] = whoc_config["hercules_comms"]["helics"]["config"]["stoptime"]
        wind_field_config["num_turbines"] = whoc_config["controller"]["num_turbines"]
        wind_field_config["preview_dt"] = int(max_controller_dt / whoc_config["simulation_dt"])
        wind_field_config["simulation_sampling_time"] = whoc_config["simulation_dt"]
        
        # wind_field_config["n_preview_steps"] = whoc_config["controller"]["n_horizon"] * int(whoc_config["controller"]["controller_dt"] / whoc_config["simulation_dt"])
        wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / whoc_config["simulation_dt"]) \
            + max_n_horizon * int(max_controller_dt/ whoc_config["simulation_dt"])
        wind_field_config["n_samples_per_init_seed"] = 1
        wind_field_config["regenerate_distribution_params"] = False
        wind_field_config["distribution_params_path"] = os.path.join(save_dir, "wind_field_data", "wind_preview_distribution_params.pkl")  
        wind_field_config["time_series_dt"] = 1
        
        # TODO check that wind field has same dt or interpolate...
        seed = 0
        if n_seeds == "auto":
            n_seeds = len(wind_field_filenames)
        if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
            logging.info("regenerating wind fields")
            wind_field_config["regenerate_distribution_params"] = True # set to True to regenerate from constructed mean and covaraicne
            full_wf = WindField(**wind_field_config)
            os.makedirs(wind_field_dir, exist_ok=True)
            wind_field_data = generate_multi_wind_ts(full_wf, wind_field_dir, init_seeds=[seed + i for i in range(n_seeds)])
            write_abl_velocity_timetable([wfd.df for wfd in wind_field_data], wind_field_dir) # then use these timetables in amr precursor
            # write_abl_velocity_timetable(wind_field_data, wind_field_dir) # then use these timetables in amr precursor
            wind_dir_lpf_alpha = np.exp(-(1 / whoc_config["controller"]["wind_dir_lpf_time_const"]) * whoc_config["simulation_dt"])
            # wind_mag_lpf_alpha = np.exp(-(1 / whoc_config["controller"]["wind_mag_lpf_time_const"]) * whoc_config["simulation_dt"])
            plot_wind_field_ts(wind_field_data[0].df, wind_field_dir, filter_func=partial(first_ord_filter, alpha=wind_dir_lpf_alpha))
            plot_ts(pd.concat([wfd.df for wfd in wind_field_data]), wind_field_dir)
            wind_field_filenames = [os.path.join(wind_field_dir, f"case_{i}.csv") for i in range(n_seeds)]
            regenerate_wind_field = True
        
        wind_field_config["regenerate_distribution_params"] = False
        
        # if wind field data exists, get it
        WIND_TYPE = "stochastic"
        wind_field_ts = []
        if os.path.exists(wind_field_dir):
            for f, fn in enumerate(wind_field_filenames):
                wind_field_ts.append(pl.read_csv(fn, try_parse_dates=True)) #index_col=0, parse_dates=["time"]))
                
                # wind_field_data[f]["time"] = pd.to_timedelta(wind_field_data[-1]["time"], unit="s") + pd.to_datetime("2025-01-01")
                # wind_field_data[f].to_csv(fn)
                
                # if WIND_TYPE == "step":
                #     # n_rows = len(wind_field_data[-1].index)
                #     wind_field_ts[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                #     wind_field_ts[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                #     wind_field_ts[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                #     wind_field_ts[-1].loc[45:, f"FreestreamWindDir"] = 270.0
        
        # write_abl_velocity_timetable(wind_field_data, wind_field_dir)
        
        # true wind disturbance time-series
        #plot_wind_field_ts(pd.concat(wind_field_data), os.path.join(wind_field_fig_dir, "seeds.png"))
        # wind_mag_ts = [wind_field_data[case_idx]["FreestreamWindMag"].to_numpy() for case_idx in range(n_seeds)]
        # wind_dir_ts = [wind_field_data[case_idx]["FreestreamWindDir"].to_numpy() for case_idx in range(n_seeds)]
        assert np.all([np.isclose(wind_field_ts[case_idx].select(pl.col("time").diff().slice(1,1).dt.total_seconds()).item(), whoc_config["simulation_dt"]) for case_idx in range(n_seeds)]), "sampling time of wind field should be equal to simulation sampling time"
        
        wind_field_ts = [wind_field_ts[case_idx].select(["time", "FreestreamWindMag", "FreestreamWindDir"]) for case_idx in range(n_seeds)] 
        
        if stoptime == "auto": 
            durations = [(df.select((pl.col("time").last() - pl.col("time").first()).dt.total_seconds())) for df in wind_field_ts]
            # whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = min([d.total_seconds() if hasattr(d, 'total_seconds') else d for d in durations])
            whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = [d.total_seconds() if hasattr(d, 'total_seconds') else d for d in durations]
        else:
            stoptime = [stoptime] * len(wind_field_ts)

    elif wf_source == "scada":
        # NOTE: we use the model config with the highest prediction length to instantiate the DataModule
        data_module = DataModule(data_path=base_model_config["dataset"]["data_path"], 
                                 normalization_consts_path=base_model_config["dataset"]["normalization_consts_path"],
                                 normalized=False, 
                                 n_splits=1, #model_config["dataset"]["n_splits"],
                                 continuity_groups=None, train_split=(1.0 - base_model_config["dataset"]["val_split"] - base_model_config["dataset"]["test_split"]),
                                 val_split=base_model_config["dataset"]["val_split"], test_split=base_model_config["dataset"]["test_split"],
                                 prediction_length=base_model_config["dataset"]["prediction_length"], context_length=base_model_config["dataset"]["context_length"],
                                 target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                 freq=f"{simulation_dt}s", target_suffixes=base_model_config["dataset"]["target_turbine_ids"],
                                 per_turbine_target=False, as_lazyframe=False, dtype=pl.Float32)
    
        if not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
        
        # pull ws_horz, ws_vert, nacelle_direction, normalization_consts from awaken data and run for ML, SVR
        data_module.generate_splits(splits=["test"], save=True, reload=reload) # TODO should reload if context/prediction length has changed
        wind_field_ts = generate_wind_field_df(data_module.test_dataset, data_module.target_cols, data_module.feat_dynamic_real_cols)
        delattr(data_module, "test_dataset")
        
        wind_field_ts = wind_field_ts.partition_by("continuity_group")
        
        wind_field_ts = sorted(wind_field_ts, reverse=True, key=lambda df: df.select(pl.col("time").last() - pl.col("time").first()).item())
        if n_seeds != "auto":
            wind_field_ts = wind_field_ts[:n_seeds]
            # wind_field_ts = wind_field_ts[143:144]
            # n_seeds = 1
            
        else:
            n_seeds = len(wind_field_ts)
        
        wind_dt = wind_field_ts[0].select(pl.col("time").diff().slice(1,1).dt.total_seconds()).item()
        logging.info(f"Loaded and normalized SCADA wind field from {base_model_config['dataset']['data_path']} with dt = {wind_dt} seconds.")
        
        # make sure wind_dt == simulation_dt
        
        if simulation_dt != wind_dt:
            logging.info(f"Resampling to {simulation_dt} seconds.")
            wind_field_ts = [wf.set_index("time").resample(f"{simulation_dt}s").mean().reset_index(names=["time"]) for wf in wind_field_ts]
            
            if wind_dt < simulation_dt:
                wind_field_ts = [wf.with_columns(
                    time=pl.col("time").dt.round(simulation_dt))\
                    .group_by("time", maintain_order=True).agg(cs.numeric().mean()) for wf in wind_field_ts]
            else:
                wind_field_ts = [wf.upsample(time_column="time", every=f"{simulation_dt}s").interpolate() for wf in wind_field_ts]
        
        
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # import polars.selectors as cs
        # target_turbine_ids = [75, 74, 5]
        # # for seed, wf in enumerate(wind_field_ts):
        # for seed, time in check_seed_times:
        #     wf = wind_field_ts[seed].copy()
        #     timestamp = wf.loc[(wf["time"] - wf["time"].iloc[0]).dt.total_seconds() == time, "time"].iloc[0]
        #     wf = wf.loc[(wf["time"] - wf["time"].iloc[0]).dt.total_seconds().between(time - 120, time + 120), :]
        #     df = wf[["time"] + [c for c in wf.columns if c.startswith("ws_") and any(c.endswith("_" + str(tid)) for tid in target_turbine_ids)]]
        #     df = pd.melt(df, id_vars=["time"], value_vars=[c for c in df.columns if c.startswith("ws_")])
        #     df["turbine_id"] = df["variable"].str.extract("_(\\d+)", expand=False)
        #     df["variable"] = df["variable"].str.extract("(\\w+)(?=_\\d+)", expand=False)
        #     fig, ax = plt.subplots(2, 1, sharex=True)
        #     sns.lineplot(data=df.loc[df["variable"]=="ws_horz", :], x="time", y="value", hue="turbine_id", ax=ax[0])
        #     sns.lineplot(data=df.loc[df["variable"]=="ws_vert", :], x="time", y="value", hue="turbine_id", ax=ax[1])
        #     ax[0].axvline(x=timestamp)
        #     ax[1].axvline(x=timestamp)
        #     ax[0].set_title("Horizontal Wind Speed (m/s)")
        #     ax[1].set_title("Vertical Wind Speed (m/s)")
        #     ax[0].set_xlabel("")
        #     ax[1].set_xlabel("Time (s)")
                
        wind_field_config = {}

        # FOR TESTING
        if False:
            import csv
            from itertools import islice
            check_seed_times = []
            with open("/Users/ahenry/Documents/toolboxes/wind_forecasting/wind_forecasting/run_scripts/perfect_testing.txt") as fp:
                csv_reader = csv.reader(fp)
                for row in islice(csv_reader, 2, None):
                    if row:
                        seed = re.search("(?<=seed_)(\\d+)", row[0])
                        time = re.search("(?<=time\\s)(\\d+)", row[1])
                        if seed and time:
                            seed = int(seed.group(0))
                            time = float(time.group(0))
                            check_seed_times.append((seed, time))
                wind_field_ts = [wind_field_ts[seed].loc[(wind_field_ts[seed]["time"] - wind_field_ts[seed]["time"].iloc[0]).dt.total_seconds().between(0, time + 120), :] for seed, time in check_seed_times]
                n_seeds = len(wind_field_ts)
        
        if stoptime == "auto":
            durations = [df.select(pl.col("time").last() - pl.col("time").first()).item() for df in wind_field_ts]
            if any(int(d / pd.Timedelta(data_module.freq)) < data_module.context_length + data_module.prediction_length for d in durations):
                logging.warning(f"One or more continuity groups are too short, delete train ready path {data_module.train_ready_path} to reload.")
            # whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = min([d.total_seconds() for d in durations])
            whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = [d.total_seconds() for d in durations]
        else:
            stoptime = [stoptime] * len(wind_field_ts)

        # TESTING START
        # wind_field_ts = [df.filter((pl.col("time") - pl.col("time").first()).dt.total_seconds() < int(stoptime[d] * 0.15)) for d, df in enumerate(wind_field_ts)]
        # durations = [df.select(pl.col("time").last() - pl.col("time").first()).item() for df in wind_field_ts]
        # whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = [d.total_seconds() for d in durations]
        # TESTING END
        
        del data_module
        gc.collect()
    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": max(d["group"] for d in case_studies[case_family].values()) + 1, "vals": [i for i in range(n_seeds)]}

    model_configs = {}
    input_dicts = []
    case_lists = []
    case_name_lists = []
    n_cases_list = []
    lut_cases = set()
    input_filenames = []
    for case_study_key in case_study_keys:
        input_df = []
        case_list, case_names = CaseGen_General(case_studies[case_study_key], namebase=case_study_key)
        case_lists = case_lists + case_list
        case_name_lists = case_name_lists + case_names
        n_cases_list.append(len(case_list))
        
        # Load default settings and make copies
        start_case_idx = len(input_dicts)
        input_dicts = input_dicts + [copy.deepcopy(whoc_config) for i in range(len(case_list))]

        # make adjustements based on case study
        for c, case in enumerate(case_list):
            logging.info(f"Processing case: {start_case_idx + c}")
            for property_name, property_value in case.items():
                if property_name in input_dicts[start_case_idx + c]["controller"]:
                    property_group = "controller"
                elif ((property_name in input_dicts[start_case_idx + c]["wind_forecast"]) 
                      or (input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"] and property_name in input_dicts[start_case_idx + c]["wind_forecast"].get(input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"], {}))):
                    property_group = "wind_forecast"
                else:
                    property_group = None
                
                if property_group:
                    if property_name == "yaw_limits":
                        input_dicts[start_case_idx + c][property_group][property_name] = tuple(int(v) for v in str(property_value).split(","))
                    elif property_name == "target_turbine_indices":
                        if property_value != "all":
                            # need to preserve order, taking first as upstream
                            target_turbine_indices = np.array([int(v) for v in property_value.split(",") if len(v)])
                            _, order_idx = np.unique(target_turbine_indices, return_index=True)
                            target_turbine_indices = target_turbine_indices[np.sort(order_idx)]
                            input_dicts[start_case_idx + c]["controller"][property_name] = tuple(target_turbine_indices)
                        else:
                            input_dicts[start_case_idx + c]["controller"][property_name] = "all"
                            
                    elif property_name == "uncertain":
                        if (case.setdefault("controller_class", whoc_config["controller"]["controller_class"])) == "GreedyController":
                            # logging.info("GreedyController cannot be run for uncertain flag. Setting uncertain to False.")
                            input_dicts[start_case_idx + c]["controller"]["uncertain"] = False
                        else:
                            input_dicts[start_case_idx + c]["controller"]["uncertain"] = property_value
                    elif isinstance(property_value, np.str_):
                        input_dicts[start_case_idx + c][property_group][property_name] = str(property_value)
                    else:
                        input_dicts[start_case_idx + c][property_group][property_name] = property_value
                else:
                    input_dicts[start_case_idx + c][property_name] = property_value
            
            assert all(input_dicts[start_case_idx + c]["controller"]["controller_dt"] <= t for t in stoptime)
            
            if input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"] or "wind_forecast_class" in case:
                model_config_path = input_dicts[start_case_idx + c]["wind_forecast"]["model_config_path"]
                if model_config_path not in model_configs:
                    with open(model_config_path, 'r') as file:
                        model_configs[model_config_path]  = yaml.safe_load(file)
                        
                if (input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"] == "MLForecast") \
                    and (input_dicts[start_case_idx + c]["wind_forecast"]["prediction_timedelta"] <=  model_configs[model_config_path]["dataset"]["prediction_length"]):
                    logging.warning(f"Provided prediction_timedelta should be less or equal to model config prediction length { model_configs[model_config_path]['dataset']['prediction_length']}. Make sure you are providing the right model config file.")
                input_dicts[start_case_idx + c]["wind_forecast"] \
                    = {**{
                        "measurements_timedelta": wind_field_ts[0].select(pl.col("time").diff().slice(1,1)).item(),
                        "context_timedelta": pd.Timedelta(seconds= model_configs[model_config_path]["dataset"]["context_length"]), # pd.Timedelta(seconds=input_dicts[start_case_idx + c]["wind_forecast"]["context_timedelta"]),
                        "prediction_timedelta": pd.Timedelta(seconds=input_dicts[start_case_idx + c]["wind_forecast"]["prediction_timedelta"]),
                        "controller_timedelta": pd.Timedelta(seconds=input_dicts[start_case_idx + c]["controller"]["controller_dt"])
                        }, 
                    **input_dicts[start_case_idx + c]["wind_forecast"].setdefault(input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"], {}),
                    }
                
                if "model_key" in input_dicts[start_case_idx + c]["wind_forecast"]:
                    db_setup_params = generate_df_setup_params(
                        model=input_dicts[start_case_idx + c]["wind_forecast"]["model_key"], 
                        model_config=model_configs[model_config_path])
                    optuna_storage = setup_optuna_storage(
                        db_setup_params=db_setup_params,
                        restart_tuning=False,
                        rank=0
                    )
                    input_dicts[start_case_idx + c]["wind_forecast"]["optuna_storage"] = optuna_storage
                
            # need to change num_turbines, floris_input_file, lut_path
            if (target_turbine_indices := input_dicts[start_case_idx + c]["controller"]["target_turbine_indices"])  != "all":
                target_turbine_indices = tuple(int(i) for i in target_turbine_indices)
                num_target_turbines = len(target_turbine_indices)
                input_dicts[start_case_idx + c]["controller"]["num_turbines"] = num_target_turbines
                # NOTE: lut tables should be regenerated for different yaw limits
                uncertain_flag = input_dicts[start_case_idx + c]["controller"]["uncertain"]
                lut_path = os.path.abspath(input_dicts[start_case_idx + c]["controller"]["lut_path"])
                floris_input_file = os.path.splitext(os.path.basename(input_dicts[start_case_idx + c]["controller"]["floris_input_file"]))[0]
                yaw_limits = (input_dicts[start_case_idx + c]["controller"]["yaw_limits"])[1]
                input_dicts[start_case_idx + c]["controller"]["lut_path"] = os.path.join(
                    os.path.dirname(lut_path), 
                    f"lut_{floris_input_file}_{target_turbine_indices}_uncertain{uncertain_flag}_yawlimits{yaw_limits}.csv")
            # **{k: v for k, v in input_dicts[start_case_idx + c]["wind_forecast"].items() if isinstance(k, str) and "_kwargs" in k} 
            assert input_dicts[start_case_idx + c]["controller"]["controller_dt"] >= input_dicts[start_case_idx + c]["simulation_dt"], "controller_dt must be greater than or equal to simulation_dt"
             
            # regenerate floris lookup tables for all wind farms included
            # generate LUT for combinations of lut_path/floris_input_file, yaw_limits, uncertain, and target_turbine_indices that arise together
            if regenerate_lut or not os.path.exists(input_dicts[start_case_idx + c]["controller"]["lut_path"]):
                
                floris_input_file = input_dicts[start_case_idx + c]["controller"]["floris_input_file"]
                lut_path = input_dicts[start_case_idx + c]["controller"]["lut_path"] 
                uncertain_flag = input_dicts[start_case_idx + c]["controller"]["uncertain"] 
                yaw_limits = tuple(input_dicts[start_case_idx + c]["controller"]["yaw_limits"])
                target_turbine_indices = input_dicts[start_case_idx + c]["controller"]["target_turbine_indices"]
                if (new_case := tuple([floris_input_file, lut_path, uncertain_flag, yaw_limits, target_turbine_indices])) in lut_cases:
                    continue
                
                logging.info(f"Regenerating LUT {lut_path}")
                LookupBasedWakeSteeringController._optimize_lookup_table(
                    floris_config_path=floris_input_file, uncertain=uncertain_flag, yaw_limits=yaw_limits, 
                    parallel=multiprocessor is not None,
                    sorted_target_tids=sorted(target_turbine_indices) if target_turbine_indices != "all" else "all", lut_path=lut_path, generate_lut=True)
                
                lut_cases.add(new_case)

                input_dicts[start_case_idx + c]["controller"]["generate_lut"] = False
            
            # rename this by index with only config updates from case inside, add dataframe csv linking case indices to names/params
            if case_lists[start_case_idx + c]["wind_case_idx"] == 0:
                # only generate input_df row for one wind seed
                input_df.append(pd.DataFrame(data={k: [v] for k, v in case.items() if k != "wind_case_idx"}))
            
            fn = f"input_config_case_{len(input_df) - 1}.pkl"
            input_filenames.append((case_study_key, case_lists[start_case_idx + c]["wind_case_idx"], fn))
            # fn = f'input_config_case_{"_".join(
            #     [f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case.items() \
            #         if key not in ["simulation_dt", "use_filtered_wind_dir", "use_lut_filtered_wind_dir", "yaw_limits", "wind_case_idx", "seed", "floris_input_file", "lut_path"]]) \
            #         if "case_names" not in case else case["case_names"]}.pkl'.replace("/", "_")

        input_df = pd.concat(input_df, ignore_index=True, axis=0)
        os.makedirs(os.path.join(save_dir, case_study_key), exist_ok=True)
        input_df.to_csv(os.path.join(save_dir, case_study_key, "case_descriptions.csv"), index=False)
        
    # TEMP change the filenames of old simulations to new
    if False:
        for case_study_key in case_study_keys:
            results_dir = os.path.join(save_dir, case_study_key)
            inp_info = pd.read_csv(os.path.join(results_dir, "case_descriptions.csv"))
            for inp_file in glob(os.path.join(results_dir, "input_config_case_*.pkl")):
                fn = os.path.basename(inp_file)
                # get info from filename, find which index in inp_info it corresponds to, and rename it
                try:
                    ctrl_cls = re.search("(?<=controller_class_)(\\w+)(?=_controller_dt)", fn).group()
                    ctrl_dt = int(re.search("(?<=controller_dt_)(\\d+)(?=_prediction_timedelta)", fn).group())
                    prediction_timedelta = int(re.search("(?<=prediction_timedelta_)(\\d+)(?=_target_turbine_indices)", fn).group())
                    tgt_turb_ind = re.search("(?<=target_turbine_indices_)(.*)(?=_uncertain)", fn).group()
                    unc_flag = True if re.search("(?<=uncertain_)(.*)(?=_wind_forecast_class)", fn).group() == "True" else False
                    wind_fct_cls = re.search("(?<=wind_forecast_class_)(\\w+)(?=.pkl)", fn).group()
                    new_case_name = inp_info.loc[(inp_info["controller_class"] == ctrl_cls) & (inp_info["controller_dt"] == ctrl_dt) & (inp_info["prediction_timedelta"] == prediction_timedelta) & (inp_info["target_turbine_indices"] == tgt_turb_ind) & (inp_info["uncertain"] == unc_flag) & (inp_info["wind_forecast_class"] == wind_fct_cls), :].index[0]
                    shutil.move(inp_file, 
                                os.path.join(results_dir, 
                                             re.sub("(?<=input_config_case_)(.*)(?=\\.pkl)", 
                                                    str(new_case_name), os.path.basename(inp_file))))
                except AttributeError:
                    continue
            
            for ts_file in glob(os.path.join(results_dir, "*.csv")):
                if os.path.basename(ts_file) == "case_descriptions.csv":
                    continue
                try:
                    fn = os.path.basename(ts_file)
                    ctrl_cls = re.search("(?<=controller_class_)(\\w+)(?=_controller_dt)", fn).group()
                    ctrl_dt = int(re.search("(?<=controller_dt_)(\\d+)(?=_prediction_timedelta)", fn).group())
                    prediction_timedelta = int(re.search("(?<=prediction_timedelta_)(\\d+)(?=_target_turbine_indices)", fn).group())
                    tgt_turb_ind = re.search("(?<=target_turbine_indices_)(.*)(?=_uncertain)", fn).group()
                    unc_flag = True if re.search("(?<=uncertain_)(.*)(?=_wind_forecast_class)", fn).group() == "True" else False
                    wind_fct_cls = re.search("(?<=wind_forecast_class_)(\\w+)(?=_seed_\\d+.csv)", fn).group()
                    new_case_name = inp_info.loc[(inp_info["controller_class"] == ctrl_cls) & (inp_info["controller_dt"] == ctrl_dt) & (inp_info["prediction_timedelta"] == prediction_timedelta) & (inp_info["target_turbine_indices"] == tgt_turb_ind) & (inp_info["uncertain"] == unc_flag) & (inp_info["wind_forecast_class"] == wind_fct_cls), :].index[0]
                    shutil.move(ts_file, 
                                os.path.join(results_dir, 
                                             re.sub("(?<=time_series_results_case_)(.*)(?=_seed_\\d+\\.csv)", 
                                                    str(new_case_name), os.path.basename(ts_file))))
                except AttributeError:
                    continue
        
    # delete any input files/time series files that don't belong
    # pattern = "(?<=input_config_case_)(.*)(?=\\.pkl)"
    pattern = "(?<=input_config_case_)(.*)(?=\\.pkl)"
    # ts_filenames = [tuple([csk, wind_case_idx, f"time_series_results_case_{re.search(pattern, fn).group()}_seed_{wind_case_idx}.csv".replace("/", "_")]) for csk, wind_case_idx, fn in input_filenames]
    ts_filenames = [tuple([csk, f"time_series_results_case_{re.search(pattern, fn).group()}_seed_{wind_case_idx}.csv"]) for csk, wind_case_idx, fn in input_filenames]
    for case_study_key in case_study_keys:
        allowed_input_files = set([fn for csk, _, fn in input_filenames if csk == case_study_key])
        allowed_ts_files = set([fn for csk, fn in ts_filenames if csk == case_study_key])
        # allowed_ts_files = set([
        #     f"time_series_results_case_{re.search('(?<=input_config_case_)(.*)(?=\\.pkl)', fn).group()}_seed_{wind_case_idx}.csv".replace("/", "_") 
        #     for csk, wind_case_idx, fn in input_filenames if csk == case_study_key])
        
        results_dir = os.path.join(save_dir, case_study_key)
        for inp_file in glob(os.path.join(results_dir, "input_config_case_*.pkl")):
            if os.path.basename(inp_file) not in allowed_input_files:
                os.remove(inp_file)
        for ts_file in glob(os.path.join(results_dir, "time_series_results_case_*.csv")):
            if os.path.basename(ts_file) not in allowed_ts_files:
                os.remove(ts_file)
        
    # prediction_timedelta = max(inp["wind_forecast"]["prediction_timedelta"] for inp in input_dicts if inp["controller"]["wind_forecast_class"]) \
    #         if any(inp["controller"]["wind_forecast_class"] for inp in input_dicts) else pd.Timedelta(seconds=0)
    # horizon_timedelta = max(pd.Timedelta(seconds=inp["controller"]["n_horizon"] * inp["controller"]["controller_dt"]) for inp in input_dicts if inp["controller"]["n_horizon"]) \
    #         if any(inp["controller"]["controller_class"] == "MPC" for inp in input_dicts) else pd.Timedelta(seconds=0)
    # stoptime -= prediction_timedelta.total_seconds()
    # assert stoptime > 0, "increase stoptime parameter and/or decresease prediction_timedetla, as stoptime < prediction_timedelta"

    # assert all([(df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() >= stoptime + prediction_timedelta + horizon_timedelta for df in wind_field_ts])
    # wind_field_ts = [df.filter((pl.col("time") - pl.col("time").first()).dt.total_seconds() 
    #                     <= stoptime[d] + prediction_timedelta.total_seconds() + horizon_timedelta.total_seconds())
    #                 for d, df in enumerate(wind_field_ts)]
    # stoptime = max(min([((df["time"].iloc[-1] - df["time"].iloc[0]) - prediction_timedelta - horizon_timedelta).total_seconds() for df in wind_field_ts]), stoptime)
    # stoptime = [min((df.select(pl.col("time").last() - pl.col("time").first()).item() - prediction_timedelta - horizon_timedelta).total_seconds(), stoptime[d]) for d, df in enumerate(wind_field_ts)]
    
    total_cases = int(len(input_filenames) / n_seeds)
    written_input_files = set()
    
    for f, ((case_study_key, wind_case_idx, fn), inp) in enumerate(zip(input_filenames, input_dicts)):
        
        inp["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime[wind_case_idx]
        if fn not in written_input_files:
            logging.info(f"Writing input_config file {len(written_input_files)} of {total_cases}")
            results_dir = os.path.join(save_dir, case_study_key)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, fn), 'wb') as fp:
                pickle.dump(inp, fp) # TODO this adds different stop times for each file
            written_input_files.add(fn)
    
    # instantiate controller and run_simulations simulation
    # with open(os.path.join(save_dir, "init_simulations.pkl"), "wb") as fp:
    #     pickle.dump({"case_lists": case_lists, "case_name_lists": case_name_lists, "input_dicts": input_dicts, "wind_field_config": wind_field_config}, fp)

    return case_lists, case_name_lists, input_dicts, wind_field_config, wind_field_ts

# 0, 1, 2, 3, 6
case_families = [
    "baseline_controllers", "solver_type", # 0, 1
     "wind_preview_type", "warm_start", # 2, 3
     "horizon_length", "cost_func_tuning",  # 4, 5
     "yaw_offset_study", "scalability", # 6, 7
     "breakdown_robustness", # 8
     "gradient_type", "n_wind_preview_samples", # 9, 10
     "generate_sample_figures", "baseline_controllers_3", # 11, 12
     "cost_func_tuning_small", "sr_solve", # 13, 14
     "baseline_controllers_ml_forecasters_awaken", "baseline_controllers_baseline_det_forecasters_awaken", "baseline_controllers_baseline_prob_forecasters_awaken", # 15, 16, 17
     "baseline_controllers_perfect_forecaster_flasc", "baseline_controllers_perfect_forecaster_awaken", # 18, 19
     "baseline_controllers_forecasters_test_flasc", "baseline_controllers_forecasters_test_awaken"] # 20, 21