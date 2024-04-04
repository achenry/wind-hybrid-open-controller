import sys
import yaml
import os

# from hercules.controller_standin import ControllerStandin
from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml

import whoc
from whoc.interfaces.hercules_actuator_disk_interface import HerculesADInterface
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.wind_field.generate_freestream_wind import generate_freestream_wind

n_seeds = 6
regenerate_wind_field = False

input_dict = load_yaml(sys.argv[1])
case_idx = int(sys.argv[2])

with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

amr_standin_data = generate_freestream_wind(".", n_seeds, regenerate_wind_field)[case_idx]

# controller = ControllerStandin(input_dict)
seed = 0
interface = HerculesADInterface(input_dict)
controller = MPC(interface, input_dict, 
                 wind_mag_ts=amr_standin_data["amr_wind_speed"], wind_dir_ts=amr_standin_data["amr_wind_direction"], 
                 lut_path=os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv"), 
                 generate_lut=False, 
                 seed=seed,
                 wind_field_config=wind_field_config)

py_sims = PySims(input_dict)


emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])
