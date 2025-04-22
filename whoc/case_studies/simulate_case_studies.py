import pandas as pd
import polars as pl
import numpy as np
import os
from time import perf_counter
from memory_profiler import profile
import re
from psutil import virtual_memory
from shutil import move

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.wind_field.WindField import first_ord_filter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @profile
def simulate_controller(controller_class, wind_forecast_class, simulation_input_dict, **kwargs):
    
    results_dir = os.path.join(kwargs["save_dir"], kwargs['case_family'])
    os.makedirs(results_dir, exist_ok=True)
    
    if simulation_input_dict["controller"]["uncertain"] and not wind_forecast_class.is_probabilistic:
        logging.info(f"Can't run with uncertain flag for {wind_forecast_class.__name__}, setting uncertainty off.")
        simulation_input_dict["controller"]["uncertain"] = simulation_input_dict["controller"]["uncertain"] and wind_forecast_class.is_probabilistic
        input_df = pd.read_csv(os.path.join(results_dir, f"case_descriptions.csv"))
        input_df.loc[int(kwargs['case_name']), "uncertain"] = False
        input_df.to_csv(os.path.join(results_dir, f"case_descriptions.csv"))
        # old_case_name = kwargs['case_name']
        # kwargs['case_name'] = re.sub("uncertain_True", "uncertain_False", kwargs['case_name'])
        # move(os.path.join(results_dir, f"input_config_case_{old_case_name}.pkl"), os.path.join(results_dir, f"input_config_case_{kwargs['case_name']}.pkl"))
    
    fn = f"time_series_results_case_{kwargs['case_name']}_seed_{kwargs['wind_case_idx']}.csv"
    save_path = os.path.join(results_dir, fn)
    temp_save_path = os.path.join(results_dir, fn.replace(".csv", "_temp.csv"))
    
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)
        
    stoptime = simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] - simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() - (simulation_input_dict["controller"]["n_horizon"] * simulation_input_dict["controller"]["controller_dt"])
    
    if not kwargs["rerun_simulations"] and os.path.exists(save_path):
        results_df = pd.read_csv(os.path.join(results_dir, fn), low_memory=False)
        # check if this saved df completed successfully
        if results_df["Time"].iloc[-1] == stoptime - simulation_input_dict["controller"]["controller_dt"] + simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds():
            logging.info(f"Loaded existing {fn} since rerun_simulations argument is false")
            return
    elif not kwargs["rerun_simulations"] and os.path.exists(os.path.join(results_dir, fn.replace("results", f"chk"))):
        # TODO HIGH setup checkpointing from temp unless restart flag is passed
        pass
    
    if os.path.exists(save_path):
        os.remove(save_path)
    
    logging.info(f"Running instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    # Load a FLORIS object for power calculations
    fi = ControlledFlorisModel(t0=kwargs["wind_field_ts"].select(pl.col("time").first()).item(),
                               yaw_limits=simulation_input_dict["controller"]["yaw_limits"],
                                offline_probability=simulation_input_dict["controller"]["offline_probability"],
                                simulation_dt=simulation_input_dict["simulation_dt"],
                                yaw_rate=simulation_input_dict["controller"]["yaw_rate"],
                                config_path=simulation_input_dict["controller"]["floris_input_file"],
                                target_turbine_indices=simulation_input_dict["controller"]["target_turbine_indices"] or "all",
                                uncertain=simulation_input_dict["controller"]["uncertain"],
                                turbine_signature=kwargs["turbine_signature"],
                                tid2idx_mapping=kwargs["tid2idx_mapping"])
     
    if simulation_input_dict["controller"]["target_turbine_indices"] != "all":
        fi_full = ControlledFlorisModel(t0=kwargs["wind_field_ts"].select(pl.col("time").first()).item(),
                                    yaw_limits=simulation_input_dict["controller"]["yaw_limits"],
                                        offline_probability=simulation_input_dict["controller"]["offline_probability"],
                                        simulation_dt=simulation_input_dict["simulation_dt"],
                                        yaw_rate=simulation_input_dict["controller"]["yaw_rate"],
                                        config_path=simulation_input_dict["controller"]["floris_input_file"],
                                        target_turbine_indices="all",
                                        uncertain=simulation_input_dict["controller"]["uncertain"],
                                        turbine_signature=kwargs["turbine_signature"],
                                        tid2idx_mapping=kwargs["tid2idx_mapping"])
    else:
        fi_full = fi
    
    if not kwargs["tid2idx_mapping"]:
        kwargs["tid2idx_mapping"] = {i: i for i in np.arange(fi_full.n_turbines)}
    idx2tid_mapping = dict([(v, k) for k, v in kwargs["tid2idx_mapping"].items()])
    
    kwargs["wind_field_config"]["preview_dt"] = int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"]) 
    kwargs["wind_field_config"]["n_preview_steps"] = simulation_input_dict["controller"]["n_horizon"] * int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"])
    kwargs["wind_field_config"]["time_series_dt"] = int(simulation_input_dict["controller"]["controller_dt"] // simulation_input_dict["simulation_dt"])
    
    if simulation_input_dict["controller"]["initial_conditions"]["yaw"] == "auto":
        if "FreestreamWindDir" in kwargs["wind_field_ts"].columns:
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = [kwargs["wind_field_ts"].select(pl.col("FreestreamWindDir").first()).item()] * fi.n_turbines
        else:
            sorted_tids = np.arange(fi_full.n_turbines) if simulation_input_dict["controller"]["target_turbine_indices"] == "all" else sorted(simulation_input_dict["controller"]["target_turbine_indices"])
            u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[i]}" for i in sorted_tids]).select(pl.all().first()).to_numpy()[0, :]
            v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[i]}" for i in sorted_tids]).select(pl.all().first()).to_numpy()[0, :]
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = 180.0 + np.rad2deg(np.arctan2(u, v))
     
    # pl.DataFrame(kwargs["wind_field_ts"])
    # simulation_input_dict["wind_forecast"]["measurement_layout"] = np.vstack([fi.env.layout_x, fi.env.layout_y]).T
    if wind_forecast_class:
        wind_forecast = wind_forecast_class(true_wind_field=kwargs["wind_field_ts"] if wind_forecast_class.__name__ == "PerfectForecast" else None,
                                            fmodel=fi_full.env, 
                                            tid2idx_mapping=kwargs["tid2idx_mapping"],
                                            turbine_signature=kwargs["turbine_signature"],
                                            use_tuned_params=kwargs["use_tuned_params"],
                                            model_config=kwargs["model_config"],
                                            **{k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" in k},
                                            kwargs={k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" not in k})
    else:
        wind_forecast = None
    ctrl = controller_class(fi, wind_forecast=wind_forecast, simulation_input_dict=simulation_input_dict, **kwargs)
    
    yaw_angles_ts = [[ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC]
    # init_yaw_angles_ts = []

    yaw_angles_change_ts = []
    turbine_powers_ts = [[np.nan] * ctrl.n_turbines]
    turbine_wind_mag_ts = [[np.nan] * ctrl.n_turbines]
    turbine_wind_dir_ts = [[np.nan] * ctrl.n_turbines]
    turbine_offline_status_ts = [[False] * ctrl.n_turbines]
    predicted_wind_speeds_ts = []
    # predicted_time_ts = []
    predicted_turbine_wind_speed_horz_ts = []
    predicted_turbine_wind_speed_vert_ts = []
    stddev_turbine_wind_speed_horz_ts = []
    stddev_turbine_wind_speed_vert_ts = []

    
    if wind_forecast_class:
        predicted_wind_speeds_ts = []
    
    convergence_time_ts = [np.nan]

    opt_cost_ts = [np.nan]
    opt_cost_terms_ts = [[np.nan] * 2]
    
    if hasattr(ctrl, "state_cons_activated"):
        lower_state_cons_activated_ts = [np.nan]
        upper_state_cons_activated_ts = [np.nan]
    else:
        lower_state_cons_activated_ts = upper_state_cons_activated_ts = None

    n_future_steps = int(ctrl.controller_dt // simulation_input_dict["simulation_dt"]) - 1
    
    t = 0
    k = 0
    
    # input to floris should be from first in target_turbine_indices (most upstream one), or mean over whole farm if no target_turbine_indices
    if kwargs["wf_source"] == "scada":
        if simulation_input_dict["controller"]["target_turbine_indices"] == "all":
            simulation_u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(idx2tid_mapping))]).select(pl.mean_horizontal(pl.all()))
            simulation_v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(idx2tid_mapping))]).select(pl.mean_horizontal(pl.all()))
        else:
            use_upstream_wind = True
            if use_upstream_wind:
                upstream_tidx = simulation_input_dict["controller"]["target_turbine_indices"][0]
                simulation_u = kwargs["wind_field_ts"].select(f"ws_horz_{idx2tid_mapping[upstream_tidx]}").to_numpy()[:, 0]
                simulation_v = kwargs["wind_field_ts"].select(f"ws_vert_{idx2tid_mapping[upstream_tidx]}").to_numpy()[:, 0]
            else:
                # use mean
                simulation_u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()[:, 0]
                simulation_v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()[:, 0]
            
        simulation_mag = (simulation_u**2 + simulation_v**2)**0.5
        simulation_dir = 180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v))
        simulation_dir[simulation_dir < 0] = 360. + simulation_dir[simulation_dir < 0]
        simulation_dir[simulation_dir > 360] = np.mod(simulation_dir[simulation_dir > 360], 360.) 
    else:
        simulation_mag = kwargs["wind_field_ts"].select("FreestreamWindMag").to_numpy()
        simulation_dir = kwargs["wind_field_ts"].select("FreestreamWindDir").to_numpy()
        simulation_u = simulation_mag * np.sin(np.deg2rad(180 + simulation_dir))
        simulation_v = simulation_mag * np.cos(np.deg2rad(180 + simulation_dir))
        
    # recompute controls and step floris forward by ctrl.controller_dt
    while t < stoptime:

        # reiniitialize and run FLORIS interface with current disturbances and disturbance up to (and excluding) next controls computation
        # using yaw angles as most recently sent from last time-step i.e. initial yaw conditions for first time step
        # for testing
        # if t == max(simulation_dir.index[-1] - 120, 0):
        #     print("hold")
        # try:
        fi.step(disturbances={"wind_speeds": simulation_mag[k:k + n_future_steps + 1],
                            "wind_directions": simulation_dir[k:k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=None if t > 0 else {"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC},
                            seed=k)
        # except Exception as e:
        #     print(f"NEGATIVE M0 ERROR for {kwargs['case_name']}_seed_{kwargs['wind_case_idx']}, time {t}, wind directions {simulation_dir[k:k + n_future_steps + 1]}")
        #     print(e)
        
        ctrl.current_freestream_measurements = [
                simulation_u[k],
                simulation_v[k]
        ]
         
        start_time = perf_counter()
        # get measurements from FLORIS int, then compute controls in controller class, set controls_dict, then send controls to FLORIS interface (calling calculate_wake)
        
        fi.run_floris = False
        # only step yaw angles by up to yaw_rate * simulation_input_dict["simulation_dt"] for each time-step
        # in ctrl.step(), get simulator measurements from FLORIS, update controls dict every simulation_dt seconds,
        # but only compute new yaw setpoints and run FLORIS with setpoints from full controllet_dt interval every controller_dt in ControllerFlorisInterface
        for tt in np.arange(t, t + ctrl.controller_dt, simulation_input_dict["simulation_dt"]):
            
            if tt == (t + ctrl.controller_dt - simulation_input_dict["simulation_dt"]):
                fi.run_floris = True
            
            # init_yaw_angles_ts += [ctrl.measurements_dict["yaw_angles"]]
            
            ctrl.step()        
            
            # Note these are results from previous time step
            yaw_angles_ts += [ctrl.measurements_dict["yaw_angles"]]
            turbine_powers_ts += [ctrl.measurements_dict["turbine_powers"]]
            turbine_wind_mag_ts += [ctrl.measurements_dict["wind_speeds"]]
            turbine_wind_dir_ts += [ctrl.measurements_dict["wind_directions"]]
            
            if wind_forecast_class and kwargs["include_prediction"] and (simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() > 0) and (ctrl.controls_dict["predicted_wind_speeds"] is not None):
                predicted_wind_speeds_ts += [ctrl.controls_dict["predicted_wind_speeds"]]
            
            turbine_offline_status_ts += [np.isclose(ctrl.measurements_dict["turbine_powers"], 0, atol=1e-3)]
            
            if hasattr(ctrl, "state_cons_activated"):
                lower_state_cons_activated_ts += [ctrl.state_cons_activated["lower"]]
                upper_state_cons_activated_ts += [ctrl.state_cons_activated["upper"]]
             
            fi.time += pd.Timedelta(seconds=simulation_input_dict["simulation_dt"])
        
        # zero turbine power could be due to low wind speed as well as formally set offline 
        # assert np.all(np.vstack(turbine_offline_status_ts)[-int(ctrl.controller_dt // simulation_input_dict["simulation_dt"]):, :] == fi.offline_status), "collected turbine_offline_status_ts should be equal to fi.offline_status in simulate_controllers"

        end_time = perf_counter()

        # convergence_time_ts.append((end_time - start_time) if ((t % ctrl.controller_dt) == 0.0) else np.nan)
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
        # add freestream wind mags/dirs provided to controller, yaw angles computed at this time-step, resulting turbine powers, wind mags, wind dirs
        # if ctrl.verbose:
        logging.info(f"Time = {t}/{stoptime} of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
        if ctrl.verbose and False:
            logging.info(f"Measured Freestream Wind Direction = {simulation_dir[k]}",
                f"Measured Freestream Wind Magnitude = {simulation_mag[k]}",
                f"Measured Turbine Wind Directions = {ctrl.measurements_dict['wind_directions'] if ctrl.measurements_dict['wind_directions'].ndim == 2 else ctrl.measurements_dict['wind_directions']}",
                f"Measured Turbine Wind Magnitudes = {ctrl.measurements_dict['wind_speeds'] if ctrl.measurements_dict['wind_speeds'].ndim == 2 else ctrl.measurements_dict['wind_speeds']}",
                f"Measured Yaw Angles = {ctrl.measurements_dict['yaw_angles'] if ctrl.measurements_dict['yaw_angles'].ndim == 2 else ctrl.measurements_dict['yaw_angles']}",
                f"Measured Turbine Powers = {ctrl.measurements_dict['turbine_powers'] if ctrl.measurements_dict['turbine_powers'].ndim == 2 else ctrl.measurements_dict['turbine_powers']}",
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
         
        t += ctrl.controller_dt
        k += int(ctrl.controller_dt / simulation_input_dict["simulation_dt"])
    
        # if RAM is running low, write existing data to dataframe and continue
        # turn data into arrays, pandas dataframe, and export to csv
        if ((ram_used := virtual_memory().percent) > kwargs["ram_limit"]) or (final := (t>=stoptime)):
            logging.info(f"Used {ram_used}% RAM.")
            # turn data into arrays, pandas dataframe, and export to csv
            write_df(case_family=kwargs["case_family"],
                    case_name=kwargs["case_name"],
                    wind_case_idx=kwargs["wind_case_idx"],
                    wf_source=kwargs["wf_source"],
                    wind_field_ts=kwargs["wind_field_ts"],
                    simulation_mag=simulation_mag, simulation_dir=simulation_dir,
                    fi_full=fi_full,
                    start_time=(k-len(turbine_powers_ts)) * simulation_input_dict["simulation_dt"],
                    turbine_wind_mag_ts=turbine_wind_mag_ts, 
                    turbine_wind_dir_ts=turbine_wind_dir_ts, 
                    turbine_offline_status_ts=turbine_offline_status_ts, 
                    yaw_angles_ts=yaw_angles_ts, 
                    turbine_powers_ts=turbine_powers_ts,
                    opt_cost_terms_ts=opt_cost_terms_ts, 
                    convergence_time_ts=convergence_time_ts,
                    predicted_wind_speeds_ts=predicted_wind_speeds_ts,
                    lower_state_cons_activated_ts=lower_state_cons_activated_ts,
                    upper_state_cons_activated_ts=upper_state_cons_activated_ts,
                    ctrl=ctrl, 
                    wind_forecast_class=wind_forecast_class, 
                    simulation_input_dict=simulation_input_dict,
                    idx2tid_mapping=idx2tid_mapping,
                    save_path=temp_save_path,
                    final=final,
                    include_prediction=kwargs["include_prediction"])
            
            if final:
                logging.info(f"Moving final result to {save_path}.")
                move(temp_save_path, save_path)
            
            turbine_powers_ts = []
            turbine_wind_mag_ts = []
            turbine_wind_dir_ts = []
            turbine_offline_status_ts = []
            if hasattr(ctrl, "state_cons_activated"):
                lower_state_cons_activated_ts = []
                upper_state_cons_activated_ts = []
            opt_cost_terms_ts = []
            convergence_time_ts = []
            yaw_angles_ts = []
            if wind_forecast_class:
                predicted_wind_speeds_ts = []

    logging.info(f"Saved {fn}")
    return
    # return results_data

# @profile
def write_df(case_family, case_name, wind_case_idx, wf_source, wind_field_ts,
             start_time, simulation_mag, simulation_dir, fi_full,
             turbine_wind_mag_ts, turbine_wind_dir_ts, turbine_offline_status_ts, yaw_angles_ts, turbine_powers_ts,
             opt_cost_terms_ts, convergence_time_ts,
             predicted_wind_speeds_ts,
             lower_state_cons_activated_ts, upper_state_cons_activated_ts,
             ctrl, wind_forecast_class, simulation_input_dict, idx2tid_mapping, save_path, 
             final=False, include_prediction=True):
    
    turbine_wind_mag_ts = np.vstack(turbine_wind_mag_ts)
    turbine_wind_dir_ts = np.vstack(turbine_wind_dir_ts)
    turbine_offline_status_ts = np.vstack(turbine_offline_status_ts)
    turbine_powers_ts = np.vstack(turbine_powers_ts)
    yaw_angles_ts = np.vstack(yaw_angles_ts)
    
    if final:
        n_truncate_steps = (int(ctrl.controller_dt - (simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] % ctrl.controller_dt)) % ctrl.controller_dt) // simulation_input_dict["simulation_dt"]
    else:
        n_truncate_steps = 0
        
    turbine_wind_mag_ts = turbine_wind_mag_ts[:(-n_truncate_steps) or None, :]
    turbine_wind_dir_ts = turbine_wind_dir_ts[:(-n_truncate_steps) or None, :]
    turbine_offline_status_ts = turbine_offline_status_ts[:(-n_truncate_steps) or None, :]
    yaw_angles_ts = yaw_angles_ts[:(-n_truncate_steps) or None, :]
    turbine_powers_ts = turbine_powers_ts[:(-n_truncate_steps) or None, :]
    opt_cost_terms_ts = opt_cost_terms_ts[:(-n_truncate_steps) or None]
    convergence_time_ts = convergence_time_ts[:(-n_truncate_steps) or None]
    
    running_opt_cost_terms_ts = np.zeros_like(opt_cost_terms_ts)
    Q = simulation_input_dict["controller"]["alpha"]
    # R = (1 - simulation_input_dict["controller"]["alpha"]) 
    
    norm_turbine_powers = turbine_powers_ts / ctrl.rated_turbine_power
    # norm_yaw_angle_changes = yaw_angles_change_ts / (ctrl.controller_dt * ctrl.yaw_rate)
    
    running_opt_cost_terms_ts[:, 0] = np.sum(np.stack([-0.5 * (norm_turbine_powers[:, i])**2 * Q for i in range(ctrl.n_turbines)], axis=1), axis=1)
    # running_opt_cost_terms_ts[:, 1] = np.sum(np.stack([0.5 * (norm_yaw_angle_changes[:, i])**2 * R for i in range(ctrl.n_turbines)], axis=1), axis=1)
    running_opt_cost_terms_ts[:, 1] = np.nan
    
    # may be longer than following: int(stoptime // simulation_input_dict["simulation_dt"]), if controller step goes beyond
    start_step = int(start_time / simulation_input_dict["simulation_dt"])
    
    if start_step >= 0:
        fs_wind_mag = simulation_mag[start_step-1:start_step-1+yaw_angles_ts.shape[0]]
        fs_wind_dir = simulation_dir[start_step-1:start_step-1+yaw_angles_ts.shape[0]]
        filtered_fs_wind_dir = first_ord_filter(fs_wind_dir,
                                                alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"]))
    else:
        fs_wind_mag = np.insert(simulation_mag[0:yaw_angles_ts.shape[0]-1], 0, np.nan)
        fs_wind_dir = np.insert(simulation_dir[0:yaw_angles_ts.shape[0]-1], 0, np.nan)
        filtered_fs_wind_dir = np.insert(first_ord_filter(fs_wind_dir[~np.isnan(fs_wind_dir)], 
                                        alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"])),
                                                        0, np.nan)
        
    start_step = max(0, start_step)
    
    results_data = {
        "CaseFamily": [case_family] * yaw_angles_ts.shape[0], 
        "CaseName": [case_name] * yaw_angles_ts.shape[0],
        "WindSeed": [wind_case_idx] * yaw_angles_ts.shape[0],
        "Time": start_time + (np.arange(0, yaw_angles_ts.shape[0]) * simulation_input_dict["simulation_dt"]),
        "FreestreamWindMag": fs_wind_mag,
        "FreestreamWindDir": fs_wind_dir,
        "FilteredFreestreamWindDir": filtered_fs_wind_dir,
        # **{
        #     f"InitTurbineYawAngle_{idx2tid_mapping[i]}": init_yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        # }, 
        **{
            f"TurbineYawAngle_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        # **{
        #     f"TurbineYawAngleChange_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": yaw_angles_change_ts[:, i] for i in range(ctrl.n_turbines)
        # },
        **{
            f"TurbinePower_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": turbine_powers_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindMag_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": turbine_wind_mag_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindDir_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": turbine_wind_dir_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineOfflineStatus_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": turbine_offline_status_ts[:, i] for i in range(ctrl.n_turbines)
        },
        # "FarmYawAngleChangeAbsSum": np.sum(np.abs(yaw_angles_change_ts), axis=1),
        # "RelativeFarmYawAngleChangeAbsSum": (np.sum(np.abs(yaw_angles_change_ts) * ~turbine_offline_status_ts, axis=1)) / (np.sum(~turbine_offline_status_ts, axis=1)),
        "FarmPower": np.sum(turbine_powers_ts, axis=1),
        # "RelativeFarmPower": np.sum(turbine_powers_ts, axis=1) / (np.sum(~turbine_offline_status_ts * np.array([max(fi.env.core.farm.turbine_definitions[i]["power_thrust_table"]["power"]) for i in range(ctrl.n_turbines)]), axis=1)),
        "OptimizationConvergenceTime": convergence_time_ts,
        **{
            f"RunningOptimizationCostTerm_{i}": running_opt_cost_terms_ts[:, i] for i in range(running_opt_cost_terms_ts.shape[1])
        },
        # "TotalRunningOptimizationCost": np.sum(running_opt_cost_terms_ts, axis=1),
    }
    
    if wf_source == "scada" and include_prediction:
        results_data.update({
            **{
                f"TrueTurbineWindSpeedHorz_{idx2tid_mapping[i]}": 
                wind_field_ts.select(f"ws_horz_{idx2tid_mapping[i]}").slice(start_step, yaw_angles_ts.shape[0]).to_numpy()[:, 0]
                for i in range(fi_full.n_turbines)
            },
            **{
                f"TrueTurbineWindSpeedVert_{idx2tid_mapping[i]}": 
                wind_field_ts.select(f"ws_vert_{idx2tid_mapping[i]}").slice(start_step, yaw_angles_ts.shape[0]).to_numpy()[:, 0]
                for i in range(fi_full.n_turbines)
            },
        })

    if hasattr(ctrl, "state_cons_activated"):
        results_data.update({
            "StateConsActivatedLower": lower_state_cons_activated_ts,
            "StateConsActivatedUpper": upper_state_cons_activated_ts,
        })

    results_data = pd.DataFrame(results_data)
    
    if wind_forecast_class and include_prediction and simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() > 0:
        
        # .group_by("time", maintain_order=True).agg(pl.all().last())\
        predicted_wind_speeds_ts = pl.concat(predicted_wind_speeds_ts, how="vertical")\
                                     .with_columns(time=((pl.col("time") - ctrl.init_time).dt.total_seconds().cast(pl.Float32)))
        # results_df = pd.concat([results_df, predicted_wind_speeds_ts], axis=1)
        # sd_ws_vert_cols
        cols = ["time"] + ctrl.mean_ws_horz_cols + ctrl.mean_ws_vert_cols + ((ctrl.sd_ws_horz_cols + ctrl.sd_ws_vert_cols) if ctrl.uncertain else [])
        predicted_wind_speeds_ts = predicted_wind_speeds_ts.select(cols)\
                .rename({
            src: f"PredictedTurbineWindSpeed{re.search('(?<=ws_)\\w+(?=_\\d+)', src).group().capitalize()}_{re.search('(?<=_)\\d+$', src).group()}"
            for src in ctrl.mean_ws_horz_cols + ctrl.mean_ws_vert_cols})\
                .rename({"time": "Time"})
        if ctrl.uncertain:
            predicted_wind_speeds_ts = predicted_wind_speeds_ts.rename({
                src: f"StddevTurbineWindSpeed{re.search('(?<=ws_)\\w+(?=_\\d+)', src).group().capitalize()}_{re.search('(?<=_)\\d+$', src).group()}"
                for src in ctrl.sd_ws_horz_cols + ctrl.sd_ws_vert_cols})
        # for key in ["CaseFamily", "CaseName", "WindSeed"]:
        #     predicted_wind_speeds_ts = predicted_wind_speeds_ts.assign(**{key: results_data[key].values[0]})
        # results_data = results_data.merge(predicted_wind_speeds_ts, on=["CaseFamily", "CaseName", "WindSeed", "Time"], how="outer")
        predicted_wind_speeds_ts = predicted_wind_speeds_ts.to_pandas()
        results_data = results_data.merge(predicted_wind_speeds_ts, on=["Time"], how="outer")
        results_data[["CaseFamily", "CaseName", "WindSeed"]] = results_data[["CaseFamily", "CaseName", "WindSeed"]].ffill()
        # results_data.loc[results_data["Time"] >= predicted_wind_speeds_ts["Time"].iloc[0], predicted_wind_speeds_ts.drop(columns=["Time"]).columns] = predicted_wind_speeds_ts.drop(columns=["Time"])
        del predicted_wind_speeds_ts
    
        # if not final:
        #     # results_data = results_data.dropna(subset=[f"TrueTurbineWindSpeedHorz_{idx2tid_mapping[i]}" for i in range(fi_full.n_turbines)])
        #     results_data = results_data.iloc[:-int(simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() / simulation_input_dict["simulation_dt"])]
    
    logging.info(f"Writing {'final' if final else 'intermediary'} result to file.")
    if final and os.path.exists(save_path):
        results_data = pd.concat([pd.read_csv(save_path, index_col=None),
                                  results_data], axis=0).groupby("Time").last()
        results_data.to_csv(save_path, mode="w", header=True, index=False)
    elif os.path.exists(save_path):
        results_data.to_csv(save_path, mode="a", header=False, index=False)
    else:
        results_data.to_csv(save_path, mode="w", header=True, index=False)