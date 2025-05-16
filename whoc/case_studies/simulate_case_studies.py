import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
import os
from time import perf_counter
from memory_profiler import profile
import re
from psutil import virtual_memory
from shutil import move

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.wind_field.WindField import first_ord_filter
from whoc.wind_field.WindField import butterworth_LPF_TFmag

from datetime import timedelta

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @ profile
def simulate_controller(controller_class, wind_forecast_class, simulation_input_dict, **kwargs):
    
    assigned_gpu = kwargs["assigned_gpu"]
    if assigned_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu)
    
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
    
    TRUNCATE_STEPS = 300 if kwargs["wf_source"] == "scada" else 0
    stoptime = simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] - simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() - (simulation_input_dict["controller"]["n_horizon"] * simulation_input_dict["controller"]["controller_dt"]) # - 2*TRUNCATE_STEPS
    
    load_from_checkpoint = not kwargs["rerun_simulations"] and os.path.exists(temp_save_path)
    load_from_final = not kwargs["rerun_simulations"] and os.path.exists(save_path)
    if load_from_final:
        results_df = pd.read_csv(save_path, low_memory=False)
        # check if this saved df completed successfully TODO add back in when all sims are uniform
        # if (results_df.shape[0] - 2) == int((stoptime - simulation_input_dict["simulation_dt"]) / simulation_input_dict["simulation_dt"]): #simulation_input_dict["controller"]["controller_dt"] + simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds():
        #     logging.info(f"Loaded existing {fn} since rerun_simulations argument is false")
        #     return
        return results_df
        if os.path.exists(save_path):
            os.remove(save_path)
            
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
            
        t = 0
        k = 0
    elif load_from_checkpoint:
        # TODO test
        logging.info(f"Loading from checkpoint {temp_save_path}")
        # set t, k to value after last in file, see how ctrl_dict is set in step, don't start save arrs with nans
        results_df = pd.read_csv(temp_save_path, low_memory=False)
        t = results_df.dropna(subset="FreestreamWindMag")["Time"].max() + simulation_input_dict["simulation_dt"]
        k = int(t // simulation_input_dict["simulation_dt"])
        simulation_input_dict["controller"]["initial_conditions"]["yaw"] = \
            list(results_df.dropna(subset="FreestreamWindMag").iloc[-1][[f"TurbineYawAngle_{idx2tid_mapping[i]}" for i in fi.sorted_tids]].astype(float).values)
    
    else:
        t = 0
        k = 0
        
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
    
    logging.info(f"Running instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    
    kwargs["wind_field_config"]["preview_dt"] = int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"]) 
    kwargs["wind_field_config"]["n_preview_steps"] = simulation_input_dict["controller"]["n_horizon"] * int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"])
    kwargs["wind_field_config"]["time_series_dt"] = int(simulation_input_dict["controller"]["controller_dt"] // simulation_input_dict["simulation_dt"])
    
    # tgt_d = 140
    # tgt_u, tgt_v = 10 * np.sin(np.deg2rad(tgt_d - 180)), 10 * np.cos(np.deg2rad(tgt_d - 180))
    # kwargs["wind_field_ts"] = kwargs["wind_field_ts"].with_columns(ws_horz_75=pl.lit(tgt_u), ws_vert_75=pl.lit(tgt_v), ws_horz_74=pl.lit(tgt_u), ws_vert_74=pl.lit(tgt_v))
    
    if isinstance(simulation_input_dict["controller"]["initial_conditions"]["yaw"], str) and simulation_input_dict["controller"]["initial_conditions"]["yaw"] == "auto":
        if "FreestreamWindDir" in kwargs["wind_field_ts"].columns:
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = [kwargs["wind_field_ts"].select(pl.col("FreestreamWindDir").first()).item()] * fi.n_turbines
        else:
            sorted_tids = fi.sorted_tids # np.arange(fi_full.n_turbines) if simulation_input_dict["controller"]["target_turbine_indices"] == "all" else sorted(simulation_input_dict["controller"]["target_turbine_indices"])
            u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[i]}" for i in sorted_tids]).select(pl.all().first()).to_numpy()[0, :]
            v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[i]}" for i in sorted_tids]).select(pl.all().first()).to_numpy()[0, :]
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = 180.0 + np.rad2deg(np.arctan2(u, v))
     
    # input to floris should be from first in target_turbine_indices (most upstream one), or mean over whole farm if no target_turbine_indices
    if kwargs["wf_source"] == "scada":
        if simulation_input_dict["controller"]["target_turbine_indices"] == "all":
            simulation_u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in idx2tid_mapping]).select(pl.mean_horizontal(pl.all()))
            simulation_v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in idx2tid_mapping]).select(pl.mean_horizontal(pl.all()))
        else:
            use_upstream_wind = simulation_input_dict["controller"]["use_upstream_wind"]
            if use_upstream_wind:
                # upstream_tidx = simulation_input_dict["controller"]["target_turbine_indices"][0]
                # rotate turbine coordinates based on most recent wind direction measurement
                # order turbines based on order of wind incidence
                layout_x = fi.env.layout_x
                layout_y = fi.env.layout_y
                # turbines_ordered_array = []
                wd = np.array(180.0 + np.rad2deg(np.arctan2(
                    np.mean(kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()),  
                    np.mean(kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()))))
                wd[wd < 0] = 360. + wd[wd < 0]
                wd[wd > 360] = np.mod(wd[wd > 360], 360.)
        
                layout_x_rot = (
                    np.cos(np.deg2rad(wd + 180.0)) * layout_y
                    + np.sin(np.deg2rad(wd + 180.0)) * layout_x
                )
                upstream_turbine_idx = np.argsort(layout_x_rot)[0]
                upstream_turbine_id = idx2tid_mapping[upstream_turbine_idx]
                logging.info(f"Using turbine id {upstream_turbine_id} as upstream turbine for wind seed {kwargs['wind_case_idx']}.")
                
                simulation_u = kwargs["wind_field_ts"].select(f"ws_horz_{upstream_turbine_id}").to_numpy()[:, 0]
                simulation_v = kwargs["wind_field_ts"].select(f"ws_vert_{upstream_turbine_id}").to_numpy()[:, 0]
            else:
                # use mean
                simulation_u = kwargs["wind_field_ts"].select([f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()[:, 0]
                simulation_v = kwargs["wind_field_ts"].select([f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]).select(pl.mean_horizontal(pl.all())).to_numpy()[:, 0]
        
        if simulation_input_dict["controller"]["filter_floris_wind"]:
            logging.info("Filtering wind passed to FLORIS.")
            # filter wind field NOTE this is not the wind field that PerfectForecast is returning...
            # FFT of raw wind direction time series
            # freq_vec_dir = np.fft.fft(simulation_dir)
            freq_vec_u = np.fft.fft(simulation_u)
            freq_vec_v = np.fft.fft(simulation_v)
            
            # fc_dir = 0.0011
            fc_mag = 0.0011
            n_lpf = 1
            ts_len = len(simulation_u)
            half_len = int(ts_len / 2)
            fs = (1 / (ts_len * simulation_input_dict["simulation_dt"])) * np.arange(1, half_len)
            
            # tf_dir_lpf = butterworth_LPF_TFmag(fs, fc_dir, n_lpf)
            tf_mag_lpf = butterworth_LPF_TFmag(fs, fc_mag, n_lpf)

            # Apply LPF magnitude
            # freq_vec_dir[1:int(ts_len / 2)] *= tf_dir_lpf
            freq_vec_u[1:half_len] *= tf_mag_lpf
            freq_vec_v[1:half_len] *= tf_mag_lpf
            
            if ts_len % 2 == 0:
                freq_vec_u[half_len] = np.sqrt(np.max([butterworth_LPF_TFmag(0.5 / simulation_input_dict["simulation_dt"], fc_mag, n_lpf), 0]))
                freq_vec_u[half_len + 1:] *= np.flip(tf_mag_lpf)
                freq_vec_v[half_len] = np.sqrt(np.max([butterworth_LPF_TFmag(0.5 / simulation_input_dict["simulation_dt"], fc_mag, n_lpf), 0]))
                freq_vec_v[half_len + 1:] *= np.flip(tf_mag_lpf)
            else:
                freq_vec_u[half_len:half_len+2] = np.sqrt(np.max([butterworth_LPF_TFmag(0.5 / simulation_input_dict["simulation_dt"], fc_mag, n_lpf), 0]))
                freq_vec_u[half_len + 2:] *= np.flip(tf_mag_lpf)
                freq_vec_v[half_len:half_len+2] = np.sqrt(np.max([butterworth_LPF_TFmag(0.5 / simulation_input_dict["simulation_dt"], fc_mag, n_lpf), 0]))
                freq_vec_v[half_len + 2:] *= np.flip(tf_mag_lpf)

            # START TEST
            # new_simulation_u = np.real(np.fft.ifft(freq_vec_u))[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            # new_simulation_v = np.real(np.fft.ifft(freq_vec_v))[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
            # axs[0].plot(simulation_u,label="Raw Wind U")
            # axs[0].plot(new_simulation_u,linewidth=2.0,color='r',label="Low-Frequency Wind U")
            # axs[0].legend()
            # axs[1].plot(simulation_v,label="Raw Wind V")
            # axs[1].plot(new_simulation_v,linewidth=2.0,color='r',label="Low-Frequency Wind V")
            # axs[1].legend()
            # plt.grid()
            # END TEST
        
            # save `originals
            all_freq_simulation_mag = ((simulation_u**2 + simulation_v**2)**0.5)#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            all_freq_simulation_dir = (180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v)))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            all_freq_simulation_dir[all_freq_simulation_dir < 0] = 360. + all_freq_simulation_dir[all_freq_simulation_dir < 0]
            all_freq_simulation_dir[all_freq_simulation_dir > 360] = np.mod(all_freq_simulation_dir[all_freq_simulation_dir > 360], 360.) 
            
            # time series of low-frequency wind direction
            simulation_u = np.real(np.fft.ifft(freq_vec_u))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            simulation_v = np.real(np.fft.ifft(freq_vec_v))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
            
        stoptime = (len(simulation_u) 
                    - int(simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() // simulation_input_dict["simulation_dt"])
                    - int((((simulation_input_dict["controller"]["n_horizon"] if controller_class.__name__ == "MPC" else 0) * simulation_input_dict["controller"]["controller_dt"])) // simulation_input_dict["simulation_dt"]))
        
        simulation_mag = (simulation_u**2 + simulation_v**2)**0.5
        simulation_dir = 180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v))
        simulation_dir[simulation_dir < 0] = 360. + simulation_dir[simulation_dir < 0]
        simulation_dir[simulation_dir > 360] = np.mod(simulation_dir[simulation_dir > 360], 360.)
        
        if not simulation_input_dict["controller"]["filter_floris_wind"]:
            logging.info("Passing raw wind to FLORIS.")
            all_freq_simulation_mag = simulation_mag
            all_freq_simulation_dir = simulation_dir
        
        # kwargs["wind_field_ts"] = kwargs["wind_field_ts"].slice(TRUNCATE_STEPS, kwargs["wind_field_ts"].select(pl.len()).item() - (2*TRUNCATE_STEPS))
        
    else:
        simulation_mag = kwargs["wind_field_ts"].select("FreestreamWindMag").to_numpy()
        simulation_dir = kwargs["wind_field_ts"].select("FreestreamWindDir").to_numpy()
        simulation_u = simulation_mag * np.sin(np.deg2rad(180. + simulation_dir))
        simulation_v = simulation_mag * np.cos(np.deg2rad(180. + simulation_dir))
     
        # pl.DataFrame(kwargs["wind_field_ts"])
    # simulation_input_dict["wind_forecast"]["measurement_layout"] = np.vstack([fi.env.layout_x, fi.env.layout_y]).T
    if wind_forecast_class:
        wind_forecast = wind_forecast_class(true_wind_field=kwargs["wind_field_ts"] if wind_forecast_class.__name__ == "PerfectForecast" else None,
                                            fmodel=fi_full.env, 
                                            tid2idx_mapping=kwargs["tid2idx_mapping"],
                                            turbine_signature=kwargs["turbine_signature"],
                                            use_tuned_params=kwargs["use_tuned_params"],
                                            **{k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" in k},
                                            kwargs={k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" not in k})
        wind_forecast.reset(assigned_gpu=assigned_gpu)
    else:
        wind_forecast = None
    ctrl = controller_class(fi, wind_forecast=wind_forecast, simulation_input_dict=simulation_input_dict, **kwargs)
    
    n_future_steps = int(ctrl.controller_dt // simulation_input_dict["simulation_dt"]) - 1
    
    if load_from_checkpoint:
        fi.time = fi.init_time + timedelta(seconds=(t - simulation_input_dict["controller"]["controller_dt"]))
        ctrl.controls_dict = {"yaw_angles": np.array([ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC)}
        previous_k = k - int(ctrl.controller_dt / simulation_input_dict["simulation_dt"])
        # fi.env.core.farm.yaw_angles = 
        fi.step(disturbances={"wind_speeds": simulation_mag[previous_k:previous_k + n_future_steps + 1],
                            "wind_directions": simulation_dir[previous_k:previous_k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=ctrl.controls_dict,
                            seed=previous_k)
        ctrl.current_freestream_measurements = [
                simulation_u[previous_k],
                simulation_v[previous_k]
        ]
        
        fi.time = fi.init_time + timedelta(seconds=(t - simulation_input_dict["simulation_dt"]))
        fi.run_floris = True
        ctrl.step()
        fi.time += timedelta(seconds=simulation_input_dict["simulation_dt"])
        ctrl.historic_measurements = kwargs["wind_field_ts"].filter(pl.col("time") < (ctrl.init_time + timedelta(seconds=t)))\
                                                            .select(["time"] + ctrl.ws_horz_cols + ctrl.ws_vert_cols + ctrl.nd_cos_cols + ctrl.nd_sin_cols)\
                                                            .with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")), cs.numeric().cast(pl.Float32))
    
    yaw_angles_ts = [[ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC] if k == 0 else []
    # init_yaw_angles_ts = []
    turbine_powers_ts = [[np.nan] * ctrl.n_turbines] if k == 0 else []
    turbine_wind_mag_ts = [[np.nan] * ctrl.n_turbines] if k == 0 else []
    turbine_wind_dir_ts = [[np.nan] * ctrl.n_turbines] if k == 0 else []
    turbine_offline_status_ts = [[False] * ctrl.n_turbines] if k == 0 else []
    
    if wind_forecast_class:
        predicted_wind_speeds_ts = []
    
    convergence_time_ts = [np.nan] if k == 0 else []

    opt_cost_ts = [np.nan] if k == 0 else []
    opt_cost_terms_ts = [[np.nan] * 2] if k == 0 else []
    
    if hasattr(ctrl, "state_cons_activated"):
        lower_state_cons_activated_ts = [np.nan] if k == 0 else []
        upper_state_cons_activated_ts = [np.nan] if k == 0 else []
    else:
        lower_state_cons_activated_ts = upper_state_cons_activated_ts = None
    
    # recompute controls and step floris forward by ctrl.controller_dt
    logging.info(f"Running for stoptime = {stoptime} for instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    while t < stoptime:

        # reiniitialize and run FLORIS interface with current disturbances and disturbance up to (and excluding) next controls computation
        # using yaw angles as most recently sent from last time-step i.e. initial yaw conditions for first time step
        
        if k == 0 :
            ctrl_dict = {"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC}
        elif k > 0:
            ctrl_dict = None
            
        fi.step(disturbances={"wind_speeds": simulation_mag[k:k + n_future_steps + 1],
                            "wind_directions": simulation_dir[k:k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=ctrl_dict,
                            seed=k)
        
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
        if ((ram_used := virtual_memory().percent) > kwargs["ram_limit"]) or (final := (t>=stoptime)) or (len(turbine_powers_ts) >= int(3600 / simulation_input_dict["simulation_dt"])):
            logging.info(f"Used {ram_used}% RAM.")
            
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(2,1)
            # w = kwargs["wind_field_ts"].filter(pl.col("time") <= ctrl.init_time + timedelta(seconds=t))
            # d_us = (np.rad2deg(np.arctan2(w.select("ws_horz_75"), w.select("ws_vert_75"))) + 180.0).flatten()
            # d_ds = (np.rad2deg(np.arctan2(w.select("ws_horz_74"), w.select("ws_vert_74"))) + 180.0).flatten()
            # fd_us = first_ord_filter(d_us, alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"]))
            # fd_ds = first_ord_filter(d_ds, alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"]))
            # y_ds = np.vstack(yaw_angles_ts)[:, 0]
            # y_us = np.vstack(yaw_angles_ts)[:, 1]
            # ax[0].plot(d_ds, label="wind dir ds", color='red')
            # ax[0].plot(fd_ds, label="filt wind dir ds", linestyle="--", color='red')
            # ax[0].plot(y_ds, label="yaw ds", color='blue')
            # ax[1].plot(d_us, label="wind dir us", color='red')
            # ax[1].plot(fd_us, label="filt wind dir ds", linestyle="--", color='red')
            # ax[1].plot(y_us, label="yaw us", color='blue')
            # ax[0].legend()
            # ax[1].legend()
            
            # version_1 = pd.read_csv(save_path.replace(".csv", "_final.csv"), low_memory=False).drop(columns="OptimizationConvergenceTime") # run to completion the first time
            # version_2 = pd.read_csv(temp_save_path, low_memory=False).drop(columns="OptimizationConvergenceTime") # loaded from checkpoint and completed
            # ((version_2 == version_1) | version_1.isna() | version_2.isna()).all()
            # version_2["TurbineWindDir_5"] - version_1["TurbineWindDir_5"]
            # version_2["TurbineWindMag_5"] - version_1["TurbineWindMag_5"]
            # version_2["TurbineWindPower_5"] - version_1["TurbineWindPower_5"]
            # bad_idx = (~version_2["TurbineWindDir_5"].isna() & ~version_1["TurbineWindDir_5"].isna() & ((version_2["TurbineWindDir_5"] - version_1["TurbineWindDir_5"]) != 0)).values
            # version_1.loc[bad_idx, ["TurbineWindMag_5","TurbineWindDir_5"]]
            # version_2.loc[bad_idx, ["TurbineWindMag_5","TurbineWindDir_5"]]
            
            # turn data into arrays, pandas dataframe, and export to csv
            write_df(wf_source=kwargs["wf_source"],
                    wind_field_ts=kwargs["wind_field_ts"],
                    simulation_mag=all_freq_simulation_mag, simulation_dir=all_freq_simulation_dir,
                    fi_full=fi_full,
                    sorted_tids=fi.sorted_tids,
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

    # logging.info(f"Saved {save_path}")
    return
    # return results_data

# @profile
def write_df(wf_source, wind_field_ts,
             start_time, simulation_mag, simulation_dir, fi_full, sorted_tids,
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
        filtered_fs_wind_mag = first_ord_filter(fs_wind_mag,
                                                alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_mag_lpf_time_const"]) * simulation_input_dict["simulation_dt"]))
    else:
        fs_wind_mag = np.insert(simulation_mag[0:yaw_angles_ts.shape[0]-1], 0, np.nan)
        fs_wind_dir = np.insert(simulation_dir[0:yaw_angles_ts.shape[0]-1], 0, np.nan)
        filtered_fs_wind_dir = np.insert(first_ord_filter(fs_wind_dir[~np.isnan(fs_wind_dir)], 
                                        alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"])),
                                                        0, np.nan)
        filtered_fs_wind_mag = np.insert(first_ord_filter(fs_wind_mag[~np.isnan(fs_wind_mag)], 
                                        alpha=np.exp(-(1 / simulation_input_dict["controller"]["wind_mag_lpf_time_const"]) * simulation_input_dict["simulation_dt"])),
                                                        0, np.nan)
        
    start_step = max(0, start_step)
    
    results_data = {
        # "CaseFamily": [case_family] * yaw_angles_ts.shape[0], 
        # "CaseName": [case_name] * yaw_angles_ts.shape[0],
        # "WindSeed": [wind_case_idx] * yaw_angles_ts.shape[0],
        "Time": start_time + (np.arange(0, yaw_angles_ts.shape[0]) * simulation_input_dict["simulation_dt"]),
        "FreestreamWindMag": fs_wind_mag,
        "FreestreamWindDir": fs_wind_dir,
        "FilteredFreestreamWindDir": filtered_fs_wind_dir,
        "FilteredFreestreamWindMag": filtered_fs_wind_mag,
        # **{
        #     f"InitTurbineYawAngle_{idx2tid_mapping[i]}": init_yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        # }, 
        **{
            f"TurbineYawAngle_{idx2tid_mapping[sorted_tids[i]]}": yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        # **{
        #     f"TurbineYawAngleChange_{idx2tid_mapping[simulation_input_dict['controller']['target_turbine_indices'][i]]}": yaw_angles_change_ts[:, i] for i in range(ctrl.n_turbines)
        # },
        **{
            f"TurbinePower_{idx2tid_mapping[sorted_tids[i]]}": turbine_powers_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindMag_{idx2tid_mapping[sorted_tids[i]]}": turbine_wind_mag_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindDir_{idx2tid_mapping[sorted_tids[i]]}": turbine_wind_dir_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineOfflineStatus_{idx2tid_mapping[sorted_tids[i]]}": turbine_offline_status_ts[:, i] for i in range(ctrl.n_turbines)
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
        # TODO not right
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
        # results_data[["CaseFamily", "CaseName", "WindSeed"]] = results_data[["CaseFamily", "CaseName", "WindSeed"]].ffill()
        # results_data.loc[results_data["Time"] >= predicted_wind_speeds_ts["Time"].iloc[0], predicted_wind_speeds_ts.drop(columns=["Time"]).columns] = predicted_wind_speeds_ts.drop(columns=["Time"])
        del predicted_wind_speeds_ts
    
        # if not final:
        #     # results_data = results_data.dropna(subset=[f"TrueTurbineWindSpeedHorz_{idx2tid_mapping[i]}" for i in range(fi_full.n_turbines)])
        #     results_data = results_data.iloc[:-int(simulation_input_dict["wind_forecast"]["prediction_timedelta"].total_seconds() / simulation_input_dict["simulation_dt"])]
    
    # TESTING START
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # # ax.plot(results_data["Time"], results_data["TurbineYawAngle_5"], label="5")
    # ax.plot(results_data["Time"], results_data["TurbineYawAngle_74"], label="74")
    # ax.plot(results_data["Time"], results_data["TurbineYawAngle_75"], label="75")
    # ax.plot(results_data["Time"], results_data["FreestreamWindDir"], label="Raw wind dir.")
    # ax.plot(results_data["Time"], results_data["FreestreamWindMag"], label="Raw wind mag.")
    # ax.plot(results_data["Time"], results_data["FilteredFreestreamWindDir"], label="Filtered wind dir.")
    # ax.plot(results_data["Time"], results_data["FilteredFreestreamWindMag"], label="Filtered wind mag.")
    # ax.legend()
    # TESTING END
    
    logging.info(f"Writing {'final' if final else 'intermediary'} result to file.")
    if final and os.path.exists(save_path):
        results_data = pd.concat([pd.read_csv(save_path, index_col=None, low_memory=False),
                                  results_data], axis=0).groupby("Time").last().reset_index(drop=False)
        # set case family and case_name first, then time etc.
        # results_data = results_data.iloc[:, [1, 2, 0] + list(range(3, len(results_data.columns)))]
        results_data.to_csv(save_path, mode="w", header=True, index=False)
    elif os.path.exists(save_path):
        results_data.to_csv(save_path, mode="a", header=False, index=False)
    else:
        results_data.to_csv(save_path, mode="w", header=True, index=False)