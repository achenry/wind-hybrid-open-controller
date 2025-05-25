"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""
import numpy as np
import pandas as pd
from datetime import timedelta
import polars as pl
import polars.selectors as cs
from scipy.signal import lfilter

from whoc.controllers.controller_base import ControllerBase
from whoc.utils import sample_utilities
from memory_profiler import profile
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from floris.tools.visualization import visualize_quiver2

# np.seterr(all="ignore")

class GreedyController(ControllerBase):
    def __init__(self, interface, wind_forecast, simulation_input_dict, verbose=False,
                 use_power_weighting=False, n_forecast_samples=100, power_curve_config=None, **kwargs):
        # print("in GreedyController.__init__")
        super().__init__(interface, verbose=verbose)
        self.n_turbines = interface.n_turbines #simulation_input_dict["controller"]["num_turbines"]
        self.yaw_limits = simulation_input_dict["controller"]["yaw_limits"]
        self.yaw_rate = simulation_input_dict["controller"]["yaw_rate"]
        self.yaw_increment = simulation_input_dict["controller"]["yaw_increment"]
        self.simulation_dt = simulation_input_dict["simulation_dt"]
        self.controller_dt = simulation_input_dict["controller"]["controller_dt"]
        self.init_time = interface.init_time
        self.wind_forecast = wind_forecast
        self.wf_source = kwargs["wf_source"]
        
        # New configuration parameters for sample-based control logic
        self.use_power_weighting = use_power_weighting
        self.n_forecast_samples = n_forecast_samples
        self.power_curve_config = power_curve_config
        
        self.prediction_timedelta_stored = max(pd.Timedelta(self.controller_dt, unit="s"), self.wind_forecast.prediction_timedelta)
        
        self.turbine_signature = kwargs["turbine_signature"]
        self.tid2idx_mapping = kwargs["tid2idx_mapping"]
        self.idx2tid_mapping = dict([(i, k) for i, k in enumerate(self.tid2idx_mapping.keys())])
        self.target_turbine_indices = simulation_input_dict["controller"]["target_turbine_indices"] or "all"
        
        if self.target_turbine_indices != "all":
            self.sorted_tids = sorted(list(self.target_turbine_indices))
        else:
            self.sorted_tids = np.arange(len(self.tid2idx_mapping))
            
        
        self.uncertain = simulation_input_dict["controller"]["uncertain"]
        
        self.previous_yaw_setpoints = None
        
        # [self.idx2tid_mapping[i] for i in self.sorted_tids]
        self.target_mean_ws_horz_cols = [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
        self.target_mean_ws_vert_cols = [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
        self.ws_horz_cols = self.mean_ws_horz_cols = [f"ws_horz_{tid}" for tid in self.tid2idx_mapping]
        self.ws_vert_cols = self.mean_ws_vert_cols = [f"ws_vert_{tid}" for tid in self.tid2idx_mapping]
        self.nd_sin_cols = [f"nd_sin_{tid}" for tid in self.tid2idx_mapping]
        self.nd_cos_cols = [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
        
        self.tgt_turbine_indices = list(self.tid2idx_mapping.values())
        self.tgt_turbine_indices = [self.tgt_turbine_indices.index(i) for i in self.sorted_tids] 
        
        self.historic_measurements = None 
        # self.historic_measurements = pd.DataFrame(columns=["time"] 
        #                                           + self.mean_ws_horz_cols 
        #                                           + self.mean_ws_vert_cols
        #                                           + [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
        #                                           + [f"nd_sin_{tid}" for tid in self.tid2idx_mapping], dtype=pd.Float64Dtype())
        
        self.wind_mag_lpf_time_const = simulation_input_dict["controller"]["wind_mag_lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=simulation_input_dict["controller"]["lpf_start_time"])
        self.wind_mag_lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["wind_mag_lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.deadband_thr = simulation_input_dict["controller"]["deadband_thr"]
        # self.deadband_thr = 0 
        self.wind_mag_use_filt = simulation_input_dict["controller"]["use_lut_filtered_wind_mag"]

        self.rated_turbine_power = simulation_input_dict["controller"]["rated_turbine_power"]
        
        self.wind_field_ts = kwargs["wind_field_ts"]
        # logging.info(f"id(wind_field_ts) in GreedyController __init__ is {id(self.wind_field_ts)}")

        self.is_yawing = np.array([False for _ in range(self.n_turbines)])

        self._last_measured_time = None

        self.yaw_norm_const = 360.0

        # Set initial conditions
        self.yaw_IC = simulation_input_dict["controller"]["initial_conditions"]["yaw"]
   
        if hasattr(self.yaw_IC, "__len__"):
            if len(self.yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": np.array(self.yaw_IC)}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": np.array([self.yaw_IC] * self.n_turbines)}
        
        self.previous_target_yaw_setpoints = self.controls_dict["yaw_angles"]
    
    # self.filtered_measurements["wind_direction"] = []
    
    def _first_ord_filter(self, x, alpha):
        b = [1 - alpha]
        a = [1, -alpha]
        return lfilter(b, a, x)
    
    def yaw_offsets_interpolant(self, wind_directions, wind_speeds):
        # return np.zeros((*wind_directions.shape, self.n_turbines))
        return np.array([[[wind_directions[i, j] for t in range(self.n_turbines)] for j in range(wind_directions.shape[1])] for i in range(wind_directions.shape[0])])
    
    def compute_controls(self):
        """
        Compute control actions using either original or sample-based logic.
        
        Routes to sample-based control logic if wind_forecast is sample-based and
        use_power_weighting is True, otherwise uses original control logic.
        """
        if self.wind_forecast and self.wind_forecast.is_sample_based and self.use_power_weighting:
            logging.info("Using sample-based control logic.")
            return self.compute_controls_sample_based()
        else:
            logging.info("Using original control logic.")
            return self.compute_controls_original()
    
    # @profile
    def compute_controls_original(self):
        # print("in GreedyController.compute_controls")
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        # if self.verbose:
        #     logging.info(f"self._last_measured_time == {self._last_measured_time}")
        #     logging.info(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self.current_time = self._last_measured_time = self.measurements_dict["time"]

        # current_wind_directions = np.broadcast_to(self.wind_dir_ts[int(self.current_time // self.simulation_dt)], (self.n_turbines,))
        if self.wf_source == "floris":
            current_wind_directions = self.measurements_dict["wind_directions"]
            current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(current_wind_directions + 180.0)) 
            current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(current_wind_directions + 180.0))
        else:
            current_row = self.wind_field_ts.filter(pl.col("time") == self.current_time)
            # self.wind_field_ts = self.wind_field_ts.filter(pl.col("time") > self.current_time)
            current_ws_horz = current_row.select([f"ws_horz_{tid}" for tid in self.tid2idx_mapping]).to_numpy()[0, :]
            current_ws_vert = current_row.select([f"ws_vert_{tid}" for tid in self.tid2idx_mapping]).to_numpy()[0, :]
            current_wind_directions = 180.0 + np.rad2deg(
                np.arctan2(
                     current_ws_horz, 
                     current_ws_vert
                )
            )
        
        if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
            # yaw angles will be set to initial values
            if self.verbose:
                logging.info("Bad wind direction measurement received, reverting to previous measurement.")
        
        # pass greedy angles to all non target turbines
        # tid2idx_mapping/idx2tid_mapping was used to form current_ws_horz/vert
        # NOTE controlled floris interface returns values in terms of sorted_tids so controls_dict must be shaped like this
        
        current_nd_cos = np.cos(np.deg2rad(current_wind_directions))
        current_nd_sin = np.sin(np.deg2rad(current_wind_directions))
        current_nd_cos[self.tgt_turbine_indices] = np.cos(np.deg2rad(self.measurements_dict["yaw_angles"]))
        current_nd_sin[self.tgt_turbine_indices] = np.sin(np.deg2rad(self.measurements_dict["yaw_angles"]))
        
        current_measurements = pl.DataFrame({
            "time": [self.current_time],
            **{f"ws_horz_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_ws_horz)},
            **{f"ws_vert_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_ws_vert)},
            **{f"nd_cos_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_nd_cos)},
            **{f"nd_sin_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_nd_sin)}
        }).with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")), cs.numeric().cast(pl.Float32))
        
        # only get wind_dirs corresponding to target_turbine_ids
        current_wind_directions = current_wind_directions[self.tgt_turbine_indices]
        
        if self.wind_mag_use_filt or self.wind_forecast:
            if self.historic_measurements is not None:
                self.historic_measurements = pl.concat([self.historic_measurements, 
                                                        current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
                                                        ], how="vertical")\
                                                            .tail(max(int(np.ceil(self.wind_mag_lpf_time_const // self.simulation_dt) * 50), 
                                                                      self.wind_forecast.n_context))
            else:
                self.historic_measurements = current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
                
        # NOTE: this is run every simulation_dt, not every controller_dt, because the yaw angle may be moving gradually towards the correct setpoint
        
        current_yaw_setpoints = self.controls_dict["yaw_angles"]
        if self.previous_yaw_setpoints is None:
            self.previous_yaw_setpoints = current_yaw_setpoints.copy()

        reached_setpoints_cond = self.is_yawing & (current_yaw_setpoints == np.mod(self.previous_target_yaw_setpoints, 360))
        if self.verbose and any(reached_setpoints_cond):
            logging.info(f"Greedy Controller turbines {np.where(reached_setpoints_cond)[0]} have reached their target setpoint of {self.previous_target_yaw_setpoints[reached_setpoints_cond]} at time {self.current_time}.")

        # flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
        self.is_yawing[reached_setpoints_cond] = False

        new_yaw_setpoints = np.array(current_yaw_setpoints)
            
        forecasted_wind_field = None
        single_forecasted_wind_field = None
        use_wind_forecast = False
        
        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            
            if self.wind_forecast and self.wind_forecast.prediction_timedelta.total_seconds() > 0:
                forecasted_wind_field = self.wind_forecast.predict_point(self.historic_measurements, self.current_time)\
                                            .with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")), cs.numeric().cast(pl.Float32))
                single_forecasted_wind_field = forecasted_wind_field.filter(pl.col("time") == self.current_time + self.wind_forecast.prediction_timedelta)
                use_wind_forecast = True
            
            # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
            if (self.current_time < self.lpf_start_time) or not self.wind_mag_use_filt:
                wind = single_forecasted_wind_field if use_wind_forecast else current_measurements.select("time", cs.starts_with("ws_"))
                
                wind_u = wind.select(self.target_mean_ws_horz_cols).to_numpy()[-1, :]
                wind_v = wind.select(self.target_mean_ws_vert_cols).to_numpy()[-1, :]
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(wind_u, wind_v))
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"unfiltered forecasted wind directions = {wind_dirs}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                
            else:
                # use filtered wind direction and speed     
                if use_wind_forecast:
                    hist_meas = self.historic_measurements
                    last_historic_time = hist_meas.select(pl.col("time").last()).item()
                    first_forecasted_time = forecasted_wind_field.select(pl.col("time").first()).item()
                    if (fcst_lead_timedelta := (first_forecasted_time - last_historic_time)) > (sim_timedelta := timedelta(seconds=self.simulation_dt)):
                        missing_forecasted_time = pl.DataFrame({"time": [last_historic_time + i * sim_timedelta for i in range(1, int(fcst_lead_timedelta / sim_timedelta))]}).with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")))
                        wind = pl.concat([
                            hist_meas.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols),
                            missing_forecasted_time, 
                            forecasted_wind_field.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols)], how="diagonal")\
                             .select(pl.col("time"), cs.numeric().interpolate_by("time"))
                    else:
                        wind = pl.concat([hist_meas.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols), 
                                            forecasted_wind_field.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols)
                                            ], how="vertical")
                        
                    assert wind.select((pl.col("time").diff().slice(1) == sim_timedelta).all()).item() and (wind.select(pl.col("time").last()).item() == single_forecasted_wind_field.select(pl.col("time").last()).item()), "DataFrame passed to low pass filter must be continuous, with sampling time equal to simulation timestep, and must end on last forecasted value."
                    del hist_meas
                else:
                    wind = self.historic_measurements
                    
                wind_u = wind.select(self.target_mean_ws_horz_cols).to_numpy()
                wind_v = wind.select(self.target_mean_ws_vert_cols).to_numpy()
                
                if self.verbose:
                    if self.wind_forecast:
                        unfilt_wind_dirs = 180.0 + np.rad2deg(np.arctan2(wind_u, wind_v))
                        logging.info(f"unfiltered forecasted wind directions = {unfilt_wind_dirs[-1, :]}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                
                               
                if self.wind_mag_use_filt:
                    # filter the wind direction, only get wind_dirs corresponding to target_turbine_ids
                    wind_u = np.array([self._first_ord_filter(wind_u[:, i], self.wind_mag_lpf_alpha)
                                                    for i in range(len(self.sorted_tids))]).T # [-int(self.controller_dt // self.simulation_dt), :]
                    wind_v = np.array([self._first_ord_filter(wind_v[:, i], self.wind_mag_lpf_alpha)
                                                    for i in range(len(self.sorted_tids))]).T
                    wind_u = wind_u[-1, :]
                    wind_v = wind_v[-1, :]
                
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(wind_u, wind_v))
                
                if self.verbose:
                    logging.info(f"filtered {'forecasted' if self.wind_forecast else 'current'} wind directions = {wind_dirs}")
            
            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            target_yaw_setpoints = np.mod(np.rint(wind_dirs / self.yaw_increment) * self.yaw_increment, 360.0)
            setpoint_change = target_yaw_setpoints - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)]) 
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0) # if == 0, need to change within 360 deg, otherwise if == 1 faster to cross 360/0 boundary
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(self.n_turbines)]
            is_target_changing = (abs_setpoint_change > self.deadband_thr) & ~self.is_yawing
            
            dir_setpoint_change = np.sign(setpoint_change)
            dir_setpoint_change[setpoint_change_idx == 1] = -dir_setpoint_change[setpoint_change_idx == 1]
            new_yaw_setpoints[is_target_changing] = new_yaw_setpoints[is_target_changing] + dir_setpoint_change[is_target_changing] * abs_setpoint_change[is_target_changing]
            self.is_yawing[is_target_changing] = True
            
            if self.verbose and any(is_target_changing):
                logging.info(f"Greedy Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} in direction {dir_setpoint_change[is_target_changing]} at time {self.current_time}")
        else:
            is_target_changing = np.zeros_like(self.is_yawing).astype(bool)
        
        reaching_setpoints_cond = self.is_yawing & ~is_target_changing 
        if self.verbose and any(reaching_setpoints_cond):
            logging.info(f"Greedy Controller continuing to yaw turbines {np.where(reaching_setpoints_cond)[0]} from {current_yaw_setpoints[reaching_setpoints_cond]} to {self.previous_target_yaw_setpoints[reaching_setpoints_cond]} at time {self.current_time}")
        
        new_yaw_setpoints[reaching_setpoints_cond] = self.previous_target_yaw_setpoints[reaching_setpoints_cond].copy()
        
        # stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
        self.previous_target_yaw_setpoints = np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        
        lb, ub = self.previous_yaw_setpoints - self.simulation_dt * self.yaw_rate, self.previous_yaw_setpoints + self.simulation_dt * self.yaw_rate
        constrained_yaw_setpoints = np.clip(new_yaw_setpoints, lb, ub)
        
        # constrained_yaw_setpoints = np.clip(constrained_yaw_setpoints, *reversed([current_wind_directions - yl for yl in self.yaw_limits]))
        self.previous_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        constrained_yaw_setpoints = np.mod(self.previous_yaw_setpoints, 360)
        
        # self.init_sol = {"states": list(constrained_yaw_setpoints / self.yaw_norm_const)}
        # self.init_sol["control_inputs"] = (constrained_yaw_setpoints - self.controls_dict["yaw_angles"]) * (self.yaw_norm_const / (self.yaw_rate * self.controller_dt))

        self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)} 
        if self.wind_forecast:
            if use_wind_forecast:
                # newest_predictions = forecasted_wind_field.filter(pl.col("time") <= self.current_time + self.prediction_timedelta_stored)\
                newest_predictions = forecasted_wind_field.filter(pl.col("time") == self.current_time + self.wind_forecast.prediction_timedelta)\
                                                        .select(["time"] + self.mean_ws_horz_cols + self.mean_ws_vert_cols)
            else:
                newest_predictions = None
            self.controls_dict["predicted_wind_speeds"] = newest_predictions
        return None
    
    def calculate_turbine_power(self, wind_speed, min_power=0.0, max_power=1.0):
        """Calculate the power output of a turbine given wind speed and considering its power curve.
        
        Args:
            wind_speed: A numpy array or torch tensor of wind speeds.
            min_power: The minimum power output (default 0.0), used for normalization or cut-in.
            max_power: The maximum power output (default 1.0), used for normalization or cut-out.
            
        Returns:
            A numpy array or torch tensor of calculated power values, matching the shape of wind_speed.
        """
        logging.debug(f"Calculating turbine power for wind speeds with shape: {np.array(wind_speed).shape}")
        
        # Check if power curve configuration is available
        if self.power_curve_config is None:
            logging.error("Power curve configuration is None, returning default power values")
            return np.ones_like(wind_speed) * 0.5
        
        # Extract power curve data from configuration
        if 'power_curve_wind_speeds' not in self.power_curve_config or 'power_curve_powers' not in self.power_curve_config:
            if 'wind_speed' in self.power_curve_config and 'power' in self.power_curve_config:
                wind_speeds_curve = np.array(self.power_curve_config['wind_speed'])
                powers_curve = np.array(self.power_curve_config['power'])
            else:
                logging.error("Power curve configuration missing required keys, returning default power values")
                return np.ones_like(wind_speed) * 0.5
        else:
            wind_speeds_curve = np.array(self.power_curve_config['power_curve_wind_speeds'])
            powers_curve = np.array(self.power_curve_config['power_curve_powers'])
        
        # Determine if working with torch tensors
        is_torch = hasattr(wind_speed, 'device') and hasattr(wind_speed, 'detach')  # Check for torch tensor
        
        if is_torch:
            # Convert to numpy for interpolation
            wind_speed_np = wind_speed.detach().cpu().numpy()
            device = wind_speed.device
            dtype = wind_speed.dtype
        else:
            wind_speed_np = np.array(wind_speed)
        
        # Normalize power curve
        if np.max(powers_curve) > 10:
            # Normalize to [0, 1] range based on rated power
            rated_power = np.max(powers_curve)
            powers_curve_normalized = powers_curve / rated_power
            logging.debug(f"Normalized power curve from kW to [0,1] range using rated power: {rated_power}")
        else:
            powers_curve_normalized = powers_curve
        
        # Interpolate power for given wind speeds
        power_output = np.interp(wind_speed_np, wind_speeds_curve, powers_curve_normalized)
        
        # Clip to specified bounds
        power_output = np.clip(power_output, min_power, max_power)
        
        # Convert back to original tensor format if needed
        if is_torch:
            import torch
            power_output = torch.tensor(power_output, device=device, dtype=dtype)
        
        logging.debug(f"Calculated power output with shape: {power_output.shape}")
        return power_output
    
    def compute_controls_sample_based(self):
        """Calculate control (yaw) angles using power-weighted wind directions from forecast samples.
        
        This method implements sample-based control logic that:
        1. Acquires forecast samples from the wind forecast model
        2. Converts wind components to wind speed and direction for each sample
        3. Calculates power output for each sample using the turbine power curve
        4. Computes power-weighted mean wind direction for each turbine
        5. Applies yaw rate limiting and bounds similar to the original control logic
        
        Returns:
            None (updates self.controls_dict)
        """
        logging.info("Starting sample-based control computation")
        
        # Use the same time and measurement logic as original method
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        self.current_time = self._last_measured_time = self.measurements_dict["time"]
        
        # Get current measurements (same as original method)
        if self.wf_source == "floris":
            current_wind_directions = self.measurements_dict["wind_directions"]
            current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(current_wind_directions + 180.0))
            current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(current_wind_directions + 180.0))
        else:
            current_row = self.wind_field_ts.filter(pl.col("time") == self.current_time)
            current_ws_horz = current_row.select([f"ws_horz_{tid}" for tid in self.tid2idx_mapping]).to_numpy()[0, :]
            current_ws_vert = current_row.select([f"ws_vert_{tid}" for tid in self.tid2idx_mapping]).to_numpy()[0, :]
            current_wind_directions = 180.0 + np.rad2deg(
                np.arctan2(current_ws_horz, current_ws_vert)
            )
        
        # Update historic measurements (same as original method)
        current_nd_cos = np.cos(np.deg2rad(current_wind_directions))
        current_nd_sin = np.sin(np.deg2rad(current_wind_directions))
        current_nd_cos[self.tgt_turbine_indices] = np.cos(np.deg2rad(self.measurements_dict["yaw_angles"]))
        current_nd_sin[self.tgt_turbine_indices] = np.sin(np.deg2rad(self.measurements_dict["yaw_angles"]))
        
        current_measurements = pl.DataFrame({
            "time": [self.current_time],
            **{f"ws_horz_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_ws_horz)},
            **{f"ws_vert_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_ws_vert)},
            **{f"nd_cos_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_nd_cos)},
            **{f"nd_sin_{self.idx2tid_mapping[i]}": [v] for i, v in enumerate(current_nd_sin)}
        }).with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")), cs.numeric().cast(pl.Float32))
        
        current_wind_directions = current_wind_directions[self.tgt_turbine_indices]
        
        if self.historic_measurements is not None:
            self.historic_measurements = pl.concat([self.historic_measurements,
                                                    current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
                                                    ], how="vertical")\
                                                        .tail(max(int(np.ceil(self.wind_mag_lpf_time_const // self.simulation_dt) * 50),
                                                                  self.wind_forecast.n_context))
        else:
            self.historic_measurements = current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
        
        # Initialize yaw setpoints
        current_yaw_setpoints = self.controls_dict["yaw_angles"]
        if self.previous_yaw_setpoints is None:
            self.previous_yaw_setpoints = current_yaw_setpoints.copy()

        # Check for turbines that have reached their setpoints
        reached_setpoints_cond = self.is_yawing & (current_yaw_setpoints == np.mod(self.previous_target_yaw_setpoints, 360))
        if self.verbose and any(reached_setpoints_cond):
            logging.info(f"Sample-based Controller turbines {np.where(reached_setpoints_cond)[0]} have reached their target setpoint of {self.previous_target_yaw_setpoints[reached_setpoints_cond]} at time {self.current_time}.")

        self.is_yawing[reached_setpoints_cond] = False
        new_yaw_setpoints = np.array(current_yaw_setpoints)
        
        # Only compute new setpoints at controller intervals
        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            
            try:
                # Sample Acquisition
                logging.info(f"Requesting {self.n_forecast_samples} forecast samples for horizon steps")
                horizon_steps = max(1, int(self.wind_forecast.prediction_timedelta.total_seconds() / self.simulation_dt))
                
                samples = self.wind_forecast.predict_sample(
                    historical_data=self.historic_measurements,
                    current_time=self.current_time,
                    n_samples=self.n_forecast_samples,
                    horizon_steps=horizon_steps
                )
                
                if samples is None:
                    logging.warning("predict_sample returned None, falling back to original control logic")
                    return self.compute_controls_original()
                
                logging.info(f"Received samples with shape: {samples.shape}")
                
                # Reshape samples based on per_turbine_target setting
                if hasattr(self.wind_forecast, 'per_turbine_target') and self.wind_forecast.per_turbine_target:
                    # Samples are [n_samples, horizon_steps, n_turbines * 2]
                    n_samples, horizon_steps_actual, num_vars = samples.shape
                    expected_vars = len(self.tid2idx_mapping) * 2  # 2 for u,v components
                    
                    if num_vars != expected_vars:
                        logging.error(f"Expected {expected_vars} variables but got {num_vars}, falling back to original control")
                        return self.compute_controls_original()
                    
                    # Reshape to [n_samples, horizon_steps, n_turbines, 2]
                    samples_reshaped = samples.reshape(n_samples, horizon_steps_actual, len(self.tid2idx_mapping), 2)
                else:
                    # Samples are [n_samples, horizon_steps, 2] - one (u,v) pair for entire farm
                    n_samples, horizon_steps_actual, num_vars = samples.shape
                    
                    if num_vars != 2:
                        logging.error(f"Expected 2 variables (u,v) but got {num_vars}, falling back to original control")
                        return self.compute_controls_original()
                    
                    # Expand to all turbines: [n_samples, horizon_steps, n_turbines, 2]
                    samples_reshaped = np.repeat(samples[:, :, np.newaxis, :], len(self.tid2idx_mapping), axis=2)
                
                logging.info(f"Reshaped samples to: {samples_reshaped.shape}")
                
                # Power-Weighted Direction Calculation for each turbine
                power_weighted_directions = np.zeros((horizon_steps_actual, len(self.sorted_tids)))
                
                for t_idx, turbine_idx in enumerate(self.sorted_tids):
                    # Extract u,v samples for this turbine: [n_samples, horizon_steps]
                    u_samples_turbine = samples_reshaped[:, :, turbine_idx, 0]  # u component
                    v_samples_turbine = samples_reshaped[:, :, turbine_idx, 1]  # v component
                    
                    # Convert wind components to wind speed and direction
                    wind_params = sample_utilities.wind_components_to_params(u_samples_turbine, v_samples_turbine)
                    wind_speed_samples = wind_params['wind_speed_samples']
                    wind_direction_samples = wind_params['wind_direction_samples']
                    
                    # Calculate power output for each sample
                    power_output_samples = self.calculate_turbine_power(wind_speed_samples)
                    
                    # Compute power-weighted mean direction for each horizon step
                    for h_step in range(horizon_steps_actual):
                        wind_dirs_step = wind_direction_samples[:, h_step]
                        power_weights_step = power_output_samples[:, h_step]
                        
                        # Use circular weighted mean
                        power_weighted_dir = sample_utilities.circular_weighted_mean(
                            wind_dirs_step, power_weights_step
                        )
                        power_weighted_directions[h_step, t_idx] = power_weighted_dir
                
                logging.info(f"Computed power-weighted directions with shape: {power_weighted_directions.shape}")
                
                # Use the first horizon step for control (similar to original method using prediction_timedelta)
                wind_dirs = power_weighted_directions[0, :]  # Use first time step
                
                if self.verbose:
                    logging.info(f"Sample-based power-weighted wind directions = {wind_dirs}")
                
            except Exception as e:
                logging.error(f"Error in sample-based control computation: {e}")
                logging.warning("Falling back to original control logic")
                return self.compute_controls_original()
            
            # Apply yaw angle calculation (same logic as original method)
            target_yaw_setpoints = np.mod(np.rint(wind_dirs / self.yaw_increment) * self.yaw_increment, 360.0)
            setpoint_change = target_yaw_setpoints - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)])
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0)
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(len(self.sorted_tids))]
            is_target_changing = (abs_setpoint_change > self.deadband_thr) & ~self.is_yawing
            
            dir_setpoint_change = np.sign(setpoint_change)
            dir_setpoint_change[setpoint_change_idx == 1] = -dir_setpoint_change[setpoint_change_idx == 1]
            new_yaw_setpoints[is_target_changing] = new_yaw_setpoints[is_target_changing] + dir_setpoint_change[is_target_changing] * abs_setpoint_change[is_target_changing]
            self.is_yawing[is_target_changing] = True
            
            if self.verbose and any(is_target_changing):
                logging.info(f"Sample-based Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} in direction {dir_setpoint_change[is_target_changing]} at time {self.current_time}")
        else:
            is_target_changing = np.zeros_like(self.is_yawing).astype(bool)
        
        # Continue yawing logic (same as original method)
        reaching_setpoints_cond = self.is_yawing & ~is_target_changing
        if self.verbose and any(reaching_setpoints_cond):
            logging.info(f"Sample-based Controller continuing to yaw turbines {np.where(reaching_setpoints_cond)[0]} from {current_yaw_setpoints[reaching_setpoints_cond]} to {self.previous_target_yaw_setpoints[reaching_setpoints_cond]} at time {self.current_time}")
        
        new_yaw_setpoints[reaching_setpoints_cond] = self.previous_target_yaw_setpoints[reaching_setpoints_cond].copy()
        
        # Apply rate limiting and bounds (same as original method)
        self.previous_target_yaw_setpoints = np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        
        lb, ub = self.previous_yaw_setpoints - self.simulation_dt * self.yaw_rate, self.previous_yaw_setpoints + self.simulation_dt * self.yaw_rate
        constrained_yaw_setpoints = np.clip(new_yaw_setpoints, lb, ub)
        
        self.previous_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        constrained_yaw_setpoints = np.mod(self.previous_yaw_setpoints, 360)
        
        self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)}
        
        # Store prediction information if available
        if hasattr(self, 'wind_forecast') and self.wind_forecast:
            self.controls_dict["predicted_wind_speeds"] = None  # Could be enhanced to store sample statistics
        
        return None