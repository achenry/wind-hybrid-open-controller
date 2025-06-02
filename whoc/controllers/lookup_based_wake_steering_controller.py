# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import numpy as np
import pandas as pd
import os
from datetime import timedelta
import re
import polars as pl
import polars.selectors as cs
from memory_profiler import profile

from whoc.controllers.controller_base import ControllerBase
from floris.floris_model import FlorisModel
from floris.uncertain_floris_model import UncertainFlorisModel

from scipy.interpolate import LinearNDInterpolator
from scipy.signal import lfilter

from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
warnings.simplefilter("error", category=FutureWarning)

class LookupBasedWakeSteeringController(ControllerBase):
    def __init__(self, interface, wind_forecast, simulation_input_dict, verbose=False, **kwargs):
        super().__init__(interface, verbose=verbose)
        self.init_time = interface.init_time
        self.wind_forecast = wind_forecast
        self.simulation_dt = simulation_input_dict["simulation_dt"]
        self.controller_dt = simulation_input_dict["controller"]["controller_dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = interface.n_turbines #simulation_input_dict["controller"]["num_turbines"]
        self.fi = interface.env
        
        self.prediction_timedelta_stored = max(pd.Timedelta(self.controller_dt, unit="s"), self.wind_forecast.prediction_timedelta)
        
        # self.filtered_measurements = pd.DataFrame(columns=["time"] + [f"ws_horz_{tid}" for tid in range(self.n_turbines)] + [f"ws_vert_{tid}" for tid in range(self.n_turbines)], dtype=pd.Float64Dtype())
        # self.ws_lpf_alpha = np.exp(-simulation_input_dict["controller"]["ws_lpf_omega_c"] * simulation_input_dict["controller"]["lpf_T"])
        self.wind_dir_use_filt = simulation_input_dict["controller"]["use_lut_filtered_wind_dir"]
        self.wind_mag_use_filt = simulation_input_dict["controller"]["use_lut_filtered_wind_mag"]
        self.wind_dir_lpf_time_const = simulation_input_dict["controller"]["wind_dir_lpf_time_const"]
        self.wind_mag_lpf_time_const = simulation_input_dict["controller"]["wind_mag_lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=simulation_input_dict["controller"]["lpf_start_time"])        
        self.wind_dir_lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["wind_dir_lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.wind_mag_lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["wind_mag_lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.deadband_thr = simulation_input_dict["controller"]["deadband_thr"]
        self.interpolation_method = simulation_input_dict["controller"]["interpolation_method"]
        # self.deadband_thr = 0
        self.floris_input_file = simulation_input_dict["controller"]["floris_input_file"]
        self.yaw_limits = simulation_input_dict["controller"]["yaw_limits"]
        self.yaw_rate = simulation_input_dict["controller"]["yaw_rate"]
        self.yaw_increment = simulation_input_dict["controller"]["yaw_increment"]
        self.max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 16
        self.rated_turbine_power = simulation_input_dict["controller"]["rated_turbine_power"]
        self.wind_field_ts = kwargs["wind_field_ts"]
        # logging.info(f"id(wind_field_ts) in LUTController __init__ is {id(self.wind_field_ts)}")
        self.wf_source = kwargs["wf_source"]
        self.use_upstream_wind = simulation_input_dict["controller"]["use_upstream_wind"]
        self.previous_yaw_setpoints = None
        
        self.turbine_signature = kwargs["turbine_signature"]
        self.tid2idx_mapping = kwargs["tid2idx_mapping"]
        self.idx2tid_mapping = dict([(i, k) for i, k in enumerate(self.tid2idx_mapping.keys())])
        self.target_turbine_indices = simulation_input_dict["controller"]["target_turbine_indices"]
        
        if self.target_turbine_indices != "all":
            self.sorted_tids = sorted(list(self.target_turbine_indices))
        else:
            self.sorted_tids = np.arange(len(self.tid2idx_mapping))
            
        self.uncertain = simulation_input_dict["controller"]["uncertain"]
        
        self.ws_horz_cols = [f"ws_horz_{tid}" for tid in self.tid2idx_mapping]
        self.ws_vert_cols = [f"ws_vert_{tid}" for tid in self.tid2idx_mapping]
        self.nd_sin_cols = [f"nd_sin_{tid}" for tid in self.tid2idx_mapping]
        self.nd_cos_cols = [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
        if self.wind_forecast and self.uncertain:
            self.mean_ws_horz_cols = [f"loc_ws_horz_{tid}" for tid in self.tid2idx_mapping]
            self.mean_ws_vert_cols = [f"loc_ws_vert_{tid}" for tid in self.tid2idx_mapping]
            self.sd_ws_horz_cols = [f"sd_ws_horz_{tid}" for tid in self.tid2idx_mapping]
            self.sd_ws_vert_cols = [f"sd_ws_vert_{tid}" for tid in self.tid2idx_mapping]
            
            self.target_mean_ws_horz_cols = [f"loc_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
            self.target_mean_ws_vert_cols = [f"loc_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
            self.target_sd_ws_horz_cols = [f"sd_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
            self.target_sd_ws_vert_cols = [f"sd_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
        else:
            self.mean_ws_horz_cols = [f"ws_horz_{tid}" for tid in self.idx2tid_mapping.values()]
            self.mean_ws_vert_cols = [f"ws_vert_{tid}" for tid in self.idx2tid_mapping.values()]
            self.sd_ws_horz_cols = self.sd_ws_vert_cols = self.target_sd_ws_horz_cols = self.target_sd_ws_horz_cols = []
            
            self.target_mean_ws_horz_cols = [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
            self.target_mean_ws_vert_cols = [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in self.sorted_tids]
         
        self.tgt_turbine_indices = list(self.tid2idx_mapping.values())
        self.tgt_turbine_indices = [self.tgt_turbine_indices.index(i) for i in self.sorted_tids] 
         
        self.historic_measurements = None

        self._last_measured_time = None
        self.is_yawing = np.array([False for _ in range(self.n_turbines)])

        # Handle yaw optimizer object
        if "df_yaw" in kwargs:
            self.wake_steering_interpolant = get_yaw_angles_interpolant(kwargs["df_yaw"])
        else:
            # optimize, unless passed existing lookup table
            # os.path.abspath(lut_path)
            # this is generated for new layout if len(target_turbine_ids) < n_turbines
            self.wake_steering_interpolant = LookupBasedWakeSteeringController._optimize_lookup_table(
                floris_config_path=self.floris_input_file, 
                lut_path=simulation_input_dict["controller"]["lut_path"], 
                yaw_limits=self.yaw_limits,
                generate_lut=simulation_input_dict["controller"]["generate_lut"],
                uncertain=self.uncertain, 
                sorted_target_tids=self.sorted_tids if self.target_turbine_indices != "all" else "all",
                parallel=kwargs["multiprocessor"] is not None)
        
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

        # For startup
        self.previous_target_yaw_setpoints = self.controls_dict["yaw_angles"]
        self.yaw_norm_const = 360.0
    
    def _first_ord_filter(self, x, alpha):
        b = [1 - alpha]
        a = [1, -alpha]
        return lfilter(b, a, x)
    
    
    @staticmethod
    def _optimize_lookup_table(floris_config_path, uncertain, yaw_limits, parallel=False, optimization="scipy", sorted_target_tids="all", lut_path=None, generate_lut=True):
        if not generate_lut and lut_path is not None and os.path.exists(lut_path):
            df_lut = pd.read_csv(lut_path, index_col=0)
            df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
            
            # start LUT inspection code
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            
            # factor = 3
            # # factor = 3.0 # single column
            # plt.rc('font', size=12*factor)          # controls default text sizes
            # plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
            # plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
            # plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
            # plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
            # plt.rc('legend', fontsize=12*factor)    # legend fontsize
            # plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize
            
            # # for dynamic lut case
            # find suspicious negative offsets
            # df_plot.loc[(df_plot["Turbine"] == 1) & (df_plot["YawOffset"] < -1) & (df_plot["wind_direction"] > 142.5), :]
            # lut_path = lut_path.replace("uncertainFalse", "uncertainTrue")
            # df_lut = pd.read_csv(lut_path, index_col=0)
            # df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
            # # # df_lut.loc[(df_lut["wind_speed"].isin(pd.unique(df_lut["wind_speed"]))) & (df_lut["wind_direction"].isin(pd.unique(df_lut["wind_direction"]))), "yaw_angles_opt"]
            # df_plot = df_lut.drop(columns=["yaw_angles_opt", "farm_power_opt", "farm_power_baseline"])
            # yaw_angles_opt = np.vstack(df_lut["yaw_angles_opt"].values)
            # df_plot = pd.concat([df_plot.assign(YawOffset=yaw_angles_opt[:, i], Turbine=i) for i in range(yaw_angles_opt.shape[1])], axis=0)
            # df_plot.loc[(df_plot["wind_direction"] > 150) & (df_plot["YawOffset"] < 0) & (df_plot["Turbine"] == 1), :]
            
            # ax = sns.lineplot(df_plot.loc[df_plot["YawOffset"] != 0, :], x="wind_direction", y="YawOffset", 
                            #   hue="wd_stddev", style="Turbine", 
                            # #   estimator=lambda arr: max(arr.min(), arr.max(), key=abs),
                            # # estimator=lambda arr: np.mean(arr[arr != 0]),
                            # estimator=lambda arr: np.sign(arr.values[np.argmax(np.abs(arr))]) * np.abs(arr).max(),
                            # #  errorbar=lambda x: (x.min(), x.max())
                            # #  estimator="median",
                            # errorbar=("pi", 95)
                            # )
            # df_plot.loc[(df_plot["Turbine"] == 1) & (df_plot["YawOffset"] < 0.0) & (df_plot["wind_direction"] > 141.0), :]
            
            # cond = (df_plot["Turbine"] == 0) & (df_plot["wind_speed"] == 5.0)
            # ax.plot(df_plot.loc[cond, "wind_direction"], df_plot.loc[cond, "YawOffset"], color="black")
            # h, l = ax.get_legend_handles_labels()
            # l[0] = "Wind Direction \nStandard Deviation ($^\\circ$)"
            # l[1] = "Static LUT"
            
            # ax.set_xlabel("Wind Direction ($^\\circ$)")
            # ax.set_ylabel("Yaw Offset ($^\\circ$)")

            # l = [ll[:-2] if ".0" in ll else ll for ll in l]
            # l[-2] = "Downstream"
            # l[-1] = "Upstream"
            # ax.legend(h, l, loc="upper right", bbox_to_anchor=(1.0, 1.05))
            # ax.set_xlim((110, 220))
            # plt.tight_layout()
            # plt.savefig(os.path.join(os.path.dirname(lut_path), "uncertain_lut_reduced.png"))
            
            # for static lut case
            # ax = sns.lineplot(df_plot, x="wind_direction", hue="wind_speed", y="YawOffset", style="Turbine")
            # # ax.legend(bbox_to_anchor=(1, 0.95), loc="upper left")
                        # h, l = ax.get_legend_handles_labels()
            # l[0] = "Wind Speed (m/s)"
            
            # ax.legend(h, l, bbox_to_anchor=(1.0, 1.01), loc="upper left")

            # ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.125, 0.12))
            # ax.legend(h, l, loc="upper right", bbox_to_anchor=(1.0, 0.95))
            # ax.set_xlim((0, 360))

            # df_plot = df_plot.groupby(["wind_speed", "wd_stddev"]).agg("mean")
            # end LUT inspection code
        else:
            # if csv to load from is not given, optimize
            # LUT optimizer wind field options 
            wind_directions_lut = np.arange(0.0, 360.0, 3.0)
            # wind_directions_lut = np.arange(0.0, 360.0, 60.0)
            wind_speeds_lut = np.arange(6.0, 22.0, 2.0)
            # wind_speeds_lut = np.array([8])
            
            ## Get optimized AEP, with wake steering
            
            # Load a FLORIS object for yaw optimization, adapting to target_turbine_ids
            # for uncertain case, add extra dimension for standard deviation of wind dir
            if uncertain:
                # wd_stddevs_lut = np.arange(1.0, 10.0, 2.0)
                wd_stddevs_lut = np.arange(0.0, 8.0, 2.0)
                # wd_stddevs_lut = np.arange(0.0, 10.0, 5.0)
                fi_lut = UncertainFlorisModel(floris_config_path,
                                                wd_resolution=0.5,
                                                ws_resolution=0.5,
                                                ti_resolution=0.01,
                                                yaw_resolution=0.5,
                                                power_setpoint_resolution=100,
                                                wd_std=3)
                # wd_grid, ws_grid, wds_grid = np.meshgrid(wind_directions_lut, wind_speeds_lut, wd_stddevs_lut, indexing="ij")
                
            else:
                fi_lut = FlorisModel(floris_config_path)  # GCH model matched to the default "legacy_gauss" of V2
            
            wd_grid, ws_grid = np.meshgrid(wind_directions_lut, wind_speeds_lut, indexing="ij")
            
            if sorted_target_tids != "all":
                fi_lut.set(layout_x=fi_lut.layout_x[sorted_target_tids], 
                           layout_y=fi_lut.layout_y[sorted_target_tids])
            
            if uncertain:
                # TESTING START check negative yaw offsets for wd=[159, 162, 162, 165, 165], ws=[10, 8, 10, 6, 8], wd_stddev=[2, 4, 6, 6, 6]
                # wind_directions_lut = np.array([162])
                # wind_speeds_lut = np.array([8])
                # wd_stddevs_lut = np.array([4])
                
                # fi_lut.set(
                #     wind_directions=wind_directions_lut,
                #     wind_speeds=wind_speeds_lut,
                #     wd_stddevs=wd_stddevs_lut,
                #     turbulence_intensities=[fi_lut.core.flow_field.turbulence_intensities[0]] * len(wind_speeds_lut)
                # )
                # TESTING END
                
                fi_lut.set(
                    wind_directions=wd_grid.flatten(),
                    wind_speeds=ws_grid.flatten(),
                    wd_stddevs=wd_stddevs_lut,
                    turbulence_intensities=[fi_lut.core.flow_field.turbulence_intensities[0]] * len(ws_grid.flatten())
                )
            else:
                fi_lut.set(
                        wind_directions=wd_grid.flatten(),
                        wind_speeds=ws_grid.flatten(),
                        turbulence_intensities=[fi_lut.core.flow_field.turbulence_intensities[0]] * len(ws_grid.flatten())
                    )
                
            # fi_lut.run()
            # turbine_powers = fi_lut.get_turbine_powers(per_wd_sample=True)
             
            if optimization == "scipy":
                yaw_opt = YawOptimizationScipy(fi_lut, 
                                            minimum_yaw_angle=yaw_limits[0],
                                            maximum_yaw_angle=yaw_limits[1], parallel=parallel,
                                            include_wd_stddev=uncertain)
                                            # opt_options={"maxiter": 100, "disp": True, 
                                            #              "iprint": 2,
                                            #              "ftol": 1e-12, "eps": 0.1})
            #     {
            #     "maxiter": 100,
            #     "disp": True,
            #     "iprint": 2,
            #     "ftol": 1e-12,
            #     "eps": 0.1,
            # }
            elif optimization == "sr":
                # TODO update this based on MPC implementation
                yaw_opt = YawOptimizationSR(fi_lut, 
                                            minimum_yaw_angle=yaw_limits[0],
                                            maximum_yaw_angle=yaw_limits[1],
                                            include_wd_stddev=uncertain)
            else:
                raise TypeError("optimization argument must equal 'scipy' or 'sr'")
            df_lut = yaw_opt.optimize()
            
            # Assume linear ramp up at 5-6 m/s and ramp down at 13-14 m/s,
            # add to table for linear interpolant
            df_copy_lb = df_lut[df_lut["wind_speed"] == 6.0].copy()
            df_copy_ub = df_lut[df_lut["wind_speed"] == 13.0].copy()
            df_copy_lb["wind_speed"] = 5.0
            df_copy_ub["wind_speed"] = 14.0
            df_copy_lb["yaw_angles_opt"] *= 0.0
            df_copy_ub["yaw_angles_opt"] *= 0.0
            df_lut = pd.concat([df_copy_lb, df_lut, df_copy_ub], axis=0).reset_index(drop=True)
            
            # Deal with 360 deg wrapping: solutions at 0 deg are also solutions at 360 deg
            df_copy_360deg = df_lut[df_lut["wind_direction"] == 0.0].copy()
            df_copy_360deg["wind_direction"] = 360.0
            df_lut = pd.concat([df_lut, df_copy_360deg], axis=0).reset_index(drop=True)
            # ['wind_direction', 'wind_speed', 'turbulence_intensity',
            #        'yaw_angles_opt', 'farm_power_opt', 'farm_power_baseline']
            
            os.makedirs(os.path.dirname(lut_path), exist_ok=True)
            if lut_path is not None:
                df_lut.to_csv(lut_path)

        # pd.unique(df_lut.iloc[np.where(np.any(np.vstack(df_lut["yaw_angles_opt"].array) != 0, axis=1))[0]]["wind_direction"])
        # Derive linear interpolant from solution space
        if uncertain:
            return LinearNDInterpolator(
                points=df_lut[["wind_direction", "wind_speed", "wd_stddev"]].values,
                values=np.vstack(df_lut["yaw_angles_opt"].values),
                fill_value=0.0,
            )
        else:
            return LinearNDInterpolator(
                points=df_lut[["wind_direction", "wind_speed"]].values,
                values=np.vstack(df_lut["yaw_angles_opt"].values),
                fill_value=0.0,
            )
    
    # @profile
    def compute_controls(self):
        # TODO update LUT for turbine breakdown
        # TODO: move data collection for filtering purposes to another method that is called every simulation_dt, also move constraints on yaw angles and incremental updates as per yaw_rate to simulator, only job of compute_controls should be to compute new yaw angles for turbines that are not in motion
        
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        # if self.verbose:
        #     logging.info(f"self._last_measured_time == {self._last_measured_time}")
        #     logging.info(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self.current_time = self._last_measured_time = self.measurements_dict["time"]

        if self.wf_source == "floris":
            current_wind_directions = self.measurements_dict["wind_directions"]
            current_wind_magnitudes = self.measurements_dict["wind_speeds"]
            # current_farm_wind_direction = self.measurements_dict["amr_wind_direction"]
            # current_farm_wind_speed = self.measurements_dict["amr_wind_speed"]
            current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
            current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
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
            current_wind_magnitudes = np.sqrt(current_ws_horz**2 + current_ws_vert**2)

        # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
        if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
            # yaw angles will be set to initial values
            if self.verbose:
                logging.info("Bad wind direction measurement received, reverting to previous measurement.")
        
        # NOTE: current_measurements collects measurements corresponding to current time step, NOT since last controller call if wind_dt < controller_dt
        # pass greedy angles to all non target turbines
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
        current_wind_magnitudes = current_wind_magnitudes[self.tgt_turbine_indices]
        
        # need historic measurements for filter or for wind forecast
        if self.wind_dir_use_filt or self.wind_mag_use_filt or self.wind_forecast:
            if self.historic_measurements is not None:
                self.historic_measurements = pl.concat([self.historic_measurements, 
                                                        current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
                                                        ], 
                                                       how="vertical")\
                                                            .tail(max(int(np.ceil(
                                                                max(self.wind_dir_lpf_time_const, self.wind_mag_lpf_time_const) 
                                                                // self.simulation_dt) * 50), self.wind_forecast.n_context))
            else:
                self.historic_measurements = current_measurements.select(["time"] + self.ws_horz_cols + self.ws_vert_cols + self.nd_cos_cols + self.nd_sin_cols)
                
        current_yaw_setpoints = self.controls_dict["yaw_angles"]
        if self.previous_yaw_setpoints is None:
            self.previous_yaw_setpoints = current_yaw_setpoints.copy()
        
        # flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
        reached_setpoints_cond = self.is_yawing & (current_yaw_setpoints == np.mod(self.previous_target_yaw_setpoints, 360))
        if self.verbose and any(reached_setpoints_cond):
            logging.info(f"LUT Controller turbines {np.where(reached_setpoints_cond)[0]} have reached their target setpoint of {self.previous_target_yaw_setpoints[reached_setpoints_cond]} at time {self.current_time}")
        
        self.is_yawing[reached_setpoints_cond] = False

        new_yaw_setpoints = np.array(current_yaw_setpoints)
        
        use_wind_forecast = False
        forecasted_wind_field = None
        single_forecasted_wind_field = None
        
        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            if self.wind_forecast and self.wind_forecast.prediction_timedelta.total_seconds() > 0:
                if self.uncertain:
                    forecasted_wind_field = self.wind_forecast.predict_distr(self.historic_measurements, self.current_time)
                else:
                    forecasted_wind_field = self.wind_forecast.predict_point(self.historic_measurements, self.current_time)
                
                forecasted_wind_field = forecasted_wind_field.with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")), cs.numeric().cast(pl.Float32))
                single_forecasted_wind_field = forecasted_wind_field.filter(pl.col("time") == self.current_time + self.wind_forecast.prediction_timedelta)
                
                use_wind_forecast = True
            
            # just hold initial yaw setpoints
            if (self.current_time < self.lpf_start_time) or not (self.wind_dir_use_filt or self.wind_mag_use_filt):
                #pass
                wind = single_forecasted_wind_field if use_wind_forecast else current_measurements.select("time", cs.starts_with("ws_"))
                
                wind_u = wind.select(self.target_mean_ws_horz_cols).to_numpy()[-1, :]
                wind_v = wind.select(self.target_mean_ws_vert_cols).to_numpy()[-1, :]
                
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(wind_u, wind_v))
                
                wind_mags = (wind_u**2 + wind_v**2)**0.5
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"unfiltered forecasted wind directions = {wind_dirs}")
                        logging.info(f"unfiltered forecasted wind directions = {wind_mags}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                        logging.info(f"unfiltered current wind directions = {current_wind_magnitudes}")
                
            else:
                # use filtered wind direction, NOTE historic_measurements includes controller_dt steps into the future such that we can run simulation in time batches
                # forecasted_wind_field.iloc[-1:].rename(columns={old_col: re.search("(?<=loc_)\\w+", old_col).group(0) for old_col in self.mean_ws_horz_cols+self.mean_ws_vert_cols})
                
                # NOTE for forecasts which don't provide continuous predictions from current time onwards, we need to interpolate for the filter
                # alternatively could adapt the filter to only consider measurments on the same scale as the forecaster
                # or only filter the historic measurements and not the forecasted ones
                if use_wind_forecast:
                    # TODO should also interpolate if prediction is multistep
                    hist_meas = self.historic_measurements.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols)
                    hist_meas = hist_meas.rename({re.search("(?<=loc_)\\w+", new_col).group(0): new_col for new_col in self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols}) if self.uncertain else hist_meas
                    last_historic_time = hist_meas.select(pl.col("time").last()).item()
                    first_forecasted_time = forecasted_wind_field.select(pl.col("time").first()).item()
                    
                    # if forecast is multistep, need to interpolate to sim_timedelta
                    # if (forecasted_wind_field.select(pl.len()).item() > 1) and forecasted_wind_field.select(pl.col("time").diff().slice(1).max()).item() > self.simulation_dt:
                    # TODO this assumes that forecast either has a single value, or once it starts has sim_dt time intervals
                    
                    if (fcst_lead_timedelta := (first_forecasted_time - last_historic_time)) > (sim_timedelta := timedelta(seconds=self.simulation_dt)):
                        missing_forecasted_time = pl.DataFrame({"time": [last_historic_time + i * sim_timedelta for i in range(1, int(fcst_lead_timedelta / sim_timedelta))]}).with_columns(pl.col("time").cast(pl.Datetime(time_unit="ns")))
                        wind = pl.concat([
                            hist_meas,
                            missing_forecasted_time, 
                            forecasted_wind_field.select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols)], how="diagonal")\
                             .select(pl.col("time"), cs.numeric().interpolate(self.interpolation_method))
                    else:
                        wind = pl.concat([hist_meas, 
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
                        unfilt_wind_mags = (wind_u**2 + wind_v**2)**0.5
                        logging.info(f"unfiltered forecasted wind directions = {unfilt_wind_dirs[-1, :]}")
                        logging.info(f"unfiltered forecasted wind magnitudes = {unfilt_wind_mags[-1, :]}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                        logging.info(f"unfiltered current wind magnitudes = {current_wind_magnitudes}")
                               
                if self.wind_mag_use_filt:
                    wind_u = np.array([self._first_ord_filter(wind_u[:, i], self.wind_mag_lpf_alpha)
                                                    for i in range(len(self.sorted_tids))]).T # [-int(self.controller_dt // self.simulation_dt), :]
                    wind_v = np.array([self._first_ord_filter(wind_v[:, i], self.wind_mag_lpf_alpha)
                                                    for i in range(len(self.sorted_tids))]).T
                wind_u = wind_u[-1, :]
                wind_v = wind_v[-1, :]
                
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(wind_u, wind_v))
                wind_mags = (wind_u**2 + wind_v**2)**0.5
                
                if self.verbose:
                    logging.info(f"filtered {'forecasted' if self.wind_forecast else 'current'} wind directions = {wind_dirs}")
                    logging.info(f"filtered {'forecasted' if self.wind_forecast else 'current'} wind magnitudes = {wind_mags}")
                
            if self.uncertain:
                ws_horz_stddevs = single_forecasted_wind_field.select(self.target_sd_ws_horz_cols).slice(0, 1).to_numpy()[0, :]
                ws_vert_stddevs = single_forecasted_wind_field.select(self.target_sd_ws_vert_cols).slice(0, 1).to_numpy()[0, :]
                forecasted_wind_norm = (single_forecasted_wind_field.select(self.target_mean_ws_horz_cols).slice(0, 1).to_numpy()[0, :]**2 
                                        + single_forecasted_wind_field.select(self.target_mean_ws_vert_cols).slice(0, 1).to_numpy()[0, :]**2)
                c1 = single_forecasted_wind_field.select(self.target_mean_ws_vert_cols).slice(0, 1).to_numpy()[0, :] / forecasted_wind_norm
                c2 = -single_forecasted_wind_field.select(self.target_mean_ws_horz_cols).slice(0, 1).to_numpy()[0, :] / forecasted_wind_norm
                wind_dir_stddevs = np.rad2deg(((c1 * ws_horz_stddevs)**2 + (c2 * ws_vert_stddevs)**2)**0.5) # to degrees

                if self.verbose:
                    logging.info(f"min wd_stdev = {min(wind_dir_stddevs)}, mean wd_stdev = {np.mean(wind_dir_stddevs)}, max wd_stdev = {max(wind_dir_stddevs)}")
                # logging.info(f"min ws_horz_stdevs = {min(ws_horz_stddevs)}, mean ws_horz_stddevs = {np.mean(ws_horz_stddevs)}, max ws_horz_stddevs = {max(ws_horz_stddevs)}")
                # logging.info(f"min ws_vert_stdevs = {min(ws_vert_stddevs)}, mean ws_vert_stdevs = {np.mean(ws_vert_stddevs)}, max ws_vert_stdevs = {max(ws_vert_stddevs)}")
           
            if self.target_turbine_indices == "all":
                wd_inp, wm_inp, wd_stddev_inp = wind_dirs.mean(), wind_mags.mean(), wind_dir_stddevs.mean() if self.uncertain else None
            else:
                # feeds wind from feed just upstream turbine to LUT, could also do mean
                # if self.use_upstream_wind:
                if len(self.target_turbine_indices) == 2:
                    # upstream_turbine_idx = np.argsort(self.target_turbine_indices)[0]
                    
                    # rotate turbine coordinates based on most recent wind direction measurement
                    # order turbines based on order of wind incidence
                    layout_x = self.fi.layout_x
                    layout_y = self.fi.layout_y
                    
                    wd = (180.0 + np.rad2deg(np.arctan2(wind_u.mean(), wind_v.mean()))).mean()
                    layout_x_rot = (
                        np.cos(np.deg2rad(wd + 180.0)) * layout_y
                        + np.sin(np.deg2rad(wd + 180.0)) * layout_x
                    )
                    upstream_turbine_idx = np.argsort(layout_x_rot)[0]
                    logging.info(f"Using turbine idx {upstream_turbine_idx} as upstream turbine for input measurements to LUT at time {self.current_time}.")
                    # if len(self.target_turbine_indices) > 2:
                    #     logging.warning("There are more than 2 target turbines under study, but only the upstream wind direction is being used.")
                    wd_inp, wm_inp, wd_stddev_inp = wind_dirs[upstream_turbine_idx], wind_mags[upstream_turbine_idx], wind_dir_stddevs[upstream_turbine_idx] if self.uncertain else None
                else:
                    logging.info(f"Using mean turbine measurements as input to LUT at time {self.current_time}.")
                    wd_inp, wm_inp, wd_stddev_inp = wind_dirs.mean(), wind_mags.mean(), wind_dir_stddevs.mean() if self.uncertain else None
            if self.uncertain:
                target_yaw_offsets = self.wake_steering_interpolant(
                    wd_inp, wm_inp, np.clip(wd_stddev_inp, self.wake_steering_interpolant.points[:, 2].min(), self.wake_steering_interpolant.points[:, 2].max()))
            else:
                target_yaw_offsets = self.wake_steering_interpolant(wd_inp, wm_inp)
            
            target_yaw_setpoints = np.mod(np.rint((wind_dirs - target_yaw_offsets) / self.yaw_increment) * self.yaw_increment, 360.0)
            
            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            setpoint_change = target_yaw_setpoints - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)]) 
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0) # if == 0, need to change within 360 deg, otherwise if == 1 faster to cross 360/0 boundary
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(self.n_turbines)]
            is_target_changing = (abs_setpoint_change > self.deadband_thr) & ~self.is_yawing
            
            # setpoint_change[setpoint_change_idx == 0] = setpoint_change[setpoint_change_idx == 0]
            # if ((setpoint_change_idx == 1) & ((target_yaw_setpoints - current_yaw_setpoints) > 0))
            dir_setpoint_change = np.sign(setpoint_change)
            dir_setpoint_change[setpoint_change_idx == 1] = -dir_setpoint_change[setpoint_change_idx == 1]
            new_yaw_setpoints[is_target_changing] = new_yaw_setpoints[is_target_changing] + dir_setpoint_change[is_target_changing] * abs_setpoint_change[is_target_changing]
            self.is_yawing[is_target_changing] = True
            
            if self.verbose and any(is_target_changing):
                logging.info(f"LUT Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} in direction {dir_setpoint_change[is_target_changing]} at time {self.current_time}") 
        else:
            is_target_changing = np.zeros_like(self.is_yawing).astype(bool)
        
        reaching_setpoints_cond = self.is_yawing & ~is_target_changing
        if self.verbose and any(reaching_setpoints_cond):
            # from [351.15 320.2 ] to [20.65 12.4 ] 
            logging.info(f"LUT Controller continuing to yaw turbines {np.where(reaching_setpoints_cond)[0]} from {current_yaw_setpoints[reaching_setpoints_cond]} to {self.previous_target_yaw_setpoints[reaching_setpoints_cond]} at time {self.current_time}")
        
        # else:
        # 	logging.info(f"LUT Controller current_setpoints = {current_yaw_setpoints}, \n previous_target_yaw_setpoints = {self.previous_target_yaw_setpoints}, \n target_setpoints={target_yaw_setpoints}")
        
        new_yaw_setpoints[reaching_setpoints_cond] = self.previous_target_yaw_setpoints[reaching_setpoints_cond].copy()
        
        # stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
        self.previous_target_yaw_setpoints = np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        
        lb, ub = self.previous_yaw_setpoints - self.simulation_dt * self.yaw_rate, self.previous_yaw_setpoints + self.simulation_dt * self.yaw_rate
        
        constrained_yaw_setpoints = np.clip(new_yaw_setpoints, lb, ub)
        # if np.all(np.diff(constrained_yaw_setpoints) == 0) and not np.all(np.diff(new_yaw_setpoints) == 0):
        # 	logging.info(f"Note: all yaw angles have been constrained by the yaw rate equally at time {self.current_time}")
        
        # if not len(is_target_changing):
        # 	logging.info(f"Note: no yaw angle setpoints surpass the deadband threshold at time {self.current_time}")

        self.previous_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        constrained_yaw_setpoints = np.mod(self.previous_yaw_setpoints, 360)
        
        self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)} 
        if self.wind_forecast:
            # wf.filter(pl.col("time") < pl.col("time").first() + preview_forecast.controller_timedelta)
            if use_wind_forecast:
                # newest_predictions = forecasted_wind_field.filter(pl.col("time") <= self.current_time + self.prediction_timedelta_stored)\
                newest_predictions = forecasted_wind_field.filter(pl.col("time") == self.current_time + self.wind_forecast.prediction_timedelta)\
                                                        .select(["time"] + self.target_mean_ws_horz_cols + self.target_mean_ws_vert_cols 
                                                                + ((self.target_sd_ws_horz_cols + self.target_sd_ws_vert_cols) if self.uncertain else []))
            else:
                newest_predictions = None
            # print(newest_predictions)
            self.controls_dict["predicted_wind_speeds"] = newest_predictions
            
        return None

def get_yaw_angles_interpolant(df_opt, ramp_up_ws=[4, 5], ramp_down_ws=[10, 12], minimum_yaw_angle=None, maximum_yaw_angle=None):
    """Create an interpolant for the optimal yaw angles from a dataframe
    'df_opt', which contains the rows 'wind_direction', 'wind_speed',
    'turbulence_intensity', and 'yaw_angles_opt'. This dataframe is typically
    produced automatically from a FLORIS yaw optimization using Serial Refine
    or SciPy. One can additionally apply a ramp-up and ramp-down region
    to transition between non-wake-steering and wake-steering operation.

    Args:
        df_opt (pd.DataFrame): Dataframe containing the rows 'wind_direction',
        'wind_speed', 'turbulence_intensity', and 'yaw_angles_opt'.
        ramp_up_ws (list, optional): List with length 2 depicting the wind
        speeds at which the ramp starts and ends, respectively, on the lower
        end. This variable defaults to [4, 5], meaning that the yaw offsets are
        zero at and below 4 m/s, then linearly transition to their full offsets
        at 5 m/s, and continue to be their full offsets past 5 m/s. Defaults to
        [4, 5].
        ramp_down_ws (list, optional): List with length 2 depicting the wind
        speeds at which the ramp starts and ends, respectively, on the higher
        end. This variable defaults to [10, 12], meaning that the yaw offsets are
        full at and below 10 m/s, then linearly transition to zero offsets
        at 12 m/s, and continue to be zero past 12 m/s. Defaults to [10, 12].

    Returns:
        LinearNDInterpolator: An interpolant function which takes the inputs
        (wind_directions, wind_speeds, turbulence_intensities), all of equal
        dimensions, and returns the yaw angles for all turbines. This function
        incorporates the ramp-up and ramp-down regions.
    """

    # Load data and set up a linear interpolant
    points = df_opt[["wind_direction", "wind_speed", "turbulence_intensity"]]
    values = np.vstack(df_opt["yaw_angles_opt"])

    # Derive maximum and minimum yaw angle (deg)
    if minimum_yaw_angle is None:
        minimum_yaw_angle = np.min(values)
    if maximum_yaw_angle is None:
        maximum_yaw_angle = np.max(values)

    # Expand wind direction range to cover 0 deg to 360 deg
    points_copied = points[points["wind_direction"] == 0.0].copy()
    points_copied.loc[points_copied.index, "wind_direction"] = 360.0
    values_copied = values[points["wind_direction"] == 0.0, :]
    points = np.vstack([points, points_copied])
    values = np.vstack([values, values_copied])

    # Copy lowest wind speed / TI solutions to -1.0 to create lower bound
    for col in [1, 2]:
        ids_to_copy_lb = points[:, col] == np.min(points[:, col])
        points_copied = np.array(points[ids_to_copy_lb, :], copy=True)
        values_copied = np.array(values[ids_to_copy_lb, :], copy=True)
        points_copied[:, col] = -1.0  # Lower bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

        # Copy highest wind speed / TI solutions to 999.0
        ids_to_copy_ub = points[:, col] == np.max(points[:, col])
        points_copied = np.array(points[ids_to_copy_ub, :], copy=True)
        values_copied = np.array(values[ids_to_copy_ub, :], copy=True)
        points_copied[:, col] = 999.0  # Upper bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

    # Now create a linear interpolant for the yaw angles
    interpolant = LinearNDInterpolator(
        points=points,
        values=values,
        fill_value=np.nan
    )

    # Now create a wrapper function with ramp-up and ramp-down
    def interpolant_with_ramps(wd_array, ws_array, ti_array=None):
        # Deal with missing ti_array
        if ti_array is None:
            ti_ref = float(np.median(interpolant.points[:, 2]))
            ti_array = np.ones(np.shape(wd_array), dtype=float) * ti_ref

        # Format inputs
        wd_array = np.array(wd_array, dtype=float)
        ws_array = np.array(ws_array, dtype=float)
        ti_array = np.array(ti_array, dtype=float)
        yaw_angles = interpolant(wd_array, ws_array, ti_array)
        yaw_angles = np.array(yaw_angles, dtype=float)

        # Define ramp down factor
        rampdown_factor = np.interp(
            x=ws_array,
            xp=[0.0, *ramp_up_ws, *ramp_down_ws, 999.0],
            fp=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        )

        # Saturate yaw offsets to threshold
        axis = len(np.shape(yaw_angles)) - 1
        nturbs = np.shape(yaw_angles)[-1]
        yaw_lb = np.expand_dims(
            minimum_yaw_angle * rampdown_factor, axis=axis
        ).repeat(nturbs, axis=axis)
        yaw_ub = np.expand_dims(
            maximum_yaw_angle * rampdown_factor, axis=axis
        ).repeat(nturbs, axis=axis)

        return np.clip(yaw_angles, yaw_lb, yaw_ub)

    return interpolant_with_ramps