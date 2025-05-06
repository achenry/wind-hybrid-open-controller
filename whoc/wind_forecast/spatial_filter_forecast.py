from typing import Union
from dataclasses import dataclass
from memory_profiler import profile

import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np

from scipy.stats import multivariate_normal as mvn

from whoc.wind_forecast.wind_forecast_base import WindForecast

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SpatialFilterForecast(WindForecast):
    """
    Reads wind speed components from upstream turbines, and computes time of arrival based on taylor's frozen wake hypothesis.
    """
    is_probabilistic = False
    def __post_init__(self):
        super().__post_init__()
        self.train_first = False
        self.n_turbines = self.fmodel.n_turbines
        self.measurement_layout = np.vstack([self.fmodel.layout_x, self.fmodel.layout_y]).T
        
        # the turbines to consider in the spatial filtering for the estimation of the wind direction at each turbine
        self.n_neighboring_turbines = self.kwargs["n_neighboring_turbines"] 
        if self.n_neighboring_turbines:
            self.cluster_turbines = [sorted(np.arange(self.n_turbines), 
                        key=lambda t: np.linalg.norm(self.measurement_layout[tid, :] - self.measurement_layout[t, :]))[:self.n_neighboring_turbines]
                                    for tid in range(self.n_turbines)]
        else:
            self.cluster_turbines = [np.arange(self.n_turbines)] * self.n_turbines
    
    def reset(self, **kwargs):
        pass
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] # pred_slice.filter(pred_slice == current_time + self.prediction_timedelta)
        outputs = self._get_ws_cols(historic_measurements)
        
        # multivariate with state matrix containing all horizontal and vertical wind speed measurements
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        
        new_measurements = historic_measurements.slice(-1, 1)
        
        pred = self.full_farm_directional_weighted_average(new_measurements)
        
        # self.last_measurement_time = historic_measurements.select(pl.col("time").last()).item()
        
        pred = {output: pred[:, o] for o, output in enumerate(outputs)}
        pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
        
        if return_pl:
            return pred
        else:
            return pred.to_pandas()
    
    
    def full_farm_directional_weighted_average(
        self,
        new_measurements: Union[pd.DataFrame, pl.DataFrame]
        # data_in,
        # wind_speeds,
        # wind_directions,
        # shift_distance,
        # is_circular=False,
        # is_bearing=False,
    ):
        """_summary_
        QUESTION
        Args:
            data_in (pd.DataFrame): num_columns = n_turbines, and indices corresponding to time
            measurement_layout (np.ndarray): 0th dim = n_turbines, 1st dim = x,y coords
            wind_directions (np.ndarray): array of wind direction estimations for each turbine
            shift_distance (float): distance from turbine at which to estimate wind direction

        Returns:
            _type_: _description_
        """

        # nTurbs = len(data_in.columns)
        # turbine_list = np.arange(0, self.n_turbines)

        # if is_bearing:  # Convert to RH CCW angle
        #     wd_mean = SpatialFilterForecast.bearing2angle(wd_mean)
        #     data_in = data_in.applymap(SpatialFilterForecast.bearing2angle)
        # re.search("(?<=ws_horz_)\\d+", "ws_horz_1")
        turbine_ids = sorted(set(re.search("(?<=\\w_)\\d+$", col).group(0) for col in new_measurements.select(cs.starts_with("ws_")).columns), key=lambda tid: int(re.search("\\d+", tid).group(0)))
        ws_horz = new_measurements.select(cs.starts_with("ws_horz_"))\
                                  .rename(lambda old_col: re.search("(?<=\\w_)\\d+$", old_col).group(0))
        ws_vert = new_measurements.select(cs.starts_with("ws_vert_"))\
                                  .rename(lambda old_col: re.search("(?<=\\w_)\\d+$", old_col).group(0))
        
        wm = new_measurements.select(**{tid: ((pl.col(f"ws_horz_{tid}")**2 + pl.col(f"ws_vert_{tid}")**2).sqrt()) for tid in turbine_ids})
        wd = new_measurements.select(**{tid: 180.0 + (pl.arctan2(pl.col(f"ws_horz_{tid}"), pl.col(f"ws_vert_{tid}")).degrees()) for tid in turbine_ids})
        
        ws_inputs = np.dstack([ws_horz.select(turbine_ids).to_numpy(), ws_vert.select(turbine_ids).to_numpy()])
        
        # farm_wind_direction = wd.select(pl.mean_horizontal(pl.all())).to_numpy().flatten()
        weights = self.neighbor_weights_directional_gaussian(cluster_turbines=self.cluster_turbines, 
                                                             wind_dirs=wd.to_numpy(), wind_speeds=wm.to_numpy() ) #, shift_distance)
        
        pred = np.zeros_like(ws_inputs)
         
        for tid in range(self.n_turbines):
            idx = self.cluster_turbines[tid]
            pred[:, tid, :] = np.einsum("ntd,nt->nd", ws_inputs[:, idx, :], weights[tid])
        
        return pred.T.reshape((2*self.n_turbines, -1)).T 

    def neighbor_weights_directional_gaussian(
        self, cluster_turbines, wind_dirs, wind_speeds, #shift_distance=0
    ):
        """
        wd_mean should be in radians, CCW
        mu = 0: no sf
        sigma = None: will default to using the standard deviation of turbine distances.
        """

        weights = dict()
        # n_turbines = np.shape(measurement_layout)[0]
       
        for i in range(self.n_turbines):
            idx = cluster_turbines[i]
            cluster_wind_direction = np.mean(np.deg2rad(wind_dirs[:, idx]), axis=1)
            cluster_layout = self.measurement_layout[idx, :]
            shift_distance = self.prediction_timedelta.total_seconds() * wind_speeds[:, i]
            
            center_point = (self.measurement_layout[i, :] + np.array(
                [
                    -shift_distance * np.sin(np.pi + cluster_wind_direction), # -cos=
                    -shift_distance * np.cos(np.pi + cluster_wind_direction),
                ]
            ).T)

            # sigma defaults to using the standard deviation of turbine distances.
            covariance = np.var(
                np.linalg.norm(cluster_layout[np.newaxis, :, :] - center_point[:, np.newaxis, :], axis=2), axis=1
            )

            f = []
            for t in range(center_point.shape[0]):
                f.append(mvn.pdf(cluster_layout, mean=center_point[t, :], cov=covariance[t] * np.identity(2)))
            f = np.array(f)
            
            fsum = f.sum(axis=1)[:, np.newaxis]
            weights[i] = np.divide(f, fsum, out=np.zeros_like(f), where=(fsum!=0))
            
            if fsum == 0:
                logging.warning(f"The center point, determined by prediction_timedelta, is too far from turbine {i}'s clusters to have any nonzero weights, assuming persistence for turbine {i}.")
                weights[i][:, np.where(idx == i)[0]] = 1

        return weights

    @staticmethod
    def bearing2angle(bearing):
        """
        Convert bearing angle in degrees to CCW positive angle in radians.
        """
        return SpatialFilterForecast.wrap_angle(np.pi / 180 * (90 - bearing), -np.pi, np.pi)
    
    @staticmethod
    def angle2bearing(angle):
        return SpatialFilterForecast.wrap_angle(90 - 180 / np.pi * angle, 0.0, 360.0)

    @staticmethod
    def wrap_angle(x, low, high):
        """
        Wrap to x to interval [low, high)
        """
        x = float(x)
        step = high - low

        while x >= high:
            x = x - step
        while x < low:
            x = x + step

        return x

    @staticmethod
    def weighted_circular_mean3(x, weights):
        """
        Find weighted mean value of x after shifting all values to
        [low, high) (method 3).

        Inputs:
            x - data for mean (list, Series, or 1D array)
            weights - weights for mean (list, Series, or 1D array) (same
                    size as x)
        Outputs:
            (x_mean) - scalar circular mean of x
        """

        # Convert list, series to array
        if type(x) == np.ndarray:
            x = np.e ** (1j * x)
            complex_mean = list(x @ np.array(weights))
        else:
            x = np.array([np.e ** (1j * x[i]) for i in range(len(x))])
            complex_mean = x @ np.array(weights)

        return np.angle(complex_mean)
 

