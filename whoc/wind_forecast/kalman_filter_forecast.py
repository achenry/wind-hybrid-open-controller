
from typing import Union
from dataclasses import dataclass

import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl

from filterpy.kalman import KalmanFilter

from whoc.wind_forecast.wind_forecast_base import WindForecast

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_palette("Paired")

class KalmanFilterWrapper(KalmanFilter):
    def __init__(self, dim_x=1, dim_z=1, **kwargs): # TODO need this st cross validation will work in Optuna objective function, ideally clone could capture dim_x, dim_z automatically
        super().__init__(dim_x=dim_x, dim_z=dim_z)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X=None, **predict_kwargs):
        if X is None:
            super().predict(**predict_kwargs)
        else:
            pred = []
            self.x = X[:1]
            # for s in range(X.shape[0]): # loop over samples
                # pred.append([])
            for i in range(X.shape[0]): # loop over history and update matrices
                z = X[i:i+1]
                super().predict()
                self.update(z)
                pred.append(self.x[0])
            return np.array(pred)

@dataclass
class KalmanFilterForecast(WindForecast):
    """Wind speed component forecasting using Kalman filtering."""
    # def read_measurements(self):
    #     pass
    is_probabilistic = True
    def __post_init__(self):
        super().__post_init__()
        self.n_prediction_interval = self.n_prediction
        self.prediction_interval = self.n_prediction_interval * self.measurements_timedelta
        self.n_turbines = self.fmodel.n_turbines
        self.dim_x = self.dim_z = self.n_targets_per_turbine * self.n_turbines
        self.last_pred = None
        self.last_var = None
        self.scaler = self.create_scaler()
        self.reset()
        
        # equal to number of n_prediction intervals for the kalman filter
        self.n_context = int(self.context_timedelta / self.prediction_timedelta)
        assert self.n_context >= 2, "For KalmanFilterForecaster, context_timedelta must be at least 2 times prediction_timedelta, since prediction_timedelta is the time interval at which it makes new estimates"
      
    def reset(self, **kwargs):
        self.model = self.create_model(
            dim_x=self.dim_x, 
            dim_z=self.dim_z,
            F=np.eye(self.dim_x), # identity matrix predicts x_t = x_t-1 + Q_t
            H=np.eye(self.dim_z) # identity matrix predicts x_t = x_t-1 + Q_t
        )
        self.last_measurement_time = None
        # store last context of w_t = x_t - x_(t-1) and v_t = z_t - H_t x_t
        self.historic_w = np.zeros((0, self.dim_x))
        self.historic_v = np.zeros((0, self.dim_z))
        self.historic_times = []
        self.initialized = False
         
    def create_scaler(self):
        return None
    
    def create_model(self, dim_x, dim_z, **kwargs):
        model = KalmanFilterWrapper(dim_x=dim_x, dim_z=dim_z)
        if "F" in kwargs:
            model.F = kwargs["F"]
        if "R" in kwargs:
            model.R = kwargs["R"]
        if "H" in kwargs:
            model.H = kwargs["H"]
        if "Q" in kwargs:
            model.Q = kwargs["Q"]
        return model
    
    def _prepare_arrays(self, historic_measurements, scaler):
        if scaler:
            historic_measurements = scaler.fit_transform(historic_measurements.to_numpy())
            
        X_train = historic_measurements[0:-self.n_context, 0]
        y_train = historic_measurements[self.n_context:, 0]
        
        return X_train, y_train
    
    def get_params(self, trial):
        return {
            # "n_prediction": np.array([[trial.suggest_float("n_prediction", 1e-3, 1e2, log=True)]]),
            # "R": np.array([[trial.suggest_float("R", 1e-3, 1e2, log=True)]]),
            # # "H": np.array([[trial.suggest_float("H", 1e-3, 1e2, log=True)]]),
            # "Q": np.array([[trial.suggest_float("Q", 1e-3, 1e2, log=True)]])
        }
    
    def predict_sample(self, n_samples: int):
        pass
        
    def _init_covariance(self, historic_noise: np.ndarray):
        cov = np.diag((1 / (self.n_context - 1)) \
            * np.sum((historic_noise[-self.n_context:, :] 
                      - (1 / self.n_context) * historic_noise[-self.n_context:, :].sum(axis=0))**2, axis=0))
        return cov
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time, return_var=False):
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:] # pred_slice.filter(pred_slice == current_time + self.prediction_timedelta)
        outputs = self._get_ws_cols(historic_measurements)
        
        # multivariate with state matrix containing all horizontal and vertical wind speed measurements
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        
        # batch predict and update based on all measurements collected since last control execution
        if self.last_measurement_time is None:
            # zs = historic_measurements.filter(pl.col("time") >= current_time)\
            #                           .gather_every(n=self.n_prediction_interval)
            zs = historic_measurements.filter(pl.col("time") >= (current_time - self.controller_timedelta)) #.filter(((current_time - pl.col("time")).dt.total_microseconds().mod(self.prediction_interval.total_seconds() * 1e6) == 0))
        else:
            # collect all the measurments, prediction_timedelta apart, taken in the last n_controller time steps since predict_point was last called
            # zs = historic_measurements.filter(pl.col("time") >= (self.last_measurement_time + self.prediction_interval))\
            #                           .gather_every(n=self.n_prediction_interval)
            # zs = historic_measurements.filter(pl.col("time") >= (self.last_measurement_time + self.prediction_interval))
            zs = historic_measurements.filter(pl.col("time") >= (self.last_measurement_time + self.controller_timedelta))
                                    #   .filter(((current_time - pl.col("time")).dt.total_microseconds().mod(self.prediction_interval.total_seconds() * 1e6) == 0))
            assert zs.select(pl.len()).item() == 0 or zs.select(pl.col("time").last()).item() == self.last_measurement_time + self.controller_timedelta #self.prediction_interval
        
        if zs.select(pl.len()).item() == 0:
            # forecaster is called every n_controller time steps
            # n_prediction time steps may not have passed since last controller step
            # in this case, no new measurements will be available, and we can return the last state estimate
            logging.info(f"No new measurements available for KalmanFilterForecaster at time {current_time}, waiting on time {(self.last_measurement_time + self.prediction_timedelta)} returning last estimated state.")
            self.last_pred = self.last_pred.with_columns(time=pred_slice)
            if return_var:
                self.last_var = self.last_var.with_columns(time=pred_slice)
        else:
            
            self.last_measurement_time = zs.select(pl.col("time").last()).item()
            # measurement_times = zs.select(pl.col("time")).to_series() 
            zs = zs.select(outputs).to_numpy()
            
            # initialize state
            if not self.initialized:
                self.model.x = np.zeros_like(zs[0, :])
                self.model.P = np.eye(self.model.dim_x)
                # Qs = [np.eye(self.model.dim_x)*1e-2 for j in range(zs.shape[0])]
                # Rs = [np.eye(self.model.dim_z)*1e-2 for j in range(zs.shape[0])]
                self.initialized = True
            else:
                # update Qt and Rt based on previous value s of process and measurement noise
                len_w = self.historic_w.shape[0]
                len_v = self.historic_v.shape[0]
                # Qs = [self._init_covariance(
                #     historic_noise=self.historic_w[len_w - j - self.n_context:len_w - j, :]) for j in range(zs.shape[0]-1, -1, -1)]
                # Rs = [self._init_covariance(
                #     historic_noise=self.historic_v[len_v - j - self.n_context:len_v - j, :]) for j in range(zs.shape[0]-1, -1, -1)]
                # for r in Rs:
                #     np.fill_diagonal(a=r, val=np.max([np.diag(r), np.ones(r.shape[0]) * 1e-2]))
            
            Qs = [np.eye(self.model.dim_x)*1e-1 for j in range(zs.shape[0])] # TODO add to config
            Rs = [np.eye(self.model.dim_z)*1e-3 for j in range(zs.shape[0])]
            
            init_x = self.model.x.copy()
            # use batch_filter to, on each controller sampling time
            # mean estimates from Kalman Filter
            # means_p = np.zeros((zs.shape[0], self.model.dim_x)) # after predict step (prior)
            # means = np.zeros((zs.shape[0], self.model.dim_x)) # after update step (posterior)
            
            # state covariances from Kalman Filter
            # covariances_p = np.zeros((zs.shape[0], self.model.dim_x, self.model.dim_x)) # (prior)
            # covariances = np.zeros((zs.shape[0], self.model.dim_x, self.model.dim_x)) # (posterior)
            
            # (means, covariances, means_p, covariances_p) = self.model.batch_filter(zs=z, Qs=Qt, Rs=Rt)
            # use single longer prediction time; by performing predict/update steps at time intervals == prediction_timedelta
            
            # for each measurement z at time step t, collected since the last controller step, 
            # predict the prior state x(t) for that time step based on the previous state x(t-1), 
            # and update the posterior estimate xhat(t) with the measurement
            # then the prediction is the persistence of that measurment into the future
            for i, z in enumerate(zs):
                # logging.info(f"Adding new measurement {i} of {zs.shape[0]} to Kalman filter at time {current_time}.")
                self.model.predict(Q=Qs[i]) # outputs new prior/prediction
                # means_p[i, :] = self.model.x
                # covariances_p[i, :, :] = self.model.P

                self.model.update(z, R=Rs[i]) # outputs new posterior
                # means[i, :] = self.model.x
                # covariances[i, :, :] = self.model.P
            
            # if np.allclose(means, means_p):
            #     print("oh")
            
            # in historic process (w) and measurment (v) noise, we only need to retain enough vectors to cover all of the measurements (spaced n_prediction apart) found in this interval of n_controller measurments, as well as the context length for each of those 
            # self.historic_times = (self.historic_times + list(measurement_times))[-int(np.ceil(self.n_controller / self.n_prediction)) - self.n_context:]
            # for testing
            # self.means_p = means_p.copy()
            # self.means = means.copy()
            # self.covariances_p = covariances_p.copy()
            # self.covariances = covariances.copy()
            
            # self.historic_v = np.vstack([self.historic_v, np.atleast_2d(zs - np.matmul(means, self.model.H))])[-int(np.ceil(self.n_controller / self.n_prediction_interval)) - self.n_context:, :]
            # means = np.vstack([init_x, means]) # concatenate initial guess of state on top to compute differences
            # self.historic_w = np.vstack([self.historic_w, np.atleast_2d(means[1:, :] - np.matmul(means[:-1, :], self.model.F))])[-int(np.ceil(self.n_controller / self.n_prediction_interval)) - self.n_context:, :]
            
            x = self.model.x #means[-1, :]
            P = self.model.P # covariances[-1, :, :]
            
            x = np.dot(self.model.F, x) # predict step outputs new prior (in this case same, due to identity F)
            P = self.model._alpha_sq * np.dot(np.dot(self.model.F, P), self.model.F.T) + Qs[-1]
        
            pred = x 
            pred = {output: pred[o:o+1] for o, output in enumerate(outputs)}
            
            self.last_pred = pl.DataFrame({"time": pred_slice}).with_columns(**pred)
            
            if return_var:
                var = np.diag(P)
                var = {output: var[o:o+1] for o, output in enumerate(outputs)}
                self.last_var = pl.DataFrame({"time": pred_slice}).with_columns(**var)
                
        if return_var:
            if return_pl:
                return self.last_pred, self.last_var
            else:
                return self.last_pred.to_pandas(), self.last_var.to_pandas()
        else:
            if return_pl:
                return self.last_pred
            else:
                return self.last_pred.to_pandas()
        
    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
        outputs = self._get_ws_cols(historic_measurements)
        
        pred, var = self.predict_point(historic_measurements, current_time, return_var=True)
        var = var.with_columns(**{output: pl.col(output).sqrt() for output in outputs})
        
        res = pred.rename({col: f"loc_{col}" for col in outputs}).join(var.rename({col: f"sd_{col}" for col in outputs}), on="time", how="inner")
        if return_pl:
            return res
        else:
            return res.to_pandas()
    
    def _train_model(self, historic_measurements, model):
        pred = []
        # batch predict and update based on all measurements collected since last control execution
        model.x = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(historic_measurements.select(pl.len()).item()):
            z = historic_measurements.slice(i, 1).to_numpy().flatten()
            model.predict()
            model.update(z)
        
        z = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(self.n_prediction):
            model.predict()
            model.update(z)
            z = model.x
            pred.append(z[0])
            
        return model, np.array(pred)