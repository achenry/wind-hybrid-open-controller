from typing import Union
from dataclasses import dataclass
from memory_profiler import profile

import pandas as pd
import polars as pl

from whoc.wind_forecast.wind_forecast_base import WindForecast

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PersistenceForecast(WindForecast):
    """ Wind speed component forecasting using persistence model that assumes future values equal current value. """
    is_probabilistic = False
    
    def reset(self, **kwargs):
        pass
    
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        
        pred_slice = self.get_pred_interval(current_time)
        pred_slice = pred_slice[-1:]
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(historic_measurements)
            return_pl = False
        else:
            return_pl = True
         
        assert historic_measurements.select((pl.col("time") == current_time).any()).item()
        last_measurement = historic_measurements.filter(pl.col("time") == current_time)
        pred = {k: [v[0]] * len(pred_slice) for k, v in last_measurement.to_dict().items() if k.startswith("ws_")}
        
        pred =  pl.concat([pred_slice.to_frame(), pl.DataFrame(pred)], how="horizontal")
        if return_pl:
            return pred
        else:
            return pred.to_pandas()
        