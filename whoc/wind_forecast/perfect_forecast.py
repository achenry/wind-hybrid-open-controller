from typing import Optional, Union
from dataclasses import dataclass
from memory_profiler import profile

import seaborn as sns
import pandas as pd
import polars as pl
import polars.selectors as cs

from whoc.wind_forecast.wind_forecast_base import WindForecast

from floris import FlorisModel

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PerfectForecast(WindForecast):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    
    col_mapping: Optional[dict] = None
    is_probabilistic = False
    
    def reset(self, **kwargs):
        pass
    
    
    def __post_init__(self):
        # logging.info(f"id(wind_field_ts) in PerfectForecast __init__ is {id(self.true_wind_field)}")
        super().__post_init__()
        self.train_first = False
        if isinstance(self.true_wind_field, pd.DataFrame):
            self.true_wind_field = pl.from_pandas(self.true_wind_field)
        elif isinstance(self.true_wind_field, pl.LazyFrame):
            self.true_wind_field = self.true_wind_field.collect()
        self.true_wind_field = self.true_wind_field.select(pl.col("time"), cs.starts_with("ws_"))
    
    # @profile
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        
        sub_df = (self.true_wind_field.rename(self.col_mapping) if self.col_mapping else self.true_wind_field)\
            .filter(pl.col("time").is_between(current_time, current_time + self.prediction_timedelta, closed="right"))
        
        # slice true_wind_field_ts to reduce memory reqs
        # self.true_wind_field = self.true_wind_field.filter(pl.col("time") > current_time)
        if isinstance(historic_measurements, pl.DataFrame):
            return sub_df
            # assert sub_df.select(pl.len()).item() == int(self.prediction_timedelta / self.measurements_timedelta)
        elif isinstance(historic_measurements, pd.DataFrame):
            return sub_df.to_pandas()
            
        return sub_df
