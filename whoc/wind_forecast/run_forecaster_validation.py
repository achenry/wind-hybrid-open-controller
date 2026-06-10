"""Module for wind speed component forecasting and sf functionality."""

import os
import yaml
import re
import argparse

import gc
from memory_profiler import profile
import glob
from itertools import cycle
from psutil import virtual_memory
from shutil import move
from functools import reduce

from scipy.signal import lfilter

import multiprocessing as mp
# from multiprocessing import get_context

# from joblib import parallel_backend

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from concurrent.futures import ProcessPoolExecutor

from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.util import period_index
from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.field_names import FieldName

from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.postprocessing.probabilistic_metrics import (
    continuous_ranked_probability_score_gaussian,
    pi_coverage_probability,
    pi_normalized_average_width,
    coverage_width_criterion,
    continuous_ranked_probability_score_samples,
    pi_coverage_probability_samples,
    pi_normalized_average_width_samples,
    coverage_width_criterion_samples,
    prediction_interval_from_samples,
)

from floris import FlorisModel

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from whoc.wind_forecast.wind_forecast_base import WindForecast
from whoc.wind_forecast.perfect_forecast import PerfectForecast
from whoc.wind_forecast.persistence_forecast import PersistenceForecast
from whoc.wind_forecast.kalman_filter_forecast import KalmanFilterForecast
from whoc.wind_forecast.spatial_filter_forecast import SpatialFilterForecast
from whoc.wind_forecast.svr_forecast import SVRForecast
from whoc.wind_forecast.ml_forecast import MLForecast

sns.set_palette("Paired")


def plot_wind_ts(
    data_df,
    save_path,
    turbine_ids="all",
    include_filtered_wind_dir=True,
    controller_timedelta=None,
    legend_loc="best",
    single_plot=False,
    fig=None,
    ax=None,
    case_label=None,
):
    # TODO only plot some turbines, not ones with overlapping yaw offsets, eg single column on farm
    colors = sns.color_palette("Paired")
    colors = [colors[1], colors[3], colors[5]]

    if not single_plot:
        fig, ax = plt.subplots(1, 1)

    # ax = np.atleast_1d(ax)

    plot_seed = data_df["continuity_group"].unique()[0]

    for seed in data_df["continuity_group"].unique():
        if seed != plot_seed:
            continue
        seed_df = data_df.filter(
            data_df["continuity_group"] == seed, data_df["feature"].is_in(["wd", "wd_filt"])
        ).select("turbine_id", "time", "feature", "value")
        if turbine_ids != "all":
            seed_df = seed_df.filter(pl.col("turbine_id").is_in(turbine_ids))

        sns.lineplot(data=seed_df, x="time", y="value", style="feature", hue="turbine_id", ax=ax)
        # if include_filtered_wind_dir:
        #     sns.lineplot(data=seed_df, x="time", y="FilteredFreestreamWindDir", label="Filtered wind dir.", color="black", linestyle="--", ax=ax[ax_idx])

    h, l = ax.get_legend_handles_labels()
    h = [handle for handle, label in zip(h, l) if label in ["wd", "wd_filt"]]
    l = ["Raw", "LPF'd"]
    ax.legend(h, l)
    ax.set(
        title="Wind Direction [$^\\circ$]",
        ylabel="",
        xlabel="Time",
        xlim=(seed_df["time"].min(), seed_df["time"].max()),
    )

    results_dir = os.path.dirname(save_path)
    plt.tight_layout()

    logging.info(f"Saving plot_wind_ts to {save_path}.")
    fig.savefig(save_path)
    return fig, ax


def first_ord_filter(x, time_const=35, dt=60):
    lpf_alpha = np.exp(-(1 / time_const) * dt)
    b = [1 - lpf_alpha]
    a = [1, -lpf_alpha]
    return lfilter(b, a, x)


def transform_wind(inp_df, added_wm=None, added_wd=None):
    original_cols = np.array(inp_df.collect_schema().names())
    if added_wm:
        inp_df = inp_df.with_columns((cs.starts_with("wm_") + added_wm).name.keep())

    if added_wd:
        inp_df = inp_df.with_columns((cs.starts_with("wd_") + added_wd).mod(360.0).name.keep())

    ws_horz = inp_df.select(cs.starts_with("wm_")).rename(
        lambda old_col: re.search("(?<=wm_)\\d+", old_col).group()
    ) * inp_df.select(((180.0 + cs.starts_with("wd_")).radians().sin()).name.keep()).rename(
        lambda old_col: re.search("(?<=wd_)\\d+", old_col).group()
    )
    ws_vert = inp_df.select(cs.starts_with("wm_")).rename(
        lambda old_col: re.search("(?<=wm_)\\d+", old_col).group()
    ) * inp_df.select(((180.0 + cs.starts_with("wd_")).radians().cos()).name.keep()).rename(
        lambda old_col: re.search("(?<=wd_)\\d+", old_col).group()
    )

    inp_df = inp_df.with_columns(ws_horz.select(pl.all().name.prefix("ws_horz_")))
    inp_df = inp_df.with_columns(ws_vert.select(pl.all().name.prefix("ws_vert_")))
    # inp_df = inp_df.select(**{col: -pl.col(f"wm_{re.search('\\d+', col).group()}") for col in inp_df.columns if col.startswith("ws_horz_")})

    return inp_df.select(original_cols)


def make_predictions(
    forecaster, test_data, prediction_type, single_cg, save_path, assigned_gpu, ram_limit
):

    logging.info(
        f"Worker process {os.getpid()}"
    )  # sees POLARS_MAX_THREADS={os.environ.get('POLARS_MAX_THREADS')}

    if assigned_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    forecasts = []

    logging.info("Getting timestamps at which controller will call forecaster.")
    controller_times = test_data.gather_every(forecaster.n_controller).select(pl.col("time"))

    if single_cg:
        splits = [test_data.select(pl.col("continuity_group").first()).item()]
        test_data = [test_data]
    else:
        logging.info("Getting number of continuity groups in data.")
        splits = test_data.select(pl.col("continuity_group").unique()).to_numpy().flatten()
        test_data = test_data.partition_by("continuity_group")
    n_splits = len(splits)

    test_idx = 0

    forecaster_name = (
        forecaster.__class__.__name__
        if forecaster.__class__.__name__ != "MLForecast"
        else f"{forecaster.model_key.capitalize()}Forecast"
    )
    save_paths = set()
    for d, ds in enumerate(test_data):
        test_data_time = ds["time"]

        start = ds.select(pl.col("time").first()).item()
        end = ds.select(pl.col("time").last()).item()
        logging.info(
            f"Getting predictions for {splits[d]}th split starting at {start} and ending at {end} using {forecaster_name} with prediction_timedelta {forecaster.prediction_timedelta}."
        )
        forecasts = []
        # split_true_wf = true_wind_field.filter(pl.col("time").is_between(start, end, closed="both"))
        # logging.info(f"Getting controller times for {splits[d]}th split.")
        # polars >= 1.x can't parse pandas.Timedelta in `>=` literal — convert to py timedelta first.
        _ctx_td = forecaster.context_timedelta
        if hasattr(_ctx_td, "to_pytimedelta"):
            _ctx_td = _ctx_td.to_pytimedelta()
        split_controller_times = controller_times.filter(
            pl.col("time").is_between(start, end, closed="both")
        ).filter((pl.col("time") - start) >= _ctx_td)
        n_controller_times = split_controller_times.select(pl.len()).item()
        # logging.info(f"Resetting forecaster state.")
        forecaster.reset(assigned_gpu=assigned_gpu)
        save_length = 0
        n_saved = 0
        for c, current_row in enumerate(split_controller_times.iter_rows(named=True)):
            current_time = current_row["time"]

            # if current_time - start >= forecaster.context_timedelta:
            logging.info(
                f"Predicting {c} of {n_controller_times} future wind fields using {forecaster_name} with prediction_timedelta {forecaster.prediction_timedelta} at time {current_time}/{end} of split {splits[d]}."
            )
            # logging.info(f"RAM 172 = {virtual_memory().percent}")
            if prediction_type == "distribution" and forecaster.is_probabilistic:
                pred = forecaster.predict_distr(
                    ds.filter(pl.col("time") <= current_time), current_time
                )
            elif prediction_type == "point" or not forecaster.is_probabilistic:
                pred = forecaster.predict_point(
                    ds.filter(pl.col("time") <= current_time), current_time
                )
            elif prediction_type == "sample":
                n_samples = 100  # Default value
                if hasattr(forecaster, "model_config"):
                    try:
                        n_samples = forecaster.model_config["model"][forecaster.model_key].get(
                            "num_parallel_samples", 100
                        )
                    except (KeyError, AttributeError):
                        pass

                pred = forecaster.predict_sample(
                    ds.filter(pl.col("time") <= current_time), current_time, n_samples=n_samples
                )

            # logging.info(f"RAM 190 = {virtual_memory().percent}")

            # logging.info(f"pred['time'] = {pred.select("time")}")
            # logging.info(f"test_data_time = {test_data_time}")
            pred = (
                pred.with_columns(cs.numeric().cast(pl.Float32))
                .with_columns(
                    test_idx=pl.lit(test_idx).cast(pl.Int32),
                    continuity_group=pl.lit(splits[d]).cast(pl.Int32),
                )
                .with_columns(time=pl.col("time").cast(pl.Datetime(time_unit="ns")))
            )
            pred = pred.filter(pred["time"].is_in(test_data_time.implode()))

            # logging.info(f"RAM 200 = {virtual_memory().percent}")

            forecasts.append(pred)

            save_length += pred.select(pl.len()).item()

            if (final := ((c == n_controller_times - 1) and (d == n_splits - 1))) or (
                ((ram_used := virtual_memory().percent) > ram_limit) and (save_length > 0)
            ):
                logging.info(f"In save conditional.")
                # sub_save_path = save_path.replace(".parquet", f"_{splits[d]}_{n_saved}.parquet")
                if callable(save_path):
                    sp = save_path(splits[d])
                else:
                    sp = save_path
                temp_sp = sp.replace(".parquet", "_temp.parquet")
                save_paths.add(temp_sp)

                if not final:
                    logging.info(
                        f"Used {ram_used}% RAM. Saving parquet of length {save_length} to {temp_sp}."
                    )
                else:
                    logging.info(
                        f"Final save for split {splits[d]}. Saving parquet of length {save_length} to {temp_sp}."
                    )

                # logging.info(f"Concatenating forecasts.")
                forecasts = pl.concat(forecasts, how="diagonal")

                # logging.info(f"diagonal concat for {save_path} columns = {forecasts.columns}")
                if not os.path.exists(temp_sp):
                    with open(temp_sp, mode="w") as fp:
                        forecasts.write_parquet(fp)
                    logging.info(
                        f"File {temp_sp} has size {os.path.getsize(temp_sp)} after first write."
                    )
                else:
                    logging.info(
                        f"File {temp_sp} has size {os.path.getsize(temp_sp)} before appending."
                    )
                    # with open(temp_sp, mode="a") as fp:
                    forecasts = pl.concat([pl.read_parquet(temp_sp), forecasts], how="vertical")
                    with open(temp_sp, mode="w") as fp:
                        forecasts.write_parquet(fp)
                    logging.info(
                        f"File {temp_sp} has size {os.path.getsize(temp_sp)} after appending."
                    )

                n_saved += 1
                forecasts = []
                save_length = 0
                ram_used = virtual_memory().percent
                logging.info(f"Used {ram_used}% RAM after saving {sp}.")

            test_idx += 1

        if len(forecasts) == 0 and n_saved == 0:
            raise Exception(
                f"{d}th dataset in data does not have sufficient data points, with {ds.select(pl.len()).item()}, to collect predictions after context_timedelta {forecaster.context_timedelta}"
            )

    for temp_sp in save_paths:
        final_sp = temp_sp.replace("_temp.parquet", ".parquet")
        logging.info(f"Moving final result to {final_sp}.")
        os.replace(temp_sp, final_sp)
        logging.info(f"Moved final result to {final_sp}.")

    return


def generate_wind_field_df(datasets, target_cols, feat_dynamic_real_cols):
    full_target = np.concatenate([ds[FieldName.TARGET] for ds in datasets], axis=-1)
    full_feat_dynamic_reals = np.concatenate(
        [ds[FieldName.FEAT_DYNAMIC_REAL] for ds in datasets], axis=-1
    )[:, : full_target.shape[1]]
    full_splits = np.atleast_2d(
        np.hstack(
            [
                np.repeat(
                    int(re.search("(?<=SPLIT)\\d+", ds[FieldName.ITEM_ID]).group()),
                    (ds[FieldName.TARGET].shape[1],),
                )
                for ds in datasets
            ]
        )
    ).astype(int)
    index = pd.concat(
        [
            period_index(
                {FieldName.START: ds[FieldName.START], FieldName.TARGET: ds[FieldName.TARGET]}
            ).to_series()
            for ds in datasets
        ]
    ).index

    wf = pd.DataFrame(
        np.concatenate([full_splits, full_target, full_feat_dynamic_reals], axis=0).transpose(),
        columns=["continuity_group"] + target_cols + feat_dynamic_real_cols,
        index=index,
    )
    wf["continuity_group"] = wf["continuity_group"].astype(int)

    wf = wf.reset_index(names="time")
    wf["time"] = wf["time"].dt.to_timestamp()
    return pl.from_pandas(wf)


def unpivot_df(df, turbine_signature):
    return (
        df.unpivot(
            index=["metric", "continuity_group", "test_idx"],
            variable_name="feature",
            value_name="score",
        )
        .with_columns(
            turbine_id=pl.col("feature").str.extract(f"({turbine_signature})$"),
            feature_type=pl.col("feature").str.extract(f"(.*)_{turbine_signature}$"),
        )
        .drop("feature")
    )


def generate_metric_per_cg(
    pred_mean, pred_stddev, true, metric_name, metric_func, cg_vals, target_cols, true_cols
):
    return pl.concat(
        [
            pl.DataFrame(
                data=np.atleast_2d(
                    metric_func(
                        pred_mean.filter(pl.col("continuity_group") == cg)
                        .select(target_cols)
                        .to_numpy(),
                        true.filter(pl.col("continuity_group") == cg).select(true_cols).to_numpy(),
                        pred_stddev.filter(pl.col("continuity_group") == cg)
                        .select(cs.starts_with("sd_"))
                        .to_numpy(),
                    )
                ),
                schema=target_cols,
            ).with_columns(continuity_group=pl.lit(cg))
            for cg in cg_vals
        ],
        how="vertical",
    ).with_columns(metric=pl.lit(metric_name), test_idx=pl.lit(-1))


def generate_sample_based_metrics_per_cg(
    pred_df, true, metric_name, metric_func, cg_vals, target_cols, true_cols
):
    """Generate sample-based metrics per continuity group.

    Args:
        pred_df: DataFrame containing sample predictions
        true: DataFrame containing true values
        metric_name: Name of the metric being calculated
        metric_func: Sample-based metric function to apply
        cg_vals: Array of continuity group values
        target_cols: List of target column names
        true_cols: List of true value column names

    Returns:
        DataFrame containing metrics for each continuity group and target
    """
    metrics = []
    for cg in cg_vals:
        cg_metrics = []
        for target_col, true_col in zip(target_cols, true_cols):
            # Get base column name without sample suffix
            base_col = target_col.split("_sample_")[0]

            # Get all sample columns for this target
            sample_cols = [col for col in pred_df.columns if col.startswith(f"{base_col}_sample_")]

            # Extract samples and reshape to (n, num_samples)
            samples = (
                pred_df.filter(pl.col("continuity_group") == cg).select(sample_cols).to_numpy()
            )
            samples = samples.reshape(samples.shape[0], -1)  # Reshape to (n, num_samples)

            # Get true values
            true_values = (
                true.filter(pl.col("continuity_group") == cg).select(true_col).to_numpy().flatten()
            )

            # Calculate metric
            metric_value = metric_func(true_values, samples)
            cg_metrics.append(metric_value)

        metrics.append(
            pl.DataFrame(data=np.atleast_2d(cg_metrics), schema=target_cols).with_columns(
                continuity_group=pl.lit(cg)
            )
        )

    return pl.concat(metrics, how="vertical").with_columns(
        metric=pl.lit(metric_name), test_idx=pl.lit(-1)
    )


def generate_forecaster_agg_results(
    forecaster, forecast_df, test_data, target_cols, prediction_type
):
    logging.info(
        f"Preparing true data for forecaster {forecaster.__class__.__name__} with prediction_timedelta = {forecaster.prediction_timedelta.total_seconds()} seconds."
    )
    # true_df_pd = test_data.collect().to_pandas()
    # true_df_pd = true_df_pd.set_index(pd.PeriodIndex(true_df_pd["time"].dt.to_period(freq=data_module.freq)))[data_module.target_cols]\
    #                     .rename(columns={src: s for s, src in enumerate(data_module.target_cols)})

    forecaster_name = (
        forecaster.__class__.__name__
        if forecaster.__class__.__name__ != "MLForecast"
        else f"{forecaster.model_key.capitalize()}Forecast"
    )
    logging.info(
        f"Preparing combined df for forecaster {forecaster_name} with prediction_timedelta = {forecaster.prediction_timedelta.total_seconds()} seconds."
    )

    fdf = forecast_df.select(["time", "test_idx"] + [cs.ends_with(tgt) for tgt in target_cols])
    if False:
        # NOTE: for multistep predictions it is possible for multiple predictions for the same timestamp to exist
        # select the first prediction found for each timestamp, i.e. the one farthest from current time
        fdf = fdf.group_by("time", maintain_order=True).first()
        # otherwise we include the errors for the same timestamp multiple times

    tdf = test_data.filter(pl.col("time").is_in(forecast_df["time"].implode())).select(
        ["time", "continuity_group"] + target_cols
    )
    combined_df = fdf.rename(
        lambda col: re.search("(?<=loc_)(\\w+)$", col).group() if col.startswith("loc_") else col
    ).join(tdf, on=["time"], suffix="_true", coalesce=False)
    true_cols = [f"{c}_true" for c in target_cols]

    # x = datetime.strptime("2023-02-19 15:54:12", "%Y-%m-%d %H:%M:%S")

    logging.info(
        f"Preparing deterministic agg_metrics for forecaster {forecaster_name} with prediction_timedelta = {forecaster.prediction_timedelta.total_seconds()} seconds."
    )

    agg_metrics = []
    err = combined_df.select(
        ["time", "continuity_group"]
        + [
            (pl.col(pred_col) - pl.col(true_col))
            for true_col, pred_col in zip(true_cols, target_cols)
        ]
    )

    rmse = (
        err.group_by("continuity_group")
        .agg(cs.numeric().pow(2).mean().sqrt())
        .with_columns(metric=pl.lit("RMSE"), test_idx=pl.lit(-1))
    )
    rmse = unpivot_df(rmse, forecaster.turbine_signature)

    mae = (
        err.group_by("continuity_group")
        .agg(cs.numeric().abs().mean())
        .with_columns(metric=pl.lit("MAE"), test_idx=pl.lit(-1))
    )
    mae = unpivot_df(mae, forecaster.turbine_signature)

    agg_metrics += [rmse, mae]

    if prediction_type == "distribution" and forecaster.is_probabilistic:
        logging.info(
            f"Preparing probabilistic agg_metrics for forecaster {forecaster_name} with prediction_timedelta = {forecaster.prediction_timedelta.total_seconds()} seconds."
        )

        pred_mean = combined_df.select(["continuity_group"] + target_cols)
        true = combined_df.select(["continuity_group"] + true_cols)
        pred_stddev = combined_df.select(pl.col("continuity_group"), cs.starts_with("sd_"))
        cg_vals = combined_df.select(pl.col("continuity_group").unique()).to_numpy().flatten()

        picp = generate_metric_per_cg(
            pred_mean,
            pred_stddev,
            true,
            "PICP",
            pi_coverage_probability,
            cg_vals,
            target_cols,
            true_cols,
        )
        picp = unpivot_df(picp, forecaster.turbine_signature)

        pinaw = generate_metric_per_cg(
            pred_mean,
            pred_stddev,
            true,
            "PINAW",
            pi_normalized_average_width,
            cg_vals,
            target_cols,
            true_cols,
        )
        pinaw = unpivot_df(pinaw, forecaster.turbine_signature)

        cwc = generate_metric_per_cg(
            pred_mean,
            pred_stddev,
            true,
            "CWC",
            coverage_width_criterion,
            cg_vals,
            target_cols,
            true_cols,
        )
        cwc = unpivot_df(cwc, forecaster.turbine_signature)

        crps = generate_metric_per_cg(
            pred_mean,
            pred_stddev,
            true,
            "CRPS",
            continuous_ranked_probability_score_gaussian,
            cg_vals,
            target_cols,
            true_cols,
        )
        crps = unpivot_df(crps, forecaster.turbine_signature)

        agg_metrics += [picp, pinaw, cwc, crps]

    elif prediction_type == "sample" and forecaster.is_probabilistic:
        logging.info(
            f"Preparing sample-based probabilistic metrics for forecaster {forecaster_name} with prediction_timedelta = {forecaster.prediction_timedelta.total_seconds()} seconds."
        )

        cg_vals = combined_df.select(pl.col("continuity_group").unique()).to_numpy().flatten()

        # Calculate sample-based metrics
        picp = generate_sample_based_metrics_per_cg(
            combined_df,
            combined_df,
            "PICP_samples",
            pi_coverage_probability_samples,
            cg_vals,
            target_cols,
            true_cols,
        )
        picp = unpivot_df(picp, forecaster.turbine_signature)

        pinaw = generate_sample_based_metrics_per_cg(
            combined_df,
            combined_df,
            "PINAW_samples",
            pi_normalized_average_width_samples,
            cg_vals,
            target_cols,
            true_cols,
        )
        pinaw = unpivot_df(pinaw, forecaster.turbine_signature)

        cwc = generate_sample_based_metrics_per_cg(
            combined_df,
            combined_df,
            "CWC_samples",
            coverage_width_criterion_samples,
            cg_vals,
            target_cols,
            true_cols,
        )
        cwc = unpivot_df(cwc, forecaster.turbine_signature)

        crps = generate_sample_based_metrics_per_cg(
            combined_df,
            combined_df,
            "CRPS_samples",
            continuous_ranked_probability_score_samples,
            cg_vals,
            target_cols,
            true_cols,
        )
        crps = unpivot_df(crps, forecaster.turbine_signature)

        agg_metrics += [picp, pinaw, cwc, crps]

    # evaluator.get_metrics_per_ts(ts, forecast, include_metrics=include_metrics)
    # evaluator.get_aggregate_metrics(metrics_per_ts, include_metrics=include_metrics)
    agg_metrics = pl.concat(agg_metrics, how="vertical_relaxed")
    agg_metrics = pl.concat(
        [
            agg_metrics.select(
                ["continuity_group", "metric", "test_idx", "feature_type", "turbine_id", "score"]
            ),
            agg_metrics.group_by(
                ["continuity_group", "metric", "test_idx", "feature_type"], maintain_order=True
            )
            .agg(pl.col("score").mean())
            .with_columns(turbine_id=pl.lit("all"))
            .select(
                ["continuity_group", "metric", "test_idx", "feature_type", "turbine_id", "score"]
            ),
            agg_metrics.group_by(
                ["continuity_group", "metric", "turbine_id", "feature_type"], maintain_order=True
            )
            .agg(pl.col("score").mean())
            .with_columns(test_idx=pl.lit(-1))
            .select(
                ["continuity_group", "metric", "test_idx", "feature_type", "turbine_id", "score"]
            ),
        ],
        how="vertical",
    ).sort(["continuity_group", "metric", "test_idx", "feature_type"])
    agg_metrics = pl.concat(
        [
            agg_metrics,
            agg_metrics.group_by(
                ["continuity_group", "metric", "feature_type"], maintain_order=True
            )
            .agg(pl.col("score").mean())
            .with_columns(test_idx=pl.lit(-1), turbine_id=pl.lit("all"))
            .select(
                ["continuity_group", "metric", "test_idx", "feature_type", "turbine_id", "score"]
            ),
        ]
    )
    return agg_metrics.with_columns(cs.float().cast(pl.Float32), cs.integer().cast(pl.Int32))


def plot_score_vs_prediction_dt(agg_df, metrics, ax_indices, fig_dir):

    n_axes = len(ax_indices)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10.9, 8.0))
    ax = sns.scatterplot(
        agg_df.filter(pl.col("metric").is_in(metrics)).to_pandas(),
        y="score",
        x="prediction_timedelta",
        style="metric",
        hue="forecaster",
        s=200,
    )
    ax.set_ylabel("Score")
    h1, l1 = ax.get_legend_handles_labels()

    ax.set_xlabel("Prediction Length (s)")
    l = l1
    h = h1
    ax.set_xticks(agg_df.select(pl.col("prediction_timedelta").unique()).to_numpy().flatten())
    new_labels = [
        " ".join(re.findall("[A-Z][^A-Z]*", re.search("(\\w+)(?=Forecast)(\\w+)", label).group()))
        if ("Forecast" in label)
        else (label.capitalize() if not label[0].isupper() else label).replace("_", " ")
        for label in l
    ]
    if "S V R Forecast" in new_labels:
        new_labels[new_labels.index("S V R Forecast")] = (
            "SVR Forecast"  # TODO automate this with re replace etc
        )
    # new_labels = ["".join(label.split(" ")) if all(label[l].isupper() for l in range(0, len(label)-1, 2) if (label[l+1].isspace() or (l+1 == len(label)-1))) else label for label in new_labels]

    l1, l2 = new_labels[: new_labels.index("Metric")], new_labels[new_labels.index("Metric") :]
    h1, h2 = h[: new_labels.index("Metric")], h[new_labels.index("Metric") :]
    leg1 = ax.legend(h1, l1, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    leg2 = plt.legend(h2, l2, loc="upper left", bbox_to_anchor=(1.01, 0.6), frameon=False)
    ax.add_artist(leg1)
    fig.subplots_adjust(right=0.6)

    fig_path = os.path.join(fig_dir, "score_vs_pred.png")
    logging.info(f"Saving plot_score_vs_prediction_dt to {fig_path}")
    fig.savefig(fig_path)
    return fig


def plot_score_vs_forecaster(agg_df, metrics, ax_indices, prediction_intervals, fig_dir, label):
    # TODO HIGH put all ML models and baseline models on same plot. Add dash boundary to baseline bars.
    sns.set_style("whitegrid")

    ax = sns.catplot(
        agg_df.filter((pl.col("metric").is_in(metrics))),
        kind="bar",
        row=0,
        hue="forecaster",
        x="metric",
        y="score",
    )
    # , hue_order=metrics) col="prediction_timedelta",

    new_labels = [
        " ".join(re.findall("[A-Z][^A-Z]*", re.search("\\w+(?=Forecast)", l._text).group()))
        for l in ax._legend.texts
    ]
    new_labels = [
        "".join(l.split(" ")) if all(l.isupper() or l.isspace() for l in l) else l
        for l in new_labels
    ]

    for p in range(ax.axes.shape[1]):
        # pred_len = re.search('(?<=prediction_timedelta = )(\\d+)', ax.axes[0, p].title.get_text()).group()
        # ax.axes[0, p].set_title(f"Prediction Length {pred_len} sec")
        ax.axes[0, p].set_ylabel("")
        ax.axes[0, p].set_xlabel("Metric")
        # ax.axes[0, p].set_xticklabels(new_xticks, rotation=25)

    start_patch_idx = 0
    for t, text in enumerate(new_labels):
        num_patches = len(ax.axes[0, 0].containers[t])
        ax._legend.texts[t].set_text(text)
        if ax._legend.texts[t]._text in ["SVR", "Persistence", "Spatial Filter", "Kalman Filter"]:
            ax._legend.get_patches()[t].set_hatch("/")
            for patch in ax.axes[0, 0].containers[t].patches:
                patch.set_hatch("/")
        start_patch_idx += num_patches

    ax.axes[0, 0].set_ylabel(f"Score")

    ax.legend.set_title("")
    ax.legend.set_loc("upper right")
    ax.legend.set_bbox_to_anchor((0.0, 0.0, 1.01, 0.9))

    fig = plt.gcf()
    fig.set_size_inches((14.1, 7.8))
    fig.subplots_adjust(right=0.85)

    fig_path = os.path.join(fig_dir, f"score_vs_forecaster{label}.png")
    logging.info(f"Saving plot_score_vs_forecaster to {fig_path}")
    fig.savefig(fig_path)

    return fig


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(prog="ModelTuning")
    parser.add_argument(
        "-mcnf",
        "--model_config",
        type=str,
        nargs="+",
        help="Filepaths to model configurations with experiment, optuna, dataset, model, callbacks, trainer keys.",
    )
    parser.add_argument(
        "-dcnf",
        "--data_config",
        type=str,
        help="Filepath to data preprocessing configuration with filters, feature_mapping, turbine_signature, nacelle_calibration_turbine_pairs, dt, raw_data_directory, processed_data_path, raw_data_file_signature, turbine_input_path, farm_input_path keys.",
    )
    parser.add_argument(
        "-sd", "--save_dir", type=str, help="Directory to save results to.", default="./"
    )
    parser.add_argument(
        "-m",
        "--model",  # type=str,
        # choices=["perfect", "persistence", "svr", "kf", "informer", "autoformer", "spacetimeformer", "sf"],
        default=None,
        nargs="+",
        help="Which model(s) to simulate, compute score for, and plot.",
    )
    parser.add_argument(
        "-rn",
        "--run_name",  # type=str,
        default="",
        help="Name to append to plots and results, e.g. for experiment tracking.",
    )
    parser.add_argument(
        "-st",
        "--simulation_timestep",
        required=True,
        type=int,
        help="Simulation time step to use (sec)",
    )
    parser.add_argument(
        "-rv",
        "--run_validation",
        action="store_true",
        help="Whether to run validation for results.",
    )
    parser.add_argument(
        "-rrv",
        "--rerun_validation",
        action="store_true",
        help="Whether to rerun validation for existing results.",
    )
    parser.add_argument(
        "-rp",
        "--run_processing",
        action="store_true",
        help="Whether to run aggregation and plotting on validation time series.",
    )
    parser.add_argument(
        "-rag",
        "--reaggregate_metrics",
        action="store_true",
        help="Whether to repeat metric aggregation for agg_ results that have already been stored.",
    )
    parser.add_argument(
        "-rld",
        "--reload_data",
        action="store_true",
        help="Whether to reload validation all_turbine simulation time step datasets or not.",
    )
    parser.add_argument(
        "-rsd",
        "--resplit_data",
        action="store_true",
        help="Whether to resplit all_turbine simulation time step datasets or not.",
    )
    # parser.add_argument("-pi", "--prediction_interval",
    #                     required=False, nargs="+", default=None,
    #                     help="Number of seconds to use as prediction_timedelta..")
    parser.add_argument(
        "-mp",
        "--multiprocessor",
        type=str,
        choices=["mpi", "cf"],
        help="which multiprocessing backend to use, omit for sequential processing",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-msp",
        "--max_splits",
        type=int,
        required=False,
        default=None,
        help="Number of test splits to use.",
    )
    parser.add_argument(
        "-mst",
        "--max_steps",
        type=int,
        required=False,
        default=None,
        help="Number of time steps to use.",
    )
    parser.add_argument(
        "-chk",
        "--checkpoint",
        type=str,
        required=False,
        default="[latest]",
        nargs="+",
        help="Which checkpoint to use: can be equal to 'latest', 'best', or a list of existing checkpoint path (one for each ML model passed to models, in same order).",
    )
    parser.add_argument(
        "-pt",
        "--prediction_type",
        type=str,
        choices=["point", "sample", "distribution"],
        default="point",
        help="Whether to make a point, sample, or distribution parameter prediction.",
    )
    parser.add_argument(
        "-awm",
        "--added_wind_mag",
        type=float,
        required=False,
        default=0.0,
        help="Wind magnitude to add to all values (after transformation to wind magnitude and direction).",
    )
    parser.add_argument(
        "-awd",
        "--added_wind_dir",
        type=float,
        required=False,
        default=0.0,
        help="Wind direction to add to all values (after transformation to wind magnitude and direction).",
    )
    parser.add_argument("-p", "--plot", action="store_true", help="Plot time series outputs.")
    # parser.add_argument("-tp", "--use_tuned_params", action="store_true",
    #                     help="Use parameters tuned from Optuna optimization, otherwise use defaults set in Module class.")
    parser.add_argument(
        "-tm",
        "--use_trained_models",
        action="store_true",
        help="Use parameters trained and stored for models that require training, e.g. SVR, read existing trained models from file.",
    )
    parser.add_argument(
        "-rl",
        "--ram_limit",
        type=int,
        default=75,
        help="Percentage of RAM usage, above which to store checkpoints.",
    )
    parser.add_argument(
        "-tti",
        "--target_turbine_indices",
        metavar="C",
        nargs="+",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "-cgs",
        "--continuity_groups",
        required=False,
        default=None,
        type=str,
        help="Continuity groups to include in validation results, separated by commas, e.g. '0,1,2,3,4'.",
    )
    parser.add_argument(
        "--cp_calibrate_stddev",
        action="store_true",
        help="If set, MLForecast.predict_distr multiplies its raw predictive stddev "
        "by the committed CP scale factors (whoc/wind_forecast/cp_scale_factors/quantile_head.json) "
        "so sd_ws_horz_* / sd_ws_vert_* columns in the validation parquet are calibrated. "
        "Default OFF — calibration is opt-in per run.",
    )
    args = parser.parse_args()

    assert args.model is None or all(
        model
        in [
            "perfect",
            "persistence",
            "svr",
            "kf",
            "informer",
            "autoformer",
            "spacetimeformer",
            "tactis",
            "sf",
        ]
        for model in args.model
    )

    if args.model is None:
        args.run_validation = args.rerun_validation = False

    if args.rerun_validation:
        args.run_validation = True

    if args.multiprocessor == "mpi":
        mpi_exists = False
        try:
            logging.info("Attempting to import MPI.")
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor

            mpi_exists = True
        except ImportError as e:
            import traceback

            logging.error(f"Failed to import mpi4py. MPI will not be available. Error: {e}")
            logging.error(traceback.format_exc())

        if not mpi_exists:
            raise RuntimeError(
                "MPI was requested (--multiprocessor mpi) but mpi4py failed to import. Check previous logs for import error details."
            )

        # elif not mpi_exists:
        #      # If MPI wasn't requested, we might not need it here, but accessing MPI.COMM_WORLD directly is still problematic.
        #      # Depending on logic flow, this might need adjustment. For now, assume it's an error if MPI isn't available.
        #      # If MPI is optional, this block might need refinement based on how `comm` is used later.
        #      comm = None # Or handle appropriately if MPI is truly optional here
        #      rank = -1   # Assign a default rank if MPI is not used
        #      print("Warning: MPI not available, proceeding without it where possible.")
        else:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

    RUN_ONCE = (
        (args.multiprocessor == "mpi" and rank == 0)
        or (args.multiprocessor != "mpi")
        or (args.multiprocessor is None)
    )

    TRANSFORM_WIND = {"added_wm": args.added_wind_mag, "added_wd": args.added_wind_dir}
    # args.fig_dir = os.path.join(os.path.dirname(whoc_file), "..", "examples", "wind_forecasting")

    if RUN_ONCE:
        os.makedirs(args.save_dir, exist_ok=True)

    model_configs = []
    for mnf_path in args.model_config:
        with open(mnf_path, "r") as file:
            model_configs.append(yaml.safe_load(file))

    prediction_timedeltas = [
        pd.Timedelta(seconds=mncf["dataset"]["prediction_length"]) for mncf in model_configs
    ]
    context_timedeltas = [
        pd.Timedelta(seconds=mncf["dataset"]["context_length"]) for mncf in model_configs
    ]
    measurements_timedelta = pd.Timedelta(seconds=args.simulation_timestep)

    # measurements_timedelta = pd.Timedelta(model_config["dataset"]["resample_freq"])

    controller_timedelta = max(pd.Timedelta(5, unit="s"), measurements_timedelta)

    with open(args.data_config, "r") as file:
        data_config = yaml.safe_load(file)

    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {
            str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())
        }
    else:
        tid2idx_mapping = {
            str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())
        }  # if more than one file type was pulled from, all turbine ids will be transformed into common type

    turbine_signature = (
        data_config["turbine_signature"][0]
        if len(data_config["turbine_signature"]) == 1
        else "\\d+"
    )

    id_var_selector = pl.exclude(
        f"^ws_horz_{turbine_signature}$",
        f"^ws_vert_{turbine_signature}$",
        f"^nd_cos_{turbine_signature}$",
        f"^nd_sin_{turbine_signature}$",
        f"^loc_ws_horz_{turbine_signature}$",
        f"^loc_ws_vert_{turbine_signature}$",
        f"^sd_ws_horz_{turbine_signature}$",
        f"^sd_ws_vert_{turbine_signature}$",
    )

    fmodel = FlorisModel(data_config["farm_input_path"])

    validation_save_dir = os.path.join(args.save_dir, "validation_results")

    logging.info("Creating datasets")

    # NOTE the dataset parts of the configs should be the same, other than context and prediction length
    # TODO ensure that we are comparing correct time stamps!!
    # base_model_config = model_configs[np.argsort([ctd + ptd for ctd, ptd in zip(context_timedeltas, prediction_timedeltas)])[-1]]

    test_data = []
    cgs = []
    joint_cgs = set()
    for mcnf in model_configs:
        data_module = DataModule(
            normalized_data_path=mcnf["dataset"]["data_path"],
            normalization_consts_path=mcnf["dataset"]["normalization_consts_path"],
            use_normalization=False,
            n_splits=1,  # model_config["dataset"]["n_splits"],
            continuity_groups=None,
            train_split=(1.0 - mcnf["dataset"]["val_split"] - mcnf["dataset"]["test_split"]),
            val_split=mcnf["dataset"]["val_split"],
            test_split=mcnf["dataset"]["test_split"],
            prediction_length=mcnf["dataset"]["prediction_length"],
            context_length=mcnf["dataset"]["context_length"],
            target_prefixes=["ws_horz", "ws_vert"],
            feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
            freq=f"{int(measurements_timedelta.total_seconds())}s",
            target_suffixes=mcnf["dataset"]["target_turbine_ids"],
            per_turbine_target=False,
            as_lazyframe=True,
            dtype=pl.Float32,
            workers=4,
            pin_memory=True,
            persistent_workers=True,
            verbose=True,
        )

        if RUN_ONCE and not os.path.exists(data_module.train_ready_data_path) or args.reload_data:
            data_module.generate_datasets()
            logging.info("Reloading test datasets.")
            reload = True
        else:
            logging.info("Reading saved test datasets.")
            reload = False

        # data_module.train_ready_data_path=data_module.train_ready_data_path.replace("awaken_data/", "awaken_data/test/")
        # reload = True
        # reload = True
        data_module.generate_splits(save=True, reload=reload or args.resplit_data, splits=["test"])

        logging.info("Sorting test datasets by duration.")
        # data_module.test_dataset = sorted(data_module.test_dataset, key=lambda ds: ds["target"].shape[1], reverse=True)
        # data_module.test_dataset = sorted(data_module.test_dataset.partition_by("item_id"), key=lambda ds: ds.select(pl.len()).item(), reverse=True)
        if args.continuity_groups is not None:
            cgs = [int(x) for x in args.continuity_groups.split(",")]
            test_dataset = data_module.datasets["test"].filter(
                pl.col("item_id").str.extract("(\\d+)").cast(int).is_in(cgs)
            )
        else:
            test_dataset = (
                data_module.datasets["test"]
                .with_columns(pl.len().over("item_id").alias("cg_size"))
                .sort("cg_size", descending=True)
                .drop("cg_size")
            )

        test_dataset = [
            test_dataset.filter(pl.col("item_id") == item_id)
            for (item_id,) in test_dataset.select(pl.col("item_id").unique(maintain_order=True))
            .collect()
            .iter_rows()
        ]
        del data_module.datasets["test"]
        if args.max_splits:
            test_dataset = test_dataset[: args.max_splits]

        # new_ds = []
        # for ds in test_dataset:
        #     cg = ds.select(pl.col("item_id").first()).collect().item()
        #     if (cg not in cgs) or ():
        #         new_ds.append(ds)
        #         cgs.append(cg)
        #         # new_idx = -1
        #         # new_pred_len = mcnf["dataset"]["prediction_length"]
        #     else:
        #         joint_cgs.add(int(re.search("(?<=SPLIT)\\d+", cg).group()))

        # if len(new_ds) > 0:
        if args.max_steps:
            assert args.max_steps >= int(
                (max(context_timedeltas) + max(prediction_timedeltas)) / measurements_timedelta
            ), (
                f"max_steps, if provided, must allow for context_timedelta + max(prediction_timedelta) = {int((max(context_timedeltas) + max(prediction_timedeltas)) / measurements_timedelta)}"
            )
            # test_dataset = [ds.slice(0, args.max_steps) for ds in test_dataset]
            # new_ds = [slice_data_entry(ds, slice(0, args.max_steps)) for ds in new_ds]
            test_dataset = [ds.slice(0, args.max_steps) for ds in test_dataset]

        # test_data.append(new_ds)
        test_data.append(test_dataset)

        # logging.info(
        #     f"Generating dataframe with prediction_timedelta {mcnf['dataset']['prediction_length']}."
        # )
        # save_path = os.path.join(os.path.dirname( base_model_config["dataset"]["data_path"]), "test_data.parquet")
        # test_data[-1] = generate_wind_field_df(test_data[-1], data_module.target_cols, data_module.feat_dynamic_real_cols)
        test_data[-1] = (
            pl.concat(test_data[-1], how="vertical")
            .rename(
                {
                    **{f"target_{i}": col for i, col in enumerate(data_module.target_cols)},
                    **{
                        f"feat_dynamic_real_{i}": col
                        for i, col in enumerate(data_module.feat_dynamic_real_cols)
                    },
                }
            )
            .with_columns(continuity_group=pl.col("item_id").str.extract("SPLIT(\\d+)").cast(int))
            .drop("item_id")
            .with_columns(prediction_timedelta=pl.lit(mcnf["dataset"]["prediction_length"]))
        )
        # test_data[-1] = test_data[-1].with_columns(prediction_timedelta=pl.lit(mcnf["dataset"]["prediction_length"]))

    if False:
        fig, ax = plt.subplots(2, 1)
        for ds, ls in zip(test_data, ["-", "--"]):
            ds = ds.sort("time")
            ax[0].plot(
                ds.select(pl.col("time")).collect().to_numpy(),
                ds.select("ws_horz_wt001", "ws_horz_wt002", "ws_horz_wt003").collect().to_numpy(),
                linestyle=ls,
            )
            ax[1].plot(
                ds.select(pl.col("time")).collect().to_numpy(),
                ds.select("ws_vert_wt001", "ws_vert_wt002", "ws_vert_wt003").collect().to_numpy(),
                linestyle=ls,
            )
        plt.show()

    if not (args.run_validation or args.run_processing):
        exit(0)

    test_data = pl.concat(test_data, how="vertical")
    # test_data = test_data.with_columns(
    #     prediction_timedelta=pl.when(pl.col("continuity_group").is_in(joint_cgs))
    #     .then(pl.lit(-1))
    #     .otherwise(pl.col("prediction_timedelta"))
    # )
    # .write_parquet(save_path, statistics=False)
    # test_data = pl.scan_parquet(save_path)

    # window_length = model_config["dataset"]["prediction_length"] + model_config["dataset"].get("lead_time", 0)
    # window_length = int(test_data[0]["target"].shape[1] * (2/3))
    # _, test_template = split(test_data, offset=-window_length)
    # test_data = test_template.generate_instances(window_length, windows=1)
    logging.info("Deleting uneccesary attributes.")
    delattr(data_module, "datasets")
    gc.collect()
    logging.info("Finished creating datasets.")

    # assert pd.Timedelta(test_data[0]["start"].freq) == measurements_timedelta
    assert (
        pd.Timedelta(test_data.select(pl.col("time").diff()).slice(1, 1).collect().item())
        == measurements_timedelta
    )

    # assert test_data.select(pl.col("time").slice(0, 2).diff()).slice(1,1).item() == measurements_timedelta

    # custom_eval_fn = {
    #             "PICP": (pi_coverage_probability, "mean", "mean"),
    #             "PINAW": (pi_normalized_average_width, "mean", "mean"),
    #             "CWC": (coverage_width_criterion, "mean", "mean"),
    #             "CRPS": (continuous_ranked_probability_score_gaussian, "mean", "mean"),
    # }
    # evaluator = MultivariateEvaluator(
    #     custom_eval_fn=custom_eval_fn,
    #     num_workers=mp.cpu_count() if args.multiprocessor == "cf" else None,
    # )
    evaluator = None

    # if GPUs are available, use one CPU and one GPU per task
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_devices = os.environ[
            "CUDA_VISIBLE_DEVICES"
        ]  # Note: must 'export' variable within nohup to find on Kestrel
        logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
        try:
            # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
            visible_gpus = [idx for idx in cuda_devices.split(",") if idx.strip()]
            num_visible_gpus = len(visible_gpus)
            if num_visible_gpus > 0:
                # max_workers = num_visible_gpus
                # max_workers = MPI.COMM_WORLD.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
                max_workers = int(os.environ.get("SLURM_NTASKS_PER_NODE", num_visible_gpus))
                logging.info(f"Found {num_visible_gpus} GPUs. Setting max_workers {max_workers}.")
            else:
                max_workers = (
                    MPI.COMM_WORLD.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
                )
                logging.warning(
                    f"CUDA_VISIBLE_DEVICES is set but no valid GPU indices found. Setting max_workers to mp.cpu_count()={max_workers}."
                )
        except Exception as e:
            logging.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")

        # Create an iterator that cycles through the available GPU IDs
        gpu_cycler = cycle(visible_gpus)

    else:
        # max_workers = MPI.COMM_WORLD.Get_size() if args.multiprocessor == "mpi" else mp.cpu_count()
        max_workers = int(os.environ.get("N_PROCESSES", mp.cpu_count()))
        gpu_cycler = None

    logging.info(f"Using max_workers={max_workers}.")

    if args.run_validation or args.run_processing:
        forecasters = []
        ## GENERATE PERFECT PREVIEW \
        if "perfect" in args.model:
            for ctd, ptd in zip(context_timedeltas, prediction_timedeltas):
                logging.info(
                    f"Instantiating PerfectForecast with context_timedelta = {ctd}, prediction_timedelta = {ptd} seconds."
                )
                forecaster = PerfectForecast(
                    measurements_timedelta=measurements_timedelta,
                    controller_timedelta=controller_timedelta,
                    prediction_timedelta=ptd,
                    context_timedelta=ctd,
                    true_wind_field=test_data,
                    fmodel=fmodel,
                    tid2idx_mapping=tid2idx_mapping,
                    turbine_signature=turbine_signature,
                    use_tuned_params=False,
                    kwargs={},
                    target_turbine_indices=args.target_turbine_indices,
                )

                forecasters.append(forecaster)

        ## GENERATE PERSISTENT PREVIEW
        if "persistence" in args.model:
            for ctd, ptd in zip(context_timedeltas, prediction_timedeltas):
                logging.info(
                    f"Instantiating PersistenceForecast with context_timedelta = {ctd}, prediction_timedelta = {ptd} seconds."
                )
                forecaster = PersistenceForecast(
                    measurements_timedelta=measurements_timedelta,
                    controller_timedelta=controller_timedelta,
                    prediction_timedelta=ptd,
                    context_timedelta=ctd,
                    fmodel=fmodel,
                    true_wind_field=None,
                    tid2idx_mapping=tid2idx_mapping,
                    turbine_signature=turbine_signature,
                    use_tuned_params=False,
                    kwargs={},
                    target_turbine_indices=args.target_turbine_indices,
                )

                forecasters.append(forecaster)

        ## GENERATE SVR PREVIEW
        if "svr" in args.model:
            for mncf, ctd, ptd in zip(model_configs, context_timedeltas, prediction_timedeltas):
                logging.info(
                    f"Instantiating SVRForecast with context_timedelta = {ctd}, prediction_timedelta = {ptd} seconds."
                )
                forecaster = SVRForecast(
                    measurements_timedelta=measurements_timedelta,
                    controller_timedelta=controller_timedelta,
                    prediction_timedelta=ptd,
                    context_timedelta=ctd,
                    fmodel=fmodel,
                    true_wind_field=None,
                    kwargs=dict(
                        kernel=mncf["model"]["svr"]["kernel"],
                        C=mncf["model"]["svr"]["C"],
                        degree=mncf["model"]["svr"]["degree"],
                        gamma=mncf["model"]["svr"]["gamma"],
                        epsilon=mncf["model"]["svr"]["epsilon"],
                        cache_size=mncf["model"]["svr"]["cache_size"],
                        n_neighboring_turbines=mncf["model"]["svr"]["n_neighboring_turbines"],
                        max_n_samples=None,
                        use_trained_models=args.use_trained_models,
                        optuna_storage=None,
                        model_config=mncf,
                    ),
                    tid2idx_mapping=tid2idx_mapping,
                    turbine_signature=turbine_signature,
                    use_tuned_params=True,
                    target_turbine_indices=args.target_turbine_indices,
                )

                forecasters.append(forecaster)

        ## GENERATE KF PREVIEW
        if "kf" in args.model:
            # tune this use single, longer, prediction time, since we have only identity state transition matrix, and must use final posterior only prediction
            for ctd, ptd in zip(context_timedeltas, prediction_timedeltas):
                logging.info(
                    f"Instantiating KalmanFilterForecast with context_timedelta = {ctd}, prediction_timedelta = {ptd} seconds."
                )

                forecaster = KalmanFilterForecast(
                    measurements_timedelta=measurements_timedelta,
                    controller_timedelta=controller_timedelta,
                    prediction_timedelta=ptd,
                    context_timedelta=ctd,
                    fmodel=fmodel,
                    true_wind_field=None,
                    tid2idx_mapping=tid2idx_mapping,
                    turbine_signature=turbine_signature,
                    use_tuned_params=False,
                    kwargs={},
                    target_turbine_indices=args.target_turbine_indices,
                )
                forecasters.append(forecaster)

        ## GENERATE KF PREVIEW
        if "sf" in args.model:
            # tune this use single, longer, prediction time, since we have only identity state transition matrix, and must use final posterior only prediction
            for mncf, ctd, ptd in zip(model_configs, context_timedeltas, prediction_timedeltas):
                logging.info(
                    f"Instantiating SpatialFilterForecast with context_timedelta = {ctd}, prediction_timedelta = {ptd} seconds."
                )

                forecaster = SpatialFilterForecast(
                    measurements_timedelta=measurements_timedelta,
                    controller_timedelta=controller_timedelta,
                    prediction_timedelta=ptd,
                    context_timedelta=ctd,
                    fmodel=fmodel,
                    true_wind_field=None,
                    tid2idx_mapping=tid2idx_mapping,
                    turbine_signature=turbine_signature,
                    use_tuned_params=False,
                    kwargs=dict(
                        n_neighboring_turbines=mncf["model"]["sf"]["n_neighboring_turbines"]
                    ),
                    target_turbine_indices=args.target_turbine_indices,
                )
                forecasters.append(forecaster)

        ## GENERATE ML PREVIEW
        if any(
            ml_model in args.model
            for ml_model in ["informer", "autoformer", "spacetimeformer", "tactis"]
        ):
            ml_models = [
                ml_model
                for ml_model in args.model
                if ml_model in ["informer", "autoformer", "spacetimeformer", "tactis"]
            ]

            for m, model in enumerate(ml_models):
                for mncf, ctd, ptd in zip(model_configs, context_timedeltas, prediction_timedeltas):
                    forecaster = MLForecast(
                        measurements_timedelta=measurements_timedelta,
                        controller_timedelta=controller_timedelta,
                        prediction_timedelta=ptd,
                        context_timedelta=ctd,
                        fmodel=fmodel,
                        true_wind_field=None,
                        tid2idx_mapping=tid2idx_mapping,
                        turbine_signature=turbine_signature,
                        use_tuned_params=True,
                        kwargs=dict(
                            model_key=model,
                            model_checkpoint=args.checkpoint[0]
                            if len(args.checkpoint) == 1
                            else args.checkpoint[m],
                            optuna_storage=None,
                            study_name=None,  # db_setup_params["study_name"],
                            model_config=mncf,
                            resample=False,
                            cp_calibrate_stddev=args.cp_calibrate_stddev,
                        ),
                        target_turbine_indices=args.target_turbine_indices,
                    )
                    forecasters.append(forecaster)

    continuity_groups = (
        test_data.group_by("prediction_timedelta")
        .agg(pl.col("continuity_group").unique())
        .collect()
    )
    continuity_groups = {
        row["prediction_timedelta"]: row["continuity_group"]
        for row in continuity_groups.iter_rows(named=True)
    }
    if -1 in continuity_groups:
        joint_cgs = continuity_groups[-1]
        del continuity_groups[-1]
        for k in continuity_groups:
            continuity_groups[k] += joint_cgs

    if args.run_validation:
        validation_to_run = []
        for forecaster in forecasters:
            prediction_timedelta = int(forecaster.prediction_timedelta.total_seconds())
            forecaster_name = (
                forecaster.__class__.__name__
                if forecaster.__class__.__name__ != "MLForecast"
                else f"{forecaster.model_key.capitalize()}Forecast"
            )
            save_dir = os.path.join(validation_save_dir, forecaster_name, str(prediction_timedelta))
            os.makedirs(save_dir, exist_ok=True)
            for c, cg in enumerate(continuity_groups[prediction_timedelta]):
                save_path = os.path.join(save_dir, f"forecast_{cg}.parquet")

                if args.rerun_validation or not os.path.exists(save_path):
                    validation_to_run.append((forecaster, cg, save_path))
                    logging.info(
                        f"Rerunning validation {forecaster_name, prediction_timedelta, save_path}"
                    )

                    temp_sp = save_path.replace(".parquet", "_temp.parquet")
                    if os.path.exists(temp_sp):
                        logging.info(f"Removing existing temp file {temp_sp}.")
                        os.remove(temp_sp)

                    if os.path.exists(save_path):
                        logging.info(f"Removing existing final file {save_path}.")
                        os.remove(save_path)

                    # if os.path.exists(save_path):
                    #     logging.info(f"Removing existing file {save_path}.")
                    #     os.remove(save_path)
                # elif os.path.exists(save_path):
                #     # TODO also delete existing files if not rerun_validation but existing files have different number of time steps
                #     forecast_df = pl.scan_parquet(save_path, glob=True, try_parse_dates=True)\
                #                     .with_columns(time=pl.col("time").cast(pl.Datetime(time_unit="ns")))
                # if (n_forecasted_timestamps := forecast_df.select(pl.col("time").n_unique()).collect().item()) < (n_true_timestamps := test_data.select(pl.col("time").n_unique()).item() - 1):
                #     validation_to_run.append((forecaster, cg, save_path))
                #     logging.info(f"Rerunning validation {forecaster_name, prediction_timedelta, save_path} since saved number of timestamps is only {n_forecasted_timestamps} whereas number in test data is {n_true_timestamps}.")
                # logging.info(f"Removing existing file {save_path}.")
                # os.remove.exists(save_path)

    if args.run_validation:
        if args.multiprocessor:
            if args.multiprocessor == "mpi":
                # max_workers = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0, max_workers=max_workers)
            elif args.multiprocessor == "cf":
                # max_workers = mp.cpu_count()
                executor = ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=mp.get_context("spawn")
                )
                # max_tasks_per_child=1)
                # executor = get_context("spawn").Pool()

            logging.info(
                f"Running generate_forecaster_results with multiprocessor {args.multiprocessor} with {max_workers} workers for cases."
            )
            for forecaster, cg, save_path in validation_to_run:
                logging.info(
                    f"- forecaster {forecaster}, continuity group {cg}, save path {save_path}"
                )

            with executor as ex:
                test_futures = [
                    ex.submit(
                        make_predictions,
                        forecaster=forecaster,
                        test_data=test_data.filter(
                            (pl.col("continuity_group") == cg)
                            & (
                                pl.col("prediction_timedelta").is_in(
                                    [forecaster.prediction_timedelta.total_seconds(), -1.0]
                                )
                            )
                        ).collect(),
                        prediction_type=args.prediction_type,
                        single_cg=True,
                        save_path=save_path,
                        assigned_gpu=next(gpu_cycler) if gpu_cycler else None,
                        ram_limit=args.ram_limit,
                    )
                    for forecaster, cg, save_path in validation_to_run
                ]

                res = [fut.result() for fut in test_futures]

                logging.info(f"Collected all generate_forecaster_results.")

        else:
            logging.info(f"Running generate_forecaster_results with loop.")
            results = []
            for forecaster, cg, save_path in validation_to_run:
                make_predictions(
                    forecaster=forecaster,
                    test_data=test_data.filter(
                        (pl.col("continuity_group") == cg)
                        & (
                            pl.col("prediction_timedelta").is_in(
                                [forecaster.prediction_timedelta.total_seconds(), -1.0]
                            )
                        )
                    ).collect(),
                    prediction_type=args.prediction_type,
                    single_cg=True,
                    # save_path=lambda cg: forecast_paths[continuity_groups.index(cg)],
                    save_path=save_path,
                    assigned_gpu=next(gpu_cycler) if gpu_cycler else None,
                    ram_limit=args.ram_limit,
                )

        logging.info("Finished all make_predictions tasks.")

    # Load generated forecast dfs
    if args.run_processing and RUN_ONCE:
        results = []
        for forecaster in forecasters:
            prediction_timedelta = int(forecaster.prediction_timedelta.total_seconds())
            forecaster_name = (
                forecaster.__class__.__name__
                if forecaster.__class__.__name__ != "MLForecast"
                else f"{forecaster.model_key.capitalize()}Forecast"
            )
            save_dir = os.path.join(validation_save_dir, forecaster_name, str(prediction_timedelta))

            logging.info(
                f"Preparing results for forecaster {forecaster_name} and prediction_timedelta {prediction_timedelta}."
            )
            # forecast_path = os.path.join(save_dir, f"forecast_*.parquet")

            # logging.info(f"Loading forecast_df from {forecast_path}.")
            # forecast_df = pl.read_parquet(forecast_path, glob=True, try_parse_dates=True)\
            #                 .with_columns(time=pl.col("time").cast(pl.Datetime(time_unit="ns")))
            results.append(
                {
                    "forecaster_name": forecaster_name,
                    "prediction_timedelta": forecaster.prediction_timedelta.total_seconds(),
                    # "forecast_df": forecast_df
                }
            )
            # logging.info(f"Finished scanning parquet files at {forecast_path}. Found {forecast_df.select(pl.col('continuity_group').unique()).to_numpy().flatten()} continuity_groups.")
        # TODO possible to replace long_df with reading from multiple files via glob?

        # Generate agg_metrics for each forecaster
        unique_cgs = {}
        for f, forecaster in enumerate(forecasters):
            prediction_timedelta = forecaster.prediction_timedelta.total_seconds()
            forecaster_name = (
                forecaster.__class__.__name__
                if forecaster.__class__.__name__ != "MLForecast"
                else f"{forecaster.model_key.capitalize()}Forecast"
            )
            save_dir = os.path.join(
                validation_save_dir, forecaster_name, str(int(prediction_timedelta))
            )

            forecast_path = os.path.join(save_dir, "forecast_*.parquet")

            logging.info(f"Loading forecast_df from {forecast_path}.")
            # schema_overrides={"test_idx": pl.Int32, "continuity_group": pl.Int32}
            forecast_df = pl.read_parquet(forecast_path, glob=True).with_columns(
                time=pl.col("time").cast(pl.Datetime(time_unit="ns"))
            )

            # if forecaster.model_key == "tactis" and forecaster.cp_calibrate_stddev and forecaster.cp_scale_factors is not None:
            #         sd = pred.distribution.stddev  # [n_leads, n_all_target_cols]
            #         pred.distribution.stddev = sd * forecaster._cp_multiplier(
            #             forecaster.data_module.target_cols, sd.shape[0], sd.device, sd.dtype
            #         )

            available_fc_cgs = (
                forecast_df.select(pl.col("continuity_group").unique())
                .to_numpy()
                .flatten()
                .astype(int)
            )
            logging.info(
                f"Finished reading parquet files at {forecast_path}. Found {len(available_fc_cgs)} continuity_groups: {available_fc_cgs}."
            )

            if prediction_timedelta in unique_cgs:
                unique_cgs[prediction_timedelta] = unique_cgs[prediction_timedelta].intersection(
                    available_fc_cgs
                )
            else:
                unique_cgs[prediction_timedelta] = set(available_fc_cgs)

        all_unique_cgs = reduce(lambda x, y: x.union(y), list(unique_cgs.values()))
        test_data = test_data.filter(pl.col("continuity_group").is_in(all_unique_cgs))

        for f, forecaster in enumerate(forecasters):
            prediction_timedelta = forecaster.prediction_timedelta.total_seconds()
            forecaster_name = (
                forecaster.__class__.__name__
                if forecaster.__class__.__name__ != "MLForecast"
                else f"{forecaster.model_key.capitalize()}Forecast"
            )
            save_dir = os.path.join(
                validation_save_dir, forecaster_name, str(int(prediction_timedelta))
            )

            forecast_path = os.path.join(save_dir, "forecast_*.parquet")
            agg_metric_path = os.path.join(save_dir, "agg_metrics.parquet")

            logging.info(f"Loading forecast_df from {forecast_path}.")
            # schema_overrides={"test_idx": pl.Int32, "continuity_group": pl.Int32})\
            forecast_df = pl.read_parquet(forecast_path, glob=True).with_columns(
                time=pl.col("time").cast(pl.Datetime(time_unit="ns"))
            )

            available_fc_cgs = set(
                forecast_df.select(pl.col("continuity_group").unique()).to_numpy().flatten()
            )
            logging.info(
                f"Finished reading parquet files at {forecast_path}. Found {available_fc_cgs} continuity_groups."
            )

            # make sure comparing common continuity groups
            logging.info(f"Filtering continuity groups to {unique_cgs[prediction_timedelta]}.")
            forecast_df = forecast_df.filter(
                pl.col("continuity_group").is_in(unique_cgs[prediction_timedelta])
            )

            # recomputes agg metrics if existing agg_metric_path doesn't contain all cgs
            if os.path.exists(agg_metric_path):
                logging.info(f"Loading agg_metrics from {agg_metric_path}.")
                agg_metrics = pl.scan_parquet(
                    agg_metric_path,
                    schema={
                        "turbine_id": pl.String,
                        "test_idx": pl.Int32,
                        "continuity_group": pl.Int32,
                        "metric": pl.String,
                        "feature_type": pl.String,
                        "score": pl.Float32,
                    },
                ).collect()
                available_agg_cgs = set(
                    agg_metrics.select(pl.col("continuity_group").unique()).to_numpy().flatten()
                )
                logging.info(
                    f"Finished scanning parquet file at {agg_metric_path} with schema {agg_metrics.schema}."
                )
                logging.info(
                    f"Found {len(available_agg_cgs)} continuity groups: {available_agg_cgs}."
                )

                # if available agg_metrics contains all the continuity groups we require
                if (available_agg_cgs != unique_cgs[prediction_timedelta]) and unique_cgs[
                    prediction_timedelta
                ].issubset(available_agg_cgs):
                    agg_metrics = agg_metrics.filter(
                        pl.col("continuity_group").is_in(unique_cgs[prediction_timedelta])
                    )
                    agg_metrics.write_parquet(agg_metric_path)
                    available_agg_cgs = set(
                        agg_metrics.select(pl.col("continuity_group").unique()).to_numpy().flatten()
                    )

            if (
                args.rerun_validation
                or args.reaggregate_metrics
                or not os.path.exists(agg_metric_path)
                or (available_agg_cgs != unique_cgs[prediction_timedelta])
            ):
                logging.info(
                    f"Generating agg_metrics for forecaster {forecaster_name} and prediction_timedelta {prediction_timedelta} since agg_metric_path {agg_metric_path} doesn't exist or doesn't contain all required continuity groups."
                )
                target_cols = (
                    data_module.target_cols
                    if forecaster.target_turbine_indices is None
                    else [
                        f"{pfx}_{forecaster.idx2tid_mapping[idx]}"
                        for pfx in data_module.target_prefixes
                        for idx in forecaster.target_turbine_indices
                    ]
                )
                agg_metrics = generate_forecaster_agg_results(
                    forecaster,
                    forecast_df,
                    test_data.filter(
                        pl.col("continuity_group").is_in(unique_cgs[prediction_timedelta])
                    ).collect(),
                    target_cols,
                    args.prediction_type,
                )
                agg_metrics.write_parquet(agg_metric_path)

            results[f]["agg_metrics"] = agg_metrics

        all_metrics = (
            results[0]["agg_metrics"].select(pl.col("metric").unique()).to_numpy().flatten()
        )
        # get the metrics we care about, there is also "MSE", "MAE", "abs_error", "QuantileLoss",
        use_metrics = ["MAE", "CRPS"]  # ["MAE", "RMSE", "PINAW", "CWC", "CRPS", "PICP"]
        metrics = [metric for metric in all_metrics if any(m in metric for m in use_metrics)]

        agg_df = pl.concat(
            [
                res["agg_metrics"]
                .with_columns(
                    forecaster=pl.lit(res["forecaster_name"]),
                    prediction_timedelta=pl.lit(res["prediction_timedelta"]),
                )
                .select(
                    [
                        "forecaster",
                        "prediction_timedelta",
                        "turbine_id",
                        "test_idx",
                        "continuity_group",
                        "metric",
                        "feature_type",
                        "score",
                    ]
                )
                for res in results
            ],
            how="vertical",
        )

        # best_cg = agg_df.filter((pl.col("forecaster") == forecaster_name)
        #                         & (pl.col("prediction_timedelta")== forecaster.prediction_timedelta.total_seconds())
        #                         & (pl.col("metric") == "RMSE")
        #                         & (pl.col("turbine_id").is_in(turbine_ids)))\
        #     .group_by("continuity_group").agg(pl.col("score").mean()).select(pl.all().sort_by("score").first()).select("continuity_group").item()

        turbine_ids = ["wt005", "wt074", "wt075"]
        assert all(tid in data_module.target_suffixes for tid in turbine_ids), (
            f"Expected target turbine IDs {turbine_ids} to be a subset of {data_module.target_suffixes}."
        )

        best_cg = (
            agg_df.filter(
                (pl.col("metric").is_in(["CRPS", "RMSE"])) & (pl.col("turbine_id") == "all")
            )
            .select("forecaster", "metric", "prediction_timedelta", "score", "continuity_group")
            .sort("score")
            .group_by(["forecaster", "metric", "prediction_timedelta"], maintain_order=True)
            .agg(pl.all().head(2))
            .explode("score", "continuity_group")["continuity_group"]
            .value_counts()
            .sort("count", descending=True)["continuity_group"][0]
        )

        true_long_path = os.path.join(validation_save_dir, f"true_long_df_{args.run_name}.parquet")
        if args.rerun_validation or not os.path.exists(true_long_path):
            test_data.drop("feat_static_cat").unpivot(
                index=["time", "continuity_group", "prediction_timedelta"],
                variable_name="feature",
                value_name="value",
            ).with_columns(
                turbine_id=pl.col("feature").str.extract(
                    f"(_)({forecaster.turbine_signature})$", group_index=2
                ),
                feature=pl.col("feature").str.extract(
                    f"(.*)(_)({forecaster.turbine_signature})$", group_index=1
                ),
                data_type=pl.lit("True"),
            ).with_columns(
                cs.float().cast(pl.Float32), cs.integer().cast(pl.Int32)
            ).collect().write_parquet(true_long_path)

        true_long = pl.scan_parquet(
            true_long_path,
            schema={
                "time": pl.Datetime(time_unit="ns"),
                "prediction_timedelta": pl.Int32,
                "turbine_id": pl.String,
                "continuity_group": pl.Int32,
                "feature": pl.String,
                "value": pl.Float32,
                "turbine_id": pl.String,
                "data_type": pl.String,
            },
            glob=True,
        ).collect()

        # plot continuity group with best rmse score
        PLOT_INDIVIDUAL = True
        forecasts_long = []
        for f, forecaster in enumerate(forecasters):
            forecaster_name = (
                forecaster.__class__.__name__
                if forecaster.__class__.__name__ != "MLForecast"
                else f"{forecaster.model_key.capitalize()}Forecast"
            )
            prediction_timedelta = int(forecaster.prediction_timedelta.total_seconds())
            save_dir = os.path.join(validation_save_dir, forecaster_name, str(prediction_timedelta))
            if args.prediction_type == "distribution" and forecaster.is_probabilistic:
                value_vars = [
                    "nd_cos",
                    "nd_sin",
                    "loc_ws_horz",
                    "loc_ws_vert",
                    "sd_ws_horz",
                    "sd_ws_vert",
                ]
                target_vars = ["loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"]
            else:
                value_vars = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
                target_vars = ["ws_horz", "ws_vert"]

            forecast_long_path = os.path.join(save_dir, "long_df.parquet")
            if (
                args.rerun_validation
                or not os.path.exists(forecast_long_path)
                or best_cg
                not in pl.scan_parquet(forecast_long_path, glob=True)
                .select(pl.col("continuity_group").unique())
                .collect()
                .to_numpy()
            ):
                forecast_path = os.path.join(save_dir, "forecast_*.parquet")
                forecast_df = pl.scan_parquet(forecast_path, glob=True).with_columns(
                    time=pl.col("time").cast(pl.Datetime(time_unit="ns"))
                )

                assert (
                    forecast_df.select((pl.col("continuity_group") == pl.lit(best_cg)).any())
                    .collect()
                    .item()
                ), (
                    f"Chosen value of best_cg {best_cg} not found in forecast_df continuity_group column."
                )
                forecast_df.filter(pl.col("continuity_group") == best_cg).select(
                    ["time", "continuity_group", "test_idx"]
                    + [cs.ends_with(f"_{tid}") for tid in turbine_ids]
                ).unpivot(
                    index=["time", "continuity_group", "test_idx"],
                    variable_name="feature",
                    value_name="value",
                ).with_columns(
                    turbine_id=pl.col("feature").str.extract(
                        f"(_)({forecaster.turbine_signature})$", group_index=2
                    ),
                    feature=pl.col("feature").str.extract(
                        f"(.*)(_)({forecaster.turbine_signature})$", group_index=1
                    ),
                    data_type=pl.lit("Forecast"),
                    forecaster=pl.lit(forecaster_name),
                    prediction_timedelta=pl.lit(prediction_timedelta),
                ).with_columns(
                    cs.float().cast(pl.Float32), cs.integer().cast(pl.Int32)
                ).collect().write_parquet(forecast_long_path)

            # forecast_df = forecast_df.with_columns(prediction_timedelta=pl.lit(prediction_timedelta))
            # forecast_df.write_parquet(forecast_long_path)

            forecasts_long.append(
                pl.scan_parquet(forecast_long_path, glob=True).with_columns(
                    time=pl.col("time").cast(pl.Datetime(time_unit="ns")),
                    turbine_id=pl.col("turbine_id").cast(pl.String),
                )
            )

            plot_distr = forecaster.is_probabilistic and args.prediction_type == "distribution"

            if PLOT_INDIVIDUAL and (
                args.rerun_validation or len(glob.glob(f"{save_dir}/*.png")) < 2
            ):
                forecast_fig = WindForecast.plot_forecast(
                    forecasts_long[-1],
                    true_long,
                    continuity_groups=[best_cg],
                    turbine_ids=turbine_ids,
                    turbine_labels=["Greedy", "LUT Ds", "LUT Us"],
                    label=f"_{forecaster.__class__.__name__}_{data_config['config_label']}",
                    fig_dir=save_dir,
                    include_turbine_legend=True,
                    feature_types=["ws_horz", "ws_vert"],
                    feature_labels=["$u$ Wind Speed (m/s)", "$v$ Wind Speed (m/s)"],
                    prediction_type="distribution" if plot_distr else "point",
                    dt=15,
                )

        # plot combined
        # cg = agg_df.select(pl.col("continuity_group").first()).item()
        mean_cols = [
            f"{feat_type}_{tid}"
            for feat_type in ["loc_ws_horz", "loc_ws_vert"]
            for tid in data_module.target_suffixes
        ]
        point_cols = [
            f"{feat_type}_{tid}"
            for feat_type in ["ws_horz", "ws_vert"]
            for tid in data_module.target_suffixes
        ]
        PLOT_ALL = True

        if PLOT_ALL:
            logging.info("Concatenating forecasts together.")
            forecasts_long = pl.concat(forecasts_long, how="vertical")
            logging.info("Finished concatenating forecasts together.")
            logging.info("Plotting all forecasts.")
            forecast_fig = WindForecast.plot_forecast(
                forecasts_long.filter(
                    pl.col("prediction_timedelta") == int(prediction_timedeltas[-1].total_seconds())
                ).with_columns(pl.col("feature").str.replace("^(ws_)", "loc_ws_")),
                true_long,
                continuity_groups=[best_cg],
                turbine_ids=turbine_ids,
                turbine_labels=["Greedy", "LUT Ds", "LUT Us"],
                label=f"_{args.run_name}_{data_config['config_label']}",
                fig_dir=validation_save_dir,
                include_turbine_legend=True,
                feature_types=["ws_horz", "ws_vert"],
                feature_labels=["$u$ Wind Speed (m/s)", "$v$ Wind Speed (m/s)"],
                prediction_type="distribution",
                multiple_forecasters=True,
                dt=15,
            )

        PLOT_METRICS = True
        if PLOT_METRICS:
            logging.info("Plotting aggregate metrics for all forecasts.")
            plotting_metrics_dirs = [
                (met, direc)
                for met, direc in zip(
                    [
                        "MAE",
                        #  "RMSE",
                        # "PINAW", "PINAW_samples",
                        # "CWC", "CWC_samples",
                        "CRPS",
                        "CRPS_samples",
                        # "PICP", "PICP_samples"
                    ],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                )
                if met in agg_df["metric"].unique()
            ]
            plotting_metrics = [v[0] for v in plotting_metrics_dirs]
            ax_indices = [v[1] for v in plotting_metrics_dirs]
            # plt.close()

            totals_agg_df = (
                agg_df.filter((pl.col("test_idx") == -1) & (pl.col("turbine_id") == "all"))
                .group_by(["forecaster", "metric", "prediction_timedelta"])
                .agg(pl.col("score").mean())
            )

            # generate scatterplot of metric vs prediction time for different models (different colors) and different metrics (different_styles) (crps, picp, pinaw, cwc, mse, mae)
            if True:
                plot_score_vs_prediction_dt(
                    totals_agg_df,
                    metrics=plotting_metrics,
                    ax_indices=ax_indices,
                    fig_dir=validation_save_dir,
                )

            # best_prediction_dt = agg_df.groupby(["metric", "prediction_timedelta"])["score"].mean().idxmax()
            # generate grouped barcharpt of metrics (crps, picp, pinaw, cwc, mse, mae) grouped together vs model on x axis for best prediction time
            totals_agg_df.filter(
                pl.col("metric").is_in(["RMSE", "MAE", "CWC", "CRPS", "PINAW"])
            ).group_by(["forecaster", "metric"]).agg(pl.all().sort_by("score").last())
            totals_agg_df.filter(pl.col("metric").is_in(["PICP"])).group_by(
                ["forecaster", "metric"]
            ).agg(pl.all().sort_by("score").last())

            # best_prediction_dt = totals_agg_df.filter(pl.col("metric").is_in(["RMSE", "MAE", "CWC", "CRPS", "PINAW"])).group_by("prediction_timedelta").agg(pl.col("score").mean()).select(pl.col("prediction_timedelta").sort_by("score").first()).item()
            # totals_agg_df.filter(pl.col("prediction_timedelta") == best_prediction_dt),
            if True:
                plot_score_vs_forecaster(
                    totals_agg_df.filter(
                        pl.col("metric").is_in(
                            [
                                # "RMSE",
                                "MAE",
                                # "CWC", "PINAW", "PICP",
                                "CRPS",
                                # "CWC_samples", "PINAW_samples", "PICP_samples",
                                "CRPS_samples",
                            ]
                        )
                    ),
                    metrics=plotting_metrics,
                    ax_indices=ax_indices,
                    prediction_intervals=totals_agg_df.select(
                        pl.col("prediction_timedelta").unique()
                    )
                    .to_numpy()
                    .flatten(),
                    fig_dir=validation_save_dir,
                    label=f"_{args.run_name}_{data_config['config_label']}",
                )

        print("here")
