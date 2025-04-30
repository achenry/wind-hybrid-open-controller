"""Utility functions for plotting wind forecast results."""

import os
import re
import logging
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Confidence Interval Z-scores (assuming normality)
CI_Z_SCORES = {
    50: norm.ppf(0.75),  # Z for 50% CI (+/- 0.674*SD)
    90: norm.ppf(0.95)   # Z for 90% CI (+/- 1.645*SD)
}
CI_COLORS = {
    50: 'lightblue',
    90: 'lightsteelblue'
}
CI_ALPHA = {
    50: 0.6,
    90: 0.3
}


def _generate_dynamic_title(base_title, model_name=None, turbine_id=None, feature_name=None, date_range=None):
    """Helper to generate dynamic plot titles."""
    title = base_title
    if turbine_id is not None:
        title += f" - Turbine {turbine_id}"
    if feature_name is not None:
        title += f" - {feature_name.replace('_', ' ').title()}"
    if model_name is not None:
        title += f" ({model_name})"
    if date_range:
        start_date, end_date = date_range
        date_format = "%Y-%m-%d %H:%M"
        title += f"\n({start_date.strftime(date_format)} to {end_date.strftime(date_format)})"
    return title

def _extract_unit(label_string):
    """Helper to extract unit from axis label like 'Speed (m/s)'."""
    match = re.search(r'\((.*?)\)', label_string)
    return f" ({match.group(1)})" if match else ""


def plot_forecast(forecast_wf, true_wf, continuity_groups, turbine_ids, label, fig_dir,
                  feature_types, feature_labels, prediction_type="point",
                  include_turbine_legend=True, multiple_forecasters=False,
                  model_name=None, date_range=None):
    """
    Plots forecasts against true values for specified features and turbines.

    Args:
        forecast_wf (pl.DataFrame): DataFrame with forecast data. Expected columns:
            time, turbine_id, feature, value, data_type, forecaster (optional),
            loc_<feature> (if prediction_type='distribution'),
            sd_<feature> (if prediction_type='distribution').
        true_wf (pl.DataFrame): DataFrame with true values. Expected columns:
            time, turbine_id, feature, value, data_type.
        continuity_groups (list): List of continuity groups to plot.
        turbine_ids (list): List of turbine IDs to plot.
        label (str): Label for saving the figure.
        fig_dir (str): Directory to save the figure.
        feature_types (list): List of base feature names (e.g., 'ws_horz').
        feature_labels (list): List of corresponding labels for y-axis (e.g., 'Horizontal Wind Speed (m/s)').
        prediction_type (str): 'point' or 'distribution'.
        include_turbine_legend (bool): Whether to include a legend for turbines.
        multiple_forecasters (bool): If True, plot forecasts from different models.
        model_name (str, optional): Name of the model for dynamic title.
        date_range (tuple, optional): Tuple of (start_datetime, end_datetime) for dynamic title.
    """
    n_features = len(feature_types)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes] # Ensure axes is always iterable

    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(turbine_ids) * (len(results.keys()) if multiple_forecasters else 1))
    color_idx = 0
    handles = []
    labels = []

    for cg in continuity_groups:
        cg_true_df = true_wf.filter(pl.col("continuity_group") == cg)
        cg_forecast_df = forecast_wf # Assuming forecast_wf is already filtered for the relevant CG or contains all

        for i, (feat_type, feat_label) in enumerate(zip(feature_types, feature_labels)):
            ax = axes[i]
            unit = _extract_unit(feat_label)
            loc_col = f"loc_{feat_type}"
            sd_col = f"sd_{feat_type}"

            # Determine forecasters to plot
            if multiple_forecasters:
                forecasters_to_plot = cg_forecast_df['forecaster'].unique().to_list()
            else:
                # If not multiple_forecasters, assume a single forecaster or point prediction
                forecasters_to_plot = [model_name if model_name else "Forecast"] # Use provided model_name or default

            # Plot True Values first
            true_plotted = False
            for tid_idx, tid in enumerate(turbine_ids):
                true_turbine_df = cg_true_df.filter((pl.col("turbine_id") == tid) & (pl.col("feature") == feat_type))
                if not true_turbine_df.is_empty():
                    line, = ax.plot(true_turbine_df["time"], true_turbine_df["value"],
                                    label=f"True T{tid}" if not true_plotted else "", # Only label true once per feature
                                    color='black', linestyle='-', linewidth=1.5, marker='.', markersize=3, zorder=10)
                    if not true_plotted:
                        handles.append(line)
                        labels.append("True")
                        true_plotted = True

            # Plot Forecasts
            forecaster_handles = {}
            forecaster_labels = {}
            turbine_handles = {}
            turbine_labels = {}

            for f_idx, forecaster_name_plot in enumerate(forecasters_to_plot):
                forecaster_df = cg_forecast_df.filter(pl.col("forecaster") == forecaster_name_plot) if multiple_forecasters else cg_forecast_df

                for tid_idx, tid in enumerate(turbine_ids):
                    color = palette[color_idx % len(palette)]
                    color_idx += 1

                    forecast_turbine_df = forecaster_df.filter(pl.col("turbine_id") == tid)

                    if prediction_type == "distribution" and loc_col in forecast_turbine_df.columns and sd_col in forecast_turbine_df.columns:
                        plot_df = forecast_turbine_df.filter(pl.col("feature") == loc_col).sort("time")
                        sd_df = forecast_turbine_df.filter(pl.col("feature") == sd_col).sort("time")

                        if not plot_df.is_empty() and not sd_df.is_empty():
                            # Ensure alignment on time if needed (should be aligned if processed correctly)
                            merged_df = plot_df.join(sd_df.select(["time", "value"]).rename({"value": "sd_val"}), on="time", how="inner")

                            # Plot mean forecast
                            line, = ax.plot(merged_df["time"], merged_df["value"],
                                            label=f"{forecaster_name_plot} T{tid}", color=color, linestyle='--', linewidth=1)

                            # Plot confidence intervals
                            for ci, z_score in sorted(CI_Z_SCORES.items(), reverse=True): # Plot wider CI first
                                lower_bound = merged_df["value"] - z_score * merged_df["sd_val"]
                                upper_bound = merged_df["value"] + z_score * merged_df["sd_val"]
                                fill = ax.fill_between(merged_df["time"], lower_bound, upper_bound,
                                                       color=CI_COLORS.get(ci, color), alpha=CI_ALPHA.get(ci, 0.3),
                                                       label=f"{ci}% CI" if tid_idx == 0 and f_idx == 0 else "", # Label CI only once
                                                       linewidth=0)
                                if tid_idx == 0 and f_idx == 0 and f"{ci}% CI" not in labels: # Add CI label to main legend once
                                     handles.append(fill)
                                     labels.append(f"{ci}% CI")

                            # Store handles/labels for legends
                            if multiple_forecasters and forecaster_name_plot not in forecaster_labels:
                                forecaster_handles[forecaster_name_plot] = line
                                forecaster_labels[forecaster_name_plot] = forecaster_name_plot
                            if include_turbine_legend and tid not in turbine_labels:
                                turbine_handles[tid] = plt.Line2D([0], [0], color=color, lw=2) # Create proxy artist
                                turbine_labels[tid] = f"Turbine {tid}"


                    elif prediction_type == "point":
                        plot_df = forecast_turbine_df.filter(pl.col("feature") == feat_type).sort("time")
                        if not plot_df.is_empty():
                            line, = ax.plot(plot_df["time"], plot_df["value"],
                                            label=f"{forecaster_name_plot} T{tid}", color=color, linestyle='--', linewidth=1)
                            # Store handles/labels for legends
                            if multiple_forecasters and forecaster_name_plot not in forecaster_labels:
                                forecaster_handles[forecaster_name_plot] = line
                                forecaster_labels[forecaster_name_plot] = forecaster_name_plot
                            if include_turbine_legend and tid not in turbine_labels:
                                turbine_handles[tid] = plt.Line2D([0], [0], color=color, lw=2) # Create proxy artist
                                turbine_labels[tid] = f"Turbine {tid}"

            # Dynamic Title and Labels for each subplot
            ax.set_ylabel(feat_label)
            ax.set_title(_generate_dynamic_title(f"Forecast vs True", model_name=model_name if not multiple_forecasters else None,
                                                 feature_name=feat_type, date_range=date_range)) # Turbine ID in legend if needed

    axes[-1].set_xlabel("Time")
    axes[-1].tick_params(axis='x', rotation=30)

    # Create Legends
    legend_elements = [handles[labels.index(lbl)] for lbl in ["True"] + [f"{ci}% CI" for ci in sorted(CI_Z_SCORES.keys())] if lbl in labels]
    legend_labels = [lbl for lbl in ["True"] + [f"{ci}% CI" for ci in sorted(CI_Z_SCORES.keys())] if lbl in labels]

    if multiple_forecasters:
        legend_elements.extend(forecaster_handles.values())
        legend_labels.extend(forecaster_labels.values())

    if include_turbine_legend:
        legend_elements.extend(turbine_handles.values())
        legend_labels.extend(turbine_labels.values())

    # Place legend outside plot
    fig.legend(legend_elements, legend_labels, loc='upper left', bbox_to_anchor=(1.01, 0.95), frameon=False, title="Legend")

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    fig_path = os.path.join(fig_dir, f'forecast_comparison{label}.png')
    logging.info(f"Saving plot_forecast to {fig_path}")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def plot_turbine_data(*args, **kwargs):
    """Placeholder for plot_turbine_data if needed, currently unused based on main block."""
    logging.warning("plot_turbine_data is called but not fully implemented based on current usage.")
    # Add implementation if required later
    pass


def plot_wind_ts(data_df, save_path, turbine_ids="all", include_filtered_wind_dir=True, controller_timedelta=None, legend_loc="best", single_plot=False, fig=None, ax=None, case_label=None, model_name=None, date_range=None):
    """Plots wind direction time series."""
    colors = sns.color_palette("Paired")
    # Ensure enough colors if many turbines
    if isinstance(turbine_ids, list) and len(turbine_ids) > len(colors):
        colors = sns.color_palette("tab20", n_colors=len(turbine_ids))
    elif turbine_ids == "all":
         unique_turbines = data_df['turbine_id'].unique().to_list()
         if len(unique_turbines) > len(colors):
              colors = sns.color_palette("tab20", n_colors=len(unique_turbines))
         turbine_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(unique_turbines)}
    else:
         turbine_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(turbine_ids)}


    if not single_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    plot_seed = data_df["continuity_group"].unique()[0] # Plot only the first continuity group if multiple exist

    plot_df = data_df.filter(data_df["continuity_group"] == plot_seed)
    if turbine_ids != "all":
        plot_df = plot_df.filter(pl.col("turbine_id").is_in(turbine_ids))

    # Plot raw wind direction
    wd_df = plot_df.filter(pl.col("feature") == "wd")
    if not wd_df.is_empty():
        sns.lineplot(data=wd_df.to_pandas(), x="time", y="value", hue="turbine_id",
                     palette=turbine_color_map if turbine_ids != "all" else None, # Use map if specific turbines
                     linestyle='-', legend=False, ax=ax) # Legend handled manually

    # Plot filtered wind direction if requested and available
    if include_filtered_wind_dir:
        wd_filt_df = plot_df.filter(pl.col("feature") == "wd_filt")
        if not wd_filt_df.is_empty():
            sns.lineplot(data=wd_filt_df.to_pandas(), x="time", y="value", hue="turbine_id",
                         palette=turbine_color_map if turbine_ids != "all" else None,
                         linestyle='--', legend=False, ax=ax)

    # Manual Legend Creation
    handles = []
    labels = []

    # Feature Handles
    if not wd_df.is_empty():
        handles.append(plt.Line2D([0], [0], color='grey', linestyle='-', lw=2))
        labels.append("Raw WD")
    if include_filtered_wind_dir and not wd_filt_df.is_empty():
        handles.append(plt.Line2D([0], [0], color='grey', linestyle='--', lw=2))
        labels.append("Filtered WD")

    # Turbine Handles (if plotting specific turbines)
    if turbine_ids != "all":
        for tid in turbine_ids:
            if tid in plot_df['turbine_id'].unique().to_list(): # Check if turbine has data
                 handles.append(plt.Line2D([0], [0], color=turbine_color_map[tid], lw=2))
                 labels.append(f"Turbine {tid}")
    elif len(plot_df['turbine_id'].unique()) <= 10: # Show all turbines if <= 10
         for tid in plot_df['turbine_id'].unique().to_list():
              handles.append(plt.Line2D([0], [0], color=turbine_color_map[tid], lw=2))
              labels.append(f"Turbine {tid}")


    # Dynamic title and labels
    ax.set_ylabel("Direction (°)")
    ax.set_xlabel("Time")
    ax.set_title(_generate_dynamic_title("Wind Direction Time Series", model_name=model_name, date_range=date_range))
    ax.tick_params(axis='x', rotation=30)

    # Place legend outside
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 0.95), frameon=False, title="Legend")

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    fig_path = os.path.join(fig_dir, f'wind_direction_ts{case_label or ""}.png')
    logging.info(f"Saving plot_wind_ts to {fig_path}")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_score_vs_prediction_dt(agg_df, metrics, ax_indices, fig_dir, model_name=None, date_range=None):
    """Plots model scores against prediction length."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    plot_data = agg_df.filter(pl.col("metric").is_in(metrics)).to_pandas()

    sns.scatterplot(data=plot_data, y="score", x="prediction_timedelta", style="metric", hue="forecaster", s=200, ax=ax)

    # Dynamic title and labels
    ax.set_title(_generate_dynamic_title("Model Performance vs. Prediction Length", model_name=model_name, date_range=date_range))
    ax.set_ylabel("Score")
    ax.set_xlabel("Prediction Length (s)")

    h, l = ax.get_legend_handles_labels()
    ax.set_xticks(agg_df.select(pl.col("prediction_timedelta").unique()).sort("prediction_timedelta").to_numpy().flatten())

    # Improve legend labels (simplified)
    new_labels = [lbl.replace("_", " ").title() if lbl != "forecaster" else "Forecaster" for lbl in l]

    # Place legend outside
    ax.legend(h, new_labels, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust layout for legend and title
    fig_path = os.path.join(fig_dir, "score_vs_pred_length.png")
    logging.info(f"Saving plot_score_vs_prediction_dt to {fig_path}")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_score_vs_forecaster(agg_df, metrics, ax_indices, prediction_intervals, fig_dir, model_name=None, date_range=None):
    """Plots forecaster performance comparison for different prediction intervals."""
    sns.set_style("whitegrid")
    figs = []
    for pred_int in prediction_intervals:
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure for each interval

        plot_data = agg_df.filter((pl.col("metric").is_in(metrics)) & (pl.col("prediction_timedelta") == pred_int)).to_pandas()

        if plot_data.empty:
            logging.warning(f"No data to plot for prediction interval {pred_int}. Skipping.")
            plt.close(fig)
            continue

        sns.barplot(data=plot_data, hue="metric", x="forecaster", y="score", log_scale=True, ax=ax)

        # Dynamic title and labels
        ax.set_title(_generate_dynamic_title(f"Forecaster Performance Comparison (Prediction Length: {int(pred_int)}s)", model_name=model_name, date_range=date_range))
        ax.set_ylabel("Score (Log Scale)")
        ax.set_xlabel("Forecaster")

        h, l = ax.get_legend_handles_labels()

        # Improve legend labels (simplified)
        new_labels = [lbl.replace("_", " ").title() if lbl != "metric" else "Metric" for lbl in l]

        # Improve x-tick labels (forecaster names) - simplified
        xtick_labels = [label.get_text().replace("Forecast", "").replace("_", " ").strip() for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels, rotation=30, ha='right')

        # Place legend outside
        ax.legend(h, new_labels, title="Metric", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust layout
        figs.append(fig) # Append the figure object

        fig_path = os.path.join(fig_dir, f"score_vs_forecaster_pred{int(pred_int)}.png")
        logging.info(f"Saving plot_score_vs_forecaster to {fig_path}")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory

    return figs


def plot_error_distribution(results_df, feature, turbine_id, unit, fig_dir, label="", model_name=None, date_range=None):
    """Plots the distribution of prediction errors."""
    loc_col = f"loc_{feature}"
    actual_col = f"actual_{feature}"

    if loc_col not in results_df.columns or actual_col not in results_df.columns:
        logging.warning(f"Required columns '{loc_col}' or '{actual_col}' not found for error distribution plot. Skipping.")
        return None

    # Ensure columns exist before filtering
    if not all(c in results_df.columns for c in [loc_col, actual_col, "turbine_id"]):
         logging.warning(f"Missing required columns for error plot (turbine_id, {loc_col}, or {actual_col}). Skipping.")
         return None

    turbine_df = results_df.filter(pl.col("turbine_id") == turbine_id)
    if turbine_df.is_empty():
        logging.warning(f"No data found for turbine {turbine_id} for error distribution plot. Skipping.")
        return None

    # Calculate errors after filtering, drop nulls specifically for error calculation
    errors = (turbine_df[loc_col] - turbine_df[actual_col]).drop_nulls()

    if errors.is_empty():
         logging.warning(f"No non-null errors calculated for turbine {turbine_id}, feature {feature}. Skipping error plot.")
         return None

    mean_error = errors.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors.to_numpy(), kde=True, ax=ax, stat="density", bins=30)
    ax.axvline(mean_error, color='r', linestyle='--', label=f'Mean Error (Bias): {mean_error:.2f}{unit}')

    # Dynamic title and labels
    ax.set_title(_generate_dynamic_title("Prediction Error Distribution", model_name=model_name, turbine_id=turbine_id, feature_name=feature, date_range=date_range))
    ax.set_xlabel(f"Error{unit}")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'error_distribution_{turbine_id}_{feature}{label}.png')
    logging.info(f"Saving plot_error_distribution to {fig_path}")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig

def plot_forecast_vs_actual_scatter(results_df, feature, turbine_id, unit, fig_dir, label="", model_name=None, date_range=None):
    """Generates a scatter plot of forecast mean vs. actual values."""
    loc_col = f"loc_{feature}"
    actual_col = f"actual_{feature}"

    # Ensure columns exist before filtering
    if not all(c in results_df.columns for c in [loc_col, actual_col, "turbine_id"]):
         logging.warning(f"Missing required columns for scatter plot (turbine_id, {loc_col}, or {actual_col}). Skipping.")
         return None

    turbine_df = results_df.filter(pl.col("turbine_id") == turbine_id).drop_nulls(subset=[loc_col, actual_col])

    if turbine_df.is_empty():
        logging.warning(f"No non-null data found for turbine {turbine_id}, feature {feature} for scatter plot. Skipping.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=turbine_df.to_pandas(), x=actual_col, y=loc_col, ax=ax, alpha=0.6, s=10, edgecolor='none') # Removed edgecolor for clarity

    # Add y=x line
    min_val = min(turbine_df[actual_col].min(), turbine_df[loc_col].min())
    max_val = max(turbine_df[actual_col].max(), turbine_df[loc_col].max())
    lims = [min_val, max_val]

    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Forecast (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    # Dynamic title and labels
    ax.set_title(_generate_dynamic_title("Forecast vs Actual", model_name=model_name, turbine_id=turbine_id, feature_name=feature, date_range=date_range))
    ax.set_xlabel(f"Actual Value{unit}")
    ax.set_ylabel(f"Forecast Mean{unit}")
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'forecast_vs_actual_{turbine_id}_{feature}{label}.png')
    logging.info(f"Saving plot_forecast_vs_actual_scatter to {fig_path}")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig

def plot_forecast_samples(forecast_samples_df, true_wf, feature, turbine_id, unit, fig_dir, label="", model_name=None, date_range=None, n_samples_to_plot=20):
    """Plots individual forecast sample paths against true values."""

    # Check required columns
    if not all(c in forecast_samples_df.columns for c in ["time", "turbine_id", "sample_id", feature]):
         logging.warning(f"Missing required columns in forecast_samples_df (time, turbine_id, sample_id, {feature}). Skipping sample plot.")
         return None
    if not all(c in true_wf.columns for c in ["time", "turbine_id", "feature", "value"]):
         logging.warning("Missing required columns in true_wf (time, turbine_id, feature, value). Skipping sample plot.")
         return None

    turbine_samples_df = forecast_samples_df.filter(pl.col("turbine_id") == turbine_id)
    if turbine_samples_df.is_empty():
        logging.warning(f"No sample data found for turbine {turbine_id}, feature {feature}. Skipping sample plot.")
        return None

    sample_ids = turbine_samples_df['sample_id'].unique().to_list()
    if len(sample_ids) > n_samples_to_plot:
        plot_sample_ids = np.random.choice(sample_ids, n_samples_to_plot, replace=False)
        plot_df = turbine_samples_df.filter(pl.col('sample_id').is_in(plot_sample_ids))
    else:
        plot_df = turbine_samples_df

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual samples
    sns.lineplot(data=plot_df.to_pandas(), x='time', y=feature, hue='sample_id', palette='viridis', alpha=0.3, legend=False, ax=ax)

    # Plot true values
    true_turbine_df = true_wf.filter((pl.col("turbine_id") == tid) & (pl.col("feature") == feature)).sort("time")
    if not true_turbine_df.is_empty():
        ax.plot(true_turbine_df["time"], true_turbine_df["value"], color='red', linewidth=2, label='True Value', zorder=10)

    # Plot forecast mean (calculated from plotted samples)
    mean_forecast = plot_df.group_by('time').agg(pl.mean(feature).alias('mean_forecast')).sort("time")
    if not mean_forecast.is_empty():
        ax.plot(mean_forecast["time"], mean_forecast["mean_forecast"], color='blue', linestyle='--', linewidth=1.5, label='Mean Forecast (Samples)', zorder=5)

    # Dynamic title and labels
    ax.set_title(_generate_dynamic_title("Forecast Samples vs Actual", model_name=model_name, turbine_id=turbine_id, feature_name=feature, date_range=date_range))
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value{unit}")
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'forecast_samples_{turbine_id}_{feature}{label}.png')
    logging.info(f"Saving plot_forecast_samples to {fig_path}")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig