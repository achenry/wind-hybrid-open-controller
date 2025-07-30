#!/usr/bin/env python3
"""
Comprehensive TACTiS wind forecast analysis and visualization suite.
Generates all deterministic and probabilistic metrics plots in one script.
"""

import os
import sys
import argparse
from datetime import timedelta
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import random
from scipy import stats

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')

def load_forecast_and_true_data(results_dir, continuity_group=0):
    """Load forecast samples and ground truth data."""
    forecast_path = os.path.join(results_dir, f"forecast_{continuity_group}.parquet")
    true_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "true_long_df_.parquet")
    
    print(f"Loading forecast data from: {forecast_path}")
    print(f"Loading true data from: {true_path}")
    
    # Load forecast data
    forecast_df = pl.read_parquet(forecast_path)
    print(f"Forecast shape: {forecast_df.shape}")
    
    # Load true data
    true_df = pl.read_parquet(true_path)
    print(f"True data shape: {true_df.shape}")
    
    return forecast_df, true_df

def load_aggregated_metrics(results_dir):
    """Load pre-computed aggregated metrics."""
    metrics_path = os.path.join(results_dir, "agg_metrics.parquet")
    if os.path.exists(metrics_path):
        return pl.read_parquet(metrics_path)
    else:
        print(f"Warning: No aggregated metrics found at {metrics_path}")
        return None

def select_random_time_regions(forecast_df, n_regions=4, region_duration_hours=1.5, seed=42):
    """Select random time regions for detailed visualization."""
    random.seed(seed)
    np.random.seed(seed)
    
    forecast_pd = forecast_df.to_pandas()
    forecast_pd['time'] = pd.to_datetime(forecast_pd['time'])
    
    unique_times = sorted(forecast_pd['time'].unique())
    region_duration = timedelta(hours=region_duration_hours)
    
    max_start_idx = len(unique_times) - int(region_duration_hours * 60 / 10)
    if max_start_idx <= 0:
        max_start_idx = len(unique_times) // 2
    
    selected_regions = []
    selected_indices = random.sample(range(0, max_start_idx), min(n_regions, max_start_idx))
    
    for idx in selected_indices:
        start_time = unique_times[idx]
        end_time = start_time + region_duration
        if end_time <= unique_times[-1]:
            selected_regions.append((start_time, end_time))
    
    return selected_regions

def compute_additional_metrics(forecast_df, true_df, continuity_group=0):
    """Compute additional deterministic and probabilistic metrics."""
    
    # Convert to pandas
    forecast_pd = forecast_df.to_pandas()
    forecast_pd['time'] = pd.to_datetime(forecast_pd['time'])
    
    true_pd = true_df.to_pandas()
    true_pd['time'] = pd.to_datetime(true_pd['time'])
    true_pd = true_pd[true_pd['continuity_group'] == continuity_group]
    
    # Pivot true data
    true_pivot = true_pd.pivot_table(index='time', columns=['feature', 'turbine_id'], values='value').reset_index()
    true_pivot.columns = ['time'] + [f"{feat}_{turb}" for feat, turb in true_pivot.columns[1:]]
    
    # Calculate sample statistics
    forecast_stats = forecast_pd.groupby('time').agg({
        col: ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 25),
              lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 95)]
        for col in forecast_pd.columns 
        if col.startswith(('ws_horz_', 'ws_vert_')) and col in forecast_pd.columns
    }).reset_index()
    
    # Flatten column names
    forecast_stats.columns = ['time'] + [f"{col[0]}_{col[1]}" for col in forecast_stats.columns[1:]]
    
    # Merge with true data
    merged_data = pd.merge(forecast_stats, true_pivot, on='time', how='inner')
    
    # Calculate metrics
    metrics_data = []
    wind_features = ['ws_horz', 'ws_vert']
    turbine_ids = [1, 2, 3, 4, 5, 6, 7]
    
    for feature in wind_features:
        for turbine_id in turbine_ids:
            col_base = f"{feature}_{turbine_id}"
            
            if f"{col_base}_mean" in merged_data.columns and col_base in merged_data.columns:
                pred_mean = merged_data[f"{col_base}_mean"].values
                pred_std = merged_data[f"{col_base}_std"].values
                true_vals = merged_data[col_base].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred_mean) | np.isnan(true_vals) | np.isnan(pred_std))
                if np.sum(valid_mask) > 0:
                    pred_mean = pred_mean[valid_mask]
                    pred_std = pred_std[valid_mask]
                    true_vals = true_vals[valid_mask]
                    
                    # Deterministic metrics
                    mae = np.mean(np.abs(pred_mean - true_vals))
                    rmse = np.sqrt(np.mean((pred_mean - true_vals)**2))
                    bias = np.mean(pred_mean - true_vals)
                    corr = np.corrcoef(pred_mean, true_vals)[0, 1] if len(pred_mean) > 1 else 0
                    
                    # Probabilistic metrics (simplified)
                    # Coverage probability for 90% interval
                    q05 = merged_data[f"{col_base}_<lambda_0>"][valid_mask]
                    q95 = merged_data[f"{col_base}_<lambda_1>"][valid_mask]
                    coverage_90 = np.mean((true_vals >= q05) & (true_vals <= q95))
                    
                    # Interval width
                    interval_width = np.mean(q95 - q05)
                    
                    # Reliability (simplified)
                    reliability = np.abs(coverage_90 - 0.9)
                    
                    metrics_data.append({
                        'feature': feature,
                        'turbine_id': turbine_id,
                        'mae': mae,
                        'rmse': rmse,
                        'bias': bias,
                        'correlation': corr,
                        'coverage_90': coverage_90,
                        'interval_width': interval_width,
                        'reliability': reliability,
                        'n_samples': len(pred_mean)
                    })
    
    return pd.DataFrame(metrics_data)

def plot_comprehensive_metrics(agg_metrics_df, computed_metrics_df=None, save_dir=None, continuity_group=0):
    """Create comprehensive metrics visualization."""
    
    if agg_metrics_df is None:
        print("No aggregated metrics available for plotting")
        return None, None
    
    # Convert to pandas if needed
    if hasattr(agg_metrics_df, 'to_pandas'):
        metrics_pd = agg_metrics_df.to_pandas()
    else:
        metrics_pd = agg_metrics_df
    
    # Create comprehensive metrics plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    # Plot 1: MAE by turbine and feature
    ax = axes[0]
    mae_data = metrics_pd[metrics_pd['metric'] == 'MAE']
    if len(mae_data) > 0:
        sns.barplot(data=mae_data, x='turbine_id', y='score', hue='feature_type', ax=ax)
        ax.set_title('Mean Absolute Error by Turbine', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (m/s)')
        ax.legend(title='Feature')
    
    # Plot 2: RMSE by turbine and feature
    ax = axes[1]
    rmse_data = metrics_pd[metrics_pd['metric'] == 'RMSE']
    if len(rmse_data) > 0:
        sns.barplot(data=rmse_data, x='turbine_id', y='score', hue='feature_type', ax=ax)
        ax.set_title('Root Mean Square Error by Turbine', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (m/s)')
        ax.legend(title='Feature')
    
    # Plot 3: Probabilistic metrics if available
    ax = axes[2]
    prob_metrics = ['PICP', 'PINAW', 'CWC', 'CRPS']
    available_prob = [m for m in prob_metrics if m in metrics_pd['metric'].values]
    
    if available_prob:
        prob_data = metrics_pd[metrics_pd['metric'].isin(available_prob)]
        sns.boxplot(data=prob_data, x='metric', y='score', ax=ax)
        ax.set_title('Probabilistic Metrics Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, 'No probabilistic\nmetrics available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Probabilistic Metrics', fontsize=12, fontweight='bold')
    
    # Plot 4: Error distribution
    ax = axes[3]
    if computed_metrics_df is not None and 'mae' in computed_metrics_df.columns:
        computed_metrics_df['mae'].hist(bins=20, alpha=0.7, ax=ax)
        ax.set_title('MAE Distribution Across Turbines', fontsize=12, fontweight='bold')
        ax.set_xlabel('MAE (m/s)')
        ax.set_ylabel('Frequency')
    else:
        ax.text(0.5, 0.5, 'Additional metrics\nnot computed', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    
    # Plot 5: Correlation analysis
    ax = axes[4]
    if computed_metrics_df is not None and 'correlation' in computed_metrics_df.columns:
        corr_data = computed_metrics_df.dropna(subset=['correlation'])
        if len(corr_data) > 0:
            sns.barplot(data=corr_data, x='turbine_id', y='correlation', hue='feature', ax=ax)
            ax.set_title('Forecast-Observation Correlation', fontsize=12, fontweight='bold')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Correlation data\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Forecast Correlation', fontsize=12, fontweight='bold')
    
    # Plot 6: Bias analysis
    ax = axes[5]
    if computed_metrics_df is not None and 'bias' in computed_metrics_df.columns:
        bias_data = computed_metrics_df.dropna(subset=['bias'])
        if len(bias_data) > 0:
            sns.barplot(data=bias_data, x='turbine_id', y='bias', hue='feature', ax=ax)
            ax.set_title('Forecast Bias by Turbine', fontsize=12, fontweight='bold')
            ax.set_ylabel('Bias (m/s)')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'Bias data\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Forecast Bias', fontsize=12, fontweight='bold')
    
    # Plot 7: Coverage probability
    ax = axes[6]
    if computed_metrics_df is not None and 'coverage_90' in computed_metrics_df.columns:
        coverage_data = computed_metrics_df.dropna(subset=['coverage_90'])
        if len(coverage_data) > 0:
            sns.barplot(data=coverage_data, x='turbine_id', y='coverage_90', hue='feature', ax=ax)
            ax.set_title('90% Prediction Interval Coverage', fontsize=12, fontweight='bold')
            ax.set_ylabel('Coverage Probability')
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
            ax.set_ylim(0, 1)
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'Coverage data\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Prediction Interval Coverage', fontsize=12, fontweight='bold')
    
    # Plot 8: Reliability
    ax = axes[7]
    if computed_metrics_df is not None and 'reliability' in computed_metrics_df.columns:
        reliability_data = computed_metrics_df.dropna(subset=['reliability'])
        if len(reliability_data) > 0:
            sns.barplot(data=reliability_data, x='turbine_id', y='reliability', hue='feature', ax=ax)
            ax.set_title('Forecast Reliability (Lower is Better)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Reliability Score')
    else:
        ax.text(0.5, 0.5, 'Reliability data\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Forecast Reliability', fontsize=12, fontweight='bold')
    
    # Plot 9: Summary statistics
    ax = axes[8]
    if len(metrics_pd) > 0:
        # Create summary table
        summary_text = f"Validation Summary\n"
        summary_text += f"Continuity Group: {continuity_group}\n\n"
        
        for metric in ['MAE', 'RMSE']:
            metric_data = metrics_pd[metrics_pd['metric'] == metric]
            if len(metric_data) > 0:
                mean_score = metric_data['score'].mean()
                summary_text += f"{metric}: {mean_score:.3f} ± {metric_data['score'].std():.3f}\n"
        
        summary_text += f"\nTurbines: {len(metrics_pd['turbine_id'].unique())}\n"
        summary_text += f"Features: {len(metrics_pd['feature_type'].unique())}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'Comprehensive Forecast Metrics Analysis - Continuity Group {continuity_group}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        save_path = os.path.join(save_dir, f"comprehensive_metrics_cg{continuity_group}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive metrics plot to: {save_path}")
        
        save_path_pdf = save_path.replace('.png', '.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive metrics plot to: {save_path_pdf}")
    
    return fig, axes

def plot_single_turbine_samples(forecast_df, true_df, time_regions, 
                               turbine_id=3, save_dir=None, continuity_group=0, 
                               n_samples_to_show=50):
    """Create clear plots of individual sample trajectories for a single turbine."""
    
    forecast_pd = forecast_df.to_pandas()
    forecast_pd['time'] = pd.to_datetime(forecast_pd['time'])
    
    true_pd = true_df.to_pandas()
    true_pd['time'] = pd.to_datetime(true_pd['time'])
    true_pd = true_pd[true_pd['continuity_group'] == continuity_group]
    
    # Pivot true data
    true_pivot = true_pd.pivot_table(index='time', columns=['feature', 'turbine_id'], values='value').reset_index()
    true_pivot.columns = ['time'] + [f"{feat}_{turb}" for feat, turb in true_pivot.columns[1:]]
    
    n_regions = len(time_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(16, 4*n_regions))
    if n_regions == 1:
        axes = axes.reshape(1, -1)
    
    features = ['ws_horz', 'ws_vert']
    feature_labels = ['u Wind Speed', 'v Wind Speed']
    
    # Colors
    sample_color = '#1f77b4'
    mean_color = '#ff7f0e'
    true_color = '#d62728'
    
    for region_idx, (start_time, end_time) in enumerate(time_regions):
        print(f"Plotting region {region_idx+1}: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')} for Turbine {turbine_id}")
        
        forecast_region = forecast_pd[
            (forecast_pd['time'] >= start_time) & 
            (forecast_pd['time'] <= end_time)
        ].copy().sort_values(['sample', 'time'])
        
        true_region = true_pivot[
            (true_pivot['time'] >= start_time) & 
            (true_pivot['time'] <= end_time)
        ].copy().sort_values('time')
        
        if len(forecast_region) == 0:
            continue
            
        for feat_idx, (feature, feature_label) in enumerate(zip(features, feature_labels)):
            ax = axes[region_idx, feat_idx]
            
            forecast_col = f"{feature}_{turbine_id}"
            true_col = f"{feature}_{turbine_id}"
            
            if forecast_col not in forecast_region.columns:
                continue
            
            unique_samples = sorted(forecast_region['sample'].unique())
            if len(unique_samples) > n_samples_to_show:
                step = len(unique_samples) // n_samples_to_show
                selected_samples = unique_samples[::step][:n_samples_to_show]
            else:
                selected_samples = unique_samples
            
            # Plot individual samples
            sample_plotted = False
            for sample in selected_samples:
                sample_data = forecast_region[forecast_region['sample'] == sample].sort_values('time')
                if len(sample_data) > 0:
                    label = f'Individual Samples ({len(selected_samples)})' if not sample_plotted else ""
                    ax.plot(sample_data['time'], sample_data[forecast_col], 
                           color=sample_color, alpha=0.4, linewidth=1.0,
                           label=label, zorder=1)
                    sample_plotted = True
            
            # Plot mean
            forecast_mean = forecast_region.groupby('time')[forecast_col].mean().reset_index()
            if len(forecast_mean) > 0:
                ax.plot(forecast_mean['time'], forecast_mean[forecast_col], 
                       color=mean_color, linewidth=3.5, alpha=0.9,
                       label=f'Sample Mean (all {len(unique_samples)} samples)', 
                       zorder=3)
            
            # Plot ground truth
            if true_col in true_region.columns:
                true_data = true_region[['time', true_col]].dropna().sort_values('time')
                if len(true_data) > 0:
                    ax.plot(true_data['time'], true_data[true_col], 
                           color=true_color, linewidth=4.0, alpha=1.0,
                           label='Ground Truth', zorder=4)
            
            # Formatting
            region_duration = (end_time - start_time).total_seconds() / 3600
            ax.set_title(f"Turbine {turbine_id} - {feature_label}\n"
                        f"Region {region_idx+1}: {start_time.strftime('%m-%d %H:%M')} - {end_time.strftime('%H:%M')} "
                        f"({region_duration:.1f}h)", 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='semibold')
            if region_idx == n_regions - 1:
                ax.set_xlabel('Time', fontsize=12, fontweight='semibold')
            
            # Format x-axis
            if region_duration <= 2:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
            else:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.5)
            
            # Set y-limits
            if len(forecast_region) > 0:
                all_values = forecast_region[forecast_col].values
                if len(all_values) > 0:
                    y_min, y_max = np.min(all_values), np.max(all_values)
                    if true_col in true_region.columns:
                        true_values = true_region[true_col].dropna().values
                        if len(true_values) > 0:
                            y_min = min(y_min, np.min(true_values))
                            y_max = max(y_max, np.max(true_values))
                    y_range = y_max - y_min
                    if y_range > 0:
                        ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
            
            # Legend
            if region_idx == 0 and feat_idx == 0:
                legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=11, frameon=True, fancybox=True, 
                                 shadow=True, ncol=1)
                legend.get_frame().set_alpha(0.95)
    
    plt.suptitle(f'Individual Sample Trajectories - Turbine {turbine_id}\n'
                 f'Continuity Group {continuity_group} | TACTiS Probabilistic Wind Forecasts', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    if save_dir:
        save_path = os.path.join(save_dir, f"single_turbine_samples_T{turbine_id}_cg{continuity_group}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved single turbine plot to: {save_path}")
        
        save_path_pdf = save_path.replace('.png', '.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    return fig, axes

def plot_uncertainty_evolution(forecast_df, true_df, turbine_ids=[1, 3, 5], 
                             features=['ws_horz', 'ws_vert'], save_dir=None, 
                             continuity_group=0):
    """Plot how forecast uncertainty evolves over time."""
    
    forecast_pd = forecast_df.to_pandas()
    forecast_pd['time'] = pd.to_datetime(forecast_pd['time'])
    
    true_pd = true_df.to_pandas()
    true_pd['time'] = pd.to_datetime(true_pd['time'])
    true_pd = true_pd[true_pd['continuity_group'] == continuity_group]
    
    # Calculate statistics for each time point
    stats_data = []
    for time_point in sorted(forecast_pd['time'].unique()):
        time_data = forecast_pd[forecast_pd['time'] == time_point]
        
        for feature in features:
            for turbine_id in turbine_ids:
                col = f"{feature}_{turbine_id}"
                if col in time_data.columns:
                    values = time_data[col].values
                    stats_data.append({
                        'time': time_point,
                        'feature': feature,
                        'turbine_id': turbine_id,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'q05': np.percentile(values, 5),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'q95': np.percentile(values, 95),
                    })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create uncertainty plot
    fig, axes = plt.subplots(len(features), 1, figsize=(15, 6*len(features)))
    if len(features) == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", len(turbine_ids))
    
    for feat_idx, feature in enumerate(features):
        ax = axes[feat_idx]
        
        for turb_idx, turbine_id in enumerate(turbine_ids):
            data = stats_df[
                (stats_df['feature'] == feature) & 
                (stats_df['turbine_id'] == turbine_id)
            ].sort_values('time')
            
            if len(data) == 0:
                continue
                
            # Plot confidence intervals
            ax.fill_between(data['time'], data['q05'], data['q95'], 
                           alpha=0.2, color=colors[turb_idx], 
                           label=f'T{turbine_id} 90% CI')
            ax.fill_between(data['time'], data['q25'], data['q75'], 
                           alpha=0.4, color=colors[turb_idx],
                           label=f'T{turbine_id} 50% CI')
            
            # Plot mean
            ax.plot(data['time'], data['mean'], color=colors[turb_idx], 
                   linewidth=2, label=f'T{turbine_id} Mean')
        
        ax.set_title(f"Probabilistic Forecast Uncertainty: {feature.replace('_', ' ').title()}", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"uncertainty_evolution_cg{continuity_group}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved uncertainty plot to: {save_path}")
        
        save_path_pdf = save_path.replace('.png', '.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    return fig, axes

def main():
    parser = argparse.ArgumentParser(description='Comprehensive TACTiS forecast analysis')
    parser.add_argument('--results_dir', required=True, 
                       help='Path to TACTiS forecast results directory')
    parser.add_argument('--continuity_group', type=int, default=0,
                       help='Continuity group to analyze (default: 0)')
    parser.add_argument('--turbine_id', type=int, default=3,
                       help='Primary turbine ID for detailed plots (default: 3)')
    parser.add_argument('--n_regions', type=int, default=4,
                       help='Number of time regions for sample plots (default: 4)')
    parser.add_argument('--region_hours', type=float, default=1.5,
                       help='Duration of each time region in hours (default: 1.5)')
    parser.add_argument('--n_samples_to_show', type=int, default=50,
                       help='Number of sample trajectories to show (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for region selection (default: 42)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("COMPREHENSIVE TACTIS FORECAST ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n1. Loading forecast and ground truth data...")
    forecast_df, true_df = load_forecast_and_true_data(args.results_dir, args.continuity_group)
    
    # Load aggregated metrics
    print("\n2. Loading aggregated metrics...")
    agg_metrics_df = load_aggregated_metrics(args.results_dir)
    
    # Compute additional metrics
    print("\n3. Computing additional metrics...")
    try:
        computed_metrics_df = compute_additional_metrics(forecast_df, true_df, args.continuity_group)
        print(f"Computed metrics for {len(computed_metrics_df)} turbine-feature combinations")
    except Exception as e:
        print(f"Warning: Could not compute additional metrics: {e}")
        computed_metrics_df = None
    
    # Create comprehensive metrics plot
    print("\n4. Creating comprehensive metrics visualization...")
    fig1, axes1 = plot_comprehensive_metrics(
        agg_metrics_df, computed_metrics_df, args.results_dir, args.continuity_group
    )
    
    # Select time regions for sample plots
    print("\n5. Selecting time regions for sample visualization...")
    time_regions = select_random_time_regions(
        forecast_df, n_regions=args.n_regions, 
        region_duration_hours=args.region_hours, seed=args.seed
    )
    
    print(f"Selected {len(time_regions)} time regions:")
    for i, (start, end) in enumerate(time_regions):
        print(f"  Region {i+1}: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}")
    
    # Create sample trajectory plots
    print(f"\n6. Creating sample trajectory plots for Turbine {args.turbine_id}...")
    fig2, axes2 = plot_single_turbine_samples(
        forecast_df, true_df, time_regions, 
        turbine_id=args.turbine_id, save_dir=args.results_dir, 
        continuity_group=args.continuity_group,
        n_samples_to_show=args.n_samples_to_show
    )
    
    # Create uncertainty evolution plot
    print("\n7. Creating uncertainty evolution plots...")
    fig3, axes3 = plot_uncertainty_evolution(
        forecast_df, true_df, 
        turbine_ids=[1, 3, 5], features=['ws_horz', 'ws_vert'],
        save_dir=args.results_dir, continuity_group=args.continuity_group
    )
    
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All plots saved to: {args.results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()