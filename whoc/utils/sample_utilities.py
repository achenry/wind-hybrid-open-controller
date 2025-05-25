"""Utility functions for processing probabilistic forecast samples.

This module provides functions for working with forecast samples from probabilistic
models, including wind component conversions, circular statistics, and interpolation.
"""

import logging
import numpy as np
import torch
from typing import Union, Dict, Tuple, Any
from scipy.interpolate import interp1d

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def circular_weighted_mean(angles: Union[np.ndarray, torch.Tensor], 
                          weights: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute the weighted mean for circular quantities (angles).
    
    Args:
        angles: Array of angles in degrees
        weights: Array of corresponding weights
        
    Returns:
        Weighted mean angle in degrees [0, 360)
        
    Raises:
        ValueError: If angles and weights have different shapes or all weights are zero
    """
    logging.debug("Computing circular weighted mean")
    
    # Input validation
    if np.array(angles).shape != np.array(weights).shape:
        raise ValueError("angles and weights must have the same shape")
    
    # Convert to numpy for processing
    is_torch = isinstance(angles, torch.Tensor)
    if is_torch:
        angles_np = angles.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        device = angles.device
    else:
        angles_np = np.array(angles)
        weights_np = np.array(weights)
    
    # Check for zero weights
    if np.sum(weights_np) == 0:
        logging.warning("All weights are zero, returning NaN")
        result = np.full_like(angles_np, np.nan)
        return torch.tensor(result, device=device) if is_torch else result
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_np)
    
    # Compute weighted mean of sine and cosine components
    weighted_sin = np.sum(weights_np * np.sin(angles_rad), axis=0 if angles_np.ndim > 0 else None)
    weighted_cos = np.sum(weights_np * np.cos(angles_rad), axis=0 if angles_np.ndim > 0 else None)
    total_weight = np.sum(weights_np, axis=0 if weights_np.ndim > 0 else None)
    
    mean_sin = weighted_sin / total_weight
    mean_cos = weighted_cos / total_weight
    
    # Reconstruct angle from mean sine and cosine
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    
    # Convert back to degrees and ensure [0, 360) range
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    mean_angle_deg = np.where(mean_angle_deg < 0, mean_angle_deg + 360, mean_angle_deg)
    
    # Return in original format
    if is_torch:
        return torch.tensor(mean_angle_deg, device=device, dtype=angles.dtype)
    else:
        return mean_angle_deg


def wind_components_to_params(u_samples: Union[np.ndarray, torch.Tensor], 
                            v_samples: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Convert (u,v) wind components to (wind speed, wind direction) samples and statistics.
    
    Args:
        u_samples: Array of u-component samples with shape [n_samples, *other_dims]
        v_samples: Array of v-component samples with shape [n_samples, *other_dims]
        
    Returns:
        Dictionary containing:
            - wind_speed_samples: Individual wind speed samples
            - wind_direction_samples: Individual wind direction samples  
            - mean_wind_speed: Mean wind speed
            - std_wind_speed: Standard deviation of wind speed
            - mean_wind_direction: Mean wind direction (circular mean)
            - std_wind_direction: Standard deviation of wind direction
            
    Raises:
        ValueError: If u_samples and v_samples have different shapes
    """
    logging.info("Converting wind components to wind speed and direction parameters")
    
    # Input validation
    if np.array(u_samples).shape != np.array(v_samples).shape:
        raise ValueError("u_samples and v_samples must have the same shape")
    
    # Determine if we're working with torch tensors
    is_torch = isinstance(u_samples, torch.Tensor)
    if is_torch:
        u_np = u_samples.detach().cpu().numpy()
        v_np = v_samples.detach().cpu().numpy()
        device = u_samples.device
        dtype = u_samples.dtype
    else:
        u_np = np.array(u_samples)
        v_np = np.array(v_samples)
    
    # Calculate wind speed for each sample
    wind_speed_samples = np.sqrt(u_np**2 + v_np**2)
    
    # Calculate wind direction for each sample
    wind_direction_samples = 180.0 + np.arctan2(u_np, v_np) * 180.0 / np.pi
    
    # Ensure wind direction is in [0, 360) range
    wind_direction_samples = np.where(wind_direction_samples < 0, 
                                    wind_direction_samples + 360, 
                                    wind_direction_samples)
    wind_direction_samples = np.where(wind_direction_samples >= 360, 
                                    wind_direction_samples - 360, 
                                    wind_direction_samples)
    
    # Calculate statistics across the first dimension (n_samples)
    mean_wind_speed = np.mean(wind_speed_samples, axis=0)
    std_wind_speed = np.std(wind_speed_samples, axis=0)
    
    # For wind direction, use circular mean and circular standard deviation
    # Create uniform weights for circular mean
    weights = np.ones_like(wind_direction_samples)
    mean_wind_direction = circular_weighted_mean(wind_direction_samples, weights)
    
    # Circular standard deviation (approximation for small spreads)
    # Convert to radians for std calculation
    wd_rad = np.deg2rad(wind_direction_samples)
    mean_wd_rad = np.deg2rad(mean_wind_direction)
    
    # Calculate circular variance and convert to std dev
    cos_diff = np.cos(wd_rad - mean_wd_rad)
    circular_var = 1 - np.mean(cos_diff, axis=0)
    std_wind_direction = np.rad2deg(np.sqrt(2 * circular_var))
    
    # Convert back to original tensor format if needed
    if is_torch:
        wind_speed_samples = torch.tensor(wind_speed_samples, device=device, dtype=dtype)
        wind_direction_samples = torch.tensor(wind_direction_samples, device=device, dtype=dtype)
        mean_wind_speed = torch.tensor(mean_wind_speed, device=device, dtype=dtype)
        std_wind_speed = torch.tensor(std_wind_speed, device=device, dtype=dtype)
        if not isinstance(mean_wind_direction, torch.Tensor):
            mean_wind_direction = torch.tensor(mean_wind_direction, device=device, dtype=dtype)
        std_wind_direction = torch.tensor(std_wind_direction, device=device, dtype=dtype)
    
    return {
        'wind_speed_samples': wind_speed_samples,
        'wind_direction_samples': wind_direction_samples,
        'mean_wind_speed': mean_wind_speed,
        'std_wind_speed': std_wind_speed,
        'mean_wind_direction': mean_wind_direction,
        'std_wind_direction': std_wind_direction
    }


def empirical_uncertainty_from_samples(samples: Union[np.ndarray, torch.Tensor], 
                                     dim: int = 0) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Calculate mean and standard deviation from raw samples along a given dimension.
    
    Args:
        samples: Array of samples
        dim: Dimension along which to compute statistics (default: 0 for n_samples)
        
    Returns:
        Dictionary containing 'mean' and 'std_dev'
    """
    logging.debug(f"Computing empirical uncertainty from samples along dimension {dim}")
    
    # Determine if we're working with torch tensors
    is_torch = isinstance(samples, torch.Tensor)
    
    if is_torch:
        mean = torch.mean(samples, dim=dim)
        std_dev = torch.std(samples, dim=dim)
    else:
        samples_np = np.array(samples)
        mean = np.mean(samples_np, axis=dim)
        std_dev = np.std(samples_np, axis=dim)
    
    return {
        'mean': mean,
        'std_dev': std_dev
    }


def interpolate_forecast_samples(samples: Union[np.ndarray, torch.Tensor], 
                               from_dt: float, 
                               to_dt: float) -> Union[np.ndarray, torch.Tensor]:
    """Interpolate sample trajectories from one time resolution to another.
    
    Preserves temporal coherence within each sample trajectory while changing
    the time resolution.
    
    Args:
        samples: Array with shape [n_samples, initial_horizon_steps, num_features]
        from_dt: Original time step duration in seconds
        to_dt: Desired new time step duration in seconds
        
    Returns:
        Interpolated samples with shape [n_samples, new_horizon_steps, num_features]
        
    Raises:
        ValueError: If from_dt is not divisible by to_dt or if inputs are invalid
    """
    logging.info(f"Interpolating forecast samples from {from_dt}s to {to_dt}s resolution")
    
    # Input validation
    if from_dt <= 0 or to_dt <= 0:
        raise ValueError("Time step durations must be positive")
    
    if from_dt % to_dt != 0:
        raise ValueError("from_dt must be an integer multiple of to_dt for simple interpolation")
    
    # Convert to numpy for processing
    is_torch = isinstance(samples, torch.Tensor)
    if is_torch:
        samples_np = samples.detach().cpu().numpy()
        device = samples.device
        dtype = samples.dtype
    else:
        samples_np = np.array(samples)
    
    if samples_np.ndim != 3:
        raise ValueError("samples must have shape [n_samples, horizon_steps, num_features]")
    
    n_samples, initial_horizon_steps, num_features = samples_np.shape
    
    # Calculate new time grid
    interpolation_factor = int(from_dt / to_dt)
    new_horizon_steps = (initial_horizon_steps - 1) * interpolation_factor + 1
    
    # Create time axes
    original_time = np.arange(initial_horizon_steps) * from_dt
    new_time = np.arange(new_horizon_steps) * to_dt
    
    # Initialize output array
    interpolated_samples = np.zeros((n_samples, new_horizon_steps, num_features))
    
    # Interpolate each sample trajectory independently
    for sample_idx in range(n_samples):
        for feature_idx in range(num_features):
            # Extract the trajectory for this sample and feature
            trajectory = samples_np[sample_idx, :, feature_idx]
            
            # Create interpolation function
            interp_func = interp1d(original_time, trajectory, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            
            # Interpolate to new time grid
            interpolated_samples[sample_idx, :, feature_idx] = interp_func(new_time)
    
    # Convert back to original tensor format if needed
    if is_torch:
        return torch.tensor(interpolated_samples, device=device, dtype=dtype)
    else:
        return interpolated_samples