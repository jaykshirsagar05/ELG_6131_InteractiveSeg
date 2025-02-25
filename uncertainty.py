import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

def calculate_uncertainty_map(predictions: np.ndarray, method: str = "entropy") -> np.ndarray:
    """
    Calculate uncertainty map from segmentation predictions.
    
    Args:
        predictions: Array of shape (N, H, W) for N Monte Carlo samples
        method: Uncertainty estimation method ("entropy" or "variance")
    
    Returns:
        uncertainty_map: Array of shape (H, W) with uncertainty values
    """
    if method == "entropy":
        # Average predictions across samples
        mean_pred = np.mean(predictions, axis=0)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon), axis=-1)
        return entropy
    
    elif method == "variance":
        # Calculate variance across MC samples
        return np.var(predictions, axis=0)
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

def get_uncertainty_overlay(
    ct_slice: np.ndarray,
    uncertainty_map: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Create an RGB overlay of uncertainty on the CT slice.
    
    Args:
        ct_slice: 2D array of CT values
        uncertainty_map: 2D array of uncertainty values
        threshold: Threshold for highlighting uncertain regions
        alpha: Transparency of the uncertainty overlay
    
    Returns:
        overlay: RGB image with uncertainty highlighted
    """
    # Normalize CT slice to 0-1
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    
    # Normalize uncertainty map to 0-1
    uncertainty_norm = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())
    
    # Create RGB image
    rgb = np.stack([ct_norm] * 3, axis=-1)
    
    # Create uncertainty mask (red overlay)
    uncertainty_mask = uncertainty_norm > threshold
    
    # Apply red tint to uncertain regions
    rgb[uncertainty_mask] = rgb[uncertainty_mask] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    
    return (rgb * 255).astype(np.uint8)

def smooth_uncertainty_map(uncertainty_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to the uncertainty map.
    
    Args:
        uncertainty_map: Raw uncertainty map
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        smoothed_map: Smoothed uncertainty map
    """
    return gaussian_filter(uncertainty_map, sigma=sigma)

def calculate_region_uncertainty(
    uncertainty_map: np.ndarray,
    segmentation: np.ndarray
) -> dict:
    """
    Calculate uncertainty statistics for different regions in the segmentation.
    
    Args:
        uncertainty_map: 2D array of uncertainty values
        segmentation: 2D array of segmentation labels
    
    Returns:
        stats: Dictionary containing uncertainty statistics per region
    """
    stats = {}
    unique_labels = np.unique(segmentation)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        region_mask = segmentation == label
        region_uncertainty = uncertainty_map[region_mask]
        
        stats[f"region_{label}"] = {
            "mean": float(np.mean(region_uncertainty)),
            "std": float(np.std(region_uncertainty)),
            "max": float(np.max(region_uncertainty)),
            "volume": int(np.sum(region_mask))
        }
    
    return stats 