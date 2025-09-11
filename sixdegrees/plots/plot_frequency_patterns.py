"""
Plot frequency patterns for different apparent velocities on polar plots.

This module provides functionality to plot frequency patterns showing
minimal and maximal frequencies for different apparent velocities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_frequency_patterns(azimuth_angles: np.ndarray, min_projections: np.ndarray, 
                          max_projections: np.ndarray, velocity_range: List[float], 
                          optional_amplitude_uncertainty: float = 1e-7,
                          log_scale: bool = False,
                          save_path: Optional[str] = None) -> None:
    """
    Plot frequency patterns for different apparent velocities on polar plots.
    Creates two subplots side by side: minimum and maximum frequencies.
    Each velocity is shown as a different color.
    
    Args:
        azimuth_angles (np.ndarray): Array of azimuth angles in degrees
        min_projections (np.ndarray): Minimum projection distances
        max_projections (np.ndarray): Maximum projection distances
        velocity_range (List[float]): List of apparent velocities in m/s
        optional_amplitude_uncertainty (float): Amplitude uncertainty (default: 1e-7)
        log_scale (bool): Whether to use logarithmic scale for frequency axis (default: False)
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    # replace inf or 0 with nan
    max_projections = np.where(np.logical_or(np.isinf(max_projections), max_projections == 0), np.nan, max_projections)
    min_projections = np.where(np.logical_or(np.isinf(min_projections), min_projections == 0), np.nan, min_projections)

    # Convert to radians for polar plot
    azimuth_rad = np.radians(azimuth_angles)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='white',
                                  subplot_kw=dict(projection='polar'))
    
    # Define colors for different velocities
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Calculate all frequencies first to determine plot limits
    all_fmin = []
    all_fmax = []
    
    for velocity in velocity_range:
        fmin = optional_amplitude_uncertainty * velocity / max_projections
        fmax = 0.25 * velocity / min_projections
        
        # Handle NaN and inf values
        fmin = np.where(np.isnan(fmin) | np.isinf(fmin), np.nan, fmin)
        fmax = np.where(np.isnan(fmax) | np.isinf(fmax), np.nan, fmax)
        
        # Only add finite values
        valid_fmin = fmin[np.isfinite(fmin)]
        valid_fmax = fmax[np.isfinite(fmax)]
        
        all_fmin.extend(valid_fmin)
        all_fmax.extend(valid_fmax)
    
    # Set plot limits
    if all_fmin:
        min_freq = min(all_fmin)
        max_freq = max(all_fmax) if all_fmax else max(all_fmin)
    else:
        min_freq, max_freq = 0.001, 1.0
    
    # Plot frequency patterns for each velocity
    for i, velocity in enumerate(velocity_range):
        # Calculate frequencies for this velocity
        fmin = optional_amplitude_uncertainty * velocity / max_projections
        fmax = 0.25 * velocity / min_projections
        
        # Handle NaN and inf values - replace with NaN for consistent handling
        fmin = np.where(np.isnan(fmin) | np.isinf(fmin), np.nan, fmin)
        fmax = np.where(np.isnan(fmax) | np.isinf(fmax), np.nan, fmax)
        
        # Create valid mask for finite values only
        valid_mask = np.isfinite(fmin) & np.isfinite(fmax)
        
        if np.any(valid_mask):
            valid_az_rad = azimuth_rad[valid_mask]
            valid_fmin = fmin[valid_mask]
            valid_fmax = fmax[valid_mask]
            
            # Ensure we have the same azimuth sampling by sorting
            sort_idx = np.argsort(valid_az_rad)
            valid_az_rad = valid_az_rad[sort_idx]
            valid_fmin = valid_fmin[sort_idx]
            valid_fmax = valid_fmax[sort_idx]
            
            # Plot min frequencies on left subplot
            ax1.plot(valid_az_rad, valid_fmin, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
            
            # Plot max frequencies on right subplot
            ax2.plot(valid_az_rad, valid_fmax, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
    
    # Configure both subplots
    for ax, title, freq_type in [(ax1, 'Minimum Frequencies', 'fmin'), (ax2, 'Maximum Frequencies', 'fmax')]:
        ax.set_title(f'{title} vs Azimuth', pad=20, fontsize=14, fontweight='bold')
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, None)  # Start from center
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(min_freq * 0.5, max_freq * 2)
        else:
            ax.set_ylim(0, max_freq * 1.1)
        
        # Add radial axis label
        ylabel_angle = 120  # degrees (East direction)
        ylabel_angle_rad = np.radians(ylabel_angle)
        
        # Position the label at the edge of the plot
        label_freq = max_freq * 1.2 if not log_scale else max_freq * 2
        ax.text(ylabel_angle_rad, label_freq, f'Frequency (Hz)', 
                fontsize=12, ha='center', va='center', 
                rotation=ylabel_angle - 90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add legends
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    # Add overall title
    fig.suptitle('Frequency Patterns vs Azimuth for Different Apparent Velocities', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_frequency_patterns_simple(azimuth_angles: np.ndarray, min_projections: np.ndarray, 
                                 max_projections: np.ndarray, velocity_range: List[float], 
                                 optional_amplitude_uncertainty: float = 1e-7,
                                 log_scale: bool = False,
                                 save_path: Optional[str] = None) -> None:
    """
    Simple version: Convert azimuthal distances to frequencies and plot on polar maps.
    Creates two subplots side by side: minimum and maximum frequencies.
    
    Args:
        azimuth_angles (np.ndarray): Array of azimuth angles in degrees
        min_projections (np.ndarray): Minimum projection distances
        max_projections (np.ndarray): Maximum projection distances
        velocity_range (List[float]): List of apparent velocities in m/s
        optional_amplitude_uncertainty (float): Amplitude uncertainty (default: 1e-7)
        log_scale (bool): Whether to use logarithmic scale for frequency axis (default: False)
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    # replace inf or 0 with nan
    max_projections = np.where(np.logical_or(np.isinf(max_projections), max_projections == 0), np.nan, max_projections)
    min_projections = np.where(np.logical_or(np.isinf(min_projections), min_projections == 0), np.nan, min_projections)
    
    # Convert to radians for polar plot
    azimuth_rad = np.radians(azimuth_angles)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='white',
                                  subplot_kw=dict(projection='polar'))
    
    # Define colors for different velocities
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Calculate and plot frequencies for each velocity
    for i, velocity in enumerate(velocity_range):
        # Calculate frequencies
        fmin = optional_amplitude_uncertainty * velocity / max_projections
        fmax = 0.25 * velocity / min_projections
        
        # Handle NaN and inf values
        fmin = np.where(np.isnan(fmin) | np.isinf(fmin), np.nan, fmin)
        fmax = np.where(np.isnan(fmax) | np.isinf(fmax), np.nan, fmax)
        
        # Create valid mask for finite values only
        valid_mask = np.isfinite(fmin) & np.isfinite(fmax)
        
        if np.any(valid_mask):
            valid_az_rad = azimuth_rad[valid_mask]
            valid_fmin = fmin[valid_mask]
            valid_fmax = fmax[valid_mask]
            
            # Sort by azimuth for smooth plotting
            sort_idx = np.argsort(valid_az_rad)
            valid_az_rad = valid_az_rad[sort_idx]
            valid_fmin = valid_fmin[sort_idx]
            valid_fmax = valid_fmax[sort_idx]
            
            # Plot min frequencies on left subplot
            ax1.plot(valid_az_rad, valid_fmin, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
            
            # Plot max frequencies on right subplot
            ax2.plot(valid_az_rad, valid_fmax, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
    
    # Configure both subplots
    for ax, title in [(ax1, 'Minimum Frequencies'), (ax2, 'Maximum Frequencies')]:
        ax.set_title(f'{title} vs Azimuth', pad=20, fontsize=14, fontweight='bold')
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, None)  # Start from center
        
        # Add radial axis label
        ylabel_angle = 120  # degrees
        ylabel_angle_rad = np.radians(ylabel_angle)
        
        # Position the label at the edge of the plot
        ax.text(ylabel_angle_rad, ax.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
                fontsize=12, ha='center', va='center', 
                rotation=ylabel_angle - 90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add legends
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    # Add overall title
    fig.suptitle('Frequency Patterns vs Azimuth for Different Apparent Velocities', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
