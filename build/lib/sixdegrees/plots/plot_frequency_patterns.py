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
    # Convert azimuth to radians for polar plot
    azimuth_rad = np.radians(azimuth_angles)
    
    # Ensure projections are positive and handle invalid values
    max_projections = np.abs(max_projections)
    min_projections = np.abs(min_projections)
    
    # Replace invalid values with NaN
    max_projections = np.where(np.isfinite(max_projections) & (max_projections > 0), max_projections, np.nan)
    min_projections = np.where(np.isfinite(min_projections) & (min_projections > 0), min_projections, np.nan)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white',
                                  subplot_kw=dict(projection='polar'))
    
    # Define colors for different velocities
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Calculate and plot frequencies for each velocity
    for i, velocity in enumerate(velocity_range):
        # Calculate frequencies
        fmin = optional_amplitude_uncertainty * velocity / max_projections
        fmax = 0.25 * velocity / max_projections
        
        # Handle invalid values
        fmin = np.where(np.isfinite(fmin) & (fmin > 0), fmin, np.nan)
        fmax = np.where(np.isfinite(fmax) & (fmax > 0), fmax, np.nan)
        
        # Find valid data points
        valid_mask = np.isfinite(fmin) & np.isfinite(fmax)
        
        if np.any(valid_mask):
            # Get valid data
            valid_az_rad = azimuth_rad[valid_mask]
            valid_fmin = fmin[valid_mask]
            valid_fmax = fmax[valid_mask]
            
            # Sort by azimuth for smooth plotting
            sort_idx = np.argsort(valid_az_rad)
            valid_az_rad = valid_az_rad[sort_idx]
            valid_fmin = valid_fmin[sort_idx]
            valid_fmax = valid_fmax[sort_idx]
            
            # Plot frequencies
            ax1.plot(valid_az_rad, valid_fmin, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
            
            ax2.plot(valid_az_rad, valid_fmax, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
    
    # Configure both subplots
    for ax, title in [(ax1, 'Minimum Frequencies'), (ax2, 'Maximum Frequencies')]:
        ax.set_title(f'{title} vs Azimuth', pad=20, fontsize=14, fontweight='bold')
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.grid(True, alpha=0.3)
        
        # Set y-axis scale
        if log_scale:
            ax.set_yscale('log')
        
        # Add radial axis label
        ylabel_angle = 120  # degrees
        ylabel_angle_rad = np.radians(ylabel_angle)
        ax.text(ylabel_angle_rad, ax.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
                fontsize=12, ha='center', va='center', 
                rotation=ylabel_angle - 90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
    
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
    # Convert azimuth to radians for polar plot
    azimuth_rad = np.radians(azimuth_angles)
    
    # Ensure projections are positive and handle invalid values
    max_projections = np.abs(max_projections)
    min_projections = np.abs(min_projections)
    
    # Replace invalid values with NaN
    max_projections = np.where(np.isfinite(max_projections) & (max_projections > 0), max_projections, np.nan)
    min_projections = np.where(np.isfinite(min_projections) & (min_projections > 0), min_projections, np.nan)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white',
                                  subplot_kw=dict(projection='polar'))
    
    # Define colors for different velocities
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Calculate and plot frequencies for each velocity
    for i, velocity in enumerate(velocity_range):
        # Calculate frequencies
        fmin = optional_amplitude_uncertainty * velocity / max_projections
        fmax = 0.25 * velocity / max_projections
        
        # Handle invalid values
        fmin = np.where(np.isfinite(fmin) & (fmin > 0), fmin, np.nan)
        fmax = np.where(np.isfinite(fmax) & (fmax > 0), fmax, np.nan)
        
        # Find valid data points
        valid_mask = np.isfinite(fmin) & np.isfinite(fmax)
        
        if np.any(valid_mask):
            # Get valid data
            valid_az_rad = azimuth_rad[valid_mask]
            valid_fmin = fmin[valid_mask]
            valid_fmax = fmax[valid_mask]
            
            # Sort by azimuth for smooth plotting
            sort_idx = np.argsort(valid_az_rad)
            valid_az_rad = valid_az_rad[sort_idx]
            valid_fmin = valid_fmin[sort_idx]
            valid_fmax = valid_fmax[sort_idx]
            
            # Plot frequencies
            ax1.plot(valid_az_rad, valid_fmin, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
            
            ax2.plot(valid_az_rad, valid_fmax, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'v={velocity:.0f} m/s')
    
    # Configure both subplots
    for ax, title in [(ax1, 'Minimum Frequencies'), (ax2, 'Maximum Frequencies')]:
        ax.set_title(f'{title} vs Azimuth', pad=20, fontsize=14, fontweight='bold')
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.grid(True, alpha=0.3)
        
        # Set y-axis scale
        if log_scale:
            ax.set_yscale('log')
        
        # Add radial axis label
        ylabel_angle = 110  # degrees
        ylabel_angle_rad = np.radians(ylabel_angle)
        ax.text(ylabel_angle_rad, ax.get_ylim()[1] * 1.1, 'Frequency (Hz)', 
                fontsize=12, ha='center', va='center')
        
        # Set rlabel position
        ax.set_rlabel_position(ylabel_angle)

        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
    
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