"""
Plot frequency limits for different velocities on polar plots.

This module provides functionality to plot frequency limits (fmin and fmax)
for different apparent velocities based on azimuthal projections.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any


def plot_frequency_limits(freq_results: Dict[str, Any], 
                         velocity_range: Optional[List[float]] = None,
                         amplitude_uncertainty: float = 0.02,
                         log_scale: bool = True,
                         figsize: tuple = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot frequency limits (fmin and fmax) for different velocities on polar plots.
    
    Args:
        freq_results (Dict): Results from convert_distances_to_frequencies containing:
            - 'azimuth_angles': Array of azimuth angles in degrees
            - 'max_projections': Maximum projection distances
            - 'min_projections': Minimum projection distances
        velocity_range (List[float], optional): List of velocities in m/s. 
                                              If None, uses [1000, 2000, 3000, 4000]
        amplitude_uncertainty (float): Amplitude uncertainty factor (default: 0.02)
        log_scale (bool): Whether to use logarithmic scale (default: True)
        figsize (tuple): Figure size (width, height) (default: (10, 8))
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    if velocity_range is None:
        velocity_range = [1000, 2000, 3000, 4000]
    
    # Extract data from results
    azimuth_angles = freq_results['azimuth_angles']
    max_projections = freq_results['max_projections']
    min_projections = freq_results['min_projections']
    
    # Convert azimuth to radians for polar plot
    rads = np.radians(azimuth_angles)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                  subplot_kw=dict(projection='polar'))
    
    # Calculate all frequencies first to determine proper y-axis range for each subplot
    all_fmin = []
    all_fmax = []
    
    for vel in velocity_range:
        fmin = amplitude_uncertainty * vel / max_projections
        fmax = 0.25 * vel / min_projections
        
        # Only add finite values
        valid_fmin = fmin[np.isfinite(fmin) & (fmin > 0)]
        valid_fmax = fmax[np.isfinite(fmax) & (fmax > 0)]
        
        all_fmin.extend(valid_fmin)
        all_fmax.extend(valid_fmax)
    
    # Set plot limits independently for each subplot
    if all_fmin:
        min_fmin = min(all_fmin)
        max_fmin = max(all_fmin)
        # Ensure reasonable limits for fmin
        if min_fmin < 1e-6:
            min_fmin = 1e-6
        if max_fmin < 1e-4:
            max_fmin = 1e-4
    else:
        min_fmin, max_fmin = 1e-6, 1e-4
    
    if all_fmax:
        min_fmax = min(all_fmax)
        max_fmax = max(all_fmax)
        # Ensure reasonable limits for fmax
        if min_fmax < 1e-3:
            min_fmax = 1e-3
        if max_fmax < 1e-1:
            max_fmax = 1e-1
    else:
        min_fmax, max_fmax = 1e-3, 1.0
    
    # Define colors using viridis
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Plot for each velocity
    for i, vel in enumerate(velocity_range):
        # Calculate frequencies
        fmin = amplitude_uncertainty * vel / max_projections
        fmax = 0.25 * vel / min_projections
        
        # Plot on both subplots with viridis colors
        ax1.plot(rads, fmin, color=colors[i], linewidth=2, alpha=0.8, label=f'v={vel} m/s')
        ax2.plot(rads, fmax, color=colors[i], linewidth=2, alpha=0.8, label=f'v={vel} m/s')
    
    # Configure subplots with independent y-axis scaling
    # Left subplot (fmin)
    ax1.set_title('Minimum Frequencies (fmin)', pad=20, fontsize=12, fontweight='bold')
    ax1.set_theta_zero_location('N')  # North at top
    ax1.set_theta_direction(-1)  # Clockwise
    ax1.grid(True, alpha=0.3)
    
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(min_fmin * 0.1, max_fmin * 10)
    else:
        ax1.set_ylim(min_fmin * 0.5, max_fmin * 1.5)
    
    # Add radial axis label for fmin
    ylabel_angle = 120  # degrees
    ylabel_angle_rad = np.radians(ylabel_angle)
    ax1.text(ylabel_angle_rad, ax1.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
            fontsize=10, ha='center', va='center', 
            rotation=ylabel_angle - 90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Right subplot (fmax)
    ax2.set_title('Maximum Frequencies (fmax)', pad=20, fontsize=12, fontweight='bold')
    ax2.set_theta_zero_location('N')  # North at top
    ax2.set_theta_direction(-1)  # Clockwise
    ax2.grid(True, alpha=0.3)
    
    if log_scale:
        ax2.set_yscale('log')
        ax2.set_ylim(min_fmax * 0.1, max_fmax * 10)
    else:
        ax2.set_ylim(min_fmax * 0.5, max_fmax * 1.5)
    
    # Add radial axis label for fmax
    ax2.text(ylabel_angle_rad, ax2.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
            fontsize=10, ha='center', va='center', 
            rotation=ylabel_angle - 90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Increase tick label sizes for both subplots
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    # Add legends
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # Add overall title
    fig.suptitle('Frequency Limits for Different Velocities', 
                 fontsize=14, fontweight='bold', y=0.93)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_frequency_limits_simple(azimuth_angles: np.ndarray, 
                                min_projections: np.ndarray, 
                                max_projections: np.ndarray,
                                velocity_range: Optional[List[float]] = None,
                                amplitude_uncertainty: float = 0.02,
                                log_scale: bool = True,
                                figsize: tuple = (10, 8),
                                save_path: Optional[str] = None) -> None:
    """
    Simple version: Plot frequency limits directly from projection data.
    
    Args:
        azimuth_angles (np.ndarray): Array of azimuth angles in degrees
        min_projections (np.ndarray): Minimum projection distances
        max_projections (np.ndarray): Maximum projection distances
        velocity_range (List[float], optional): List of velocities in m/s. 
                                              If None, uses [1000, 2000, 3000, 4000]
        amplitude_uncertainty (float): Amplitude uncertainty factor (default: 0.02)
        log_scale (bool): Whether to use logarithmic scale (default: True)
        figsize (tuple): Figure size (width, height) (default: (10, 8))
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    if velocity_range is None:
        velocity_range = [1000, 2000, 3000, 4000]
    
    # Convert azimuth to radians for polar plot
    rads = np.radians(azimuth_angles)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                  subplot_kw=dict(projection='polar'))
    
    # Calculate all frequencies first to determine proper y-axis range for each subplot
    all_fmin = []
    all_fmax = []
    
    for vel in velocity_range:
        fmin = amplitude_uncertainty * vel / max_projections
        fmax = 0.25 * vel / min_projections
        
        # Only add finite values
        valid_fmin = fmin[np.isfinite(fmin) & (fmin > 0)]
        valid_fmax = fmax[np.isfinite(fmax) & (fmax > 0)]
        
        all_fmin.extend(valid_fmin)
        all_fmax.extend(valid_fmax)
    
    # Set plot limits independently for each subplot
    if all_fmin:
        min_fmin = min(all_fmin)
        max_fmin = max(all_fmin)
        # Ensure reasonable limits for fmin
        if min_fmin < 1e-6:
            min_fmin = 1e-6
        if max_fmin > 1:
            max_fmin = 1
    else:
        min_fmin, max_fmin = 1e-6, 1
    
    if all_fmax:
        min_fmax = min(all_fmax)
        max_fmax = max(all_fmax)
        # Ensure reasonable limits for fmax
        if min_fmax < 1e-3:
            min_fmax = 1e-3
        if max_fmax > 1e2:
            max_fmax = 1e2
    else:
        min_fmax, max_fmax = 1e-3, 1e2
    
    # Define colors using viridis
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocity_range)))
    
    # Plot for each velocity
    for i, vel in enumerate(velocity_range):
        # Calculate frequencies
        fmin = amplitude_uncertainty * vel / max_projections
        fmax = 0.25 * vel / min_projections
        
        # Plot on both subplots with viridis colors
        ax1.plot(rads, fmin, color=colors[i], linewidth=2, alpha=0.8, label=f'v={vel} m/s')
        ax2.plot(rads, fmax, color=colors[i], linewidth=2, alpha=0.8, label=f'v={vel} m/s')
    
    # Configure subplots with independent y-axis scaling
    # Left subplot (fmin)
    ax1.set_title('Minimum Frequencies (fmin)', pad=20, fontsize=12, fontweight='bold')
    ax1.set_theta_zero_location('N')  # North at top
    ax1.set_theta_direction(-1)  # Clockwise
    ax1.grid(True, alpha=0.3)
    
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(min_fmin * 0.1, max_fmin * 10)
    else:
        ax1.set_ylim(min_fmin * 0.5, max_fmin * 1.5)
    
    # Add radial axis label for fmin
    ylabel_angle = 120  # degrees
    ylabel_angle_rad = np.radians(ylabel_angle)
    ax1.text(ylabel_angle_rad, ax1.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
            fontsize=10, ha='center', va='center', 
            rotation=ylabel_angle - 90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Right subplot (fmax)
    ax2.set_title('Maximum Frequencies (fmax)', pad=20, fontsize=12, fontweight='bold')
    ax2.set_theta_zero_location('N')  # North at top
    ax2.set_theta_direction(-1)  # Clockwise
    ax2.grid(True, alpha=0.3)
    
    if log_scale:
        ax2.set_yscale('log')
        ax2.set_ylim(min_fmax * 0.1, max_fmax * 10)
    else:
        ax2.set_ylim(min_fmax * 0.5, max_fmax * 1.5)
    
    # Add radial axis label for fmax
    ax2.text(ylabel_angle_rad, ax2.get_ylim()[1] * 0.8, 'Frequency (Hz)', 
            fontsize=10, ha='center', va='center', 
            rotation=ylabel_angle - 90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Increase tick label sizes for both subplots
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=9)

    # Set rlabel position
    ax1.set_rlabel_position(110)
    ax2.set_rlabel_position(110)

    # Add legends
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # Add overall title
    fig.suptitle('Frequency Limits for Different Velocities', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()