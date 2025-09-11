"""
Plot azimuth distance range results as polar plots.

This module provides functionality to plot azimuth distance range results
showing minimal and maximal distances with respect to the reference station.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_azimuth_distance_range(results: Dict, save_path: Optional[str] = None, 
                               show_station_labels: bool = True) -> None:
    """
    Plot the azimuth distance range results as a polar plot with scatter and lines.
    Shows both radial distances and projections.
    
    Args:
        results (Dict): Results from compute_azimuth_distance_range
        save_path (str, optional): Path to save the plot
        show_station_labels (bool): Whether to show station labels on the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white', 
                          subplot_kw=dict(projection='polar'))
    
    # Convert azimuth to radians for polar plot
    azimuth_rad = np.radians(results['azimuth_angles'])
    
    # Create valid masks for non-NaN values
    valid_proj_mask = ~(np.isnan(results['min_projections']) | np.isnan(results['max_projections']))
    
    # Plot projections (new method)
    if np.any(valid_proj_mask):
        min_az_rad = azimuth_rad[valid_proj_mask]
        min_proj = results['min_projections'][valid_proj_mask]
        max_proj = results['max_projections'][valid_proj_mask]
        
        # Scatter plots for projections
        ax.scatter(min_az_rad, min_proj, 
                  c='cyan', s=60, alpha=0.8, 
                  label='Min Projection', zorder=4, marker='v')
        ax.scatter(min_az_rad, max_proj, 
                  c='blue', s=60, alpha=0.8, 
                  label='Max Projection', zorder=4, marker='d')
        
        # Line plots for projections
        ax.plot(min_az_rad, min_proj, 'c-', 
               linewidth=3, alpha=0.8, zorder=3)
        ax.plot(min_az_rad, max_proj, 'blue', 
               linewidth=3, alpha=0.8, zorder=3)
        
        # Fill between min and max projections
        # ax.fill_between(min_az_rad, min_proj, max_proj, 
        #                alpha=0.2, color='cyan', label='Projection Range', zorder=2)
    
    # Add station positions as scatter points
    station_data = results['station_data']
    for station in station_data:
        az_rad = np.radians(station['azimuth'])
        ax.scatter(az_rad, station['distance'], 
                   c='red', s=150, alpha=0.9, zorder=6, marker='^')
        
        # Add station labels only if requested
        if show_station_labels:
            ax.annotate(station['station'].split('.')[1], 
                        (az_rad, station['distance']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    # Configure polar plot
    ax.set_title('Maximum and Minimum Range vs Azimuth\nfrom Reference Station', 
                pad=25, fontsize=16, fontweight='bold')
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(0, None)  # Start from center
    
    # Add radial axis label positioned at a specific azimuth angle
    # Position the ylabel at 120 degrees azimuth
    ylabel_angle = 110  # degrees
    ylabel_angle_rad = np.radians(ylabel_angle)
    
    # Get the maximum distance for positioning
    max_distance = np.nanmax([np.nanmax(results['min_projections']), 
                             np.nanmax(results['max_projections'])])
    
    # Position the label at the edge of the plot
    ax.text(ylabel_angle_rad, max_distance * 1.2, 'Distance (m)', 
            fontsize=14, ha='center', va='center')

    # Set rlabel position
    ax.set_rlabel_position(110)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
