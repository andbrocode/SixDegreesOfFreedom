"""
Plot frequency limits vs apparent velocity.

This module provides functionality to plot frequency limits (fmin and fmax)
as a function of apparent velocity, showing both optimistic and conservative bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from ..seismicarray import seismicarray


def plot_frequency_limits_vs_velocity(
    array: seismicarray,
    velocity_min: float = 500.0,
    velocity_max: float = 5000.0,
    velocity_step: float = 50,
    amplitude_uncertainty: float = 0.01,
    figsize: Tuple[float, float] = (10, 5),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot frequency limits vs apparent velocity for a seismic array.
    
    Computes frequency limits for a range of velocities and visualizes them
    as filled areas showing optimistic and conservative bounds.
    
    Parameters:
    -----------
    array : seismicarray
        SeismicArray instance with computed azimuthal distances
    velocity_min : float, optional
        Minimum velocity in m/s (default: 500.0)
    velocity_max : float, optional
        Maximum velocity in m/s (default: 5000.0)
    n_velocities : int, optional
        Number of velocities to compute (default: 50)
    amplitude_uncertainty : float, optional
        Amplitude uncertainty factor (default: 0.01)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8))
    verbose : bool, optional
        Print progress information (default: True)
    save_path : str, optional
        Path to save the figure. If None, figure is shown (default: None)
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    
    Raises:
    -------
    ValueError
        If azimuthal distances are not available in the array
    """
    # Check if azimuthal distances are available
    if array.azimuthal_distances['azimuth_angles'] is None:
        raise ValueError(
            "Azimuthal distances not available. "
            "Run compute_azimuth_distance_range first."
        )
    
    # Create velocity array
    velocity_range = np.linspace(velocity_min, velocity_max, velocity_step)
    
    if verbose:
        print(f"Computing frequency limits for {len(velocity_range)} velocities...")
        print(f"Velocity range: {velocity_range[0]:.0f} to {velocity_range[-1]:.0f} m/s")
        print(f"Amplitude uncertainty: {amplitude_uncertainty}")
    
    # Compute frequency limits for each velocity
    fmin_optimistic_list = []
    fmax_optimistic_list = []
    fmin_conservative_list = []
    fmax_conservative_list = []
    
    for vel in velocity_range:
        freq_results = array.convert_distances_to_frequencies(
            apparent_velocity=vel,
            optional_amplitude_uncertainty=amplitude_uncertainty
        )
        
        fmin_optimistic_list.append(freq_results['fmin_optimistic'])
        fmax_optimistic_list.append(freq_results['fmax_optimistic'])
        fmin_conservative_list.append(freq_results['fmin_conservative'])
        fmax_conservative_list.append(freq_results['fmax_conservative'])
    
    # Convert to numpy arrays
    fmin_optimistic = np.array(fmin_optimistic_list)
    fmax_optimistic = np.array(fmax_optimistic_list)
    fmin_conservative = np.array(fmin_conservative_list)
    fmax_conservative = np.array(fmax_conservative_list)
    
    if verbose:
        print(f"\nFrequency ranges:")
        print(f"fmin_optimistic: {np.nanmin(fmin_optimistic):.3f} to {np.nanmax(fmin_optimistic):.3f} Hz")
        print(f"fmax_optimistic: {np.nanmin(fmax_optimistic):.3f} to {np.nanmax(fmax_optimistic):.3f} Hz")
        print(f"fmin_conservative: {np.nanmin(fmin_conservative):.3f} to {np.nanmax(fmin_conservative):.3f} Hz")
        print(f"fmax_conservative: {np.nanmin(fmax_conservative):.3f} to {np.nanmax(fmax_conservative):.3f} Hz")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot frequency ranges as filled areas
    ax.fill_between(
        velocity_range, 
        fmin_conservative, 
        fmax_conservative,
        alpha=0.2, 
        color='red', 
        label='Conservative Range (fmin to fmax)'
    )
    ax.fill_between(
        velocity_range, 
        fmin_optimistic, 
        fmax_optimistic,
        alpha=0.3, 
        color='blue', 
        label='Optimistic Range (fmin to fmax)'
    )
    
    # Add boundary lines
    ax.plot(
        velocity_range, 
        fmin_conservative, 
        'r-', 
        linewidth=2, 
        alpha=0.8, 
        label='Conservative fmin'
    )
    ax.plot(
        velocity_range, 
        fmax_conservative, 
        'r--', 
        linewidth=2, 
        alpha=0.8, 
        label='Conservative fmax'
    )
    ax.plot(
        velocity_range, 
        fmin_optimistic, 
        'b-', 
        linewidth=2, 
        alpha=0.8, 
        label='Optimistic fmin'
    )
    ax.plot(
        velocity_range, 
        fmax_optimistic, 
        'b--', 
        linewidth=2, 
        alpha=0.8, 
        label='Optimistic fmax'
    )

    
    # Configure axes
    ax.set_xlabel('Apparent Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(
        'Frequency Limits vs Apparent Velocity\n(Conservative vs Optimistic Bounds)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim(velocity_range[0], velocity_range[-1])
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close();
    else:
        plt.show();
    
    return fig
