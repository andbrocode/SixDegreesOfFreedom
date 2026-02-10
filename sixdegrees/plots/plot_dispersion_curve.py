"""
Functions for plotting dispersion curves from trace dispersion analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure


def plot_dispersion_curve(
    love_results: Optional[Dict] = None,
    rayleigh_results: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = (8, 6),
    xlog: bool = False,
    ylog: bool = False,
    markersize: float = 7,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    show_errors: bool = False
) -> Figure:
    """
    Plot dispersion curves for Love and/or Rayleigh waves.
    
    Parameters:
    -----------
    love_results : dict, optional
        Dictionary from plot_trace_dispersion output for Love waves.
        Must contain 'frequencies' and 'velocities' keys.
    rayleigh_results : dict, optional
        Dictionary from plot_trace_dispersion output for Rayleigh waves.
        Must contain 'frequencies' and 'velocities' keys.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    xlog : bool, optional
        Use logarithmic x-axis (frequency) if True. Default is False.
    ylog : bool, optional
        Use logarithmic y-axis (velocity) if True. Default is False.
    markersize : float, optional
        Size of markers. Default is 8.
    linewidth : float, optional
        Width of lines connecting points. Default is 1.5.
    title : str, optional
        Plot title. If None, generates automatic title.
    show_errors : bool, optional
        If True, display error bars when 'velocity_errors' are available in results.
        Default is False.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with dispersion curve plot.
    
    Examples:
    ---------
    >>> # Plot only Rayleigh dispersion
    >>> fig, out_rayleigh = plot_trace_dispersion(..., wave_type="rayleigh", output=True)
    >>> fig = plot_dispersion_curve(rayleigh_results=out_rayleigh)
    
    >>> # Plot both Love and Rayleigh
    >>> fig, out_love = plot_trace_dispersion(..., wave_type="love", output=True)
    >>> fig, out_rayleigh = plot_trace_dispersion(..., wave_type="rayleigh", output=True)
    >>> fig = plot_dispersion_curve(love_results=out_love, rayleigh_results=out_rayleigh)
    """
    
    # Validate inputs
    if love_results is None and rayleigh_results is None:
        raise ValueError("At least one of 'love_results' or 'rayleigh_results' must be provided")
    
    # Validate dictionary structure
    if love_results is not None:
        if 'frequencies' not in love_results or 'velocities' not in love_results:
            raise ValueError("love_results must contain 'frequencies' and 'velocities' keys")
    
    if rayleigh_results is not None:
        if 'frequencies' not in rayleigh_results or 'velocities' not in rayleigh_results:
            raise ValueError("rayleigh_results must contain 'frequencies' and 'velocities' keys")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot settings
    font = 12
    
    # Plot Love dispersion if provided
    if love_results is not None:
        freq_love = love_results['frequencies']
        vel_love = love_results['velocities']
        
        # Get velocity errors if available and show_errors is True
        vel_err_love = None
        if show_errors and 'velocity_errors' in love_results:
            vel_err_love = love_results['velocity_errors']
        
        # Filter out NaN values
        if vel_err_love is not None:
            valid_mask = ~(np.isnan(freq_love) | np.isnan(vel_love) | np.isnan(vel_err_love))
        else:
            valid_mask = ~(np.isnan(freq_love) | np.isnan(vel_love))
        freq_love = freq_love[valid_mask]
        vel_love = vel_love[valid_mask]
        if vel_err_love is not None:
            vel_err_love = vel_err_love[valid_mask]
        
        if len(freq_love) > 0:
            # Sort by frequency for better line plotting
            sort_idx = np.argsort(freq_love)
            freq_love = freq_love[sort_idx]
            vel_love = vel_love[sort_idx]
            if vel_err_love is not None:
                vel_err_love = vel_err_love[sort_idx]
            
            # Plot with or without error bars
            if show_errors and vel_err_love is not None:
                ax.errorbar(freq_love, vel_love, yerr=vel_err_love, fmt='o-', 
                           color='tab:red', label='Love', markersize=markersize, 
                           linewidth=linewidth, capsize=3, capthick=1, zorder=3)
            else:
                ax.plot(freq_love, vel_love, 'o-', color='tab:red', 
                       label='Love', markersize=markersize, linewidth=linewidth, zorder=3)
    
    # Plot Rayleigh dispersion if provided
    if rayleigh_results is not None:
        freq_rayleigh = rayleigh_results['frequencies']
        vel_rayleigh = rayleigh_results['velocities']
        
        # Get velocity errors if available and show_errors is True
        vel_err_rayleigh = None
        if show_errors and 'velocity_errors' in rayleigh_results:
            vel_err_rayleigh = rayleigh_results['velocity_errors']
        
        # Filter out NaN values
        if vel_err_rayleigh is not None:
            valid_mask = ~(np.isnan(freq_rayleigh) | np.isnan(vel_rayleigh) | np.isnan(vel_err_rayleigh))
        else:
            valid_mask = ~(np.isnan(freq_rayleigh) | np.isnan(vel_rayleigh))
        freq_rayleigh = freq_rayleigh[valid_mask]
        vel_rayleigh = vel_rayleigh[valid_mask]
        if vel_err_rayleigh is not None:
            vel_err_rayleigh = vel_err_rayleigh[valid_mask]
        
        if len(freq_rayleigh) > 0:
            # Sort by frequency for better line plotting
            sort_idx = np.argsort(freq_rayleigh)
            freq_rayleigh = freq_rayleigh[sort_idx]
            vel_rayleigh = vel_rayleigh[sort_idx]
            if vel_err_rayleigh is not None:
                vel_err_rayleigh = vel_err_rayleigh[sort_idx]
            
            # Plot with or without error bars
            if show_errors and vel_err_rayleigh is not None:
                ax.errorbar(freq_rayleigh, vel_rayleigh, yerr=vel_err_rayleigh, fmt='s-', 
                           color='tab:blue', label='Rayleigh', markersize=markersize, 
                           linewidth=linewidth, capsize=3, capthick=1, zorder=2)
            else:
                ax.plot(freq_rayleigh, vel_rayleigh, 's-', color='tab:blue', 
                       label='Rayleigh', markersize=markersize, linewidth=linewidth, zorder=2)
    
    # Set axis scales
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    
    # Labels and formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=font)
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=font)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=font-1)
    
    # Set title
    if title is None:
        wave_types = []
        if love_results is not None:
            wave_types.append('Love')
        if rayleigh_results is not None:
            wave_types.append('Rayleigh')
        title = f"Dispersion Curves: {', '.join(wave_types)}"
    
    ax.set_title(title, fontsize=font+1)
    
    plt.tight_layout()
    return fig

