"""
Functions for plotting dispersion curves from compute_dispersion_curve output.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure


def plot_dispersion_curves(dispersion_results: Optional[Dict] = None,
                           love_results: Optional[Dict] = None,
                           rayleigh_results: Optional[Dict] = None,
                           figsize: Optional[Tuple[float, float]] = (8, 6),
                           xlog: bool = False,
                           ylog: bool = False,
                           markersize: float = 7,
                           linewidth: float = 1.5,
                           title: Optional[str] = None,
                           show_errors: bool = True,
                           vel_min: float = 0,
                           vel_max: float = 5000) -> Figure:
    """
    Plot dispersion curves from compute_dispersion_curve output.
    
    This function takes the output from compute_dispersion_curve and creates
    a dispersion curve plot showing phase velocity vs frequency, similar to
    plot_dispersion_curve but using KDE peak velocities and deviations.
    
    Parameters:
    -----------
    dispersion_results : dict, optional
        Output dictionary from compute_dispersion_curve function for a single wave type.
        If provided, will extract wave_type and plot accordingly.
    love_results : dict, optional
        Output dictionary from compute_dispersion_curve for Love waves.
        Must contain 'frequency_bands' key with KDE peak velocities.
    rayleigh_results : dict, optional
        Output dictionary from compute_dispersion_curve for Rayleigh waves.
        Must contain 'frequency_bands' key with KDE peak velocities.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    xlog : bool, optional
        Use logarithmic x-axis (frequency) if True. Default is False.
    ylog : bool, optional
        Use logarithmic y-axis (velocity) if True. Default is False.
    markersize : float, optional
        Size of markers. Default is 7.
    linewidth : float, optional
        Width of lines connecting points. Default is 1.5.
    title : str, optional
        Plot title. If None, generates automatic title.
    show_errors : bool, optional
        If True, display error bars using KDE deviations. Default is True.
    vel_min : float, optional
        Minimum velocity for y-axis limit. Default is 0.
    vel_max : float, optional
        Maximum velocity for y-axis limit. Default is 5000.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with dispersion curve plot.
    
    Examples:
    ---------
    >>> # Plot from single compute_dispersion_curve output
    >>> results = sd.compute_dispersion_curve(wave_type="love", ...)
    >>> fig = plot_dispersion_curves(dispersion_results=results)
    
    >>> # Plot both Love and Rayleigh
    >>> love_results = sd.compute_dispersion_curve(wave_type="love", ...)
    >>> rayleigh_results = sd.compute_dispersion_curve(wave_type="rayleigh", ...)
    >>> fig = plot_dispersion_curves(love_results=love_results, rayleigh_results=rayleigh_results)
    """
    
    # Handle single dispersion_results input
    if dispersion_results is not None:
        wave_type = dispersion_results.get('wave_type', 'love').lower()
        if wave_type == 'love':
            love_results = dispersion_results
        elif wave_type == 'rayleigh':
            rayleigh_results = dispersion_results
        else:
            raise ValueError(f"Unknown wave_type: {wave_type}. Must be 'love' or 'rayleigh'")
    
    # Validate inputs
    if love_results is None and rayleigh_results is None:
        raise ValueError("At least one of 'dispersion_results', 'love_results', or 'rayleigh_results' must be provided")
    
    # Extract frequencies and velocities from results
    freq_love = None
    vel_love = None
    vel_err_love = None
    
    if love_results is not None:
        if 'frequency_bands' not in love_results:
            raise ValueError("love_results must contain 'frequency_bands' key")
        
        frequency_bands = love_results['frequency_bands']
        freq_love = np.array([band['f_center'] for band in frequency_bands])
        vel_love = np.array([band['kde_peak_velocity'] for band in frequency_bands])
        vel_err_love = np.array([band['kde_deviation'] for band in frequency_bands])
    
    freq_rayleigh = None
    vel_rayleigh = None
    vel_err_rayleigh = None
    
    if rayleigh_results is not None:
        if 'frequency_bands' not in rayleigh_results:
            raise ValueError("rayleigh_results must contain 'frequency_bands' key")
        
        frequency_bands = rayleigh_results['frequency_bands']
        freq_rayleigh = np.array([band['f_center'] for band in frequency_bands])
        vel_rayleigh = np.array([band['kde_peak_velocity'] for band in frequency_bands])
        vel_err_rayleigh = np.array([band['kde_deviation'] for band in frequency_bands])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot settings
    font = 12
    
    # Plot Love dispersion if provided
    if freq_love is not None and vel_love is not None:
        # Filter out NaN values
        if show_errors and vel_err_love is not None:
            valid_mask = ~(np.isnan(freq_love) | np.isnan(vel_love) | np.isnan(vel_err_love))
        else:
            valid_mask = ~(np.isnan(freq_love) | np.isnan(vel_love))
        
        freq_love_valid = freq_love[valid_mask]
        vel_love_valid = vel_love[valid_mask]
        if vel_err_love is not None:
            vel_err_love_valid = vel_err_love[valid_mask]
        else:
            vel_err_love_valid = None
        
        if len(freq_love_valid) > 0:
            # Sort by frequency for better line plotting
            sort_idx = np.argsort(freq_love_valid)
            freq_love_valid = freq_love_valid[sort_idx]
            vel_love_valid = vel_love_valid[sort_idx]
            if vel_err_love_valid is not None:
                vel_err_love_valid = vel_err_love_valid[sort_idx]
            
            # Plot with or without error bars
            if show_errors and vel_err_love_valid is not None:
                ax.errorbar(freq_love_valid, vel_love_valid, yerr=vel_err_love_valid, fmt='o-', 
                           color='tab:red', label='Love', markersize=markersize, 
                           linewidth=linewidth, capsize=3, capthick=1, zorder=3)
            else:
                ax.plot(freq_love_valid, vel_love_valid, 'o-', color='tab:red', 
                       label='Love', markersize=markersize, linewidth=linewidth, zorder=3)
    
    # Plot Rayleigh dispersion if provided
    if freq_rayleigh is not None and vel_rayleigh is not None:
        # Filter out NaN values
        if show_errors and vel_err_rayleigh is not None:
            valid_mask = ~(np.isnan(freq_rayleigh) | np.isnan(vel_rayleigh) | np.isnan(vel_err_rayleigh))
        else:
            valid_mask = ~(np.isnan(freq_rayleigh) | np.isnan(vel_rayleigh))
        
        freq_rayleigh_valid = freq_rayleigh[valid_mask]
        vel_rayleigh_valid = vel_rayleigh[valid_mask]
        if vel_err_rayleigh is not None:
            vel_err_rayleigh_valid = vel_err_rayleigh[valid_mask]
        else:
            vel_err_rayleigh_valid = None
        
        if len(freq_rayleigh_valid) > 0:
            # Sort by frequency for better line plotting
            sort_idx = np.argsort(freq_rayleigh_valid)
            freq_rayleigh_valid = freq_rayleigh_valid[sort_idx]
            vel_rayleigh_valid = vel_rayleigh_valid[sort_idx]
            if vel_err_rayleigh_valid is not None:
                vel_err_rayleigh_valid = vel_err_rayleigh_valid[sort_idx]
            
            # Plot with or without error bars
            if show_errors and vel_err_rayleigh_valid is not None:
                ax.errorbar(freq_rayleigh_valid, vel_rayleigh_valid, yerr=vel_err_rayleigh_valid, fmt='s-', 
                           color='tab:blue', label='Rayleigh', markersize=markersize, 
                           linewidth=linewidth, capsize=3, capthick=1, zorder=2)
            else:
                ax.plot(freq_rayleigh_valid, vel_rayleigh_valid, 's-', color='tab:blue', 
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
    
    # Set y-axis limits (only for linear scale)
    if not ylog:
        ax.set_ylim(vel_min, vel_max)
    
    # Set title
    if title is None:
        wave_types = []
        if freq_love is not None:
            wave_types.append('Love')
        if freq_rayleigh is not None:
            wave_types.append('Rayleigh')
        title = f"Dispersion Curves: {', '.join(wave_types)}"
    
    ax.set_title(title, fontsize=font+1)
    
    plt.tight_layout()
    return fig
