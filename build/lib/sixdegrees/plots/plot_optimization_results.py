"""
Functions for plotting optimization results.
"""
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm, ListedColormap

def plot_optimization_results(sd, params: Dict, wave_type: str='love', 
                            vel_max_threshold: float=5000, cc_threshold: float=0.8, 
                            baz_theo: Optional[float]=None) -> plt.Figure:
    """
    Plot optimization results including frequency bands, backazimuth, and velocities.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    params : Dict
        Dictionary containing optimization results from optimize_parameters() with keys:
            - times: Time points
            - frequency: Dictionary with center frequencies
            - backazimuth: Dictionary with optimal backazimuth values
            - velocity: Array of velocity values
            - cross_correlation: Dictionary with optimal CC values
    wave_type : str
        Type of wave to analyze ('love' or 'rayleigh')
    vel_max_threshold : float
        Maximum velocity threshold in m/s. Points above this will be plotted in grey
    cc_threshold : float
        Minimum cross-correlation coefficient threshold
    baz_theo : float, optional
        Theoretical backazimuth for comparison
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing frequency, backazimuth, and velocity plots
    """
    font = 12

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig, 
                width_ratios=[1, 0.05], 
                height_ratios=[1, 1, 1], 
                hspace=0.25)
    
    # Convert data to arrays if needed
    times = np.array(params['times'])
    freqs = np.array(params['frequency']['center'])
    baz = np.array(params['backazimuth']['optimal'])
    vel = np.array(params['velocity'])
    cc = np.array(params['cross_correlation']['optimal'])
    
    # Create velocity mask
    vel_mask = vel <= vel_max_threshold
    
    # Set colorbar parameters
    vmin, vmax, vstep = cc_threshold, 1.0, 0.01
    levels = np.arange(vmin, vmax + vstep, vstep)  # steps of 0.01
    
    # Create discrete colormap
    n_bins = len(levels) - 1
    viridis = plt.cm.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, n_bins))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    
    # Create subplots
    ax_freq = fig.add_subplot(gs[0, 0])
    ax_baz = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[2, 0])
    
    # Plot frequency bands
    sc_freq = ax_freq.scatter(times, freqs, c=cc, cmap=cmap, norm=norm,
                            alpha=0.8, s=30)
    ax_freq.set_ylabel('Frequency (Hz)', fontsize=font)
    ax_freq.set_yscale('log')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.set_xticklabels([])
    
    # Plot backazimuth
    sc_baz = ax_baz.scatter(times, baz, c=cc, cmap=cmap, norm=norm,
                           alpha=0.8, s=30)
    if baz_theo is not None:
        ax_baz.axhline(y=baz_theo, color='r', linestyle='--', alpha=0.8,
                      label=f'Theoretical: {baz_theo:.1f}°')
        ax_baz.legend()
    ax_baz.set_ylabel('Backazimuth (°)', fontsize=font)
    ax_baz.set_ylim(0, 360)
    ax_baz.grid(True, alpha=0.3)
    ax_baz.set_xticklabels([])
    
    # Plot velocities
    # Plot points above threshold in grey
    if not np.all(vel_mask):
        ax_vel.scatter(times[~vel_mask], vel[~vel_mask], color='grey',
                      alpha=0.3, s=30, label='Above threshold')
    # Plot valid points with color
    sc_vel = ax_vel.scatter(times[vel_mask], vel[vel_mask], 
                           c=cc[vel_mask], cmap=cmap, norm=norm,
                           alpha=0.8, s=30)
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=font)
    ax_vel.set_ylim(bottom=0)
    ax_vel.grid(True, alpha=0.3)
    if not np.all(vel_mask):
        ax_vel.legend()
    
    # Add timestamp to x-axis
    if 'starttime' in params:
        ax_vel.set_xlabel(f"Time (s) from {params['starttime'].date} "
                         f"{str(params['starttime'].time).split('.')[0]} UTC",
                         fontsize=font)
    else:
        ax_vel.set_xlabel('Time (s)', fontsize=font)
    
    # Add colorbar
    cax = fig.add_subplot(gs[:, 1])
    plt.colorbar(sc_freq, cax=cax, label='Cross-correlation')
    
    # Calculate statistics if data available
    if len(times) > 0:
        # Frequency statistics
        freq_stats = f"Freq: {np.min(freqs):.2f}-{np.max(freqs):.2f} Hz"
        ax_freq.text(0.02, 0.98, freq_stats, transform=ax_freq.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Backazimuth statistics
        baz_mean = np.average(baz, weights=cc)
        baz_std = np.sqrt(np.average((baz - baz_mean)**2, weights=cc))
        baz_stats = f"BAZ: {baz_mean:.1f}° ± {baz_std:.1f}°"
        if baz_theo is not None:
            dev = abs(baz_mean - baz_theo)
            if dev > 180:
                dev = 360 - dev
            baz_stats += f"\nDev: {dev:.1f}°"
        ax_baz.text(0.02, 0.98, baz_stats, transform=ax_baz.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Velocity statistics
        vel_valid = vel[vel_mask]
        cc_valid = cc[vel_mask]
        if len(vel_valid) > 0:
            vel_mean = np.average(vel_valid, weights=cc_valid)
            vel_std = np.sqrt(np.average((vel_valid - vel_mean)**2, weights=cc_valid))
            vel_stats = f"Vel: {vel_mean:.0f} ± {vel_std:.0f} m/s"
            ax_vel.text(0.02, 0.98, vel_stats, transform=ax_vel.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig