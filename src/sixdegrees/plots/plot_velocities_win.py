"""
Functions for plotting velocity and backazimuth estimates in time windows.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

def plot_velocities_win(sd, results_velocities: Dict, cc_threshold: float = 0.0, 
                       baz_theo: Union[float, None] = None, 
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot Love wave velocity and backazimuth estimates with correlation coefficient coloring
    
    Parameters
    ----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    results_velocities : Dict
        Dictionary containing velocity analysis results with keys:
        'time', 'backazimuth', 'velocity', 'ccoef'
    cc_threshold : float, optional
        Cross-correlation threshold for filtering (default: 0.0)
    baz_theo : float, optional
        Theoretical backazimuth to plot as reference line (default: None)
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches (default: (12, 8))
        
    Returns:
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Convert arrays if needed
    times = np.array(results_velocities['time'])
    baz = np.array(results_velocities['backazimuth'])
    vel = np.array(results_velocities['velocity'])
    cc = np.array(results_velocities['ccoef'])

    # Apply threshold mask
    mask = cc > cc_threshold

    # Create figure with space for colorbar
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[15, 0.5], hspace=0.1)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[:, 1])  # colorbar axis

    # Plot backazimuth estimates
    sc1 = ax1.scatter(results_velocities['time'][mask], 
                     results_velocities['backazimuth'][mask], 
                     c=results_velocities['ccoef'][mask], 
                     cmap='viridis', 
                     alpha=1, 
                     label='Estimated BAZ', 
                     zorder=2, 
                     vmin=0, 
                     vmax=1,
                     edgecolors='black', 
                     linewidths=0.5)
    
    # Add theoretical backazimuth line if provided
    if baz_theo is not None:
        ax1.axhline(y=baz_theo, color='r', ls='--', label='Theoretical BAZ', zorder=0)

    # Plot velocity estimates
    sc2 = ax2.scatter(results_velocities['time'][mask], 
                     results_velocities['velocity'][mask], 
                     c=results_velocities['ccoef'][mask], 
                     cmap='viridis', 
                     alpha=1, 
                     label='Phase Velocity', 
                     zorder=2, 
                     vmin=0, 
                     vmax=1, 
                     edgecolors='black', 
                     linewidths=0.5)

    # Configure axes
    ax1.set_ylim(0, 360)
    ax1.set_ylabel('Backazimuth (°)')
    ax1.grid(which='both', zorder=0, alpha=0.5)
    ax1.legend()

    ax2.set_ylim(0, 5000)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(which='both', zorder=0, alpha=0.5)
    ax2.legend()

    # Add minor ticks
    for ax in [ax1, ax2]:
        ax.minorticks_on()

    # Add colorbar
    cb = plt.colorbar(sc1, cax=cax, label='Cross-Correlation Coefficient')

    plt.tight_layout()
    return fig