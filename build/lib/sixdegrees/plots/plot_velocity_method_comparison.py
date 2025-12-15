"""
Functions for comparing different velocity estimation methods.
"""
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

def plot_velocity_method_comparison(velocities1: Dict, velocities2: Dict, 
                                  cc_threshold: float = 0.75, 
                                  vel_max: Optional[float] = None,
                                  labels: Tuple[str, str] = ('RANSAC', 'ODR'),
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comparison plot of RANSAC and ODR velocity estimates.
    
    Parameters:
    -----------
    velocities1 : Dict
        Dictionary containing RANSAC velocity results with keys:
            - time: Time points
            - velocity: Velocity estimates
            - ccoef: Cross-correlation coefficients
    velocities2 : Dict
        Dictionary containing ODR velocity results with same structure
    cc_threshold : float
        Cross-correlation threshold for filtering (default: 0.75)
    vel_max : float or None
        Maximum velocity for y-axis limit in velocity subplots (default: None, auto-scale)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing velocity comparison plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[1, 1, 1], 
                  hspace=0.2, wspace=0.05)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Create mask based on correlation threshold
    mask1 = velocities1['ccoef'] > cc_threshold
    mask2 = velocities2['ccoef'] > cc_threshold

    # Create colormap
    vmin, vmax, vstep = 0.5, 1.0, 0.05
    levels = np.arange(vmin, vmax + vstep, vstep)
    n_bins = len(levels) - 1
    viridis = plt.cm.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, n_bins))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=len(levels)-1)

    # Plot RANSAC velocities
    cm1 = ax1.scatter(velocities1['time'][mask1],
                     velocities1['velocity'][mask1],
                     c=velocities1['ccoef'][mask1],
                     cmap=cmap, norm=norm, alpha=0.8, zorder=2)
    plt.colorbar(cm1, cax=cax1, label='CC')
    ax1.set_ylabel(f'{labels[0]} Velocity (m/s)')
    ax1.set_ylim(bottom=0)
    if vel_max is not None:
        ax1.set_ylim(top=vel_max)
    ax1.grid(True, alpha=0.3)

    # Plot ODR velocities
    cm2 = ax2.scatter(velocities2['time'][mask2],
                     velocities2['velocity'][mask2],
                     c=velocities2['ccoef'][mask2],
                     cmap=cmap, norm=norm, alpha=0.8, zorder=2)
    plt.colorbar(cm2, cax=cax2, label='CC')
    ax2.set_ylabel(f'{labels[1]} Velocity (m/s)')
    ax2.set_ylim(bottom=0)
    if vel_max is not None:
        ax2.set_ylim(top=vel_max)
    ax2.grid(True, alpha=0.3)

    # Calculate and plot velocity difference
    common_times = np.intersect1d(velocities1['time'][mask1],
                                  velocities2['time'][mask2])
    
    if len(common_times) > 0:
        # Get velocities at common times
        vel1 = np.interp(common_times, 
                        velocities1['time'][mask1],
                        velocities1['velocity'][mask1])
        vel2 = np.interp(common_times,
                         velocities2['time'][mask2],
                         velocities2['velocity'][mask2])
        
        # Calculate difference
        vel_diff = vel2 - vel1
        
        # Calculate bar width based on time spacing
        if len(common_times) > 1:
            bar_width = np.min(np.diff(common_times)) * 0.8
        else:
            bar_width = 1.0
        
        # Plot difference as bar plot
        ax3.bar(common_times, vel_diff, width=bar_width, zorder=3,
                color='black', alpha=1, edgecolor='black', linewidth=0.5)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.7)
        ax3.set_ylabel(f'{labels[1]} - {labels[0]} (m/s)')
        ax3.grid(True, which='both', ls='-', alpha=0.3)

    if vel_max is not None:
        ax3.set_ylim(-vel_max, vel_max)

    # Set common x-label
    ax3.set_xlabel('Time (s)')

    # Add statistics to velocity plots
    if np.any(mask1):
        vel1_mean = np.mean(velocities1['velocity'][mask1])
        vel1_std = np.std(velocities1['velocity'][mask1])
        vel1_stats = f"Mean: {vel1_mean:.1f} m/s\nStd: {vel1_std:.1f} m/s"
        ax1.text(0.02, 0.98, vel1_stats, transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if np.any(mask2):
        vel2_mean = np.mean(velocities2['velocity'][mask2])
        vel2_std = np.std(velocities2['velocity'][mask2])
        vel2_stats = f"Mean: {vel2_mean:.1f} m/s\nStd: {vel2_std:.1f} m/s"
        ax2.text(0.02, 0.98, vel2_stats, transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig