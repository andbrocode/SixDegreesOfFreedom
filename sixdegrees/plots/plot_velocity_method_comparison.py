"""
Functions for comparing different velocity estimation methods.
"""
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

def plot_velocity_method_comparison(sd, love_velocities_ransac: Dict, love_velocities_odr: Dict, 
                                  cc_threshold: float = 0.75, 
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comparison plot of RANSAC and ODR velocity estimates.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    love_velocities_ransac : Dict
        Dictionary containing RANSAC velocity results with keys:
            - time: Time points
            - velocity: Velocity estimates
            - ccoef: Cross-correlation coefficients
    love_velocities_odr : Dict
        Dictionary containing ODR velocity results with same structure
    cc_threshold : float
        Cross-correlation threshold for filtering (default: 0.75)
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
    mask_ransac = love_velocities_ransac['ccoef'] > cc_threshold
    mask_odr = love_velocities_odr['ccoef'] > cc_threshold

    # Create colormap
    vmin, vmax, vstep = 0.5, 1.0, 0.05
    levels = np.arange(vmin, vmax + vstep, vstep)
    n_bins = len(levels) - 1
    viridis = plt.cm.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, n_bins))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=len(levels)-1)

    # Plot RANSAC velocities
    cm1 = ax1.scatter(love_velocities_ransac['time'][mask_ransac],
                     love_velocities_ransac['velocity'][mask_ransac],
                     c=love_velocities_ransac['ccoef'][mask_ransac],
                     cmap=cmap, norm=norm, alpha=0.8)
    plt.colorbar(cm1, cax=cax1, label='CC')
    ax1.set_ylabel('RANSAC Velocity (m/s)')
    ax1.grid(True, alpha=0.3)

    # Plot ODR velocities
    cm2 = ax2.scatter(love_velocities_odr['time'][mask_odr],
                     love_velocities_odr['velocity'][mask_odr],
                     c=love_velocities_odr['ccoef'][mask_odr],
                     cmap=cmap, norm=norm, alpha=0.8)
    plt.colorbar(cm2, cax=cax2, label='CC')
    ax2.set_ylabel('ODR Velocity (m/s)')
    ax2.grid(True, alpha=0.3)

    # Calculate and plot velocity difference
    common_times = np.intersect1d(love_velocities_ransac['time'][mask_ransac],
                                love_velocities_odr['time'][mask_odr])
    
    if len(common_times) > 0:
        # Get velocities at common times
        ransac_vel = np.interp(common_times, 
                              love_velocities_ransac['time'][mask_ransac],
                              love_velocities_ransac['velocity'][mask_ransac])
        odr_vel = np.interp(common_times,
                           love_velocities_odr['time'][mask_odr],
                           love_velocities_odr['velocity'][mask_odr])
        
        # Calculate difference
        vel_diff = odr_vel - ransac_vel
        
        # Plot difference
        ax3.plot(common_times, vel_diff, 'k-', alpha=0.8)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_ylabel('ODR - RANSAC (m/s)')
        ax3.grid(True, alpha=0.3)
        
        # Calculate statistics
        mean_diff = np.mean(vel_diff)
        std_diff = np.std(vel_diff)
        
        # Add statistics text
        stats = f"Mean diff: {mean_diff:.1f} m/s\n"
        stats += f"Std diff: {std_diff:.1f} m/s"
        ax3.text(0.02, 0.98, stats, transform=ax3.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set common x-label
    ax3.set_xlabel('Time (s)')

    # Add statistics to velocity plots
    if np.any(mask_ransac):
        ransac_mean = np.mean(love_velocities_ransac['velocity'][mask_ransac])
        ransac_std = np.std(love_velocities_ransac['velocity'][mask_ransac])
        ransac_stats = f"Mean: {ransac_mean:.1f} m/s\nStd: {ransac_std:.1f} m/s"
        ax1.text(0.02, 0.98, ransac_stats, transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if np.any(mask_odr):
        odr_mean = np.mean(love_velocities_odr['velocity'][mask_odr])
        odr_std = np.std(love_velocities_odr['velocity'][mask_odr])
        odr_stats = f"Mean: {odr_mean:.1f} m/s\nStd: {odr_std:.1f} m/s"
        ax2.text(0.02, 0.98, odr_stats, transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig