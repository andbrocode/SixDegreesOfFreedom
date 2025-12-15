"""
Functions for plotting frequency-dependent analysis results.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_dependent(results_freq, data='vel', figsize=(12, 5), minors=True, fmin=None, fmax=None):
    """
    Plot frequency-dependent velocity results with scatter points and connecting lines.
    
    Parameters:
    -----------
    results_freq : dict
        Results dictionary from compute_frequency_dependent_parameters containing:
            - frequency: Array of frequencies
            - velocity: Array of velocities
            - cc_values: Array of correlation coefficients
            - backazimuth: Array of backazimuth values
            - parameters: Dictionary with wave_type, t_win_factor, overlap, method
    data : str, optional
        Data to plot: 'vel' or 'baz', by default 'vel'
    figsize : tuple, optional
        Figure size (width, height), by default (12, 5)
    minors : bool, optional
        Add minor ticks to axes if True, by default True
    fmin : float, optional
        Minimum frequency to plot, by default None
    fmax : float, optional
        Maximum frequency to plot, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get data
    frequencies = np.array(results_freq['frequency'])
    velocities = np.array(results_freq['velocity'])
    cc_values = np.array(results_freq['cc_values'])
    backazimuths = np.array(results_freq['backazimuth'])

    # Sort by frequency to ensure line plot connects points in order
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    velocities = velocities[sort_idx]
    cc_values = cc_values[sort_idx]
    backazimuths = backazimuths[sort_idx]

    # Create scatter plot with colorbar for CC values
    if data == 'vel':
        scatter = ax.scatter(frequencies, velocities, c=cc_values, 
                            cmap='viridis', s=100, alpha=1,
                            edgecolor='black', linewidth=1, zorder=5,
                            vmax=1)
        # Add connecting line
        ax.plot(frequencies, velocities, 'k-', alpha=0.3, zorder=2)

    elif data == 'baz':
        scatter = ax.scatter(frequencies, backazimuths, c=cc_values, 
                            cmap='viridis', s=100, alpha=1,
                            edgecolor='black', linewidth=1, zorder=5,
                            vmax=1)
        # Add connecting line
        ax.plot(frequencies, backazimuths, 'k-', alpha=0.3, zorder=2)

    # add error bars for mad
    if data == 'vel':
        try:
            mad = results_freq['vel_mad']
            ax.errorbar(frequencies, velocities, yerr=mad, fmt='none', ecolor='black', alpha=0.3, zorder=4)
        except:
            pass
    elif data == 'baz':
        try:
            mad = results_freq['baz_mad']
            ax.errorbar(frequencies, backazimuths, yerr=mad, fmt='none', ecolor='black', alpha=0.3, zorder=4)
        except:
            pass

    # Configure axes
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)

    if data == 'vel':
        ax.set_ylabel('Apparent Phase Velocity (m/s)', fontsize=12)
    elif data == 'baz':
        ax.set_ylabel('Backazimuth (°)', fontsize=12)
    
    # Add grid
    ax.grid(True, which='major', alpha=0.3, zorder=1)
    if minors:
        ax.grid(True, which='minor', alpha=0.1, zorder=1)
        ax.minorticks_on()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, pad=0.01)
    cbar.set_label('Cross-correlation coefficient', fontsize=12)

    # Set x-axis limits
    if fmin is not None:
        ax.set_xlim(left=fmin)
    if fmax is not None:
        ax.set_xlim(right=fmax)

    # Set y-axis limits based on data type
    if data == 'vel':
        ax.set_ylim(bottom=0)
    if data == 'baz':
        ax.set_ylim(bottom=0, top=360)

    # Add title
    if data == 'vel':
        title = f"{results_freq['parameters']['wave_type'].title()} Wave Velocity vs Frequency "
    elif data == 'baz':
        title = f"{results_freq['parameters']['wave_type'].title()} Wave Backazimuth vs Frequency "
    title += f"(win={results_freq['parameters']['t_win_factor']}/f, "
    title += f"overlap={results_freq['parameters']['overlap']*100:.0f}%, "
    title += f"method={results_freq['parameters']['method']})"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig
