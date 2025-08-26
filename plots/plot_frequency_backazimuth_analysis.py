"""
Functions for plotting frequency-dependent backazimuth analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_frequency_backazimuth_analysis(results, event_info=None, vmax_percentile=95,
                                      figsize=(12, 10), show_peak_line=True):
    """
    Plot frequency-dependent backazimuth analysis results.
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_dependent_backazimuth containing:
            - wave_types: Dictionary of wave type results
            - frequency_bands: Center frequencies
            - parameters: Analysis parameters
    event_info : dict, optional
        Event information with 'backazimuth' key for theoretical comparison
    vmax_percentile : float
        Percentile for color scale maximum (to avoid outliers)
    figsize : tuple
        Figure size (width, height)
    show_peak_line : bool
        Whether to show line connecting peak estimates
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    wave_types = list(results['wave_types'].keys())
    n_wave_types = len(wave_types)
    
    if n_wave_types == 0:
        print("No wave type results to plot")
        return None
    
    # Create figure
    fig, axes = plt.subplots(n_wave_types, 1, figsize=figsize, sharex=True)
    if n_wave_types == 1:
        axes = [axes]
    
    # Get frequency data
    center_freqs = results['frequency_bands']['center']
    baz_grid = results['baz_grid']
    
    # Create meshgrid for pcolormesh
    freq_edges = np.logspace(np.log10(center_freqs.min()), np.log10(center_freqs.max()), len(center_freqs) + 1)
    baz_edges = np.arange(0, 361, np.diff(baz_grid)[0])
    
    colors = {'love': 'Blues', 'rayleigh': 'Reds'}
    
    for i, wave_type in enumerate(wave_types):
        ax = axes[i]
        data = results['wave_types'][wave_type]
        
        # Get KDE values and normalize for better visualization
        kde_matrix = data['kde_values'].T  # Transpose for correct orientation
        
        # Set colormap limits
        kde_nonzero = kde_matrix[kde_matrix > 0]
        if len(kde_nonzero) > 0:
            vmax = np.percentile(kde_nonzero, vmax_percentile)
            vmin = np.percentile(kde_nonzero, 5)
        else:
            vmax = 1.0
            vmin = 0.01
        
        # Create pcolormesh plot
        colormap = colors.get(wave_type, 'viridis')
        im = ax.pcolormesh(center_freqs, baz_grid, kde_matrix, 
                          cmap=colormap, shading='auto',
                          vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='KDE Density', pad=0.02)
        
        # Plot peak line if requested
        if show_peak_line:
            valid_peaks = ~np.isnan(data['peak_baz'])
            if np.any(valid_peaks):
                ax.plot(center_freqs[valid_peaks], data['peak_baz'][valid_peaks], 
                       'k-', linewidth=2, alpha=0.8, label='Peak BAZ')
                ax.scatter(center_freqs[valid_peaks], data['peak_baz'][valid_peaks], 
                       color='k', marker='o', alpha=0.8, facecolor='white', zorder=3)
        
        # Plot theoretical backazimuth if available
        if event_info and 'backazimuth' in event_info:
            ax.axhline(y=event_info['backazimuth'], color='grey', 
                      linestyle='--', linewidth=2, alpha=0.9, label='Theoretical BAZ')
        
        # Customize axes
        ax.set_xscale('log')
        ax.set_ylabel('Backazimuth (°)')
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(0, 361, 60))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{wave_type.upper()} Wave Backazimuth vs Frequency', 
                    fontsize=12, fontweight='bold')
        
        # Add legend if there are lines to show
        if show_peak_line or (event_info and 'backazimuth' in event_info):
            ax.legend(loc='upper right')
        
        # Add statistics text
        n_bands_with_data = np.sum(data['n_estimates'] > 0)
        stats_text = f'Bands with data: {n_bands_with_data}/{len(center_freqs)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    # Set x-label only for bottom subplot
    axes[-1].set_xlabel('Frequency (Hz)')
    
    # Main title
    octave_frac = results['parameters']['octave_fraction']
    plt.suptitle(f'Frequency-Dependent Backazimuth Analysis (1/{octave_frac} Octave Bands)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig
