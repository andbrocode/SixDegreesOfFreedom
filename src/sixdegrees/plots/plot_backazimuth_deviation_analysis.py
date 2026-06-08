"""
Functions for plotting backazimuth deviation analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

def plot_backazimuth_deviation_analysis(results, event_info, figsize=(15, 8), bin_step=None):
    """
    Plot deviation analysis between estimated and theoretical backazimuth.
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_dependent_backazimuth containing:
            - detailed_results: Dictionary of wave type results
            - frequency_bands: Center frequencies
    event_info : dict
        Event information with 'backazimuth' key for theoretical comparison
    figsize : tuple
        Figure size (width, height)
    bin_step : float, optional
        Bin spacing in degrees (e.g., 5 for bins every 5 degrees). 
        If None, uses automatic binning with 20 bins.
        
    Returns:
    --------
    tuple : (figure, analysis_results)
        Figure object and dictionary with deviation analysis results
    """
    if 'backazimuth' not in event_info:
        print("No theoretical backazimuth available in event_info")
        return None, {}
    
    theoretical_baz = event_info['backazimuth']
    wave_types = list(results['wave_types'].keys())
    n_wave_types = len(wave_types)
    
    if n_wave_types == 0:
        print("No wave type results to analyze")
        return None, {}
    
    # Calculate deviations for each wave type
    deviations = {}
    center_freqs = results['frequency_bands']['center']
    
    for wave_type in wave_types:
        peak_baz = results['wave_types'][wave_type]['peak_baz']
        valid_mask = ~np.isnan(peak_baz)
        
        if np.any(valid_mask):
            # Calculate angular deviation (considering circular nature of angles)
            deviation = peak_baz[valid_mask] - theoretical_baz
            # Wrap to [-180, 180] range
            deviation = ((deviation + 180) % 360) - 180
            
            deviations[wave_type] = {
                'deviation': deviation,
                'frequencies': center_freqs[valid_mask],
                'all_deviation': np.full_like(peak_baz, np.nan),
                'mean_deviation': np.mean(deviation),
                'std_deviation': np.std(deviation),
                'rms_deviation': np.sqrt(np.mean(deviation**2))
            }
            
            # Store all deviations (including NaN for missing estimates)
            all_dev = np.full_like(peak_baz, np.nan)
            dev_calc = peak_baz - theoretical_baz
            dev_calc = ((dev_calc + 180) % 360) - 180
            all_dev[valid_mask] = deviation
            deviations[wave_type]['all_deviation'] = dev_calc
        else:
            print(f"No valid estimates for {wave_type}")
            continue
    
    if not deviations:
        print("No valid deviations to plot")
        return None, {}
    
    # Create figure with layout: main plot + single merged histogram
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[3, 0.8, 0.1, 0.3], 
                 hspace=0.0, wspace=0.0)
    ax_freq = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_freq)
    
    # Colors for different wave types
    colors = {'love': 'blue', 'rayleigh': 'red'}
    
    # Plot 1: Deviation vs Frequency with lines to zero (no regression)
    for i, (wave_type, data) in enumerate(deviations.items()):
        color = colors.get(wave_type, f'C{i}')
        
        # Plot vertical lines from zero to each point
        for freq, dev in zip(data['frequencies'], data['deviation']):
            ax_freq.plot([freq, freq], [0, dev], color=color, alpha=0.3, linewidth=1)
        
        # Plot deviation vs frequency markers
        ax_freq.semilogx(data['frequencies'], data['deviation'], 
                        'o', color=color, alpha=0.8, markersize=8,
                        label=f'{wave_type.upper()} waves', markeredgecolor='black',
                        markeredgewidth=0.5)
    
    # Reference line at zero deviation
    ax_freq.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax_freq.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_freq.set_ylabel('Deviation from Theoretical BAZ (°)', fontsize=12)
    ax_freq.set_title('Backazimuth Deviation vs Frequency', fontsize=14, fontweight='bold')
    
    # Add grid and subgrid
    ax_freq.grid(True, which='major', alpha=0.5, linewidth=1)
    ax_freq.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    ax_freq.minorticks_on()
    
    ax_freq.legend(loc='upper left')
    
    # Determine binning strategy
    all_deviations = np.concatenate([data['deviation'] for data in deviations.values()])
    
    if bin_step is not None:
        # Use fixed degree spacing
        data_min, data_max = np.min(all_deviations), np.max(all_deviations)
        
        # Extend range to nearest bin_step boundaries
        bin_min = np.floor(data_min / bin_step) * bin_step
        bin_max = np.ceil(data_max / bin_step) * bin_step
        
        # Create bins every bin_step degrees
        common_bins = np.arange(bin_min, bin_max + bin_step, bin_step)
        bin_info = f"(bins every {bin_step}°)"
    else:
        # Use automatic binning
        n_bins = 20
        bin_range = (np.min(all_deviations), np.max(all_deviations))
        common_bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
        bin_info = f"({len(common_bins)-1} bins)"
    
    # Plot 2: Optimized histogram with KDE overlay
    if n_wave_types == 1:
        # Single wave type
        wave_type = list(deviations.keys())[0]
        data = deviations[wave_type]
        color = colors.get(wave_type, 'blue')
        
        # Create histogram
        counts, bins, patches = ax_hist.hist(data['deviation'], bins=common_bins, alpha=0.6, color=color, 
                                           edgecolor='black', density=True, orientation='horizontal')
        
        # Add KDE overlay
        if len(data['deviation']) > 1:
            kde = stats.gaussian_kde(data['deviation'])
            y_kde = np.linspace(data['deviation'].min(), data['deviation'].max(), 100)
            kde_values = kde(y_kde)
            ax_hist.plot(kde_values, y_kde, color=color, linewidth=2, alpha=0.8, label='KDE')
    
    else:
        # Multiple wave types - optimized layout with bars left/right of bin centers
        bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
        bin_width = np.diff(common_bins)[0]
        
        # Calculate bar positioning
        bar_width = bin_width * 0.35  # Narrower bars
        positions = [-bar_width/2, bar_width/2] if n_wave_types == 2 else [0]  # Left/right positioning
        
        kde_curves = {}  # Store KDE curves for overlay
        
        for i, (wave_type, data) in enumerate(deviations.items()):
            color = colors.get(wave_type, f'C{i}')
            
            # Calculate histogram counts
            counts, _ = np.histogram(data['deviation'], bins=common_bins, density=True)
            
            # Position bars left/right of bin centers
            if n_wave_types > 1:
                offset_bins = bin_centers + positions[i]
            else:
                offset_bins = bin_centers
            
            # Plot bars
            bars = ax_hist.barh(offset_bins, counts, height=bar_width, 
                              color=color, alpha=0.6, edgecolor='black', linewidth=0.5,
                              label=f'{wave_type.upper()}')
            
            # Calculate and store KDE for overlay
            if len(data['deviation']) > 1:
                kde = stats.gaussian_kde(data['deviation'])
                y_kde = np.linspace(common_bins[0], common_bins[-1], 100)
                kde_values = kde(y_kde)
                kde_curves[wave_type] = {'y': y_kde, 'kde': kde_values, 'color': color}
        
        # Overlay KDE curves
        for wave_type, kde_data in kde_curves.items():
            ax_hist.plot(kde_data['kde'], kde_data['y'], 
                        color=kde_data['color'], linewidth=2.5, alpha=0.9,
                        linestyle='-', label=f'{wave_type.upper()} KDE')
    
    # Zero reference line in histogram
    ax_hist.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax_hist.set_xlabel('Density', fontsize=10)
    ax_hist.set_title(f'Distribution {bin_info}', fontsize=11)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.tick_params(labelleft=False)  # Remove y-axis labels
    
    # Remove 0.00 tick label from density axis
    xticks = ax_hist.get_xticks()
    xticks_filtered = xticks[xticks > 0.001]  # Remove ticks close to zero
    if len(xticks_filtered) > 0:
        ax_hist.set_xticks(xticks_filtered)
    
    # Add legend to histogram
    if n_wave_types > 1:
        ax_hist.legend(loc='upper right', fontsize=8)
    elif n_wave_types == 1 and len(list(deviations.values())[0]['deviation']) > 1:
        ax_hist.legend(loc='upper right', fontsize=8)
    
    # Overall title
    plt.suptitle('Backazimuth Estimation Deviation Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Return results for further analysis
    analysis_results = {
        'deviations': deviations,
        'theoretical_baz': theoretical_baz,
        'center_frequencies': center_freqs,
        'bin_info': bin_info
    }
    
    return fig, analysis_results
