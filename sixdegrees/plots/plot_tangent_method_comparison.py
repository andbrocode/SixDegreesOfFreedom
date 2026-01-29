def plot_tangent_method_comparison(results_rot, results_acc, event_info=None, figsize=(12, 6)):
    """
    Simple comparison plot of rotation vs acceleration tangent methods
    
    Parameters:
    -----------
    results_rot : dict
        Results from tangent method using rotation components
    results_acc : dict  
        Results from tangent method using acceleration components
    event_info : dict, optional
        Event information with theoretical backazimuth
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Simple comparison plot figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Extract data
    baz_rot = results_rot['cc_max_y'] 
    cc_rot = results_rot['cc_max']
    
    baz_acc = results_acc['cc_max_y']
    cc_acc = results_acc['cc_max']
    
    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins every 10 degrees
    bins = np.arange(0, 361, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = 10
    
    # Plot histograms with offset positioning
    bar_width = bin_width * 0.35
    
    # Rotation histogram (left side of bins)
    counts_rot, _ = np.histogram(baz_rot, bins=bins, density=True)
    ax.bar(bin_centers - bar_width/2, counts_rot, width=bar_width, 
           alpha=0.6, color='blue', edgecolor='darkblue', linewidth=0.5,
           label=f'Rotation (N={len(baz_rot)})')
    
    # Acceleration histogram (right side of bins)
    counts_acc, _ = np.histogram(baz_acc, bins=bins, density=True)
    ax.bar(bin_centers + bar_width/2, counts_acc, width=bar_width, 
           alpha=0.6, color='red', edgecolor='darkred', linewidth=0.5,
           label=f'Acceleration (N={len(baz_acc)})')
    
    # Add KDE curves
    if len(baz_rot) > 1:
        kde_rot = stats.gaussian_kde(baz_rot, weights=cc_rot)
        x_kde = np.linspace(0, 360, 360)
        kde_values_rot = kde_rot(x_kde)
        ax.plot(x_kde, kde_values_rot, color='darkblue', linewidth=2.5, 
                alpha=0.9, label='Rotation KDE')
    
    if len(baz_acc) > 1:
        kde_acc = stats.gaussian_kde(baz_acc, weights=cc_acc)
        x_kde = np.linspace(0, 360, 360)
        kde_values_acc = kde_acc(x_kde)
        ax.plot(x_kde, kde_values_acc, color='darkred', linewidth=2.5, 
                alpha=0.9, label='Acceleration KDE')
    
    # Add theoretical backazimuth if available
    if event_info and 'backazimuth' in event_info:
        theo_baz = event_info['backazimuth']
        ax.axvline(theo_baz, color='green', linestyle='--', 
                   linewidth=3, label=f'Theoretical: {theo_baz:.1f}°')
    
    # Calculate and display statistics

    # calculated maximum of kde and its index and the half width at half maximum
    kde_rot = stats.gaussian_kde(baz_rot, weights=cc_rot)
    kde_acc = stats.gaussian_kde(baz_acc, weights=cc_acc)
    max_rot = np.max(kde_rot.pdf(x_kde))
    max_rot_index = np.where(kde_rot.pdf(x_kde) == max_rot)[0][0]
    max_acc = np.max(kde_acc.pdf(x_kde))
    max_acc_index = np.where(kde_acc.pdf(x_kde) == max_acc)[0][0]
    hwhm_rot = np.abs(x_kde[np.where(kde_rot.pdf(x_kde) > max_rot/2)[0][0]] - x_kde[np.where(kde_rot.pdf(x_kde) > max_rot/2)[0][-1]])
    hwhm_acc = np.abs(x_kde[np.where(kde_acc.pdf(x_kde) > max_acc/2)[0][0]] - x_kde[np.where(kde_acc.pdf(x_kde) > max_acc/2)[0][-1]])
    
    # Add statistics text
    stats_text = f"Rotation: {max_rot_index:.1f}° ± {hwhm_rot:.1f}°\n"
    stats_text += f"Acceleration: {max_acc_index:.1f}° ± {hwhm_acc:.1f}°\n"
    
    # Calculate difference
    diff = abs(max_rot_index - max_acc_index)
    if diff > 180:
        diff = 360 - diff
    stats_text += f"Difference: {diff:.1f}°"
    
    # Add deviations if theoretical available
    if event_info and 'backazimuth' in event_info:
        dev_rot = abs(max_rot_index - theo_baz)
        if dev_rot > 180:
            dev_rot = 360 - dev_rot
        dev_acc = abs(max_acc_index - theo_baz)
        if dev_acc > 180:
            dev_acc = 360 - dev_acc
        stats_text += f"\nRot. Dev.: {dev_rot:.1f}°\nAcc. Dev.: {dev_acc:.1f}°"
    
    # add max_rot and max_acc as vertical lines between 0 and max value
    ax.plot([max_rot_index, max_rot_index], [0, max_rot],
            color='darkblue', linestyle='--', linewidth=2,
            label=f'Rotation Max: {max_rot_index:.1f}°'
            )
    ax.plot([max_acc_index, max_acc_index], [0, max_acc],
            color='darkred', linestyle='--', linewidth=2,
            label=f'Acceleration Max: {max_acc_index:.1f}°'
            )

    # Position statistics text
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Configure plot
    ax.set_xlabel('Backazimuth (°)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Tangent Method Comparison: Rotation vs Acceleration Components', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Remove 0.00 tick label from density axis
    yticks = ax.get_yticks()
    yticks_filtered = yticks[yticks > 0.001]
    if len(yticks_filtered) > 0:
        ax.set_yticks(yticks_filtered)
    
    plt.tight_layout()
    return fig