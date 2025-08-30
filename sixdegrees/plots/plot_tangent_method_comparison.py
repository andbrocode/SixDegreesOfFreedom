
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
    from sixdegrees.utils.get_kde_stats import get_kde_stats
    
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
           alpha=0.6, color='tab:blue', edgecolor='darkblue', linewidth=1.5,
           label=f'Rotation (N={len(baz_rot)})')
    
    # Acceleration histogram (right side of bins)
    counts_acc, _ = np.histogram(baz_acc, bins=bins, density=True)
    ax.bar(bin_centers + bar_width/2, counts_acc, width=bar_width, 
           alpha=0.6, color='tab:orange', edgecolor='brown', linewidth=1.5,
           label=f'Acceleration (N={len(baz_acc)})')
    
    # Add KDE curves
    if len(baz_rot) > 1:
        # get kde stats (pad)
        kde_stats_rot = get_kde_stats(baz_rot, cc_rot, _baz_steps=0.5, Ndegree=60)
        baz_estimate_rot = kde_stats_rot['baz_estimate']
        baz_std_rot = kde_stats_rot['kde_dev']
        kde_max_rot = max(kde_stats_rot['kde_values'])

        ax.plot(kde_stats_rot['kde_angles'], 
                kde_stats_rot['kde_values'],
                color='darkblue', linewidth=2.5,
                alpha=0.9, label='Rotation KDE'
                )

    if len(baz_acc) > 1:
        # get kde stats (pad)
        kde_stats_acc = get_kde_stats(baz_acc, cc_acc, _baz_steps=0.5, Ndegree=60)
        baz_estimate_acc = kde_stats_acc['baz_estimate']
        baz_std_acc = kde_stats_acc['kde_dev']
        kde_max_acc = max(kde_stats_acc['kde_values'])

        ax.plot(kde_stats_acc['kde_angles'], 
                kde_stats_acc['kde_values'],
                color='brown', linewidth=2.5,
                alpha=0.9, label='Acceleration KDE'
                )

    # Add theoretical backazimuth if available
    if event_info and 'backazimuth' in event_info:
        theo_baz = event_info['backazimuth']
        ax.axvline(theo_baz, color='green', linestyle='--', 
                   linewidth=3, label=f'Theoretical: {theo_baz:.1f}°')
    
    # get estatimates from results_rot and results_acc
    rot_baz_estimate = round(results_rot['baz_estimate'], 0)
    acc_baz_estimate = round(results_acc['baz_estimate'], 0)
    rot_baz_estimate_std = round(results_rot['baz_estimate_std'], 0)
    acc_baz_estimate_std = round(results_acc['baz_estimate_std'], 0)

    print(f"rot_baz_estimate: {rot_baz_estimate}, acc_baz_estimate: {acc_baz_estimate}")
    
    # Add statistics text
    stats_text = f"Rotation: {rot_baz_estimate}° ± {rot_baz_estimate_std}°\n"
    stats_text += f"Acceleration: {acc_baz_estimate}° ± {acc_baz_estimate_std}°\n"
    
    # Calculate difference
    diff = abs(rot_baz_estimate - acc_baz_estimate)
    if diff > 180:
        diff = 360 - diff
    # stats_text += f"Difference: {diff:.1f}°"
    
    # Add deviations if theoretical available
    if event_info and 'backazimuth' in event_info:
        dev_rot = abs(rot_baz_estimate - theo_baz)
        if dev_rot > 180:
            dev_rot = 360 - dev_rot
        dev_acc = abs(acc_baz_estimate - theo_baz)
        if dev_acc > 180:
            dev_acc = 360 - dev_acc
        stats_text += f"\nRot. Dev.: {dev_rot}°\nAcc. Dev.: {dev_acc}°"
    
    # add max_rot and max_acc as vertical lines between 0 and max value
    ax.plot([rot_baz_estimate, rot_baz_estimate], [0, kde_max_rot],
            color='darkblue', linestyle='--', linewidth=2,
            label=f'Rotation Max: {rot_baz_estimate} ± {rot_baz_estimate_std}°'
            )
    ax.plot([acc_baz_estimate, acc_baz_estimate], [0, kde_max_acc],
            color='darkred', linestyle='--', linewidth=2,
            label=f'Acceleration Max: {acc_baz_estimate} ± {acc_baz_estimate_std}°'
            )

    # Position statistics text
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         verticalalignment='top', fontsize=11, fontfamily='monospace',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Configure plot
    ax.set_xlabel('Backazimuth (°)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Tangent Method Comparison: Rotation vs Acceleration Components', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.minorticks_on()

    # Remove 0.00 tick label from density axis
    yticks = ax.get_yticks()
    yticks_filtered = yticks[yticks > 0.001]
    if len(yticks_filtered) > 0:
        ax.set_yticks(yticks_filtered)
    
    plt.tight_layout()
    return fig

