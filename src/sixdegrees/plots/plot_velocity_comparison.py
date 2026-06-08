"""
Functions for plotting velocity comparison results.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_comparison(results_vel1, results_vel2, labels=None, title=None):
    """
    Create a comparison plot of two velocity results with a difference subplot.
    
    Parameters:
    -----------
    results_vel1 : dict
        First velocity results dictionary containing:
            - time: array of time points
            - velocity: array of velocities
            - parameters: dict with 'win_time_s' and 'overlap'
    results_vel2 : dict
        Second velocity results dictionary (same structure as results_vel1)
    labels : tuple, optional
        Tuple of (label1, label2) for the two datasets
        Default: ('With Expected Backazimuth', 'Adaptive Backazimuth per Window')
    title : str, optional
        Base title for the plot (default: "Velocity Estimates Comparison")
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default labels if not provided
    if labels is None:
        labels = ('With Expected Backazimuth', 'Adaptive Backazimuth per Window')
    
    # Create figure with specific size ratio and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), 
                                  gridspec_kw={'height_ratios': [3, 1]},
                                  sharex=True)
    
    # Extract data
    times1 = results_vel1['time']
    vel1 = results_vel1['velocity']
    times2 = results_vel2['time']
    vel2 = results_vel2['velocity']
    
    # Get window and overlap from parameters
    window_time = results_vel1['parameters']['win_time_s']
    overlap = results_vel1['parameters']['overlap'] * 100  # Convert to percentage
    
    # Plot velocities - note the order of plotting matches the label order
    ax1.plot(times1, vel1, 'ko', label=labels[0], alpha=0.7,
             markersize=6, markerfacecolor='gray', markeredgecolor='black')
    
    ax1.plot(times2, vel2, 'rs', label=labels[1], alpha=0.7,
             markersize=6, markerfacecolor='red', markeredgecolor='black')
    
    # Calculate velocity difference in percentage
    # Interpolate to common time points if necessary
    if not np.array_equal(times1, times2):
        # Use the intersection of time ranges
        t_min = max(times1.min(), times2.min())
        t_max = min(times1.max(), times2.max())
        mask1 = (times1 >= t_min) & (times1 <= t_max)
        mask2 = (times2 >= t_min) & (times2 <= t_max)
        
        # Calculate percentage difference where both velocities exist
        common_times = times1[mask1]
        vel1_masked = vel1[mask1]
        vel2_masked = np.interp(common_times, times2[mask2], vel2[mask2])
    else:
        common_times = times1
        vel1_masked = vel1
        vel2_masked = vel2
    
    # Calculate percentage difference
    diff_percent = ((vel2_masked - vel1_masked) / vel1_masked) * 100
    
    # Plot difference as bars
    ax2.bar(common_times, diff_percent, color='black', width=window_time/5)
    
    # Customize velocity plot
    ax1.set_ylabel('Velocity (m/s)')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 3000)
    
    # Customize difference plot
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Difference (%)')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_ylim(-50, 50)
    
    # Set title with parameters
    if title is None:
        title = "Velocity Estimates Comparison"
    full_title = f"{title} (Window: {window_time:.1f}s | Overlap: {overlap:.0f}%)"
    fig.suptitle(full_title, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    
    return fig
