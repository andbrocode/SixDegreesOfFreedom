"""
Functions for plotting frequency vs time maps with adaptive windows.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_time_map_adaptive(results, plot_type='backazimuth', event_info=None, 
                                   figsize=(12, 8), vmin=None, vmax=None):
    """
    Plot frequency vs time map for adaptive time windows using grid-based approach.
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_backazimuth_adaptive containing:
            - frequency_bands: Array of frequency bands
            - time_windows: List of time window arrays
            - backazimuth_data: List of backazimuth arrays
            - correlation_data: List of correlation arrays
            - window_factor: Window factor used
    plot_type : str
        'backazimuth' (shows deviation from theoretical) or 'correlation'
    event_info : dict, optional
        Event info with theoretical 'backazimuth' for comparison
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float, optional
        Color scale limits
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Check if this is adaptive window data
    if not results.get('adaptive_windows', False):
        print("Warning: This function is designed for adaptive window results")
    
    freq_bands = results['frequency_bands']
    
    # Find the smallest time window to determine grid resolution
    min_window_length = float('inf')
    all_times = []
    
    for freq_idx, freq in enumerate(freq_bands):
        time_windows = results['time_windows'][freq_idx]
        all_times.extend(time_windows)
        
        # Calculate window length for this frequency
        window_length = results['window_factor'] / freq
        min_window_length = min(min_window_length, window_length)
    
    # Create time grid based on smallest window
    all_times = np.array(all_times)
    time_min, time_max = np.min(all_times), np.max(all_times)
    
    # Grid resolution: use half the minimum window length for fine resolution
    grid_time_step = min_window_length / 2
    n_time_bins = int((time_max - time_min) / grid_time_step) + 1
    time_grid = np.linspace(time_min, time_max, n_time_bins)
    
    # Create frequency grid (log scale)
    freq_grid = freq_bands
    
    # Create meshgrid
    TIME_GRID, FREQ_GRID = np.meshgrid(time_grid, freq_grid)
    
    # Initialize data grid with NaN
    data_grid = np.full(TIME_GRID.shape, np.nan)
    
    # Fill grid with data from each frequency band
    for freq_idx, freq in enumerate(freq_bands):
        time_windows = results['time_windows'][freq_idx]
        
        if plot_type == 'backazimuth':
            baz_data = results['backazimuth_data'][freq_idx]
            
            if event_info and 'backazimuth' in event_info:
                theoretical_baz = event_info['backazimuth']
                # Calculate deviation
                deviation = baz_data - theoretical_baz
                deviation = ((deviation + 180) % 360) - 180
                values = deviation
            else:
                values = baz_data
        else:  # correlation
            values = results['correlation_data'][freq_idx]
        
        # Ensure arrays have same length
        min_length = min(len(time_windows), len(values))
        time_windows_trimmed = time_windows[:min_length]
        values_trimmed = values[:min_length]
        
        # Calculate window length for this frequency
        window_length = results['window_factor'] / freq
        half_window = window_length / 2
        
        # Fill grid cells for each time window
        for t_center, value in zip(time_windows_trimmed, values_trimmed):
            if np.isnan(value):
                continue
                
            # Find time range for this window
            t_start = t_center - half_window
            t_end = t_center + half_window
            
            # Find grid indices that fall within this time window
            time_mask = (time_grid >= t_start) & (time_grid <= t_end)
            time_indices = np.where(time_mask)[0]
            
            # Fill all grid cells within this time window
            for t_idx in time_indices:
                data_grid[freq_idx, t_idx] = value
    
    # Set up plot parameters
    if plot_type == 'backazimuth':
        if event_info and 'backazimuth' in event_info:
            label = 'Backazimuth Deviation (°)'
            cmap = 'RdBu_r'
            if vmin is None and vmax is None:
                valid_data = data_grid[~np.isnan(data_grid)]
                if len(valid_data) > 0:
                    max_abs_dev = np.max(np.abs(valid_data))
                    vmin, vmax = -max_abs_dev, max_abs_dev
                else:
                    vmin, vmax = -10, 10
        else:
            label = 'Backazimuth (°)'
            cmap = 'hsv'
            if vmin is None: vmin = 0
            if vmax is None: vmax = 360
    else:
        label = 'Cross-Correlation'
        cmap = 'viridis'
        if vmin is None: vmin = 0
        if vmax is None: 
            valid_data = data_grid[~np.isnan(data_grid)]
            vmax = np.max(valid_data) if len(valid_data) > 0 else 1.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using pcolormesh
    im = ax.pcolormesh(time_grid, freq_bands, data_grid, 
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=label, pad=0.02)
    
    # Show adaptive window boundaries
    try:
        for freq_idx, freq in enumerate(freq_bands):
            time_windows = results['time_windows'][freq_idx]
            window_length = results['window_factor'] / freq
    except Exception as e:
        print(f"Warning: Could not plot window boundaries: {e}")
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_yscale('log')
    
    title = f'Frequency-Time Map: {plot_type.title()} (Adaptive Windows: {results["window_factor"]:.1f}/fc)'
    if plot_type == 'backazimuth' and event_info and 'backazimuth' in event_info:
        title += f'\n(Theoretical BAZ: {event_info["backazimuth"]:.1f}°)'
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig
