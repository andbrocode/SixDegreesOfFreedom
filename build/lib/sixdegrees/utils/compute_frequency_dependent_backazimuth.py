"""
Functions for computing frequency-dependent backazimuth analysis.
"""
import numpy as np
from obspy import Stream
from obspy.signal.array_analysis import array_processing
import matplotlib.pyplot as plt

def compute_frequency_dependent_backazimuth(st, params, plot=False):
    """
    Compute frequency-dependent backazimuth analysis using array processing.
    
    Parameters:
    -----------
    st : obspy.Stream
        Stream containing array data
    params : dict
        Dictionary containing:
            - freq_min: List of minimum frequencies
            - freq_max: List of maximum frequencies
            - slowness_max: Maximum slowness value
            - slowness_step: Slowness step size
            - window_length: Window length in seconds
            - window_fraction: Window overlap fraction
            - prewhitening: Prewhitening flag
    plot : bool, optional
        Whether to plot the results (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing:
            - times: Array of time points
            - frequency: Dictionary with center, min, max frequencies
            - backazimuth: Dictionary with optimal, mean, std values
            - velocity: Array of velocities
            - cross_correlation: Dictionary with optimal, mean, std values
    """
    results = {
        'times': [],
        'frequency': {'center': [], 'min': [], 'max': []},
        'backazimuth': {'optimal': [], 'mean': [], 'std': []},
        'velocity': [],
        'cross_correlation': {'optimal': [], 'mean': [], 'std': []}
    }
    
    # Process each frequency band
    for fmin, fmax in zip(params['freq_min'], params['freq_max']):
        
        # Configure array processing parameters with explicit types
        kwargs = {
            'sll_x': float(-params['slowness_max']),
            'slm_x': float(params['slowness_max']),
            'sll_y': float(-params['slowness_max']), 
            'slm_y': float(params['slowness_max']),
            'sl_s': float(params['slowness_step']),
            'win_len': float(params['window_length']),
            'win_frac': float(params['window_fraction']),
            'frqlow': float(fmin),
            'frqhigh': float(fmax),
            'prewhiten': int(params['prewhitening']),
            'semb_thres': -1e9,
            'vel_thres': -1e9,
            'timestamp': 'mlabday',
            'stime': st[0].stats.starttime,
            'etime': st[0].stats.endtime,
            'method': 0,  # Explicitly set method
            'coordsys': 'lonlat',  # Explicitly set coordinate system
            'verbose': False  # Explicitly set verbose
        }

        # Perform array processing
        out = array_processing(st, **kwargs)
        
        # Extract results
        times = out[:, 0]
        rel_power = out[:, 1]
        abs_power = out[:, 2] 
        baz = out[:, 3]
        slowness = out[:, 4]
        
        # Fix backazimuth values
        baz[baz < 0.0] += 360
        
        # Calculate velocities
        velocity = 1.0 / slowness
        
        # Store results
        results['times'].extend(times)
        results['frequency']['center'].extend([np.mean([fmin, fmax])] * len(times))
        results['frequency']['min'].extend([fmin] * len(times))
        results['frequency']['max'].extend([fmax] * len(times))
        results['backazimuth']['optimal'].extend(baz)
        results['velocity'].extend(velocity)
        results['cross_correlation']['optimal'].extend(abs_power)
        
        # Calculate statistics
        results['backazimuth']['mean'].append(np.mean(baz))
        results['backazimuth']['std'].append(np.std(baz))
        results['cross_correlation']['mean'].append(np.mean(abs_power))
        results['cross_correlation']['std'].append(np.std(abs_power))

    if plot:
        fig = plot_frequency_backazimuth_analysis(results)
        return results, fig
    
    return results
