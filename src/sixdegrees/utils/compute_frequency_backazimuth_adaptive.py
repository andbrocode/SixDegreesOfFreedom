"""
Functions for computing frequency-dependent backazimuth analysis with adaptive windows.
"""
import numpy as np
import gc
from acoustics.octave import Octave
from obspy import Stream
from .get_kde_stats import get_kde_stats

def compute_frequency_backazimuth_adaptive(sd_object, wave_type='love', fmin=0.01, fmax=0.5, 
                                         octave_fraction=3, baz_step=1, 
                                         window_factor=1.0, overlap_fraction=0.5,
                                         baz_win_overlap=0.5, verbose=True,
                                         cc_threshold=None):
    """
    Compute backazimuth for octave frequency bands with adaptive time windows (1/fc).
    
    Parameters:
    -----------
    sd_object : sixdegrees object
        Object containing seismic data and processing methods
    wave_type : str
        'love', 'rayleigh', or 'tangent'
    fmin, fmax : float
        Frequency range in Hz
    octave_fraction : int
        Octave fraction (3 for 1/3 octave)
    window_factor : float
        Multiplier for 1/fc to determine time window length (default: 1.0)
    overlap_fraction : float
        Overlap fraction between time windows (0-1)
    cc_threshold : float, optional
        Correlation coefficient threshold for filtering results
    
    Returns:
    --------
    dict
        Results with frequency bands, adaptive time windows, correlations, backazimuths,
        and statistical estimates including histogram, KDE, and uncertainty
    """
    if verbose:
        print(f"Computing {wave_type} backazimuth with adaptive time windows (factor={window_factor})...")
    
    # Generate octave bands
    octave = Octave(fraction=octave_fraction, fmin=fmin, fmax=fmax)
    center_freqs = octave.center
    lower_freqs = octave.lower
    upper_freqs = octave.upper
    
    # Store original stream
    original_stream = sd_object.get_stream('all', raw=True).copy()
    total_duration = original_stream[0].stats.endtime - original_stream[0].stats.starttime
    
    # Initialize results
    results = {
        'frequency_bands': center_freqs,
        'frequency_lower': lower_freqs,
        'frequency_upper': upper_freqs,
        'backazimuth_data': [],
        'correlation_data': [],
        'time_windows': [],  # Will be list of arrays, one per frequency
        'adaptive_windows': True,
        'window_factor': window_factor,
    }
    
    # Process each frequency band with adaptive time windows
    for i, (fl, fu, fc) in enumerate(zip(lower_freqs, upper_freqs, center_freqs)):
        # Calculate adaptive time window length
        time_window_sec = max(int(window_factor / fc), 1)
   
        if verbose:
            print(f"  Processing {fc:.3f} Hz ({fl:.3f}-{fu:.3f} Hz), window={time_window_sec:.1f}s")
        
        try:
            # Filter data for this frequency band
            filtered_stream = original_stream.copy()
            filtered_stream.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
            
            # Temporarily replace stream in sd_object
            sd_object.st = filtered_stream
            
            # Compute backazimuth for this frequency band with adaptive window
            results_baz = sd_object.compute_backazimuth(
                wave_type=wave_type,
                baz_step=baz_step,
                baz_win_sec=time_window_sec,
                baz_win_overlap=overlap_fraction,
                verbose=False,
                out=True
            )
 
            if results_baz and 'cc_max_y' in results_baz:
                # Store time windows for this frequency
                results['time_windows'].append(results_baz['twin_center'])
                
                # Store backazimuth and correlation data
                results['backazimuth_data'].append(results_baz['cc_max_y'])
                results['correlation_data'].append(results_baz['cc_max'])
            else:
                # Fill with NaN if no results
                # Create dummy time windows based on expected length
                n_windows = max(1, int(total_duration / time_window_sec * (1 - overlap_fraction) + 1))
                dummy_times = np.linspace(time_window_sec/2, total_duration - time_window_sec/2, n_windows)
                
                results['time_windows'].append(dummy_times)
                results['backazimuth_data'].append(np.full(len(dummy_times), np.nan))
                results['correlation_data'].append(np.full(len(dummy_times), np.nan))
                
        except Exception as e:
            if verbose:
                print(f"    Error processing {fc:.3f} Hz: {e}")
            
            # Create dummy data for failed processing
            time_window_sec = window_factor / fc
            n_windows = max(1, int(total_duration / time_window_sec * (1 - overlap_fraction) + 1))
            dummy_times = np.linspace(time_window_sec/2, total_duration - time_window_sec/2, n_windows)
            
            results['time_windows'].append(dummy_times)
            results['backazimuth_data'].append(np.full(len(dummy_times), np.nan))
            results['correlation_data'].append(np.full(len(dummy_times), np.nan))
        
        finally:
            # Restore original stream
            sd_object.st = original_stream
        
        gc.collect()
    
    if verbose:
        total_points = sum(len(baz_data) for baz_data in results['backazimuth_data'])
        valid_points = sum(np.sum(~np.isnan(baz_data)) for baz_data in results['backazimuth_data'])
        coverage = valid_points / total_points * 100 if total_points > 0 else 0
        print(f"Completed: {coverage:.1f}% coverage ({valid_points}/{total_points} points)")

    # Compute statistical estimates
    if verbose:
        print("Computing statistical estimates...")
    
    # Flatten all backazimuth and correlation data
    all_baz = np.concatenate(results['backazimuth_data'])
    all_cc = np.concatenate(results['correlation_data'])

    # Apply correlation threshold if specified
    if cc_threshold is not None:
        mask = all_cc >= cc_threshold
        all_baz = all_baz[mask]
        all_cc = all_cc[mask]

    # Remove NaN values
    valid_mask = ~np.isnan(all_baz)
    all_baz = all_baz[valid_mask]
    all_cc = all_cc[valid_mask]

    if len(all_baz) > 5:
        # get kde stats for backazimuth
        kde_stats = get_kde_stats(all_baz, all_cc, _baz_steps=0.5, Ndegree=60)
        baz_estimate = kde_stats['baz_estimate']
        baz_std = kde_stats['kde_dev']

        # Store statistical results
        results.update({
            'baz_estimate': baz_estimate,
            'baz_std': baz_std,
            'n_measurements': len(all_baz),
        })
        
        if verbose:
            print(f"Estimated backazimuth: {baz_estimate:.1f}° ± {baz_std:.1f}°")
            print(f"Based on {len(all_baz)} measurements with mean CC: {np.mean(all_cc):.3f}")
    else:
        if verbose:
            print("Warning: No valid measurements for statistical estimation")
        results.update({
            'baz_estimate': np.nan,
            'baz_std': np.nan,
            'n_measurements': 0,
        })
    
    gc.collect()
    return results
