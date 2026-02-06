"""
Functions for plotting filtered traces in frequency bands for Rayleigh or Love waves.
Includes optimized backazimuth search for each frequency band.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Tuple
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from obspy import Stream
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.cross_correlation import correlate, xcorr_max
from ..utils.regression import regression


def _generate_octave_bands(fmin: float, fmax: float) -> List[Tuple[float, float]]:
    """
    Generate octave frequency bands between fmin and fmax.
    
    Parameters:
    -----------
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns:
    --------
    bands : list of tuples
        List of (fmin, fmax) tuples for each octave band
    """
    bands = []
    f = fmin
    
    while f < fmax:
        f_next = f * 2  # Next octave
        if f_next > fmax:
            f_next = fmax
        bands.append((f, f_next))
        f = f_next
    
    return bands

def _optimize_backazimuth(
    rot_filtered: Stream,
    acc_filtered: Stream,
    wave_type: str,
    baz_center: float,
    baz_range: float = 20.0,
    baz_step: float = 1.0
) -> Tuple[float, float]:
    """
    Optimize backazimuth by maximizing correlation coefficient.
    
    Parameters:
    -----------
    rot_filtered : Stream
        Filtered rotation stream
    acc_filtered : Stream
        Filtered acceleration stream
    wave_type : str
        Wave type: "rayleigh" or "love"
    baz_center : float
        Center backazimuth (event backazimuth)
    baz_range : float
        Search range in degrees (±baz_range around baz_center)
    baz_step : float
        Step size for backazimuth search in degrees
        
    Returns:
    --------
    optimized_baz : float
        Optimized backazimuth that maximizes correlation
    max_cc : float
        Maximum correlation coefficient
    """
    # Generate backazimuth search range
    baz_min = baz_center - baz_range
    baz_max = baz_center + baz_range
    baz_test = np.arange(baz_min, baz_max + baz_step, baz_step)
    
    max_cc = -np.inf
    optimized_baz = baz_center
    
    wave_type = wave_type.lower()
    
    for test_baz in baz_test:
        try:
            if wave_type == "rayleigh":
                # Rayleigh: correlate vertical acceleration with transverse rotation
                acc_z = acc_filtered.select(channel="*Z")[0].data
                rot_r, rot_t = rotate_ne_rt(
                    rot_filtered.select(channel="*N")[0].data,
                    rot_filtered.select(channel="*E")[0].data,
                    test_baz
                )
                
                # Compute correlation
                ccorr = correlate(
                    acc_z,
                    rot_t,
                    0,  # max shift
                    demean=True,
                    normalize='naive',
                    method='auto'
                )
                
                # Get maximum correlation
                xshift, cc_max = xcorr_max(ccorr, abs_max=False)
                
            elif wave_type == "love":
                # Love: correlate vertical rotation with transverse acceleration
                rot_z = rot_filtered.select(channel="*Z")[0].data
                acc_r, acc_t = rotate_ne_rt(
                    acc_filtered.select(channel="*N")[0].data,
                    acc_filtered.select(channel="*E")[0].data,
                    test_baz
                )
                
                # Compute correlation
                ccorr = correlate(
                    rot_z,
                    acc_t,
                    0,  # max shift
                    demean=True,
                    normalize='naive',
                    method='auto'
                )
                
                # Get maximum correlation
                xshift, cc_max = xcorr_max(ccorr, abs_max=False)
            else:
                continue
            
            # Update if this is the best correlation
            if cc_max > max_cc:
                max_cc = cc_max
                optimized_baz = test_baz
                
        except Exception:
            # Skip this backazimuth if there's an error
            continue
    
    return optimized_baz, max_cc


def plot_trace_dispersion(
    sd_object: Optional['sixdegrees'] = None,
    rot: Optional[Stream] = None,
    acc: Optional[Stream] = None,
    wave_type: str = "rayleigh",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    fraction_of_octave: int = 1,
    frequency_bands: Optional[List[Tuple[float, float]]] = None,
    baz: Optional[float] = None,
    unitscale: str = "nano",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    raw: bool = True,
    output: bool = False,
    optimized: bool = True,
    baz_range: float = 20.0,
    baz_step: float = 1.0,
    regression_method: str = "odr",
    zero_intercept: bool = True,
    data_type: str = "acceleration",
    bootstrap: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot filtered traces in frequency bands for Rayleigh or Love waves.
    
    Creates subpanels showing filtered traces for each frequency band, with velocity
    (black) on the left y-axis and rotation/angle (red) on the right y-axis.
    Both axes are aligned at zero.
    
    Parameters:
    -----------
    sd_object : sixdegrees, optional
        sixdegrees object. If provided, will extract rot and acc from sd_object.get_stream(),
        and extract baz, fmin, fmax from the object if not explicitly provided.
    rot : Stream, optional
        Rotation rate/rotation stream. Required if sd_object is not provided.
    acc : Stream, optional
        Acceleration/velocity stream. Required if sd_object is not provided.
    data_type : str
        Type of data: "acceleration" (rotation rate and acceleration) or "velocity" (rotation and velocity).
        Default is "acceleration". This determines units and labels.
    wave_type : str
        Wave type: "rayleigh" or "love"
    fmin : float, optional
        Minimum frequency for octave band generation. Required if frequency_bands is not provided.
    fmax : float, optional
        Maximum frequency for octave band generation. Required if frequency_bands is not provided.
    fraction_of_octave : int, optional
        Fraction of octave for octave band generation. If not provided, uses 1 (octaves).
    frequency_bands : list of tuples, optional
        List of (fmin, fmax) tuples for frequency bands. If provided, overrides fmin/fmax.
    baz : float, optional
        Backazimuth. If not provided and sd_object is given, will try to extract from sd_object.
        If optimized=True, this is used as the center for the search range.
    unitscale : str
        Unit scale: "nano" or "micro"
    figsize : tuple, optional
        Figure size (width, height). If None, uses default based on number of bands.
    title : str, optional
        Custom title for the plot. If None, generates automatic title.
    raw : bool
        If True and sd_object is provided, uses raw (unfiltered) stream data.
    output : bool
        If True, returns a dictionary with velocities, frequencies, and backazimuths.
    optimized : bool
        If True, optimize backazimuth for each frequency band within ±baz_range degrees.
    baz_range : float
        Search range in degrees for backazimuth optimization (±baz_range around baz).
    baz_step : float
        Step size in degrees for backazimuth optimization search.
    regression_method : str
        Method to use for regression: "odr", "ransac", or "theilsen".
    zero_intercept : bool
        Force intercept to be zero if True.
    bootstrap : dict, optional
        Bootstrap options dictionary for regression. If None, bootstrap is disabled.
        Valid keys:
        - 'n_iterations': int, number of bootstrap iterations (default: 1000)
        - 'stat': str, statistic to use ('mean' or 'median', default: 'mean')
        - 'random_seed': int, random seed for reproducibility (default: 42)
        - 'sample_fraction': float, fraction of data to use in each bootstrap iteration (0.0 to 1.0, default: 0.8)
        Example: bootstrap={'n_iterations': 2000, 'stat': 'median', 'random_seed': 123, 'sample_fraction': 0.8}
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    # Validate wave_type
    wave_type = wave_type.lower()
    if wave_type not in ["rayleigh", "love"]:
        raise ValueError(f"wave_type must be 'rayleigh' or 'love', got '{wave_type}'")
    
    # Extract streams and parameters from sd_object if provided
    if sd_object is not None:
        # Extract streams if not provided
        if rot is None:
            rot = sd_object.get_stream("rotation", raw=raw).copy()
        if acc is None:
            acc = sd_object.get_stream("translation", raw=raw).copy()
        
        # Extract parameters if not explicitly provided
        if fmin is None and hasattr(sd_object, 'fmin') and sd_object.fmin is not None:
            fmin = sd_object.fmin
        if fmax is None and hasattr(sd_object, 'fmax') and sd_object.fmax is not None:
            fmax = sd_object.fmax
        
        # Extract baz if not provided
        if baz is None:
            # Try theoretical baz first
            if hasattr(sd_object, 'baz_theo') and sd_object.baz_theo is not None:
                baz = sd_object.baz_theo
            # Try theoretical_baz attribute
            elif hasattr(sd_object, 'theoretical_baz') and sd_object.theoretical_baz is not None:
                baz = sd_object.theoretical_baz
            # Try baz_estimated (may be a dict with wave_type keys)
            elif hasattr(sd_object, 'baz_estimated') and sd_object.baz_estimated is not None:
                if isinstance(sd_object.baz_estimated, dict):
                    baz_val = sd_object.baz_estimated.get(wave_type.lower(), None)
                    if baz_val is None:
                        baz_val = next(iter(sd_object.baz_estimated.values()), None)
                    baz = baz_val
                else:
                    baz = sd_object.baz_estimated
            # Try event_info
            elif hasattr(sd_object, 'event_info') and sd_object.event_info is not None:
                if isinstance(sd_object.event_info, dict) and 'backazimuth' in sd_object.event_info:
                    baz = sd_object.event_info['backazimuth']
    
    # Validate that we have required streams
    if rot is None or acc is None:
        raise ValueError("Either provide rot and acc directly, or provide sd_object (sixdegrees object)")
    
    # Validate that we have baz
    if baz is None:
        raise ValueError("baz (backazimuth) must be provided either directly or extractable from sd_object")
    
    # Determine frequency bands
    if frequency_bands is None:
        if fmin is None or fmax is None:
            raise ValueError("Either provide frequency_bands or both fmin and fmax")
        
        # Generate octave bands using sd_object method if available
        if sd_object is not None and hasattr(sd_object, 'get_octave_bands'):
            f_lower, f_upper, f_center = sd_object.get_octave_bands(fmin=fmin, fmax=fmax, faction_of_octave=fraction_of_octave)
            frequency_bands = [(fl, fu) for fl, fu in zip(f_lower, f_upper)]
        else:
            # Simple octave band generation if sd_object method not available
            frequency_bands = _generate_octave_bands(fmin, fmax)
    else:
        # Validate frequency_bands format
        if not isinstance(frequency_bands, list) or len(frequency_bands) == 0:
            raise ValueError("frequency_bands must be a non-empty list of (fmin, fmax) tuples")
        for band in frequency_bands:
            if not isinstance(band, (tuple, list)) or len(band) != 2:
                raise ValueError("Each frequency band must be a tuple/list of (fmin, fmax)")
    
    n_bands = len(frequency_bands)
    
    # Define scaling factors based on data_type
    mu = r"$\mu$"
    if data_type.lower() == "velocity":
        # Velocity mode: rotation (rad) and velocity (m/s)
        if unitscale == "nano":
            acc_scaling, acc_unit = 1e6, f"{mu}m/s"
            rot_scaling, rot_unit = 1e9, f"nrad"
        elif unitscale == "micro":
            acc_scaling, acc_unit = 1e3, f"mm/s"
            rot_scaling, rot_unit = 1e6, f"{mu}rad"
        else:
            raise ValueError(f"Invalid unitscale: {unitscale}. Valid options are: 'nano', 'micro'")
        tra_label_prefix = "v"
        rot_label_prefix = "r"
    else:
        # Acceleration mode (default): rotation rate (rad/s) and acceleration (m/s²)
        if unitscale == "nano":
            acc_scaling, acc_unit = 1e6, f"{mu}m/s$^2$"
            rot_scaling, rot_unit = 1e9, f"nrad/s"
        elif unitscale == "micro":
            acc_scaling, acc_unit = 1e3, f"mm/s$^2$"
            rot_scaling, rot_unit = 1e6, f"{mu}rad/s"
        else:
            raise ValueError(f"Invalid unitscale: {unitscale}. Valid options are: 'nano', 'micro'")
        tra_label_prefix = "a"
        rot_label_prefix = r"$\dot{r}$"
    
    # Set figure size
    if figsize is None:
        figsize = (15, 2 * n_bands)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]
    
    plt.subplots_adjust(hspace=0.0)
    
    # Plot settings
    font = 12
    lw = 0.8
    
    # Get sampling rate
    dt = rot[0].stats.delta
    times = rot[0].times()
    
    out = {}
    out['velocities'] = np.ones(len(frequency_bands))*np.nan
    out['frequencies'] = np.ones(len(frequency_bands))*np.nan
    out['backazimuths'] = np.ones(len(frequency_bands))*np.nan

    # Process each frequency band
    for i, (fl, fu) in enumerate(frequency_bands):
        ax = axes[i]
        
        # Remove bottom and top spines
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Filter streams for this frequency band
        rot_filtered = rot.copy()
        acc_filtered = acc.copy()
        
        # Detrend and taper before filtering
        rot_filtered = rot_filtered.detrend('linear')
        rot_filtered = rot_filtered.detrend('demean')
        rot_filtered = rot_filtered.taper(0.1, type='cosine')

        acc_filtered = acc_filtered.detrend('linear')
        acc_filtered = acc_filtered.detrend('demean')
        acc_filtered = acc_filtered.taper(0.1, type='cosine')
        
        # Apply bandpass filter
        rot_filtered = rot_filtered.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
        acc_filtered = acc_filtered.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
        
        # Detrend and taper after filtering
        rot_filtered = rot_filtered.detrend('demean')
        rot_filtered = rot_filtered.taper(0.1, type='cosine')

        acc_filtered = acc_filtered.detrend('demean')
        acc_filtered = acc_filtered.taper(0.1, type='cosine')
        

        # Optimize backazimuth if requested
        if optimized:
            optimized_baz, max_cc = _optimize_backazimuth(
                rot_filtered, acc_filtered, wave_type, baz, baz_range, baz_step
            )
            baz_used = optimized_baz
        else:
            baz_used = baz
        
        # Store backazimuth in output
        out['backazimuths'][i] = baz_used
        
        # Get components based on wave type
        if wave_type == "rayleigh":
            # Rayleigh: vertical velocity (acc_z) vs horizontal rotation (rot_t)
            acc_z = acc_filtered.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(
                rot_filtered.select(channel="*N")[0].data,
                rot_filtered.select(channel="*E")[0].data,
                baz_used
            )
            
            # Apply scaling
            acc_z_scaled = acc_z * acc_scaling
            rot_t_scaled = rot_t * rot_scaling
            
            # Linear regression with only slope: tra = slope * rot
            # Use regression function for slope estimation
            reg_result = regression(rot_t_scaled, acc_z_scaled, method=regression_method, 
                                   zero_intercept=zero_intercept, verbose=False, bootstrap=bootstrap)
            slope = reg_result['slope']
            
            out['velocities'][i] = slope*1e3
            out['frequencies'][i] = np.sqrt(2)*fu

            # Scale rotation data by the slope
            rot_t_scaled = rot_t_scaled * slope
            
            # Plot velocity/acceleration (black) on left axis
            ax.plot(times, acc_z_scaled, color="black", lw=lw, label=f"{tra_label_prefix}_Z")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_t_scaled, color="red", lw=lw, label=f"{rot_label_prefix}_H")
                        
            # Remove bottom and top spines
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            # Get max values for symmetric ylim
            acc_max = max([abs(np.min(acc_z_scaled)), abs(np.max(acc_z_scaled))])
            rot_max = max([abs(np.min(rot_t_scaled)), abs(np.max(rot_t_scaled))])
            
            # Set ylims symmetric around zero (ensures zero alignment)
            if acc_max > 0:
                ax.set_ylim(-acc_max * 1.05, acc_max * 1.05)
            else:
                ax.set_ylim(-1, 1)  # Default range if all zeros
            if rot_max > 0:
                ax2.set_ylim(-rot_max * 1.05, rot_max * 1.05)
            else:
                ax2.set_ylim(-1, 1)  # Default range if all zeros
            
            # Remove y-axis labels and ticks
            ax.set_ylabel("")
            ax2.set_ylabel("")
            ax.set_yticks([])
            ax2.set_yticks([])
            
            # Show scale factor and backazimuth as text
            text_y_pos = 0.8
            scale_unit = "m/s" if data_type.lower() == "velocity" else "m/s²"
            # Format scale with bootstrap uncertainty if available
            if bootstrap is not None:
                uncertainty = reg_result.get('slope_dev', reg_result.get('slope_dev', None))
                ax.text(0.02, text_y_pos, f"scale: {slope*1e3:.0f}±{uncertainty*1e3:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            else:
                ax.text(0.02, text_y_pos, f"scale: {slope*1e3:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            if optimized:
                ax.text(0.02, abs(1-text_y_pos), f"baz: {baz_used:.1f}°", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            
        elif wave_type == "love":
            # Love: vertical rotation (rot_z) vs horizontal velocity (acc_t)
            rot_z = rot_filtered.select(channel="*Z")[0].data
            acc_r, acc_t = rotate_ne_rt(
                acc_filtered.select(channel="*N")[0].data,
                acc_filtered.select(channel="*E")[0].data,
                baz_used
            )
            
            # Apply scaling
            rot_z_scaled = 2*rot_z * rot_scaling
            acc_t_scaled = acc_t * acc_scaling
            
            # Linear regression with only slope: tra = slope * rot
            # Use regression function for slope estimation
            reg_result = regression(rot_z_scaled, acc_t_scaled, method=regression_method, 
                                   zero_intercept=zero_intercept, verbose=False, bootstrap=bootstrap)
            slope = reg_result['slope']

            out['velocities'][i] = slope*1e3
            out['frequencies'][i] = np.sqrt(2)*fu
            
            # Scale rotation data by the slope
            rot_z_scaled = rot_z_scaled * slope
            
            # Plot velocity/acceleration (black) on left axis
            ax.plot(times, acc_t_scaled, color="black", lw=lw, label=f"{tra_label_prefix}_H")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_z_scaled, color="red", lw=lw, label=f"{rot_label_prefix}_Z")
            
            # Remove bottom and top spines
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            
            # Get max values for symmetric ylim
            acc_max = max([abs(np.min(acc_t_scaled)), abs(np.max(acc_t_scaled))])
            rot_max = max([abs(np.min(rot_z_scaled)), abs(np.max(rot_z_scaled))])
            
            # Set ylims symmetric around zero (ensures zero alignment)
            if acc_max > 0:
                ax.set_ylim(-acc_max * 1.05, acc_max * 1.05)
            else:
                ax.set_ylim(-1, 1)  # Default range if all zeros
            if rot_max > 0:
                ax2.set_ylim(-rot_max * 1.05, rot_max * 1.05)
            else:
                ax2.set_ylim(-1, 1)  # Default range if all zeros
            
            # Remove y-axis labels and ticks
            ax.set_ylabel("")
            ax2.set_ylabel("")
            ax.set_yticks([])
            ax2.set_yticks([])
            
            # Show scale factor and backazimuth as text
            text_y_pos = 0.8
            scale_unit = "m/s" if data_type.lower() == "velocity" else "m/s²"
            # Format scale with bootstrap uncertainty if available
            if bootstrap is not None:
                uncertainty = reg_result.get('slope_dev', None)
                ax.text(0.02, text_y_pos, f"scale: {slope*1e3:.0f}±{uncertainty*1e3:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            else:
                ax.text(0.02, text_y_pos, f"scale: {slope*1e3:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            if optimized:
                ax.text(0.02, abs(1-text_y_pos), f"baz: {baz_used:.1f}°", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
        
        # Set x-axis label only on bottom subplot
        if i == n_bands - 1:
            ax.set_xlabel("Time (s)", fontsize=font)
        
        # Add frequency band label on the right
        # Format frequency based on magnitude
        if fl < 0.1:
            fl_str = f"{fl:.3f}"
        elif fl < 1:
            fl_str = f"{fl:.2f}"
        else:
            fl_str = f"{fl:.1f}"
        
        if fu < 0.1:
            fu_str = f"{fu:.3f}"
        elif fu < 1:
            fu_str = f"{fu:.2f}"
        else:
            fu_str = f"{fu:.1f}"
        
        ax.text(0.99, 0.8, f"{fl_str}-{fu_str} Hz", 
                transform=ax.transAxes, fontsize=font-2,
                rotation=0, va='center', ha='right')
        
        # Add grid
        # ax.grid(True, which='both', ls='--', alpha=0.3)
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.spines['bottom'].set_visible(True)

    # Set title
    if title is None:
        # Generate automatic title
        start_time = acc[0].stats.starttime
        title = f"{wave_type.capitalize()} waves"
        if baz is not None:
            title += f" | BAz = {baz:.1f}°"
        if optimized:
            title += " (optimized)"
        title += f" | {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    fig.suptitle(title, fontsize=font+2, y=0.93)
    
    if output:
        return fig, out
    else:
        return fig


