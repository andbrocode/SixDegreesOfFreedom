"""
Functions for plotting filtered traces in frequency bands for Rayleigh or Love waves.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Tuple
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from obspy import Stream
from obspy.signal.rotate import rotate_ne_rt


def plot_filtered_traces_frequency_bands(
    sd_object: Optional['sixdegrees'] = None,
    rot: Optional[Stream] = None,
    acc: Optional[Stream] = None,
    wave_type: str = "rayleigh",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    frequency_bands: Optional[List[Tuple[float, float]]] = None,
    baz: Optional[float] = None,
    unitscale: str = "nano",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    raw: bool = True
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
        Rotation rate stream. Required if sd_object is not provided.
    acc : Stream, optional
        Acceleration stream. Required if sd_object is not provided.
    wave_type : str
        Wave type: "rayleigh" or "love"
    fmin : float, optional
        Minimum frequency for octave band generation. Required if frequency_bands is not provided.
    fmax : float, optional
        Maximum frequency for octave band generation. Required if frequency_bands is not provided.
    frequency_bands : list of tuples, optional
        List of (fmin, fmax) tuples for frequency bands. If provided, overrides fmin/fmax.
    baz : float, optional
        Backazimuth. If not provided and sd_object is given, will try to extract from sd_object.
    unitscale : str
        Unit scale: "nano" or "micro"
    figsize : tuple, optional
        Figure size (width, height). If None, uses default based on number of bands.
    title : str, optional
        Custom title for the plot. If None, generates automatic title.
    raw : bool
        If True and sd_object is provided, uses raw (unfiltered) stream data.
        
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
            rot = sd_object.get_stream("rotation", raw=raw)
        if acc is None:
            acc = sd_object.get_stream("translation", raw=raw)
        
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
            f_lower, f_upper, f_center = sd_object.get_octave_bands(fmin=fmin, fmax=fmax, faction_of_octave=1)
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
    
    # Define scaling factors
    mu = r"$\mu$"
    if unitscale == "nano":
        acc_scaling, acc_unit = 1e6, f"{mu}m/s"
        rot_scaling, rot_unit = 1e9, f"nrad"
    elif unitscale == "micro":
        acc_scaling, acc_unit = 1e3, f"mm/s"
        rot_scaling, rot_unit = 1e6, f"{mu}rad"
    else:
        raise ValueError(f"Invalid unitscale: {unitscale}. Valid options are: 'nano', 'micro'")
    
    # Set figure size
    if figsize is None:
        figsize = (15, 2 * n_bands)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]
    
    plt.subplots_adjust(hspace=0.15)
    
    # Plot settings
    font = 12
    lw = 1.0
    
    # Get sampling rate
    dt = rot[0].stats.delta
    times = rot[0].times()
    
    # Process each frequency band
    for i, (fl, fu) in enumerate(frequency_bands):
        ax = axes[i]
        
        # Filter streams for this frequency band
        rot_filtered = rot.copy()
        acc_filtered = acc.copy()
        
        # Detrend and taper before filtering
        rot_filtered.detrend('linear')
        rot_filtered.detrend('demean')
        rot_filtered.taper(0.05, type='cosine')
        acc_filtered.detrend('linear')
        acc_filtered.detrend('demean')
        acc_filtered.taper(0.05, type='cosine')
        
        # Apply bandpass filter
        rot_filtered.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
        acc_filtered.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
        
        # Get components based on wave type
        if wave_type == "rayleigh":
            # Rayleigh: vertical velocity (acc_z) vs horizontal rotation (rot_t)
            acc_z = acc_filtered.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(
                rot_filtered.select(channel="*N")[0].data,
                rot_filtered.select(channel="*E")[0].data,
                baz
            )
            
            # Apply scaling
            acc_z_scaled = acc_z * acc_scaling
            rot_t_scaled = rot_t * rot_scaling
            
            # Plot velocity (black) on left axis
            ax.plot(times, acc_z_scaled, color="black", lw=lw, label=f"v_Z")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_t_scaled, color="red", lw=lw, label=f"r_H")
            
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
            
            # Add zero line for clarity
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
            
            # Set labels
            ax.set_ylabel(f"velocity [{acc_unit}]", fontsize=font, color="black")
            ax2.set_ylabel(f"angle [{rot_unit}]", fontsize=font, color="red")
            
        elif wave_type == "love":
            # Love: vertical rotation (rot_z) vs horizontal velocity (acc_t)
            rot_z = rot_filtered.select(channel="*Z")[0].data
            acc_r, acc_t = rotate_ne_rt(
                acc_filtered.select(channel="*N")[0].data,
                acc_filtered.select(channel="*E")[0].data,
                baz
            )
            
            # Apply scaling
            rot_z_scaled = rot_z * rot_scaling
            acc_t_scaled = acc_t * acc_scaling
            
            # Plot velocity (black) on left axis
            ax.plot(times, acc_t_scaled, color="black", lw=lw, label=f"v_H")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_z_scaled, color="red", lw=lw, label=f"r_Z")
            
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
            
            # Add zero line for clarity
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
            
            # Set labels
            ax.set_ylabel(f"velocity [{acc_unit}]", fontsize=font, color="black")
            ax2.set_ylabel(f"angle [{rot_unit}]", fontsize=font, color="red")
        
        # Set x-axis label only on bottom subplot
        if i == n_bands - 1:
            ax.set_xlabel("Time [s]", fontsize=font)
        
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
        
        ax.text(1.02, 0.5, f"{fl_str}-{fu_str} Hz", 
                transform=ax.transAxes, fontsize=font-2,
                rotation=0, va='center', ha='left')
        
        # Set tick colors
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add grid
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Set title
    if title is None:
        # Generate automatic title
        start_time = acc[0].stats.starttime
        title = f"{wave_type.capitalize()} waves"
        if baz is not None:
            title += f" | BAZ = {baz:.1f}°"
        title += f" | {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    fig.suptitle(title, fontsize=font+2, y=0.995)
    
    return fig


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
