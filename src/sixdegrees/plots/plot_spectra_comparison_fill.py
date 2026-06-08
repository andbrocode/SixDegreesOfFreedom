"""
Plot power spectral density comparison between rotation and acceleration data.
"""

from typing import Union, Tuple, Optional
import numpy as np
from numpy import ndarray, reshape
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from obspy import Stream
import multitaper as mt


def plot_spectra_comparison_fill(rot: Optional[Stream]=None, acc: Optional[Stream]=None, sd_object: Optional['sixdegrees']=None,
                                 fmin: Union[float, None]=None, fmax: Union[float, None]=None, 
                                 ylog: bool=False, xlog: bool=False, fill: bool=False, data_type: str="acceleration") -> Figure:
    """
    Plot power spectral density comparison between rotation and acceleration/velocity data with horizontal layout
    
    Parameters:
    -----------
    rot : Stream, optional
        Rotation rate/rotation stream. Required if sd_object is not provided.
    acc : Stream, optional
        Acceleration/velocity stream. Required if sd_object is not provided.
    sd_object : sixdegrees, optional
        sixdegrees object. If provided, will extract rot and acc from sd_object.get_stream(),
        and extract fmin, fmax from the object if not explicitly provided.
    fmin : float or None, optional
        Minimum frequency for bandpass filter. If not provided and sd_object is given, will use sd_object.fmin.
    fmax : float or None, optional
        Maximum frequency for bandpass filter. If not provided and sd_object is given, will use sd_object.fmax.
    ylog : bool
        Use logarithmic y-axis scale if True
    xlog : bool
        Use logarithmic x-axis scale if True
    fill : bool
        Fill the area under curves if True
    data_type : str
        Type of data: "acceleration" (rotation rate and acceleration) or "velocity" (rotation and velocity).
        Default is "acceleration". This determines units and labels.
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    
    # Extract streams and parameters from sd_object if provided
    if sd_object is not None:
        # Extract streams if not provided
        if rot is None:
            rot = sd_object.get_stream("rotation")
        if acc is None:
            acc = sd_object.get_stream("translation")
        
        # Extract parameters if not explicitly provided (provided parameters have higher priority)
        if fmin is None and hasattr(sd_object, 'fmin') and sd_object.fmin is not None:
            fmin = sd_object.fmin
        if fmax is None and hasattr(sd_object, 'fmax') and sd_object.fmax is not None:
            fmax = sd_object.fmax
    
    # Validate that we have required streams
    if rot is None or acc is None:
        raise ValueError("Either provide rot and acc directly, or provide sd_object (sixdegrees object)")
    
    # Determine units and labels based on data_type
    if data_type.lower() == "velocity":
        # Velocity mode: rotation (rad) and velocity (m/s)
        rot_unit_label = r"rad$^2$/s$^2$/Hz"  # PSD of rotation
        tra_unit_label = r"m$^2$/s$^2$/Hz"  # PSD of velocity
    else:
        # Acceleration mode (default): rotation rate (rad/s) and acceleration (m/s²)
        rot_unit_label = r"rad$^2$/s$^2$/Hz"  # PSD of rotation rate
        tra_unit_label = r"m$^2$/s$^4$/Hz"  # PSD of acceleration
    
    def _multitaper_psd(arr: ndarray, dt: float, n_win: int=5, time_bandwidth: float=4.0) -> Tuple[ndarray, ndarray]:
        """Calculate multitaper power spectral density"""
        out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=2)
        _f, _psd = out_psd.rspec()
        return reshape(_f, _f.size), reshape(_psd, _psd.size)

    # Calculate PSDs for each component
    Tsec = 5
    components = [
        ('Z', '*Z'), ('N', '*N'), ('E', '*E')
    ]
    psds = {}
    for comp_name, comp_pattern in components:
        f1, psd1 = _multitaper_psd(
            rot.select(channel=comp_pattern)[0].data, 
            rot[0].stats.delta,
            n_win=Tsec
        )
        f2, psd2 = _multitaper_psd(
            acc.select(channel=comp_pattern)[0].data, 
            acc[0].stats.delta,
            n_win=Tsec
        )
        psds[comp_name] = {'rot': (f1, psd1), 'acc': (f2, psd2)}

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)

    # Plot settings
    font = 12
    lw = 1
    rot_color = "darkred"
    acc_color = "black"
    alpha = 0.5 if fill else 1.0

    # Add title with time information
    title = f"{rot[0].stats.starttime.date} {str(rot[0].stats.starttime.time).split('.')[0]} UTC"
    if fmin is not None and fmax is not None:
        title += f" | {fmin}-{fmax} Hz"
    fig.suptitle(title, fontsize=font+2, y=1.02)

    # Plot each component
    for i, (comp_name, comp_data) in enumerate(psds.items()):
        # Get component labels
        rot_label = f"{rot[0].stats.station}.{rot.select(channel=f'*{comp_name}')[0].stats.channel}"
        acc_label = f"{acc[0].stats.station}.{acc.select(channel=f'*{comp_name}')[0].stats.channel}"
        
        if fill:
            # Plot with fill
            axes[i].fill_between(
                comp_data['rot'][0],
                comp_data['rot'][1],
                lw=lw,
                label=rot_label,
                color=rot_color,
                alpha=alpha,
                zorder=3
            )
            ax2 = axes[i].twinx()
            ax2.fill_between(
                comp_data['acc'][0],
                comp_data['acc'][1],
                lw=lw,
                label=acc_label,
                color=acc_color,
                alpha=alpha,
                zorder=2
            )
        else:
            # Plot lines
            axes[i].plot(
                comp_data['rot'][0],
                comp_data['rot'][1],
                lw=lw,
                label=rot_label,
                color=rot_color,
                ls="-",
                zorder=3
            )
            ax2 = axes[i].twinx()
            ax2.plot(
                comp_data['acc'][0],
                comp_data['acc'][1],
                lw=lw,
                label=acc_label,
                color=acc_color,
                zorder=2
            )
        
        # Configure axes
        axes[i].legend(loc=1, ncols=4)
        if xlog:
            axes[i].set_xscale("log")
        if ylog:
            axes[i].set_yscale("log")
            ax2.set_yscale("log")
        
        # axes[i].grid(which="both", alpha=0.5)
        axes[i].tick_params(axis='y', colors=rot_color)
        axes[i].set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        # Set frequency limits
        xlim_right = fmax if fmax else rot[0].stats.sampling_rate * 0.5
        axes[i].set_xlim(left=fmin, right=xlim_right)
        ax2.set_xlim(left=fmin, right=xlim_right)
        axes[i].set_xlabel("Frequency (Hz)", fontsize=font)

        # Set legends
        ax2.legend(loc=2)

        # For the last panel (E component), don't create new y-axis ticks on the right
        if i == 2:
            ax2.set_ylabel(f"PSD ({tra_unit_label})", fontsize=font)
        if i == 0:
            axes[i].set_ylabel(f"PSD ({rot_unit_label})", fontsize=font, color=rot_color)
        
        # Add component label
        axes[i].set_title(f"Component {comp_name}", fontsize=font)

    # Adjust layout to accommodate supertitle
    plt.subplots_adjust(top=0.90)
    
    return fig
