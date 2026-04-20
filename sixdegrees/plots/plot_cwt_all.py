"""
Functions for plotting continuous wavelet transform analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional
from matplotlib.gridspec import GridSpec
from obspy import Stream

def plot_cwt_all(rot: Optional[Stream]=None, acc: Optional[Stream]=None, sd_object: Optional['sixdegrees']=None,
                 cwt_output: Optional[Dict]=None, clog: bool=False, 
                fmin: Union[float, None]=None, fmax: Union[float, None]=None,
                ylim: Union[float, None]=None, data_type: str="acceleration") -> plt.Figure:
    """
    Plot continuous wavelet transform analysis for all components of rotation and translation
    
    Parameters:
    -----------
    rot : Stream, optional
        Rotation rate/rotation stream. Required if sd_object is not provided.
    acc : Stream, optional
        Acceleration/velocity stream. Required if sd_object is not provided.
    sd_object : sixdegrees, optional
        sixdegrees object. If provided, will extract rot and acc from sd_object.get_stream(),
        and extract fmin, fmax from the object if not explicitly provided.
    cwt_output : Dict, optional
        Dictionary containing CWT results for each component. Required if not provided.
    clog : bool
        Use logarithmic colorscale if True
    fmin : float or None
        Minimum frequency limit for y-axis (Hz). If None, uses minimum frequency from data.
    fmax : float or None
        Maximum frequency limit for y-axis (Hz). If None, uses maximum frequency from data.
    ylim : float or None, deprecated
        Upper frequency limit for plotting (deprecated, use fmax instead).
        If provided, will override fmax for backward compatibility.
    data_type : str
        Type of data: "acceleration" (rotation rate and acceleration) or "velocity" (rotation and velocity).
        Default is "acceleration". This determines units and labels.
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
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
    if cwt_output is None:
        raise ValueError("cwt_output dictionary is required")
    
    # Plot settings
    tscale = 1
    font = 12
    cmap = plt.get_cmap("viridis")
    
    # Determine scaling and units based on data_type
    if data_type.lower() == "velocity":
        # Velocity mode: rotation (rad) and velocity (m/s)
        rot_scale = 1e6  # micro-radians
        acc_scale = 1e3  # mm/s
        rot_unit_base = r"$\mu$rad"
        tra_unit_base = r"mm/s"
        rot_label_base = r"$\Omega"
        tra_label_base = r"$v"
    else:
        # Acceleration mode (default): rotation rate (rad/s) and acceleration (m/s²)
        rot_scale = 1e6  # micro-rad/s
        acc_scale = 1e3  # mm/s²
        rot_unit_base = r"$\mu$rad/s"
        tra_unit_base = r"mm/s$^2$"
        rot_label_base = r"$\dot{\Omega}"
        tra_label_base = r"$a"

    # Count total components and calculate needed subplots
    n_panels = len(cwt_output.keys())
    n_components = len(rot) + len(acc)
    
    # Create figure with GridSpec
    # Each component needs 2 rows - one for waveform and one for CWT
    fig = plt.figure(figsize=(15, 4*n_panels))
    gs = GridSpec(2*n_panels, 1, figure=fig, height_ratios=[1, 3]*n_panels, hspace=0.3)

    # Component mapping
    components = []
    for tr in rot:
        components.append((tr.stats.channel, 'Rotation'))
    for tr in acc:
        components.append((tr.stats.channel, 'Translation'))
    
    # Set colormap limits
    if clog:
        vmin, vmax, norm = 0.01, 1, "log"
    else:
        vmin, vmax, norm = 0.0, 0.9, None
        
    # Plot each component
    for i, (comp, data_type_comp) in enumerate(components):
        wave_ax = fig.add_subplot(gs[2*i])
        cwt_ax = fig.add_subplot(gs[2*i+1])
        
        # Get data and scale
        if data_type_comp == 'Rotation':
            tr = rot.select(channel=f"*{comp}")[0]
            data = tr.data * rot_scale
            unit = rot_unit_base
            label = f"{rot_label_base}_{comp[-1]}$"
        else:
            tr = acc.select(channel=f"*{comp}")[0]
            data = tr.data * acc_scale
            unit = tra_unit_base
            label = f"{tra_label_base}_{comp[-1]}$"
        
        # Get times from the current trace instead of rotation stream
        times = tr.times() * tscale
        
        # Plot waveform
        wave_ax.plot(times, data, color="k", label=label, lw=1)
        wave_ax.set_xlim(min(times), max(times))
        wave_ax.legend(loc=1)
        wave_ax.set_xticklabels([])
        wave_ax.set_ylabel(f"{label}\n({unit})", fontsize=font)
        wave_ax.grid(True, alpha=0.3)
        
        # Plot CWT
        # Construct key matching the format used in CWT computation: "{component}_{data_type_comp}"
        # e.g., "Z_Rotation", "N_Translation", etc.
        component_letter = comp[-1]  # Get last character (Z, N, E)
        key = f"{component_letter}_{data_type_comp}"
        
        # Check if key exists, if not try alternative formats
        if key not in cwt_output:
            # Try with full channel name
            if comp in cwt_output:
                key = comp
            else:
                # Try with just the component letter
                if component_letter in cwt_output:
                    key = component_letter
                else:
                    raise KeyError(f"CWT output key '{key}' not found. Available keys: {list(cwt_output.keys())}")
        
        im = cwt_ax.pcolormesh(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['frequencies'],
            cwt_output[key]['cwt_power'],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            rasterized=True,
            # shading='nearest'
        )
        
        # Add cone of influence
        cwt_ax.plot(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['cone'],
            color="white",
            ls="--",
            alpha=0.7
        )
        cwt_ax.fill_between(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['cone'],
            min(cwt_output[key]['frequencies']) * np.ones(len(cwt_output[key]['cone'])),
            color="white",
            alpha=0.2,
            rasterized=True
        )
        
        # Set frequency limits
        freq_min = min(cwt_output[key]['frequencies'])
        freq_max = max(cwt_output[key]['frequencies'])
        
        # Handle backward compatibility: ylim overrides fmax if provided
        if ylim is not None:
            fmax_effective = ylim
        else:
            fmax_effective = fmax
        
        # Set y-axis limits
        ylim_min = fmin if fmin is not None else freq_min
        ylim_max = fmax_effective if fmax_effective is not None else freq_max
        
        cwt_ax.set_ylim(ylim_min, ylim_max)
        
        cwt_ax.set_yscale('log')
        cwt_ax.set_ylabel("Frequency (Hz)", fontsize=font)
        cwt_ax.grid(True, alpha=0.3)
        
        # Only add xlabel to bottom subplot
        if i == len(components) - 1:
            cwt_ax.set_xlabel(f"Time (s) from {rot[0].stats.starttime.date} "
                            f"{str(rot[0].stats.starttime.time).split('.')[0]} UTC",
                            fontsize=font)
        
        # Add subplot labels
        wave_ax.text(.005, .97, f"({chr(97+i*2)})", ha='left', va='top',
                        transform=wave_ax.transAxes, fontsize=font+2)
        cwt_ax.text(.005, .97, f"({chr(98+i*2)})", ha='left', va='top',
                    transform=cwt_ax.transAxes, fontsize=font+2, color="w")
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Normalized CWT Power", fontsize=font)
    
    plt.subplots_adjust(right=0.9)
    return fig

