"""
Functions for plotting continuous wavelet transform analysis for a single component.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union
from matplotlib.gridspec import GridSpec
from obspy import Stream

def plot_cwt(st: Stream, cwt_output: Dict, clog: bool=False, 
            ylim: Union[float, None]=None, scale: float=1e6) -> plt.Figure:
    """
    Plot continuous wavelet transform analysis for all components of rotation and translation
    
    Parameters:
    -----------
    st : Stream
        Stream of data to plot
    cwt_output : Dict
        Dictionary containing CWT results for each component
    clog : bool
        Use logarithmic colorscale if True
    ylim : float or None
        Upper frequency limit for plotting
    scale : float
        Scale factor for data
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Plot settings
    tscale = 1
    font = 12
    cmap = plt.get_cmap("viridis")

    # decide if rotation or translation data 
    if "J" in st[0].stats.channel:
        if scale == 1e9:
            unit = r"nrad"
        elif scale == 1e6:
            unit = r"$\mu$rad"
        elif scale == 1e3:
            unit = r"mrad"
        else:
            unit = r"rad"
            scale = 1
            print(f"WARNING: unknown scale factor (1e3, 1e6, 1e9): {scale}. Using 1 for scale")
    else:
        if scale == 1e9:
            unit = r"nm/s$^2$"
        elif scale == 1e6:
            unit = r"mm/s$^2$"
        elif scale == 1e3:
            unit = r"m/s$^2$"
        else:
            unit = r"m/s$^2$"
            print(f"WARNING: unknown scale factor (1e3, 1e6, 1e9): {scale}. Using 1 for scale")
            scale = 1


    # Create figure with GridSpec
    # Each component needs 2 rows - one for waveform and one for CWT
    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)

    # Set colormap limits
    if clog:
        vmin, vmax, norm = 0.01, 1, "log"
    else:
        vmin, vmax, norm = 0.0, 0.9, None
        
    # Plot each component
    wave_ax = fig.add_subplot(gs[0])
    cwt_ax = fig.add_subplot(gs[1])
        
    # Get data and scale
    tr = st.copy()[0]
    data = tr.data * scale
    label = fr"$\Omega_{tr.stats.channel[-1]}$"
    key = f"{tr.stats.channel}"
    
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
    im = cwt_ax.pcolormesh(
        cwt_output[key]['times'] * tscale,
        cwt_output[key]['frequencies'],
        cwt_output[key]['cwt_power'],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        rasterized=True
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
        alpha=0.2
    )
    
    # Set frequency limits
    if ylim is None:
        cwt_ax.set_ylim(min(cwt_output[key]['frequencies']),
                        max(cwt_output[key]['frequencies']))
    else:
        cwt_ax.set_ylim(min(cwt_output[key]['frequencies']), ylim)
    
    cwt_ax.set_yscale('log')
    cwt_ax.set_ylabel("Frequency (Hz)", fontsize=font)
    cwt_ax.grid(True, alpha=0.3)

    # Only add xlabel to bottom subplot
    cwt_ax.set_xlabel(
        f"Time (s) from {st[0].stats.starttime.date} "
        f"{str(st[0].stats.starttime.time).split('.')[0]} UTC",
        fontsize=font
    )
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Normalized CWT Power", fontsize=font)
    
    plt.subplots_adjust(right=0.9)
    return fig
