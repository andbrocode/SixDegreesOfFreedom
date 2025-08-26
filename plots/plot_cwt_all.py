"""
Functions for plotting continuous wavelet transform analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union
from matplotlib.gridspec import GridSpec
from obspy import Stream

def plot_cwt_all(rot: Stream, acc: Stream, cwt_output: Dict, clog: bool=False, 
                ylim: Union[float, None]=None) -> plt.Figure:
    """
    Plot continuous wavelet transform analysis for all components of rotation and translation
    
    Parameters:
    -----------
    rot : Stream
        Rotation rate stream
    acc : Stream
        Acceleration stream
    cwt_output : Dict
        Dictionary containing CWT results for each component
    clog : bool
        Use logarithmic colorscale if True
    ylim : float or None
        Upper frequency limit for plotting
        
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
    rot_scale = 1e6
    acc_scale = 1e3

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
    for i, (comp, data_type) in enumerate(components):
        wave_ax = fig.add_subplot(gs[2*i])
        cwt_ax = fig.add_subplot(gs[2*i+1])
        
        # Get data and scale
        if data_type == 'Rotation':
            tr = rot.select(channel=f"*{comp}")[0]
            data = tr.data * rot_scale
            unit = r"$\mu$rad"
            label = f"$\Omega_{comp[-1]}$"
        else:
            tr = acc.select(channel=f"*{comp}")[0]
            data = tr.data * acc_scale
            unit = r"mm/s$^2$"
            label = f"$a_{comp[-1]}$"
        
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
        key = f"{comp}"
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

