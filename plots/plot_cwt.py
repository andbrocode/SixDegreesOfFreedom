"""
Functions for plotting continuous wavelet transform analysis for a single component.
"""
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from obspy import Stream

def plot_cwt(sd, st: Stream, cwt_output: Dict, clog: bool=False, 
             unitscale: str='nano') -> plt.Figure:
    """
    Plot continuous wavelet transform analysis for a single component.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    st : obspy.Stream
        Input stream containing the component to analyze
    cwt_output : Dict
        Dictionary containing CWT results with keys:
            - time: Time points
            - period: Period values
            - power: Power spectrum
            - coi: Cone of influence
    clog : bool
        Use logarithmic colorscale if True
    unitscale : str
        Unit scale for rotation rate ('nano' or 'micro')
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing time series and CWT plots
    """
    # Set scale factor based on unit
    if unitscale == 'nano':
        scale = 1e9
        unit = 'nrad/s'
    elif unitscale == 'micro':
        scale = 1e6
        unit = '$\mu$rad/s'
    else:
        scale = 1
        unit = 'rad/s'
        print(f"-> warning: unknown unit scale ({unitscale}). Using rad/s")
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2], 
                 width_ratios=[1, 0.05], hspace=0.1)
    
    # Create time series subplot
    ax_ts = fig.add_subplot(gs[0, 0])
    
    # Plot waveform
    tr = st[0]
    t = np.arange(len(tr.data)) * tr.stats.delta
    ax_ts.plot(t, tr.data * scale, 'k-', alpha=0.8)
    
    ax_ts.set_xlim(min(t), max(t))
    ax_ts.set_xticklabels([])
    ax_ts.set_ylabel(f'Amplitude ({unit})')
    ax_ts.grid(True, alpha=0.3)
    
    # Create CWT subplot
    ax_cwt = fig.add_subplot(gs[1, 0])
    
    # Get CWT data
    times = cwt_output['time']
    periods = cwt_output['period']
    power = cwt_output['power']
    coi = cwt_output['coi']
    
    # Plot CWT power
    if clog:
        power_plot = np.log2(power)
        cmap_label = 'log2(Power)'
    else:
        power_plot = power
        cmap_label = 'Power'
    
    mesh = ax_cwt.pcolormesh(times, periods, power_plot, 
                            cmap='viridis', shading='auto')
    
    # Plot COI
    ax_cwt.fill_between(times, coi, periods[-1], color='white', alpha=0.5)
    ax_cwt.plot(times, coi, 'w--', alpha=0.5)
    
    # Configure CWT subplot
    ax_cwt.set_yscale('log')
    ax_cwt.set_ylabel('Period (s)')
    ax_cwt.set_xlabel('Time (s)')
    
    # Add colorbar
    cax = fig.add_subplot(gs[1, 1])
    plt.colorbar(mesh, cax=cax, label=cmap_label)
    
    plt.tight_layout()
    return fig