"""
Functions for plotting spectral comparison with fill between curves.
"""
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, AutoMinorLocator
from obspy import Stream
from multitaper import mtspec

def plot_spectra_comparison_fill(sd, rot: Stream, acc: Stream, fmin: Union[float, None]=None, 
                               fmax: Union[float, None]=None, ylog: bool=False,
                               xlog: bool=False, fill: bool=False) -> plt.Figure:
    """
    Plot spectral comparison with optional fill between curves.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    rot : obspy.Stream
        Rotation rate stream
    acc : obspy.Stream
        Acceleration stream
    fmin : float, optional
        Minimum frequency for plotting
    fmax : float, optional
        Maximum frequency for plotting
    ylog : bool
        Use logarithmic y-axis if True
    xlog : bool
        Use logarithmic x-axis if True
    fill : bool
        Fill area between curves if True
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing spectral comparison plots
    """
    def __sync_axes(ax1, ax2, ylog=False):
        """
        Synchronize grid lines between two axes by calculating appropriate scale ratios.
        
        Args:
            ax1: Primary axis (rotation)
            ax2: Secondary axis (acceleration)
            ylog: Whether using logarithmic scale
        """
        if ylog:
            # For log scale, calculate ratio in log space
            ax1_range = np.log10(ax1.get_ylim())
            ax2_range = np.log10(ax2.get_ylim())
            
            # Calculate scale ratio
            scale_ratio = (ax2_range[1] - ax2_range[0]) / (ax1_range[1] - ax1_range[0])
            
            # Set locators based on ratio
            ax1.yaxis.set_major_locator(LogLocator(numticks=8))
            ax2.yaxis.set_major_locator(LogLocator(numticks=int(8 * scale_ratio)))
            
            # Set minor locators
            ax1.yaxis.set_minor_locator(LogLocator(numticks=8, subs=(.2, .4, .6, .8)))
            ax2.yaxis.set_minor_locator(LogLocator(numticks=int(8 * scale_ratio), subs=(.2, .4, .6, .8)))
        else:
            # For linear scale, calculate ratio directly
            ax1_range = ax1.get_ylim()
            ax2_range = ax2.get_ylim()
            
            scale_ratio = (ax2_range[1] - ax2_range[0]) / (ax1_range[1] - ax1_range[0])
            
            # Use MaxNLocator for even spacing
            n_ticks = 5
            ax1.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
            ax2.yaxis.set_major_locator(plt.MaxNLocator(int(n_ticks * scale_ratio)))
            
            # Set minor locators
            ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.2)
    
    # Define components to plot
    components = [
        ('Vertical', 'Z'),
        ('Radial', 'R'),
        ('Transverse', 'T')
    ]
    
    # Calculate PSDs and find global min/max values
    rot_min, rot_max = np.inf, -np.inf
    acc_min, acc_max = np.inf, -np.inf
    
    for comp_name, comp_pattern in components:
        # Get rotation rate component
        tr1 = rot.select(component=comp_pattern)
        if len(tr1) == 0:
            continue
        tr1 = tr1[0]
        
        # Get acceleration component
        tr2 = acc.select(component=comp_pattern)
        if len(tr2) == 0:
            continue
        tr2 = tr2[0]
        
        # Calculate PSDs
        nfft = int(tr1.stats.npts/2)
        dt = tr1.stats.delta
        
        # Use multitaper spectral estimation
        spec1, freq = mtspec(tr1.data, delta=dt, time_bandwidth=4, nfft=nfft)
        spec2, _ = mtspec(tr2.data, delta=dt, time_bandwidth=4, nfft=nfft)
        
        # Convert to PSD
        psd1 = np.sqrt(spec1)
        psd2 = np.sqrt(spec2)
        
        # Update global min/max, excluding zeros and negative values for log scale
        valid_rot = psd1[psd1 > 0] if ylog else psd1
        valid_acc = psd2[psd2 > 0] if ylog else psd2
        
        if len(valid_rot) > 0:
            rot_min = min(rot_min, np.min(valid_rot))
            rot_max = max(rot_max, np.max(valid_rot))
        if len(valid_acc) > 0:
            acc_min = min(acc_min, np.min(valid_acc))
            acc_max = max(acc_max, np.max(valid_acc))
    
    # Add padding to limits
    if ylog:
        rot_min *= 0.1
        rot_max *= 10
        acc_min *= 0.1
        acc_max *= 10
    else:
        rot_padding = (rot_max - rot_min) * 0.1
        acc_padding = (acc_max - acc_min) * 0.1
        rot_min = max(0, rot_min - rot_padding)
        rot_max += rot_padding
        acc_min = max(0, acc_min - acc_padding)
        acc_max += acc_padding
    
    # Plot each component
    for i, (comp_name, comp_pattern) in enumerate(components):
        ax = fig.add_subplot(gs[i])
        ax1 = ax  # Primary y-axis for rotation
        ax2 = ax.twinx()  # Secondary y-axis for acceleration
        
        # Get traces
        tr1 = rot.select(component=comp_pattern)
        tr2 = acc.select(component=comp_pattern)
        
        if len(tr1) > 0 and len(tr2) > 0:
            tr1 = tr1[0]
            tr2 = tr2[0]
            
            # Calculate PSDs
            nfft = int(tr1.stats.npts/2)
            dt = tr1.stats.delta
            
            spec1, freq = mtspec(tr1.data, delta=dt, time_bandwidth=4, nfft=nfft)
            spec2, _ = mtspec(tr2.data, delta=dt, time_bandwidth=4, nfft=nfft)
            
            psd1 = np.sqrt(spec1)
            psd2 = np.sqrt(spec2)
            
            # Plot PSDs
            if fill:
                ax1.fill_between(freq, psd1, alpha=0.3, color='blue', label='Rotation')
                ax2.fill_between(freq, psd2, alpha=0.3, color='red', label='Translation')
            else:
                ax1.plot(freq, psd1, color='blue', label='Rotation', alpha=0.8)
                ax2.plot(freq, psd2, color='red', label='Translation', alpha=0.8)
            
            # Set frequency limits
            if fmin is not None:
                ax1.set_xlim(left=fmin)
            if fmax is not None:
                ax1.set_xlim(right=fmax)
            
            # Set scales
            if xlog:
                ax1.set_xscale('log')
            if ylog:
                ax1.set_yscale('log')
                ax2.set_yscale('log')
            
            # Set limits
            ax1.set_ylim(rot_min, rot_max)
            ax2.set_ylim(acc_min, acc_max)
            
            # Synchronize axes and grid lines
            __sync_axes(ax1, ax2, ylog)
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add labels
            if i == len(components) - 1:
                ax1.set_xlabel('Frequency (Hz)')
            if i == 1:
                ax1.set_ylabel('Rotation Rate (rad/s)')
                ax2.set_ylabel('Translation (m/s)')
            
            # Add component label
            ax1.text(0.02, 0.98, comp_name, transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig