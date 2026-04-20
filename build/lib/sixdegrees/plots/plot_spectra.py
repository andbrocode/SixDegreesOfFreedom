"""
Functions for plotting spectral comparison with fill between curves.
"""
from typing import Optional, Union, Dict, Tuple, List, Literal
from enum import Enum
import numpy as np
from numpy import ndarray
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from obspy import Stream, Trace
import multitaper as mt

class SpectralMethod(str, Enum):
    """Spectral estimation methods."""
    FFT = 'fft'
    MULTITAPER = 'multitaper'
    WELCH = 'welch'

def _fft(data: ndarray, dt: float) -> Tuple[ndarray, ndarray]:
    """
    Calculate power spectral density using FFT.
    
    Parameters
    ----------
    data : ndarray
        Input time series data
    dt : float
        Sampling interval
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        Frequency and PSD arrays
    """
    n = len(data)
    freq = np.fft.rfftfreq(n, dt)
    fft = np.fft.rfft(data * np.hanning(n))
    asd = 2.0 * np.abs(fft)**2 / (n * n)  # Normalized PSD
    return freq, asd

def _multitaper(data: ndarray, dt: float, n_win: int = 5, time_bandwidth: float = 4.0) -> Tuple[ndarray, ndarray]:
    """
    Calculate power spectral density using multitaper method.
    
    Parameters
    ----------
    data : ndarray
        Input time series data
    dt : float
        Sampling interval
    n_win : int, optional
        Number of windows, by default 5
    time_bandwidth : float, optional
        Time-bandwidth product, by default 4.0
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        Frequency and PSD arrays
    """
    out_psd = mt.MTSpec(data, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=2)
    f, psd = out_psd.rspec()
    asd = np.sqrt(psd.reshape(-1))
    return f.reshape(-1), asd

def _welch(data: ndarray, dt: float, nperseg: Optional[int] = None) -> Tuple[ndarray, ndarray]:
    """
    Calculate power spectral density using Welch's method.
    
    Parameters
    ----------
    data : ndarray
        Input time series data
    dt : float
        Sampling interval
    nperseg : int, optional
        Length of each segment. If None, defaults to data.size//8
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        Frequency and PSD arrays
    """
    if nperseg is None:
        nperseg = len(data) // 8
    freq, psd = signal.welch(data, fs=1/dt, nperseg=nperseg, window='hann')
    asd = np.sqrt(psd)
    return freq, asd

def calculate_spectrum(data: ndarray, dt: float, method: SpectralMethod = SpectralMethod.MULTITAPER, 
                 **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Calculate power spectral density using specified method.
    
    Parameters
    ----------
    data : ndarray
        Input time series data
    dt : float
        Sampling interval
    method : SpectralMethod, optional
        Spectral estimation method, by default SpectralMethod.MULTITAPER
    **kwargs : dict
        Additional arguments passed to specific PSD methods:
        - FFT: no additional parameters
        - Multitaper: n_win, time_bandwidth
        - Welch: nperseg
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        Frequency and PSD arrays
    """
    if method == SpectralMethod.FFT:
        return _fft(data, dt)
    elif method == SpectralMethod.MULTITAPER:
        return _multitaper(data, dt, **kwargs)
    elif method == SpectralMethod.WELCH:
        return _welch(data, dt, **kwargs)
    else:
        raise ValueError(f"Unknown spectral method: {method}")

def plot_spectra(rot: Stream, acc: Stream, 
                fmin: Optional[float] = None, 
                fmax: Optional[float] = None,
                ylog: bool = False, 
                xlog: bool = False, 
                fill: bool = False,
                method: Union[SpectralMethod, str] = SpectralMethod.MULTITAPER,
                **kwargs) -> Figure:
    """
    Plot power spectral density comparison between rotation and acceleration data with horizontal layout.
    
    Parameters
    ----------
    rot : Stream
        Rotation rate stream
    acc : Stream
        Acceleration stream
    fmin : float, optional
        Minimum frequency for plotting
    fmax : float, optional
        Maximum frequency for plotting
    ylog : bool, optional
        Use logarithmic y-axis scale, by default False
    xlog : bool, optional
        Use logarithmic x-axis scale, by default False
    fill : bool, optional
        Fill the area under curves, by default False
    method : SpectralMethod or str, optional
        Spectral estimation method to use, by default SpectralMethod.MULTITAPER
        Options:
        - 'fft': Fast Fourier Transform with Hanning window
        - 'multitaper': Thomson multitaper method (better for non-stationary signals)
        - 'welch': Welch's method (good for noise reduction)
    **kwargs : dict
        Additional arguments for specific spectral methods:
        - FFT: no additional parameters
        - Multitaper: n_win (int, default=5), time_bandwidth (float, default=4.0)
        - Welch: nperseg (int, default=data.size//8)
        
    Returns
    -------
    Figure
        Matplotlib figure containing the plots
    """

    # Plot settings
    PLOT_SETTINGS = {
        'font_size': 12,
        'line_width': 1.5,
        'rot_color': 'darkred',  # Dark red
        'acc_color': 'black',  # Dark slate gray
        'alpha_fill': 0.4,
        'alpha_line': 1.0,
        'components': [('Z', '*Z'), ('N', '*N'), ('E', '*E')],
        'n_windows': 5
    }

    # Create figure with improved layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3, top=0.85)

    # Convert string method to enum if needed
    if isinstance(method, str):
        method = SpectralMethod(method.lower())

    # Calculate PSDs for all components
    psds = {}
    for comp_name, comp_pattern in PLOT_SETTINGS['components']:
        # Get traces for this component
        rot_tr = rot.select(channel=comp_pattern)[0]
        acc_tr = acc.select(channel=comp_pattern)[0]
        
        # Calculate PSDs using selected method
        f_rot, psd_rot = calculate_spectrum(
            rot_tr.data, 
            rot_tr.stats.delta,
            method=method,
            **kwargs
        )
        f_acc, psd_acc = calculate_spectrum(
            acc_tr.data, 
            acc_tr.stats.delta,
            method=method,
            **kwargs
        )
        psds[comp_name] = {
            'rot': {'freq': f_rot, 'psd': psd_rot, 'trace': rot_tr},
            'acc': {'freq': f_acc, 'psd': psd_acc, 'trace': acc_tr}
        }

    # Add title with metadata
    start_time = rot[0].stats.starttime
    end_time = rot[0].stats.endtime
    title = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}-{end_time.strftime('%H:%M:%S')} UTC | {method.value.title()}"
    if fmin is not None and fmax is not None:
        title += f" | {fmin}-{fmax} Hz"
    
    # Add method-specific parameters to title
    if method == SpectralMethod.MULTITAPER:
        n_win = kwargs.get('n_win', 5)
        time_bandwidth = kwargs.get('time_bandwidth', 4.0)
        title += f" (N={n_win}, NW={time_bandwidth})"
    elif method == SpectralMethod.WELCH:
        nperseg = kwargs.get('nperseg', None)
        if nperseg:
            title += f" (Seg={nperseg})"
    fig.suptitle(title, fontsize=PLOT_SETTINGS['font_size']+2, y=1.0)

    # Plot each component
    for i, (comp_name, comp_data) in enumerate(psds.items()):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Get trace info for labels
        rot_tr = comp_data['rot']['trace']
        acc_tr = comp_data['acc']['trace']
        rot_label = f"{rot_tr.stats.station}.{rot_tr.stats.channel}"
        acc_label = f"{acc_tr.stats.station}.{acc_tr.stats.channel}"
        
        # Plot data
        if fill:
            ax1.fill_between(
                comp_data['rot']['freq'],
                comp_data['rot']['psd'],
                color=PLOT_SETTINGS['rot_color'],
                alpha=PLOT_SETTINGS['alpha_fill'],
                label=rot_label,
                zorder=3
            )
            ax2.fill_between(
                comp_data['acc']['freq'],
                comp_data['acc']['psd'],
                color=PLOT_SETTINGS['acc_color'],
                alpha=PLOT_SETTINGS['alpha_fill'],
                label=acc_label,
                zorder=2
            )
        else:
            ax1.plot(
                comp_data['rot']['freq'],
                comp_data['rot']['psd'],
                color=PLOT_SETTINGS['rot_color'],
                alpha=PLOT_SETTINGS['alpha_line'],
                linewidth=PLOT_SETTINGS['line_width'],
                label=rot_label,
                zorder=3
            )
            ax2.plot(
                comp_data['acc']['freq'],
                comp_data['acc']['psd'],
                color=PLOT_SETTINGS['acc_color'],
                alpha=PLOT_SETTINGS['alpha_line'],
                linewidth=PLOT_SETTINGS['line_width'],
                label=acc_label,
                zorder=2
            )

        # Configure axes
        if xlog:
            ax1.set_xscale('log')
        if ylog:
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        # Set frequency limits
        xlim_right = fmax if fmax else rot[0].stats.sampling_rate * 0.5
        ax1.set_xlim(left=fmin, right=xlim_right)
        
        # Set y-limits to start from 0
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        # Style the axes
        # ax1.grid(True, which='both', alpha=0.2, linestyle='--')
        ax1.tick_params(axis='y', colors=PLOT_SETTINGS['rot_color'])
        ax2.tick_params(axis='y', colors=PLOT_SETTINGS['acc_color'])
        
        for spine in ax1.spines.values():
            spine.set_alpha(0.3)
        for spine in ax2.spines.values():
            spine.set_alpha(0.3)

        # Add labels
        ax1.set_xlabel('Frequency (Hz)', fontsize=PLOT_SETTINGS['font_size'])
        if i == 0:
            ax1.set_ylabel(r'ASD (rad/s/$\sqrt{Hz}$)', 
                         fontsize=PLOT_SETTINGS['font_size'],
                         color=PLOT_SETTINGS['rot_color'])
        if i == 2:
            ax2.set_ylabel(r'ASD (m/s$^2$/$\sqrt{Hz}$)',
                         fontsize=PLOT_SETTINGS['font_size'],
                         color=PLOT_SETTINGS['acc_color'])

        # Add legends with improved styling
        ax1.legend(loc='upper right', framealpha=0.8, 
                  fontsize=PLOT_SETTINGS['font_size']-1)
        ax2.legend(loc='upper left', framealpha=0.8,
                  fontsize=PLOT_SETTINGS['font_size']-1)

        # Add component title
        ax1.set_title(f'Component {comp_name}',
                     fontsize=PLOT_SETTINGS['font_size'],
                     pad=10)

    plt.tight_layout()
    return fig