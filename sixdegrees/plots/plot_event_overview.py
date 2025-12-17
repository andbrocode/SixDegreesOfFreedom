"""
Function for plotting a comprehensive event overview combining multiple analysis plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from typing import Dict, Optional, Union
from obspy import Stream
from obspy.signal.rotate import rotate_ne_rt
from obspy.core.utcdatetime import UTCDateTime


def plot_event_overview(sd, baz_results: Dict, velocity_results: Dict, 
                       event_info: Optional[Dict] = None,
                       wave_type: str = 'love',
                       baz_theo: Optional[float] = None,
                       baz_theo_margin: float = 10,
                       unitscale: str = 'nano',
                       cc_threshold: Optional[float] = None,
                       cc_method: str = 'mid',
                       vmax: Optional[float] = None,
                       minors: bool = True,
                       fmin: Optional[float] = None,
                       fmax: Optional[float] = None,
                       twin_sec: int = 5,
                       twin_overlap: float = 0.5,
                       map_projection: str = 'orthographic',
                       zoom_t1: float = 2.0,
                       zoom_t2: float = 20.0,
                       figsize: Optional[tuple] = None,
                       rot_scale_factor: float = 3.0) -> plt.Figure:
    """
    Create a comprehensive event overview plot combining waveform comparison, 
    backazimuth estimates, velocity estimates, and a geographic map.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    baz_results : Dict
        Dictionary containing backazimuth results from compute_backazimuth
    velocity_results : Dict
        Dictionary containing velocity results from compute_velocities
    event_info : Dict, optional
        Event information dictionary containing:
            - origin_time: Event origin time
            - magnitude: Event magnitude
            - distance_km: Epicentral distance in km
            - backazimuth: Theoretical backazimuth
            - latitude: Event latitude
            - longitude: Event longitude
    wave_type : str
        Wave type: 'love' or 'rayleigh' (default: 'love')
    baz_theo : float, optional
        Theoretical backazimuth in degrees. If None, uses event_info['backazimuth']
    baz_theo_margin : float
        Margin around theoretical backazimuth in degrees (default: 10)
    unitscale : str
        Unit scale for rotation rate ('nano' or 'micro', default: 'nano')
    cc_threshold : float, optional
        Minimum cross-correlation coefficient threshold
    cc_method : str
        Type of cc to choose ('mid' or 'max', default: 'mid')
    vmax : float, optional
        Maximum velocity for plot scaling
    minors : bool
        Add minor ticks to axes if True (default: True)
    fmin : float, optional
        Minimum frequency for bandpass filter
    fmax : float, optional
        Maximum frequency for bandpass filter
    twin_sec : int
        Time window length in seconds for waveform CC (default: 5)
    twin_overlap : float
        Time window overlap for waveform CC (default: 0.5)
    map_projection : str
        Map projection type ('orthographic' or 'platecarree', default: 'orthographic')
    zoom_t1 : float
        Time before P/S arrival to show in zoom window (default: 2.0 seconds)
    zoom_t2 : float
        Time after P/S arrival to show in zoom window (default: 20.0 seconds)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-determined.
    rot_scale_factor : float
        Scaling factor for rotation data in plots (default: 3.0)

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the comprehensive overview plot
    """
    # Validate wave_type
    wave_type = wave_type.lower()
    if wave_type not in ['love', 'rayleigh']:
        raise ValueError(f"Invalid wave_type: {wave_type}. Use 'love' or 'rayleigh'.")
    
    # Get theoretical backazimuth
    if baz_theo is None and event_info is not None:
        baz_theo = event_info.get('backazimuth', None)
    
    # Get station coordinates for map
    station_coords = {
        'latitude': sd.station_latitude,
        'longitude': sd.station_longitude
    }
    
    # Prepare baz_estimates for map
    baz_estimates = {}
    if baz_results:
        if cc_method == 'mid':
            baz_est = baz_results.get('baz_mid', None)
        else:
            baz_est = baz_results.get('baz_max', None)
        
        if baz_est is not None:
            # Get mean or median of estimates
            if isinstance(baz_est, (list, np.ndarray)):
                if len(baz_est) > 0:
                    # Use weighted mean if CC values available
                    if cc_method == 'mid':
                        cc_vals = baz_results.get('cc_mid', np.ones(len(baz_est)))
                    else:
                        cc_vals = baz_results.get('cc_max', np.ones(len(baz_est)))
                    
                    if cc_threshold is not None:
                        mask = np.array(cc_vals) > cc_threshold
                        if np.any(mask):
                            baz_est = np.average(np.array(baz_est)[mask], weights=np.array(cc_vals)[mask])
                        else:
                            baz_est = np.mean(baz_est)
                    else:
                        baz_est = np.average(baz_est, weights=cc_vals)
            baz_estimates[wave_type] = baz_est
    
    # Determine figure size
    if figsize is None:
        figsize = (20, 16)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create figure with GridSpec
    # New layout:
    # Row 0-2: Event info (left, col 0, rows 0-2) + empty (col 1) + Map (right, col 5+, rows 0-2)
    # Row 3-4: Waveform comparison
    # Row 5: P and S wave zoom windows (side by side)
    # Row 6: Backazimuth estimates
    # Row 7: Velocity estimates
    gs = GridSpec(8, 10, figure=fig, hspace=0.5, wspace=0.3, 
                  height_ratios=[0.6, 0.6, 0.6, 1.2, 1.2, 0.8, 1.0, 1.0],
                  width_ratios=[0.6, 0.1, 1, 1, 1, 1, 1, 1, 1, 0.15])
    
    # ========== TOP ROWS 0-2: Event info and map ==========
    # Event info box (left, first column, spans rows 0-2) - smaller to make map bigger
    ax_info = fig.add_subplot(gs[0:3, 0])
    ax_info.axis('off')
    _plot_event_info_box(ax_info, event_info, baz_theo)
    
    # Map (right, column 5 onwards, spans rows 0-2, bigger)
    ax_map = _create_map_subplot(fig, gs[0:3, 5:9], map_projection, station_coords, event_info)
    if event_info:
        _plot_spherical_map_backazimuth(
            ax_map, event_info, baz_estimates,
            station_coords.get('latitude', 0),
            station_coords.get('longitude', 0),
            map_projection
        )
    
    # ========== WAVEFORM COMPARISON ==========
    ax_wave = fig.add_subplot(gs[3:5, :9])
    p_arrival_time, s_arrival_time, max_time = _plot_waveform_comparison(ax_wave, sd, baz_theo, wave_type, unitscale, 
                             fmin, fmax, twin_sec, twin_overlap, event_info, rot_scale_factor)
    
    # ========== P AND S WAVE ZOOM WINDOWS ==========
    # Side by side below waveform comparison (with small gap in column 4)
    ax_p_zoom = fig.add_subplot(gs[5, :4])
    ax_s_zoom = fig.add_subplot(gs[5, 5:9])
    
    if p_arrival_time is not None:
        starttime = sd.get_stream("translation")[0].stats.starttime
        times = sd.get_stream("translation").select(channel="*Z")[0].times()
        if p_arrival_time - zoom_t1 <= 0:
            t1 = 0
        else:
            t1 = p_arrival_time - zoom_t1
        if p_arrival_time + zoom_t2 >= times[-1]:
            t2 = times[-1]
        else:
            t2 = p_arrival_time + zoom_t2
        _plot_zoom_window(ax_p_zoom, sd, baz_theo, wave_type, unitscale, fmin, fmax,
                            t1, t2, 'P', 12, arrival_time=p_arrival_time, 
                            zoom_t1=zoom_t1, zoom_t2=zoom_t2, rot_scale_factor=rot_scale_factor)
    
    if s_arrival_time is not None:
        starttime = sd.get_stream("translation")[0].stats.starttime
        times = sd.get_stream("translation").select(channel="*Z")[0].times()
        if s_arrival_time - zoom_t1 <= 0:
            t1 = 0
        else:
            t1 = s_arrival_time - zoom_t1
        if s_arrival_time + zoom_t2 >= times[-1]:
            t2 = times[-1]
        else:
            t2 = s_arrival_time + zoom_t2
        _plot_zoom_window(ax_s_zoom, sd, baz_theo, wave_type, unitscale, fmin, fmax,
                        t1, t2, 'S', 12, arrival_time=s_arrival_time,
                        zoom_t1=zoom_t1, zoom_t2=zoom_t2, rot_scale_factor=rot_scale_factor)
    
    # ========== BACKAZIMUTH ESTIMATES ==========
    ax_baz = fig.add_subplot(gs[6, :9])
    _plot_backazimuth_panel(ax_baz, sd, baz_results, wave_type, 
                           baz_theo, baz_theo_margin, unitscale, cc_threshold, 
                           cc_method, minors, max_time=max_time)
    
    # ========== VELOCITY ESTIMATES ==========
    ax_vel = fig.add_subplot(gs[7, :9])
    _plot_velocity_panel(ax_vel, None, sd, velocity_results, 
                        wave_type, vmax, cc_threshold, minors, max_time=max_time)
    
    # Add overall title
    title = f"{wave_type.capitalize()} Wave Event Analysis"
    # if event_info and 'origin_time' in event_info:
    #     origin_time = event_info['origin_time']
    #     if isinstance(origin_time, UTCDateTime):
    #         title += f" | {origin_time.date} {str(origin_time.time).split('.')[0]} UTC"
    #     else:
    #         title += f" | {origin_time}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.93)
    
    return fig


def _plot_event_info_box(ax, event_info, baz_theo):
    """Plot event information in a text box"""
    if event_info is None:
        ax.text(0.5, 0.5, "No event information available", 
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes)
        return
    
    info_lines = []
    info_lines.append(f"Event Information:")
    info_lines.append("")

    # Origin time
    if 'origin_time' in event_info:
        origin_time = event_info['origin_time']
        if isinstance(origin_time, UTCDateTime):
            info_lines.append(f"Origin Time: {origin_time.date} {str(origin_time.time).split('.')[0]} UTC")
        else:
            info_lines.append(f"Origin Time: {origin_time}")
    
    # Magnitude
    if 'magnitude' in event_info:
        mag_type = event_info.get('magnitude_type', '')
        info_lines.append(f"Magnitude: {event_info['magnitude']} {mag_type}")
    
    # Distance
    if 'distance_km' in event_info:
        info_lines.append(f"Distance: {event_info['distance_km']:.1f} km")
    if 'distance_deg' in event_info:
        info_lines.append(f"Distance: {event_info['distance_deg']:.1f}°")
    
    # Theoretical backazimuth
    if baz_theo is not None:
        info_lines.append(f"Theoretical BAZ: {baz_theo:.1f}°")
    elif 'backazimuth' in event_info:
        info_lines.append(f"Theoretical BAZ: {event_info['backazimuth']:.1f}°")
    
    # Location
    if 'latitude' in event_info and 'longitude' in event_info:
        info_lines.append(f"Location: {event_info['latitude']:.2f}°N, {event_info['longitude']:.2f}°E")
    
    # Depth
    if 'depth_km' in event_info:
        info_lines.append(f"Depth: {event_info['depth_km']:.1f} km")
    
    # Create prettified text box with gray background - bigger font with increased vertical spacing
    # Add extra spacing between lines
    text_str = '\n\n'.join(info_lines)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
           fontsize=16, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9, 
                    edgecolor='darkgray', linewidth=1.5, pad=0.8),
           family='monospace', linespacing=0.7)


def _plot_waveform_comparison(ax, sd, baz, wave_type, unitscale, fmin, fmax, 
                             twin_sec, twin_overlap, event_info=None, rot_scale_factor=3.0):
    """Plot waveform comparison similar to plot_waveform_cc with crosscorrelation dots"""
    from numpy import linspace, ones, array
    from obspy.signal.cross_correlation import correlate, xcorr_max
    from matplotlib.colors import BoundaryNorm
    
    def _cross_correlation_windows(arr1, arr2, dt, Twin, overlap=0, lag=0, demean=True):
        from numpy import arange, roll
        N = len(arr1)
        n_interval = int(Twin/dt)
        n_overlap = int(overlap*Twin/dt)
        
        times, samples = [], []
        n1, n2 = 0, n_interval
        while n2 <= N:
            samples.append((n1, n2))
            times.append(int(n1+(n2-n1)/2)*dt)
            n1 = n1 + n_interval - n_overlap
            n2 = n2 + n_interval - n_overlap
        
        cc = []
        for _n, (n1, n2) in enumerate(samples):
            _arr1 = roll(arr1[n1:n2], lag)
            _arr2 = arr2[n1:n2]
            ccf = correlate(_arr1, _arr2, 0, demean=demean, normalize='naive', method='fft')
            shift, val = xcorr_max(ccf, abs_max=False)
            cc.append(val)
        
        return array(times), array(cc)
    
    # Get streams
    rot = sd.get_stream("rotation").copy()
    acc = sd.get_stream("translation").copy()
    
    # Apply filtering if needed
    if fmin is not None and fmax is not None:
        rot.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
        acc.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
    
    # Define scaling factors
    if unitscale == "nano":
        acc_scaling, acc_unit = 1e6, f"{sd.mu}{sd.tunit}"
        rot_scaling, rot_unit = 1e9, f"n{sd.runit}"
    elif unitscale == "micro":
        acc_scaling, acc_unit = 1e3, f"m{sd.tunit}"
        rot_scaling, rot_unit = 1e6, f"{sd.mu}{sd.runit}"
    else:
        raise ValueError(f"Invalid unitscale: {unitscale}")
    
    font = 12
    lw = 1.0
    
    # Use baz from event_info or default to 0
    if baz is None:
        baz = 0
    
    # Get sampling rate
    dt = rot[0].stats.delta
    
    # Get components based on wave type
    if wave_type == "love":
        rot_z = rot.select(channel="*Z")[0].data
        acc_r, acc_t = rotate_ne_rt(
            acc.select(channel="*N")[0].data,
            acc.select(channel="*E")[0].data,
            baz
        )
        # Apply scaling
        rot_z *= rot_scaling
        acc_t *= acc_scaling
        
        # Calculate max values for ylim
        rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
        acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])
        
        # Calculate cross-correlation
        tt0, cc0 = _cross_correlation_windows(rot_z, acc_t, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        
        # Calculate overall CC value
        from obspy.signal.cross_correlation import correlate
        cc_overall = max(correlate(rot_z, acc_t, 0, demean=True, normalize='naive', method='fft'))
        
        # Plot
        times = rot.select(channel="*Z")[0].times()
        ax.plot(times, rot_z*rot_scale_factor, label=f"{rot_scale_factor:.1f}x ROT-Z", color="tab:red", lw=lw, zorder=3)
        ax2 = ax.twinx()
        ax2.plot(times, acc_t, label=f"ACC-T", color="black", lw=lw)
        
        # Add crosscorrelation dots at bottom
        ax3 = ax.twinx()
        cmap = plt.get_cmap("coolwarm", 16)
        boundaries = np.arange(-1.0, 1.2, 0.2)
        norm = BoundaryNorm(boundaries, cmap.N)
        scatter_cc = ax3.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, norm=norm, s=20, zorder=4)
        ax3.set_ylim(-1, 1)
        ax3.yaxis.set_visible(False)
        
        ax.set_ylim(-rot_z_max, rot_z_max)
        ax2.set_ylim(-acc_t_max, acc_t_max)
        
        ax.set_xlim(left=0)
        ax2.set_xlim(left=0)
        
        ax.set_ylabel(f"Rotation rate ({rot_unit})", fontsize=font, color="tab:red")
        ax2.set_ylabel(f"Acceleration ({acc_unit})", fontsize=font, color="black")
        ax.tick_params(axis='y', labelcolor="tab:red")
        ax2.tick_params(axis='y', labelcolor="black")
        
        # Add colorbar for cross-correlation in right upper corner
        cbar_cc_ax = ax.inset_axes([0.9, 1.07, 0.1, 0.07]) # [left, bottom, width, height]
        boundaries_cbar = np.arange(-1.0, 1.1, 0.1)  # Steps of 0.1 from -1 to 1 (20 bins)
        # Create colormap with enough colors for the boundaries (20 bins need at least 20 colors)
        cmap_cbar = plt.get_cmap("coolwarm", len(boundaries_cbar)-1)
        norm_cbar = BoundaryNorm(boundaries_cbar, cmap_cbar.N)
        sm = plt.cm.ScalarMappable(cmap=cmap_cbar, norm=norm_cbar)
        sm.set_array([])
        cbar_cc = plt.colorbar(sm, cax=cbar_cc_ax, boundaries=boundaries_cbar, 
                              ticks=[-1, 0, 1], format='%.0f', orientation='horizontal')
        cbar_cc.set_label("CC-Coefficient", fontsize=font-2, rotation=0, labelpad=-50)
        
        # Add overall CC value as text
        ax.text(0.98, 0.2, f"CC = {cc_overall:.2f}", 
               transform=ax.transAxes, fontsize=font,
               horizontalalignment='right', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='black', linewidth=1, pad=0.5), zorder=10)
        
    elif wave_type == "rayleigh":
        acc_z = acc.select(channel="*Z")[0].data
        rot_r, rot_t = rotate_ne_rt(
            rot.select(channel="*N")[0].data,
            rot.select(channel="*E")[0].data,
            baz
        )
        # Apply scaling
        acc_z *= acc_scaling
        rot_t *= rot_scaling
        
        # Calculate max values for ylim
        acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
        rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])
        
        # Calculate cross-correlation
        tt1, cc1 = _cross_correlation_windows(rot_t, acc_z, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        
        # Calculate overall CC value
        from obspy.signal.cross_correlation import correlate
        cc_overall = max(correlate(rot_t, acc_z, 0, demean=True, normalize='naive', method='fft'))
        
        # Plot
        times = acc.select(channel="*Z")[0].times()
        ax.plot(times, rot_t*rot_scale_factor, label=f"{rot_scale_factor:.1f}x ROT-T", color="tab:red", lw=lw, zorder=3)
        ax2 = ax.twinx()
        ax2.plot(times, acc_z, label=f"ACC-Z", color="black", lw=lw)
        
        # Add crosscorrelation dots at bottom
        ax3 = ax.twinx()
        cmap = plt.get_cmap("coolwarm", 16)
        boundaries = np.arange(-1.0, 1.2, 0.2)
        norm = BoundaryNorm(boundaries, cmap.N)
        scatter_cc = ax3.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, norm=norm, s=20, zorder=4)
        ax3.set_ylim(-1, 1)
        ax3.yaxis.set_visible(False)
        
        ax.set_ylim(-rot_t_max, rot_t_max)
        ax2.set_ylim(-acc_z_max, acc_z_max)

        ax.set_xlim(left=0)
        ax2.set_xlim(left=0)

        ax.set_ylabel(f"Rotation rate ({rot_unit})", fontsize=font, color="tab:red")
        ax2.set_ylabel(f"Acceleration ({acc_unit})", fontsize=font, color="black")
        ax.tick_params(axis='y', labelcolor="tab:red")
        ax2.tick_params(axis='y', labelcolor="black")
        
        # Add colorbar for cross-correlation in right upper corner
        cbar_cc_ax = ax.inset_axes([0.9, 1.07, 0.1, 0.07]) # [left, bottom, width, height]
        boundaries_cbar = np.arange(-1.0, 1.1, 0.1)  # Steps of 0.1 from -1 to 1 (20 bins)
        # Create colormap with enough colors for the boundaries (20 bins need at least 20 colors)
        cmap_cbar = plt.get_cmap("coolwarm", len(boundaries_cbar)-1)
        norm_cbar = BoundaryNorm(boundaries_cbar, cmap_cbar.N)
        sm = plt.cm.ScalarMappable(cmap=cmap_cbar, norm=norm_cbar)
        sm.set_array([])
        cbar_cc = plt.colorbar(sm, cax=cbar_cc_ax, boundaries=boundaries_cbar, 
                              ticks=[-1, 0, 1], format='%.0f', orientation='horizontal')
        cbar_cc.set_label("CC-Coefficient", fontsize=font-2, rotation=0, labelpad=-50)
        
        # Add overall CC value as text
        ax.text(0.98, 0.2, f"CC = {cc_overall:.2f}", 
               transform=ax.transAxes, fontsize=font,
               horizontalalignment='right', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='black', linewidth=1, pad=0.5), zorder=10)
    
    # Sync twin axes using sd.sync_twin_axes
    sd.sync_twin_axes(ax, ax2)
    
    ax.legend(loc=1, ncols=2)
    ax2.legend(loc=4, ncols=2, bbox_to_anchor=(1, 0.8))
    ax.grid(which="both", alpha=0.5)
    ax.set_xlabel("Time (s)", fontsize=font)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Get max time for synchronizing other panels and set xlim
    max_time = times[-1] if len(times) > 0 else None
    if max_time is not None:
        ax.set_xlim(right=max_time)
        ax2.set_xlim(right=max_time)
    
    # Add P and S wave arrival lines (zoom windows will be in separate subplots)
    p_arrival_time = None
    s_arrival_time = None
    
    if event_info is not None:
        starttime = sd.get_stream("translation")[0].stats.starttime
        
        # Get P arrival
        try:
            p_arrival_utc = sd.get_theoretical_arrival(phase='P')
            if p_arrival_utc is not None:
                p_arrival_time = (p_arrival_utc - starttime)
        except:
            try:
                p_arrival_utc = sd.get_theoretical_arrival(phase='Pdiff')
                if p_arrival_utc is not None:
                    p_arrival_time = (p_arrival_utc - starttime)
            except:
                pass
        
        # Get S arrival
        try:
            s_arrival_utc = sd.get_theoretical_arrival(phase='S')
            if s_arrival_utc is not None:
                s_arrival_time = (s_arrival_utc - starttime)
        except:
            pass
        
        # Plot P arrival line
        if p_arrival_time is not None and 0 <= p_arrival_time <= times[-1]:
            ax.axvline(x=p_arrival_time, color='blue', linestyle='--', linewidth=2, 
                      alpha=0.7, zorder=5, label='P arrival')
            ax.text(p_arrival_time+20, ax.get_ylim()[1]*0.95, 'P', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=font+2, fontweight='bold', color='blue', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1))
        
        # Plot S arrival line
        if s_arrival_time is not None and 0 <= s_arrival_time <= times[-1]:
            ax.axvline(x=s_arrival_time, color='tab:red', linestyle='--', linewidth=2, 
                      alpha=0.7, zorder=5, label='S arrival')
            ax.text(s_arrival_time+20, ax.get_ylim()[1]*0.95, 'S', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=font+2, fontweight='bold', color='tab:red', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1))
    
    return p_arrival_time, s_arrival_time, max_time


def _plot_zoom_window(ax_zoom, sd, baz, wave_type, unitscale, fmin, fmax, 
                     t_start, t_end, phase_label, font, arrival_time=None, zoom_t1=None, zoom_t2=None, rot_scale_factor=3.0):
    """Plot zoom window for P or S wave arrival with time axis relative to arrival time"""
    from obspy.signal.rotate import rotate_ne_rt

    # Get streams
    rot = sd.get_stream("rotation").copy()
    acc = sd.get_stream("translation").copy()
    
    # Apply filtering if needed
    if fmin is not None and fmax is not None:
        rot.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
        acc.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
    
    # Define scaling factors
    if unitscale == "nano":
        acc_scaling, acc_unit = 1e6, f"{sd.mu}{sd.tunit}"
        rot_scaling, rot_unit = 1e9, f"n{sd.runit}"
    elif unitscale == "micro":
        acc_scaling, acc_unit = 1e3, f"m{sd.tunit}"
        rot_scaling, rot_unit = 1e6, f"{sd.mu}{sd.runit}"
    else:
        raise ValueError(f"Invalid unitscale: {unitscale}")
    
    # Use baz from event_info or default to 0
    if baz is None:
        baz = 0
    
    starttime = acc[0].stats.starttime
    t1_utc = starttime + t_start
    t2_utc = starttime + t_end

    # Trim streams to zoom window
    rot_zoom = rot.trim(t1_utc, t2_utc)
    acc_zoom = acc.trim(t1_utc, t2_utc)
    
    if len(rot_zoom) == 0 or len(acc_zoom) == 0:
        return
    
    lw = 0.9
    
    # Get components based on wave type
    if wave_type == "love":
        rot_z = rot_zoom.select(channel="*Z")[0].data * rot_scaling
        acc_r, acc_t = rotate_ne_rt(
            acc_zoom.select(channel="*N")[0].data,
            acc_zoom.select(channel="*E")[0].data,
            baz
        )
        acc_t *= acc_scaling
        
        # Times relative to arrival_time (arrival_time = 0)
        # rot_zoom.times() is relative to trimmed starttime (starttime + t_start)
        # Convert to relative to arrival_time: times + t_start - arrival_time
        times_raw = rot_zoom.select(channel="*Z")[0].times()
        times = times_raw + t_start - arrival_time
        ax_zoom.plot(times, rot_z*rot_scale_factor, label=f"{rot_scale_factor:.1f}x ROT-Z", color="tab:red", lw=lw, zorder=3)
        ax_zoom2 = ax_zoom.twinx()
        ax_zoom2.plot(times, acc_t, label=f"ACC-T", color="black", lw=lw)
        
    elif wave_type == "rayleigh":
        acc_z = acc_zoom.select(channel="*Z")[0].data * acc_scaling
        rot_r, rot_t = rotate_ne_rt(
            rot_zoom.select(channel="*N")[0].data,
            rot_zoom.select(channel="*E")[0].data,
            baz
        )
        rot_t *= rot_scaling
        
        # Times relative to arrival_time (arrival_time = 0)
        # acc_zoom.times() is relative to trimmed starttime (starttime + t_start)
        # Convert to relative to arrival_time: times + t_start - arrival_time
        times_raw = acc_zoom.select(channel="*Z")[0].times()
        times = times_raw + t_start - arrival_time
        ax_zoom.plot(times, rot_t*rot_scale_factor, label=f"{rot_scale_factor:.1f}x ROT-T", color="tab:red", lw=lw, zorder=3)
        ax_zoom2 = ax_zoom.twinx()
        ax_zoom2.plot(times, acc_z, label=f"ACC-Z", color="black", lw=lw)
    
    # Mark arrival time at x=0 (arrival_time is now the reference)
    if arrival_time is not None:
        ax_zoom.axvline(x=0, color='blue' if phase_label == 'P' else 'red', 
                       linestyle='--', linewidth=2, alpha=0.8, zorder=5)
    
    # Set x-axis limits relative to arrival time
    # Minimum: -zoom_t1, Maximum: +zoom_t2, Zero: arrival_time
    if zoom_t1 is not None and zoom_t2 is not None:
        ax_zoom.set_xlim(left=-zoom_t1, right=zoom_t2)
        ax_zoom2.set_xlim(left=-zoom_t1, right=zoom_t2)
    else:
        # Fallback: use t_start and t_end relative to arrival
        ax_zoom.set_xlim(left=t_start - arrival_time, right=t_end - arrival_time)
        ax_zoom2.set_xlim(left=t_start - arrival_time, right=t_end - arrival_time)

    # Sync axes
    sd.sync_twin_axes(ax_zoom, ax_zoom2)
    
    ax_zoom.grid(which="both", alpha=0.5)
    ax_legend = ax_zoom.legend(loc=1, ncols=1)
    ax_zoom2.legend(loc=4, ncols=1)
    ax_zoom.set_title(f'{phase_label}-wave zoom', fontsize=font-2, fontweight='bold')
    ax_zoom.set_xlabel("Time relative to arrival (s)", fontsize=font)
    ax_zoom.tick_params(labelsize=font-3)
    ax_zoom2.tick_params(labelsize=font-3)

    # add y-axis label to ax_zoom and ax_zoom2
    ax_zoom.set_ylabel(f"ROT ({rot_unit})", fontsize=font, color="tab:red")
    ax_zoom2.set_ylabel(f"ACC ({acc_unit})", fontsize=font, color="black")

    # make rotation y-axis label and text red 
    ax_zoom.tick_params(axis='y', labelcolor="tab:red")

def _plot_backazimuth_panel(ax_baz, sd, baz_results, wave_type, 
                           baz_theo, baz_theo_margin, unitscale, cc_threshold,
                           cc_method, minors, max_time=None):
    """Plot backazimuth estimation panel with histogram inside"""
    font = 12
    
    # Get backazimuth data
    if cc_method == 'mid':
        ccc = baz_results.get('cc_mid', [])
        baz = baz_results.get('baz_mid', [])
        time = baz_results.get('twin_center', [])
    else:
        ccc = baz_results.get('cc_max', [])
        baz = baz_results.get('baz_max', [])
        time = baz_results.get('twin_center', [])
    
    # Convert to arrays
    time = np.array(time)
    baz = np.array(baz)
    ccc = np.array(ccc)
    
    # Apply threshold
    if cc_threshold is not None:
        mask = ccc > cc_threshold
        time = time[mask]
        baz = baz[mask]
        cc = ccc[mask]
    else:
        cc = ccc
    
    if len(time) == 0:
        ax_baz.text(0.5, 0.5, "No backazimuth data available", 
                   ha='center', va='center', transform=ax_baz.transAxes)
        return
    
    # Plot backazimuth estimates
    cmap = plt.get_cmap("viridis", 10)
    scatter = ax_baz.scatter(time, baz, c=cc, s=50, cmap=cmap,
                            edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
    
    # Add theoretical backazimuth
    if baz_theo is not None:
        try:
            times_all = sd.get_stream("translation").select(channel="*Z")[0].times()
            if len(times_all) > 0:
                ax_baz.plot([min(times_all), max(times_all)], [baz_theo, baz_theo],
                           color='k', ls='--', label='Theoretical BAz', zorder=1, linewidth=2)
                ax_baz.fill_betweenx([baz_theo-baz_theo_margin, baz_theo+baz_theo_margin],
                                min(times_all), max(times_all),
                                color='grey', alpha=0.3, zorder=0)
        except:
            # If we can't get times, use the time range from baz_results
            if len(time) > 0:
                ax_baz.plot([min(time), max(time)], [baz_theo, baz_theo],
                           color='k', ls='--', label='Theoretical BAz', zorder=1, linewidth=2)
                ax_baz.fill_betweenx([baz_theo-baz_theo_margin, baz_theo+baz_theo_margin],
                                min(time), max(time),
                                color='grey', alpha=0.3, zorder=0)
    
    # Configure axis
    ax_baz.set_ylim(-5, 365)
    ax_baz.set_xlim(left=0)
    if max_time is not None:
        ax_baz.set_xlim(right=max_time)
    ax_baz.set_yticks(range(0, 360+60, 60))
    ax_baz.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
    ax_baz.set_ylabel(f"{wave_type.capitalize()} BAz (°)", fontsize=font)
    ax_baz.set_xlabel("Time (s)", fontsize=font)
    ax_baz.legend(loc='upper right')
    
    if minors:
        ax_baz.minorticks_on()

    # Add colorbar outside with minimal pad
    cbar_ax = ax_baz.inset_axes([1.01, 0., 0.015, 1])
    cb = plt.colorbar(scatter, cax=cbar_ax, pad=0.01)
    cb.set_label("CC-Coefficient", fontsize=font-2)
    cb.set_ticks([0, 0.5, 1])


def _plot_velocity_panel(ax_vel, ax_cbar, sd, velocity_results, wave_type,
                        vmax, cc_threshold, minors, max_time=None):
    """Plot velocity estimation panel"""
    font = 12
    
    # Check if we have data
    if len(velocity_results.get('time', [])) == 0:
        ax_vel.text(0.5, 0.5, "No velocity data available",
                   ha='center', va='center', transform=ax_vel.transAxes)
        return
    
    # Prepare mask
    if cc_threshold is not None:
        mask = np.array(velocity_results['ccoef']) > cc_threshold
    else:
        mask = np.array(velocity_results['ccoef']) >= 0
    
    if not np.any(mask):
        ax_vel.text(0.5, 0.5, "No velocity data after filtering",
                   ha='center', va='center', transform=ax_vel.transAxes)
        return
    
    # Plot velocities
    cmap = plt.get_cmap("viridis", 10)
    scatter = ax_vel.scatter(velocity_results['time'][mask],
                           velocity_results['velocity'][mask],
                           c=velocity_results['ccoef'][mask],
                           cmap=cmap, s=70, alpha=1.0,
                           vmin=0, vmax=1, edgecolors="k", lw=1, zorder=2)
    
    # Add error bars
    if 'terr' in velocity_results:
        ax_vel.errorbar(velocity_results['time'][mask],
                       velocity_results['velocity'][mask],
                       xerr=velocity_results['terr'][mask],
                       color='black', alpha=0.4, ls='none', zorder=1)
    
    # Configure axis
    ax_vel.set_ylabel("Velocity (m/s)", fontsize=font)
    ax_vel.set_xlabel("Time (s)", fontsize=font)
    ax_vel.set_ylim(bottom=0)
    ax_vel.set_xlim(left=0)
    if max_time is not None:
        ax_vel.set_xlim(right=max_time)
    if vmax is not None:
        ax_vel.set_ylim(top=vmax)
    ax_vel.grid(True, which='both', ls='--', alpha=0.3)
    
    if minors:
        ax_vel.minorticks_on()
    
    # Add colorbar outside with minimal pad
    if ax_cbar is None:
        cbar_ax = ax_vel.inset_axes([1.01, 0., 0.015, 1])
    else:
        cbar_ax = ax_cbar
    cb = plt.colorbar(scatter, cax=cbar_ax, pad=0.01)
    cb.set_label("CC-Coefficient", fontsize=font-2)
    cb.set_ticks([0, 0.5, 1])


def _create_map_subplot(fig, gridspec, projection, station_coords=None, event_info=None):
    """Create map subplot with appropriate projection"""
    try:
        import cartopy.crs as ccrs
        import numpy as np
        
        if projection == 'orthographic':
            # Calculate optimal center point
            center_lon = station_coords.get('longitude', 0) if station_coords else 0
            center_lat = station_coords.get('latitude', 0) if station_coords else 0
            
            # Ensure coordinates are valid numbers
            if not (np.isfinite(center_lon) and np.isfinite(center_lat)):
                center_lon, center_lat = 0, 0
            
            if event_info and 'latitude' in event_info and 'longitude' in event_info:
                event_lat = event_info['latitude']
                event_lon = event_info['longitude']
                
                if not (np.isfinite(event_lat) and np.isfinite(event_lon)):
                    proj = ccrs.Orthographic(center_lon, center_lat)
                    ax = fig.add_subplot(gridspec, projection=proj)
                    # Fix for cartopy/matplotlib compatibility
                    if not hasattr(ax, '_autoscaleXon'):
                        try:
                            ax._autoscaleXon = ax.get_autoscalex_on()
                        except:
                            ax._autoscaleXon = True
                    if not hasattr(ax, '_autoscaleYon'):
                        try:
                            ax._autoscaleYon = ax.get_autoscaley_on()
                        except:
                            ax._autoscaleYon = True
                    return ax
                
                # Normalize longitudes
                event_lon = ((event_lon + 180) % 360) - 180
                station_lon = ((center_lon + 180) % 360) - 180
                
                # Convert to radians
                lat1, lon1 = np.radians(center_lat), np.radians(station_lon)
                lat2, lon2 = np.radians(event_lat), np.radians(event_lon)
                
                try:
                    # Calculate midpoint
                    Bx = np.cos(lat2) * np.cos(lon2 - lon1)
                    By = np.cos(lat2) * np.sin(lon2 - lon1)
                    center_lat = np.degrees(np.arctan2(np.sin(lat1) + np.sin(lat2),
                                                      np.sqrt((np.cos(lat1) + Bx)**2 + By**2)))
                    dlon = lon2 - lon1
                    if abs(dlon) > np.pi:
                        dlon = -(2*np.pi - abs(dlon)) * np.sign(dlon)
                    center_lon = np.degrees(lon1 + dlon/2)
                    center_lon = ((center_lon + 180) % 360) - 180
                except:
                    center_lat = (center_lat + event_lat) / 2
                    center_lon = (station_lon + event_lon) / 2
                
                if not (np.isfinite(center_lon) and np.isfinite(center_lat)):
                    center_lon, center_lat = 0, 0
                
                proj = ccrs.Orthographic(center_lon, center_lat)
                ax = fig.add_subplot(gridspec, projection=proj)
            else:
                proj = ccrs.Orthographic(center_lon, center_lat)
                ax = fig.add_subplot(gridspec, projection=proj)
            
            # Fix for cartopy/matplotlib compatibility
            if not hasattr(ax, '_autoscaleXon'):
                try:
                    ax._autoscaleXon = ax.get_autoscalex_on()
                except:
                    ax._autoscaleXon = True
            if not hasattr(ax, '_autoscaleYon'):
                try:
                    ax._autoscaleYon = ax.get_autoscaley_on()
                except:
                    ax._autoscaleYon = True
            return ax
        else:
            ax = fig.add_subplot(gridspec, projection=ccrs.PlateCarree())
            # Fix for cartopy/matplotlib compatibility
            if not hasattr(ax, '_autoscaleXon'):
                try:
                    ax._autoscaleXon = ax.get_autoscalex_on()
                except:
                    ax._autoscaleXon = True
            if not hasattr(ax, '_autoscaleYon'):
                try:
                    ax._autoscaleYon = ax.get_autoscaley_on()
                except:
                    ax._autoscaleYon = True
            return ax
    except ImportError:
        return fig.add_subplot(gridspec)


def _plot_spherical_map_backazimuth(ax, event_info, baz_estimates, station_lat, station_lon,
                                   projection='orthographic'):
    """Plot spherical map with backazimuth information"""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except ImportError:
        use_cartopy = False
    
    # Fix for cartopy/matplotlib compatibility issue with _autoscaleXon/_autoscaleYon
    # These attributes were removed in newer matplotlib versions
    if use_cartopy and hasattr(ax, 'get_autoscalex_on'):
        if not hasattr(ax, '_autoscaleXon'):
            try:
                ax._autoscaleXon = ax.get_autoscalex_on()
            except:
                ax._autoscaleXon = True
        if not hasattr(ax, '_autoscaleYon'):
            try:
                ax._autoscaleYon = ax.get_autoscaley_on()
            except:
                ax._autoscaleYon = True
    
    # Set up map features
    if use_cartopy:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
        
        if projection == 'orthographic':
            ax.gridlines(alpha=0.5)
            ax.set_global()
        
        transform = ccrs.PlateCarree()
    else:
        if projection == 'orthographic':
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.axis('off')
        transform = None
    
    # Normalize station longitude
    station_lon_norm = ((station_lon + 180) % 360) - 180
    
    # Plot station
    if use_cartopy:
        ax.scatter(station_lon_norm, station_lat, s=100, marker='^', color='red',
               label='Station', edgecolors='black', linewidths=1.5,
               transform=transform, zorder=5)
    else:
        if projection == 'orthographic':
            x_st, y_st = _project_to_sphere(station_lat, station_lon_norm, station_lat, station_lon_norm)
            ax.scatter(x_st, y_st, s=100, marker='^', color='red',
                   label='Station', edgecolors='black', linewidths=1.5, zorder=5)
        else:
            ax.scatter(station_lon_norm, station_lat, s=100, marker='^', color='red',
                   label='Station', edgecolors='black', linewidths=1.5, zorder=5)
    
    # Plot event if available
    if event_info and 'latitude' in event_info and 'longitude' in event_info:
        event_lat = event_info['latitude']
        event_lon = ((event_info['longitude'] + 180) % 360) - 180
        
        if use_cartopy:
            ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=15,
                   label='Event', markeredgecolor='black', markeredgewidth=1.5,
                   transform=transform, zorder=5)
        else:
            if projection == 'orthographic':
                x_ev, y_ev = _project_to_sphere(event_lat, event_lon, station_lat, station_lon_norm)
                ax.plot(x_ev, y_ev, marker='*', color='yellow', markersize=15,
                       label='Event', markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            else:
                ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=15,
                       label='Event', markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    # Plot great circles
    colors = {'love': 'orange', 'rayleigh': 'green', 'tangent': 'purple'}
    
    # Theoretical great circle
    if event_info and 'backazimuth' in event_info:
        theo_baz = event_info['backazimuth']
        try:
            if use_cartopy:
                gc_lons, gc_lats = _great_circle_path_2d(station_lat, station_lon_norm, theo_baz)
                ax.plot(gc_lons, gc_lats, color='black', linewidth=4,
                       linestyle=':', label=f'Theoretical BAz: {theo_baz:.0f}°', alpha=0.9,
                       transform=transform, zorder=3)
        except Exception as e:
            pass
    
    # Estimated great circles
    for wave_type, baz_deg in baz_estimates.items():
        try:
            color = colors.get(wave_type, 'purple')
            if use_cartopy:
                gc_lons, gc_lats = _great_circle_path_2d(station_lat, station_lon_norm, baz_deg)
                ax.plot(gc_lons, gc_lats, color=color, linewidth=2.5,
                       label=f'{wave_type.upper()} BAz: {baz_deg:.0f}°', alpha=0.8,
                       transform=transform, zorder=4)
        except Exception as e:
            pass
    
    ax.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left', fontsize=11)


def _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=120, num_points=100):
    """Calculate great circle path points"""
    azimuth = azimuth % 360
    
    # Convert to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    azimuth_rad = np.radians(azimuth)
    
    distances = np.linspace(0.0, np.radians(max_distance_deg), num_points)
    
    # Calculate great circle points
    lats_rad = np.arcsin(
        np.sin(lat0_rad) * np.cos(distances) +
        np.cos(lat0_rad) * np.sin(distances) * np.cos(azimuth_rad)
    )
    
    dlon = np.arctan2(
        np.sin(azimuth_rad) * np.sin(distances) * np.cos(lat0_rad),
        np.cos(distances) - np.sin(lat0_rad) * np.sin(lats_rad)
    )
    
    lons_rad = lon0_rad + dlon
    
    # Convert back to degrees
    lats_deg = np.degrees(lats_rad)
    lons_deg = np.degrees(lons_rad)
    
    # Unwrap longitudes to keep paths continuous when crossing dateline
    # This allows great circles to extend beyond ±180 for proper plotting
    lons_deg = np.degrees(np.unwrap(np.radians(lons_deg)))
    
    # Force exact match for first point
    lats_deg[0] = lat0
    lons_deg[0] = lon0
    
    return lons_deg, lats_deg


def _project_to_sphere(lat, lon, center_lat, center_lon):
    """Project lat/lon to sphere coordinates for orthographic-like view"""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    cos_c = (np.sin(center_lat_rad) * np.sin(lat_rad) +
             np.cos(center_lat_rad) * np.cos(lat_rad) * np.cos(lon_rad - center_lon_rad))
    
    x = np.full_like(lat, np.nan, dtype=float)
    y = np.full_like(lat, np.nan, dtype=float)
    
    visible = cos_c >= 0
    
    if np.any(visible):
        x[visible] = np.cos(lat_rad[visible]) * np.sin(lon_rad[visible] - center_lon_rad)
        y[visible] = (np.cos(center_lat_rad) * np.sin(lat_rad[visible]) -
                     np.sin(center_lat_rad) * np.cos(lat_rad[visible]) *
                     np.cos(lon_rad[visible] - center_lon_rad))
    
    return x, y

