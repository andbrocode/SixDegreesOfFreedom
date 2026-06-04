"""
Functions for plotting waveform cross-correlation analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, Tuple
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from obspy import Stream
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.rotate import rotate_ne_rt
from obspy.core.utcdatetime import UTCDateTime


def plot_waveform_cc(rot: Optional[Stream]=None, acc: Optional[Stream]=None, sd_object: Optional['sixdegrees']=None, 
                    baz: Optional[float]=None, fmin: Optional[float]=None, fmax: Optional[float]=None, 
                    wave_type: str="both", pol_dict: Union[None, Dict]=None, distance: Union[None, float]=None, 
                    runit: Optional[str]=None, tunit: Optional[str]=None, twin_sec: int=5, twin_overlap: float=0.5, 
                    unitscale: str="nano", data_type: str="acceleration", t1:UTCDateTime=None, t2:UTCDateTime=None,
                    scaled: bool=False) -> plt.Figure:

    """
    Plot waveform cross-correlation.

    Parameters:
    -----------
    rot : Stream, optional
        Rotation rate/rotation stream. Required if sd_object is not provided.
    acc : Stream, optional
        Acceleration/velocity stream. Required if sd_object is not provided.
    sd_object : sixdegrees, optional
        sixdegrees object. If provided, will extract rot and acc from sd_object.get_stream(),
        and extract baz, distance, fmin, fmax, runit, tunit from the object if not explicitly provided.
    baz : float, optional
        Backazimuth. If not provided and sd_object is given, will try to extract from sd_object.
    fmin : float or None, optional
        Minimum frequency for bandpass filter. If not provided and sd_object is given, will use sd_object.fmin.
    fmax : float or None, optional
        Maximum frequency for bandpass filter. If not provided and sd_object is given, will use sd_object.fmax.
    wave_type : str
        Wave type: "love", "rayleigh", or "both"
    pol_dict : dict or None
        Polarity dictionary
    distance : float or None, optional
        Distance. If not provided and sd_object is given, will try to extract from sd_object.event_info.
    runit : str, optional
        Unit for rotation rate/rotation. If None, will be determined from sd_object or data_type.
    tunit : str, optional
        Unit for acceleration/velocity. If None, will be determined from sd_object or data_type.
    twin_sec : int
        Time window length
    twin_overlap : float
        Time window overlap
    unitscale : str
        Unit scale: "nano" or "micro"
    data_type : str
        Type of data: "acceleration" (rotation rate and acceleration) or "velocity" (rotation and velocity).
        Default is "acceleration". This determines default units and labels if not explicitly provided.
    t1 : UTCDateTime or None
        Start time to trim data
    t2 : UTCDateTime or None
        End time to trim data
    scaled : bool
        If True, use the same y-limits and tick positions on the translation (left) and
        rotation (right) axes. Default is False (independent axis limits).
    
    Returns:
    --------
    fig : plt.Figure
        Figure object

    """
    import matplotlib.pyplot as plt
    from obspy.signal.cross_correlation import correlate
    from obspy.signal.rotate import rotate_ne_rt
    from numpy import linspace, ones, array
    from matplotlib.ticker import AutoMinorLocator
    from obspy.core.utcdatetime import UTCDateTime
    from matplotlib.colors import BoundaryNorm

    # Extract streams and parameters from sd_object if provided
    if sd_object is not None:
        # Extract streams if not provided
        if rot is None:
            rot = sd_object.get_stream("rotation").copy()
        if acc is None:
            acc = sd_object.get_stream("translation").copy()
        
        # Extract parameters if not explicitly provided (provided parameters have higher priority)
        if fmin is None and hasattr(sd_object, 'fmin') and sd_object.fmin is not None:
            fmin = sd_object.fmin
        if fmax is None and hasattr(sd_object, 'fmax') and sd_object.fmax is not None:
            fmax = sd_object.fmax
        
        # Extract units from sd_object if not provided
        if runit is None and hasattr(sd_object, 'runit'):
            runit = sd_object.runit
        if tunit is None and hasattr(sd_object, 'tunit'):
            tunit = sd_object.tunit
        
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
                    # Try to get baz for the current wave_type
                    baz_val = sd_object.baz_estimated.get(wave_type.lower(), None)
                    if baz_val is None:
                        # Try any available baz value
                        baz_val = next(iter(sd_object.baz_estimated.values()), None)
                    baz = baz_val
                else:
                    baz = sd_object.baz_estimated
            # Try event_info
            elif hasattr(sd_object, 'event_info') and sd_object.event_info is not None:
                if isinstance(sd_object.event_info, dict) and 'backazimuth' in sd_object.event_info:
                    baz = sd_object.event_info['backazimuth']
        
        # Extract distance if not provided
        if distance is None:
            # Try event_info
            if hasattr(sd_object, 'event_info') and sd_object.event_info is not None:
                if isinstance(sd_object.event_info, dict):
                    if 'distance_km' in sd_object.event_info:
                        distance = sd_object.event_info['distance_km']
    
    # Determine units and labels based on data_type if not explicitly provided
    if data_type.lower() == "velocity":
        # Velocity mode: rotation (rad) and velocity (m/s)
        if runit is None:
            runit = r"rad"
        if tunit is None:
            tunit = r"m/s"
        rot_label_symbol = "Angle"  # Rotation angle
        tra_label_symbol = "Velocity"  # Velocity
    else:
        # Acceleration mode (default): rotation rate (rad/s) and acceleration (m/s²)
        if runit is None:
            runit = r"rad/s"
        if tunit is None:
            tunit = r"m/s$^2$"
        rot_label_symbol = "Rotation rate"  # Rotation rate
        tra_label_symbol = "Acceleration"  # Acceleration
    
    # Validate that we have required streams
    if rot is None or acc is None:
        raise ValueError("Either provide rot and acc directly, or provide sd_object (sixdegrees object)")
    
    # Validate that we have baz
    if baz is None:
        raise ValueError("baz (backazimuth) must be provided either directly or extractable from sd")

    def _cross_correlation_windows(arr1: array, arr2: array, dt: float, Twin: float, overlap: float=0, lag: int=0, demean: bool=True, plot: bool=False) -> Tuple[array, array]:

        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import arange, array, roll

        N = len(arr1)
        n_interval = int(Twin/dt)
        n_overlap = int(overlap*Twin/dt)

        # time = arange(0, N*dt, dt)

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

    _rot = rot.copy()
    _acc = acc.copy()

    if t1 is not None and not isinstance(t1, UTCDateTime):
        t1 = UTCDateTime(t1)
    if t2 is not None and not isinstance(t2, UTCDateTime):
        t2 = UTCDateTime(t2)

    # trim data
    if t1 is not None and t2 is not None:
        _rot = _rot.trim(t1, t2)
        _acc = _acc.trim(t1, t2)

    # get sampling rate
    dt = _rot[0].stats.delta

    # define polarity
    pol = {"HZ":1,"HN":1,"HE":1,"HR":1,"HT":1,
            "JZ":1,"JN":1,"JE":1,"JR":1,"JT":1,
            }
    
    # update polarity dictionary
    if pol_dict is not None:
        for k in pol_dict.keys():
            pol[k] = pol_dict[k]

    # Change number of rows based on wave type
    if wave_type == "both":
        Nrow, Ncol = 2, 1
        fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5*Nrow), sharex=True)
        plt.subplots_adjust(hspace=0.1)
        ax = axes  # axes is already an array for multiple subplots
    else:
        Nrow, Ncol = 1, 1
        fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5), sharex=True)
        ax = [axes]  # wrap single axes in list for consistent indexing
    
    # Define scaling factors and units based on data_type and unitscale
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

    # define linewidth and fontsize
    lw = 1
    font = 12
    rot_color = "tab:red"
    tra_color = "black"

    cc = []
    cc_all = []

    # Get vertical and rotated components
    if wave_type == "both" or wave_type == "love":
        # get vertical component
        rot_z = _rot.select(channel="*Z")[0].data
        # rotate components
        acc_r, acc_t = rotate_ne_rt(_acc.select(channel="*N")[0].data, _acc.select(channel="*E")[0].data, baz)
        # apply scaling
        rot_z *= rot_scaling
        acc_r *= acc_scaling
        acc_t *= acc_scaling
        # calculate max values
        acc_r_max = max([abs(min(acc_r)), abs(max(acc_r))])
        acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])
        rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
        # update polarity and labels
        rot_prefix = "ROT" if data_type.lower() == "velocity" else "ROT"
        tra_prefix = "VEL" if data_type.lower() == "velocity" else "ACC"
        rot0, acc0, rot0_lbl, acc0_lbl = pol['JZ']*rot_z, pol['HT']*acc_t, f"{pol['JZ']}x {rot_prefix}-Z", f"{pol['HT']}x {tra_prefix}-T"
        # calculate cross-correlation
        tt0, cc0 = _cross_correlation_windows(rot0, acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        cc.append(cc0)
        cc_all.append(max(correlate(rot0, acc0, 0, demean=True, normalize='naive', method='fft')))

    if wave_type == "both" or wave_type == "rayleigh":
        # get vertical component
        acc_z = _acc.select(channel="*Z")[0].data
        # rotate components
        rot_r, rot_t = rotate_ne_rt(_rot.select(channel="*N")[0].data, _rot.select(channel="*E")[0].data, baz)
        # apply scaling
        acc_z *= acc_scaling
        rot_r *= rot_scaling
        rot_t *= rot_scaling
        # calculate max values
        acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
        rot_r_max = max([abs(min(rot_r)), abs(max(rot_r))])
        rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])
        # update polarity and labels
        rot_prefix = "ROT" if data_type.lower() == "velocity" else "ROT"
        tra_prefix = "VEL" if data_type.lower() == "velocity" else "ACC"
        rot1, acc1, rot1_lbl, acc1_lbl = pol['JT']*rot_t, pol['HZ']*acc_z, f"{pol['JT']}x {rot_prefix}-T", f"{pol['HZ']}x {tra_prefix}-Z"
        # calculate cross-correlation
        tt1, cc1 = _cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        cc.append(cc1)
        cc_all.append(max(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='fft')))

    # rot2, acc2, rot2_lbl, acc2_lbl = pol['JZ']*rot_z, pol['HR']*acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
    # tt2, cc2 = _cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

    # Discrete colormap spanning -1 to 1 in 0.1 steps (20 bins, 21 edges)
    boundaries = np.round(np.arange(-1.0, 1.0 + 0.1, 0.1), 2)  # [-1.0, -0.9, ..., 0.9, 1.0]
    cmap = plt.get_cmap("coolwarm", len(boundaries) - 1)  # one color per bin -> 20 bins
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    # Show only these tick labels on the colorbar
    cbar_ticks = [-1.0, -0.7, -0.3, 0.0, 0.3, 0.7, 1.0]

    def _plot_cc_panel(
        ax,
        times_tra,
        tra_data,
        tra_lbl,
        tra_max,
        times_rot,
        rot_data,
        rot_lbl,
        rot_max,
        tt,
        cc_vals,
    ):
        """Translation on left axis; rotation on right (red) axis."""
        ax.plot(times_tra, tra_data, label=tra_lbl, color=tra_color, lw=lw)
        ax_rot = ax.twinx()
        ax_rot.plot(times_rot, rot_data, label=rot_lbl, color=rot_color, lw=lw, zorder=3)
        ax_cc = ax.twinx()
        cm = ax_cc.scatter(
            tt, ones(len(tt)) * -0.9, c=cc_vals, alpha=abs(cc_vals),
            cmap=cmap, norm=norm, label="",
        )

        if scaled:
            panel_max = max(tra_max, rot_max)
            if panel_max <= 0:
                panel_max = 1.0
            ylim = (-panel_max, panel_max)
            ax.set_ylim(*ylim)
            ax_rot.set_ylim(*ylim)
            ticks = linspace(-panel_max, panel_max, 5)
            ax.set_yticks(ticks)
            ax_rot.set_yticks(ticks)
        else:
            ax.set_ylim(-tra_max, tra_max)
            ax_rot.set_ylim(-rot_max, rot_max)

        ax_cc.set_ylim(-1, 1)
        ax_cc.yaxis.set_visible(False)

        ax_rot.spines["right"].set_color(rot_color)
        ax_rot.tick_params(axis="y", colors=rot_color)
        ax_rot.yaxis.label.set_color(rot_color)

        return ax_rot, cm

    twinaxs = []
    cms = []

    if wave_type == "love":
        ax_rot, cm1 = _plot_cc_panel(
            ax[0],
            _acc.select(channel="*N")[0].times(), acc0, acc0_lbl, acc_t_max,
            _rot.select(channel="*Z")[0].times(), rot0, rot0_lbl, rot_z_max,
            tt0, cc0,
        )
        twinaxs = [ax_rot]
        cms = [cm1]

    elif wave_type == "rayleigh":
        ax_rot, cm2 = _plot_cc_panel(
            ax[0],
            _acc.select(channel="*Z")[0].times(), acc1, acc1_lbl, acc_z_max,
            _rot.select(channel="*N")[0].times(), rot1, rot1_lbl, rot_t_max,
            tt1, cc1,
        )
        twinaxs = [ax_rot]
        cms = [cm2]

    elif wave_type == "both":
        ax_rot0, cm1 = _plot_cc_panel(
            ax[0],
            _acc.select(channel="*N")[0].times(), acc0, acc0_lbl, acc_t_max,
            _rot.select(channel="*Z")[0].times(), rot0, rot0_lbl, rot_z_max,
            tt0, cc0,
        )
        ax_rot1, cm2 = _plot_cc_panel(
            ax[1],
            _acc.select(channel="*Z")[0].times(), acc1, acc1_lbl, acc_z_max,
            _rot.select(channel="*N")[0].times(), rot1, rot1_lbl, rot_t_max,
            tt1, cc1,
        )
        twinaxs = [ax_rot0, ax_rot1]
        cms = [cm1, cm2]

    # Set labels and grid
    if wave_type == "both":
        names = ["love", "rayleigh"]
    else:
        names = [wave_type]

    for i, wt in zip(range(Nrow), names):
        ax[i].legend(loc=1, ncols=4)
        ax[i].grid(which="both", alpha=0.5)
        ax[i].set_ylabel(f"{tra_label_symbol} ({acc_unit})", fontsize=font)
        ax[i].text(
            0.05, 0.9,
            f"{wt.capitalize()}: CC={cc_all[i]:.2f}",
            ha='left', va='top', 
            transform=ax[i].transAxes, 
            fontsize=font-1,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
        )

    for _ax in twinaxs:
        _ax.legend(loc=1, bbox_to_anchor=(1, 0.9))
        _ax.set_ylabel(f"{rot_label_symbol} ({rot_unit})", fontsize=font, color=rot_color)

    # Add colorbar
    cax = ax[Nrow-1].inset_axes([0.8, -0.25, 0.2, 0.1], transform=ax[Nrow-1].transAxes)
    
    # Create a ScalarMappable for the colorbar (norm already defined above)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, location="bottom", orientation="horizontal", 
                        boundaries=boundaries, ticks=cbar_ticks)
    cbar.ax.set_xticklabels([f"{t:g}" for t in cbar_ticks])
    cbar.ax.tick_params(labelsize=font-2)

    cbar.set_label("Cross-Correlation Value", fontsize=font-1, loc="left", labelpad=-55, color="k")

    # Set limits for scatter plots
    for cm in cms:
        cm.set_clim(-1, 1)

    # set subticks for x axis
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator())

    # Add xlabel to bottom subplot
    ax[Nrow-1].set_xlabel("Time (s)", fontsize=font)

    # Set title
    tbeg = _acc[0].stats.starttime
    title = f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC"
    if wave_type != "both":
        title += f" | {wave_type}"
    if fmin is not None and fmax is not None:
        title += f" | f = {fmin}-{fmax} Hz"
    elif fmin is not None:
        title += f" | f ≥ {fmin} Hz"
    elif fmax is not None:
        title += f" | f ≤ {fmax} Hz"
    if baz is not None:
        title += f"  |  BAz = {round(baz, 1)}°"
    if distance is not None:
        title += f"  |  ED = {round(distance, 0)} km"
    title += f"  |  T = {twin_sec}s ({int(100*twin_overlap)}%)"
    ax[0].set_title(title)

    # plt.show()
    return fig