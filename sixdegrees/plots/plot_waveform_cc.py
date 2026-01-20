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


def plot_waveform_cc(rot0: Optional[Stream]=None, acc0: Optional[Stream]=None, sd_object: Optional['sixdegrees']=None, 
                    baz: Optional[float]=None, fmin: Optional[float]=None, fmax: Optional[float]=None, 
                    wave_type: str="both", pol_dict: Union[None, Dict]=None, distance: Union[None, float]=None, 
                    runit: str=r"rad/s", tunit: str=r"m/s$^2$", twin_sec: int=5, twin_overlap: float=0.5, 
                    unitscale: str="nano", t1:UTCDateTime=None, t2:UTCDateTime=None) -> plt.Figure:

    """
    Plot waveform cross-correlation.

    Parameters:
    -----------
    rot0 : Stream, optional
        Rotation rate stream. Required if sd_object is not provided.
    acc0 : Stream, optional
        Acceleration stream. Required if sd_object is not provided.
    sd_object : sixdegrees, optional
        sixdegrees object. If provided, will extract rot0 and acc0 from sd_object.get_stream(),
        and extract baz, distance, fmin, fmax from the object if not explicitly provided.
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
    runit : str
        Unit for rotation rate
    tunit : str
        Unit for acceleration
    twin_sec : int
        Time window length
    twin_overlap : float
        Time window overlap
    unitscale : str
        Unit scale: "nano" or "micro"
    t1 : UTCDateTime or None
        Start time to trim data
    t2 : UTCDateTime or None
        End time to trim data
    
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

    # Extract streams and parameters from sd_object if provided
    if sd_object is not None:
        # Extract streams if not provided
        if rot0 is None:
            rot0 = sd_object.get_stream("rotation")
        if acc0 is None:
            acc0 = sd_object.get_stream("translation")
        
        # Extract parameters if not explicitly provided (provided parameters have higher priority)
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
    
    # Validate that we have required streams
    if rot0 is None or acc0 is None:
        raise ValueError("Either provide rot0 and acc0 directly, or provide sd_object (sixdegrees object)")
    
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

    rot = rot0.copy()
    acc = acc0.copy()

    if t1 is not None and not isinstance(t1, UTCDateTime):
        t1 = UTCDateTime(t1)
    if t2 is not None and not isinstance(t2, UTCDateTime):
        t2 = UTCDateTime(t2)

    # trim data
    if t1 is not None and t2 is not None:
        rot = rot.trim(t1, t2)
        acc = acc.trim(t1, t2)

    # get sampling rate
    dt = rot[0].stats.delta

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
    
    # define scaling factors
    mu = r"$\mu$"
    if unitscale == "nano":
        acc_scaling, acc_unit = 1e6, f"{mu}{tunit}"
        rot_scaling, rot_unit = 1e9, f"n{runit}"
    elif unitscale == "micro":
        acc_scaling, acc_unit = 1e3, f"m{tunit}"
        rot_scaling, rot_unit = 1e6, f"{mu}{runit}"
    else:
        raise ValueError(f"Invalid unitscale: {unitscale}. Valid options are: 'nano', 'micro'")

    # define linewidth and fontsize
    lw = 1
    font = 12

    cc = []
    cc_all = []

    # Get vertical and rotated components
    if wave_type == "both" or wave_type == "love":
        # get vertical component
        rot_z = rot.select(channel="*Z")[0].data
        # rotate components
        acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data, acc.select(channel="*E")[0].data, baz)
        # apply scaling
        rot_z *= rot_scaling
        acc_r *= acc_scaling
        acc_t *= acc_scaling
        # calculate max values
        acc_r_max = max([abs(min(acc_r)), abs(max(acc_r))])
        acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])
        rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
        # update polarity
        rot0, acc0, rot0_lbl, acc0_lbl = pol['JZ']*rot_z, pol['HT']*acc_t, f"{pol['JZ']}x ROT-Z", f"{pol['HT']}x ACC-T"
        # calculate cross-correlation
        tt0, cc0 = _cross_correlation_windows(rot0, acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        cc.append(cc0)
        cc_all.append(max(correlate(rot0, acc0, 0, demean=True, normalize='naive', method='fft')))

    if wave_type == "both" or wave_type == "rayleigh":
        # get vertical component
        acc_z = acc.select(channel="*Z")[0].data
        # rotate components
        rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data, rot.select(channel="*E")[0].data, baz)
        # apply scaling
        acc_z *= acc_scaling
        rot_r *= rot_scaling
        rot_t *= rot_scaling
        # calculate max values
        acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
        rot_r_max = max([abs(min(rot_r)), abs(max(rot_r))])
        rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])
        # update polarity
        rot1, acc1, rot1_lbl, acc1_lbl = pol['JT']*rot_t, pol['HZ']*acc_z, f"{pol['JT']}x ROT-T", f"{pol['HZ']}x ACC-Z"
        # calculate cross-correlation
        tt1, cc1 = _cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
        cc.append(cc1)
        cc_all.append(max(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='fft')))

    # rot2, acc2, rot2_lbl, acc2_lbl = pol['JZ']*rot_z, pol['HR']*acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
    # tt2, cc2 = _cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

    # Create colormap with boundaries at 0.2 steps from -1 to 1
    cmap = plt.get_cmap("coolwarm", 16)
    boundaries = np.arange(-1.0, 1.2, 0.2)  # Creates: [-1.0, -0.8, -0.6, ..., 0.8, 1.0]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(boundaries, cmap.N)
    # Only label specific values on colorbar
    cbar_ticks = [-1, -0.6, 0, 0.6, 1]

    if wave_type == "love":
        ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
        ax00 = ax[0].twinx()
        ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
        ax01 = ax[0].twinx()
        cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, norm=norm, label="")

        ax[0].set_ylim(-rot_z_max, rot_z_max)
        ax00.set_ylim(-acc_t_max, acc_t_max)
        ax01.set_ylim(-1, 1)
        ax01.yaxis.set_visible(False)

        twinaxs = [ax00]
        cms = [cm1]

    elif wave_type == "rayleigh":
        ax[0].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
        ax11 = ax[0].twinx()
        ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
        ax12 = ax[0].twinx()
        cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, norm=norm, label="")

        ax[0].set_ylim(-rot_t_max, rot_t_max)
        ax11.set_ylim(-acc_z_max, acc_z_max)
        ax12.set_ylim(-1, 1)
        ax12.yaxis.set_visible(False)

        twinaxs = [ax11]
        cms = [cm2]

    elif wave_type == "both":
        # First subplot
        ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
        ax00 = ax[0].twinx()
        ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
        ax01 = ax[0].twinx()
        cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, norm=norm, label="")

        ax[0].set_ylim(-rot_z_max, rot_z_max)
        ax00.set_ylim(-acc_t_max, acc_t_max)
        ax01.set_ylim(-1, 1)
        ax01.yaxis.set_visible(False)

        # Second subplot
        ax[1].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
        ax11 = ax[1].twinx()
        ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
        ax12 = ax[1].twinx()
        cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, norm=norm, label="")

        ax[1].set_ylim(-rot_t_max, rot_t_max)
        ax11.set_ylim(-acc_z_max, acc_z_max)
        ax12.set_ylim(-1, 1)
        ax12.yaxis.set_visible(False)

        twinaxs = [ax00, ax11]
        cms = [cm1, cm2]

    # Sync twinx axes
    ax[0].set_yticks(linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-1], len(ax[0].get_yticks())))
    twinaxs[0].set_yticks(linspace(twinaxs[0].get_yticks()[0], twinaxs[0].get_yticks()[-1], len(ax[0].get_yticks())))

    if wave_type == "both":
        ax[1].set_yticks(linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-1], len(ax[1].get_yticks())))
        twinaxs[1].set_yticks(linspace(twinaxs[1].get_yticks()[0], twinaxs[1].get_yticks()[-1], len(ax[1].get_yticks())))

    # Set labels and grid
    rot_rate_label = r"$\dot{\Omega}$"
    if wave_type == "both":
        names = ["love", "rayleigh"]
    else:
        names = [wave_type]

    for i, wt in zip(range(Nrow), names):
        ax[i].legend(loc=1, ncols=4)
        ax[i].grid(which="both", alpha=0.5)
        ax[i].set_ylabel(f"{rot_rate_label} ({rot_unit})", fontsize=font)
        ax[i].text(0.05, 0.9,
                    f"{wt.capitalize()}: CC={cc_all[i]:.2f}",
                    ha='left', va='top', 
                    transform=ax[i].transAxes, 
                    fontsize=font-1,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
                    )

    for _ax in twinaxs:
        _ax.legend(loc=1, bbox_to_anchor=(1, 0.9))
        _ax.set_ylabel(f"$a$ ({acc_unit})", fontsize=font)

    # Add colorbar
    cax = ax[Nrow-1].inset_axes([0.8, -0.25, 0.2, 0.1], transform=ax[Nrow-1].transAxes)
    
    # Create a ScalarMappable for the colorbar (norm already defined above)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, location="bottom", orientation="horizontal", 
                        boundaries=boundaries, ticks=cbar_ticks)

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
    tbeg = acc[0].stats.starttime
    title = f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC"
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