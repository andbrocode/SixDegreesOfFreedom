"""
Functions for plotting backazimuth estimation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from typing import Dict
from matplotlib.gridspec import GridSpec
from obspy.signal.rotate import rotate_ne_rt
from numpy import arange, histogram, average, cov

def plot_backazimuth_results(sd, baz_results: Dict, wave_type: str='love', 
                            baz_theo: float=None, baz_theo_margin: float=10, unitscale: str='nano',
                            cc_threshold: float=None, minors: bool=True, cc_method: str='mid',
                            terr: bool=True) -> plt.Figure:
    """
    Plot backazimuth estimation results
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object
    baz_results : Dict
        Dictionary containing backazimuth results
    wave_type : str
        Type of wave ('love' or 'rayleigh')
    baz_theo : float, optional
        Theoretical backazimuth in degrees
    baz_theo_margin : float, optional
        Margin around theoretical backazimuth in degrees
    cc_threshold : float, optional
        Minimum cross-correlation coefficient threshold
    minors : bool, optional
        Add minor ticks to axes if True
    cc_method : str
        Type of cc to choose ('mid' or 'max')
    unitscale : str
        Unit scale for rotation rate ('nano' or 'micro')
    terr : bool, optional
        Add error bars to backazimuth estimates if True

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """

    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(4, 8, figure=fig, hspace=0.2)
    
    # Create subplots
    ax_wave = fig.add_subplot(gs[0:2, :])  # Waveform panel
    ax_baz = fig.add_subplot(gs[2:3, :])  # Backazimuth panel
    ax_hist = fig.add_subplot(gs[2:3, 7:])  # Histogram panel
    ax_hist.set_axis_off()
    
    # Plot settings
    font = 12
    lw = 1.0
    if unitscale == 'nano':
        rot_scale, rot_unit = 1e9, f"n{sd.runit}"
        trans_scale, trans_unit = 1e6, f"{sd.mu}{sd.tunit}"
    elif unitscale == 'micro':
        rot_scale, rot_unit = 1e6, f"{sd.mu}{sd.runit}"
        trans_scale, trans_unit = 1e3, f"m{sd.tunit}"

    
    # Get streams and apply filtering if needed
    rot = sd.get_stream("rotation").copy()
    acc = sd.get_stream("translation").copy()

    # Get components
    if wave_type == "love":
        hn = acc.select(channel="*HN")[0].data
        he = acc.select(channel="*HE")[0].data
        jz = rot.select(channel="*JZ")[0].data
    elif wave_type == "rayleigh":
        hz = acc.select(channel="*HZ")[0].data
        je = rot.select(channel="*JE")[0].data
        jn = rot.select(channel="*JN")[0].data
    else:
        raise ValueError(f"Invalid wave_type: {wave_type}. Use 'love' or 'rayleigh'.")
    
    # Rotate to radial-transverse
    if baz_theo is not None:
        if wave_type == "love":
            hr, ht = rotate_ne_rt(hn, he, baz_theo)
        elif wave_type == "rayleigh":
            jr, jt = rotate_ne_rt(jn, je, baz_theo)
    else:
        print("No theoretical backazimuth provided")
        return

    # get times
    time = baz_results['twin_center']

    # select maximal or mid approach results
    if cc_method == 'mid':
        ccc = baz_results['cc_mid']
        baz = baz_results['baz_mid']
    elif cc_method == 'max':
        ccc = baz_results['cc_max']
        baz = baz_results['baz_max']
    
    # apply cc threshold if provided
    if cc_threshold is not None:
        mask = ccc > cc_threshold
        time = time[mask]
        baz = baz[mask]
        cc = ccc[mask]

    # Plot transverse components
    times = acc.select(channel="*HZ")[0].times()

    if wave_type == "love":

        # Plot translational data
        ax_wave.plot(times, ht*trans_scale, 'black', label=f"{sd.tra_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[-1][:-1]}T", lw=lw)
        ax_wave.set_ylim(-max(abs(ht*trans_scale)), max(abs(ht*trans_scale)))

        # Add rotational data on twin axis
        ax_wave2 = ax_wave.twinx()
        ax_wave2.plot(times, jz*rot_scale, 'darkred', label=f"{sd.rot_seed[0].split('.')[1]}.{sd.rot_seed[0].split('.')[-1][:-1]}Z", lw=lw)
        ax_wave2.set_ylim(-max(abs(jz*rot_scale)), max(abs(jz*rot_scale)))

    elif wave_type == "rayleigh":
        ax_wave.plot(times, hz*trans_scale, 'black', label=f"{sd.tra_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[-1][:-1]}Z", lw=lw)
        ax_wave.set_ylim(-max(abs(hz*trans_scale)), max(abs(hz*trans_scale)))

        # Add rotational data on twin axis
        ax_wave2 = ax_wave.twinx()
        ax_wave2.plot(times, jt*rot_scale, 'darkred', label=f"{sd.rot_seed[0].split('.')[1]}.{sd.rot_seed[0].split('.')[-1][:-1]}T", lw=lw)
        ax_wave2.set_ylim(-max(abs(jt*rot_scale)), max(abs(jt*rot_scale)))
        
    # Configure waveform axes
    # ax_wave.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
    ax_wave.legend(loc=1)
    ax_wave.set_ylabel(f"Acceleration ({trans_unit})", fontsize=font)
    ax_wave2.tick_params(axis='y', colors="darkred")
    ax_wave2.set_ylabel(f"Rotation rate ({rot_unit})", color="darkred", fontsize=font)
    ax_wave2.legend(loc=4)
    
    # Plot backazimuth estimates
    cmap = plt.get_cmap("viridis", 10)
    scatter = ax_baz.scatter(
        time,
        baz,
        c=cc,
        s=50,
        cmap=cmap,
        edgecolors="k",
        lw=1,
        vmin=0,
        vmax=1,
        zorder=2
    )
    
    if terr:
        ax_baz.errorbar(time, baz, xerr=baz_results['parameters']['baz_win_sec']/2, fmt='.', color='gray', alpha=0.6, zorder=0)
    
    # Configure backazimuth axis
    ax_baz.set_ylim(-5, 365)
    ax_baz.set_yticks(range(0, 360+60, 60))
    ax_baz.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
    ax_baz.set_ylabel(f"{wave_type.capitalize()} BAz (°)", fontsize=font)
    
    # Add theoretical backazimuth
    ax_baz.plot(
        [min(times), max(times)], 
        [baz_theo, baz_theo],
        color='k',
        ls='--',
        label='Theoretical BAz',
        zorder=1
    )

    ax_baz.fill_between([baz_theo-baz_theo_margin, baz_theo+baz_theo_margin],
                        [min(times), min(times)],
                        color='grey', alpha=0.5, zorder=1)

    # Compute statistics
    deltaa = 10
    angles1 = arange(0, 365, deltaa)

    # Compute histogram
    hist = histogram(
        baz,
        bins=len(angles1)-1,
        range=[min(angles1), max(angles1)], 
        weights=cc, 
        density=True
    )

    # get kde stats
    try:
        kde_stats = sd.get_kde_stats(baz, cc, _baz_steps=0.5, Ndegree=60, plot=False)
        # get max and std
        baz_max = kde_stats['baz_estimate']
        baz_std = kde_stats['kde_dev']
        print(f"baz_max = {baz_max}, baz_std = {baz_std}")
        got_kde = True
    except:
        got_kde = False

    # Add histogram
    # ax_hist2.plot(kernel_density(np.linspace(0, 360, 100)), np.linspace(0, 360, 100), 'k-', lw=2)
    ax_hist.hist(baz, bins=len(angles1)-1, range=[min(angles1), max(angles1)],
                    weights=cc, orientation="horizontal", density=True, color="grey")
    if got_kde:
        ax_hist.plot(kde_stats['kde_values'],
                    kde_stats['kde_angles'],
                    c="k",
                    lw=2,
                    label='KDE'
                    )
    ax_hist.set_ylim(-5, 365)
    ax_hist.invert_xaxis()
    ax_hist.set_axis_off()
    
    # Add colorbar
    cbar_ax = ax_baz.inset_axes([1.02, 0., 0.02, 1])
    cb = plt.colorbar(scatter, cax=cbar_ax)
    cb.set_label("CC coefficient", fontsize=font)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels([0, 0.5, 1])

    # Add title and labels
    title = f"{sd.tbeg.date} {str(sd.tbeg.time).split('.')[0]} UTC"
    title += f" | {wave_type.capitalize()} Waves"
    if sd.fmin is not None and sd.fmax is not None:
        title += f" | f = {sd.fmin}-{sd.fmax} Hz"
    if cc_threshold is not None:
        title += f" | CC > {cc_threshold}"
    if baz_theo is not None:
        title += f" | Theo. BAz = {round(baz_theo, 1)}°"
    if baz_results['parameters']['baz_win_sec'] is not None:
        title += f" | T = {baz_results['parameters']['baz_win_sec']} s ({baz_results['parameters']['baz_win_overlap']*100}%)"
    fig.suptitle(title, fontsize=font+2, y=0.93)
    
    ax_baz.set_xlabel("Time (s)", fontsize=font)

    # Adjust x-axis limits
    ax_wave.set_xlim(min(times), max(times)+0.15*max(times))
    ax_wave2.set_xlim(min(times), max(times)+0.15*max(times))
    ax_baz.set_xlim(min(times), max(times)+0.15*max(times))

    # Add minor ticks
    if minors:
        ax_wave.minorticks_on()
        ax_baz.minorticks_on()
        ax_wave2.minorticks_on()

    return fig