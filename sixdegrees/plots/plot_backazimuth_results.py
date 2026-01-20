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
    
    # Convert to numpy arrays if not already
    time = np.asarray(time)
    baz = np.asarray(baz)
    ccc = np.asarray(ccc)
    
    # apply cc threshold if provided
    if cc_threshold is not None:
        mask = ccc > cc_threshold
        time = time[mask]
        baz = baz[mask]
        cc = ccc[mask]
    else:
        cc = ccc
    
    # Check for empty or all-NaN backazimuth estimates
    valid_mask = ~np.isnan(baz)
    if len(baz) == 0 or not np.any(valid_mask):
        # Handle empty or all-NaN case
        print(f"Warning: No valid backazimuth estimates found for {wave_type} waves.")
        if len(baz) == 0:
            print("  Backazimuth array is empty.")
        else:
            print(f"  All {len(baz)} backazimuth estimates are NaN.")
        
        # Create empty arrays for plotting
        time = np.array([])
        baz = np.array([])
        cc = np.array([])
        has_valid_data = False
    else:
        # Filter out NaN values
        if not np.all(valid_mask):
            print(f"Warning: {np.sum(~valid_mask)} NaN backazimuth estimates filtered out.")
            time = time[valid_mask]
            baz = baz[valid_mask]
            cc = cc[valid_mask]
        has_valid_data = True

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
    
    # Plot backazimuth estimates (only if we have valid data)
    if has_valid_data and len(baz) > 0:
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
            try:
                xerr = baz_results.get('parameters', {}).get('baz_win_sec', None)
                if xerr is not None:
                    ax_baz.errorbar(time, baz, xerr=xerr/2, fmt='.', color='gray', alpha=0.6, zorder=0)
            except (KeyError, TypeError):
                pass  # Skip errorbar if parameters are missing
    else:
        # Create a dummy scatter for colorbar (will be empty)
        cmap = plt.get_cmap("viridis", 10)
        scatter = ax_baz.scatter([], [], c=[], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1)
        ax_baz.text(0.5, 0.5, 'No valid backazimuth estimates', 
                   transform=ax_baz.transAxes, ha='center', va='center', 
                   fontsize=font+2, color='red', weight='bold')
    
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

    # Compute statistics (only if we have valid data)
    got_kde = False
    kde_stats = None
    
    if has_valid_data and len(baz) > 0:
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
            
            # Validate KDE stats before using them
            kde_values = kde_stats.get('kde_values', None)
            kde_angles = kde_stats.get('kde_angles', None)
            
            if (kde_values is not None and kde_angles is not None and
                len(kde_values) > 0 and len(kde_angles) > 0 and
                len(kde_values) == len(kde_angles)):
                # get max and std
                baz_max = kde_stats.get('baz_estimate', None)
                baz_std = kde_stats.get('kde_dev', None)
                if baz_max is not None and baz_std is not None:
                    print(f"baz_max = {baz_max}, baz_std = {baz_std}")
                got_kde = True
            else:
                got_kde = False
                if hasattr(sd, 'verbose') and sd.verbose:
                    print("Warning: KDE stats returned invalid or empty arrays. Skipping KDE plot.")
        except Exception as e:
            got_kde = False
            kde_stats = None
            if hasattr(sd, 'verbose') and sd.verbose:
                print(f"Could not compute KDE stats: {e}")

        # Add histogram
        # ax_hist2.plot(kernel_density(np.linspace(0, 360, 100)), np.linspace(0, 360, 100), 'k-', lw=2)
        ax_hist.hist(baz, bins=len(angles1)-1, range=[min(angles1), max(angles1)],
                        weights=cc, orientation="horizontal", density=True, color="grey")
        if got_kde and kde_stats is not None:
            # Validate KDE stats before plotting
            kde_values = kde_stats.get('kde_values', None)
            kde_angles = kde_stats.get('kde_angles', None)
            
            if kde_values is not None and kde_angles is not None:
                # Convert to numpy arrays and check dimensions
                kde_values = np.asarray(kde_values)
                kde_angles = np.asarray(kde_angles)
                
                # Check that arrays are not empty, have matching dimensions, and contain valid values
                if (len(kde_values) > 0 and len(kde_angles) > 0 and 
                    len(kde_values) == len(kde_angles) and
                    not np.all(np.isnan(kde_values)) and
                    not np.all(np.isnan(kde_angles))):
                    ax_hist.plot(
                        kde_values,
                        kde_angles,
                        c="k",
                        lw=2,
                        label='KDE'
                    )
                else:
                    if hasattr(sd, 'verbose') and sd.verbose:
                        if len(kde_values) != len(kde_angles):
                            print(f"Warning: KDE arrays have mismatched dimensions "
                                  f"(values: {len(kde_values)}, angles: {len(kde_angles)}). Skipping KDE plot.")
                        elif len(kde_values) == 0:
                            print("Warning: KDE arrays are empty. Skipping KDE plot.")
                        else:
                            print("Warning: KDE arrays contain only NaN values. Skipping KDE plot.")
            else:
                if hasattr(sd, 'verbose') and sd.verbose:
                    print("Warning: KDE stats missing 'kde_values' or 'kde_angles'. Skipping KDE plot.")
    else:
        # Create empty histogram with proper range
        deltaa = 10
        angles1 = arange(0, 365, deltaa)
        ax_hist.hist([], bins=len(angles1)-1, range=[min(angles1), max(angles1)],
                    orientation="horizontal", density=True, color="grey")
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
    try:
        baz_win_sec = baz_results.get('parameters', {}).get('baz_win_sec', None)
        baz_win_overlap = baz_results.get('parameters', {}).get('baz_win_overlap', None)
        if baz_win_sec is not None and baz_win_overlap is not None:
            title += f" | T = {baz_win_sec} s ({baz_win_overlap*100}%)"
    except (KeyError, TypeError):
        pass  # Skip if parameters are missing
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