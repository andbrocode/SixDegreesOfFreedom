"""
Functions for plotting velocity estimation results.
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from obspy.signal.rotate import rotate_ne_rt
from typing import Dict, Optional

def plot_velocities(sd, velocity_results: Dict, vmax: Optional[float]=None, 
                   minors: bool=True, cc_threshold: Optional[float]=None) -> plt.Figure:
    """
    Plot waveforms and velocity estimates
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    velocity_results : Dict
        Results dictionary from compute_velocities
    vmax : float or None
        Maximum velocity for plot scaling
    minors : bool
        Add minor ticks to axes if True
    cc_threshold : float, optional
        Minimum cross-correlation coefficient threshold
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    wave_type = velocity_results['parameters']['wave_type'].lower()

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(4, 8, figure=fig, hspace=0.2)
    
    # Create subplots
    ax_wave = fig.add_subplot(gs[0:2, :7])  # Waveform panel
    ax_vel = fig.add_subplot(gs[2:4, :7])   # Velocity panel
    
    # Plot settings
    font = 12
    lw = 1.0
    rot_scale, rot_unit = 1e9, f"n{sd.runit}"
    tra_scale, tra_unit = 1e6, f"{sd.mu}{sd.tunit}"
    
    # Get time vector
    times = np.arange(len(sd.st[0])) / sd.sampling_rate
    
    # get streams
    acc = sd.get_stream("translation").copy()
    rot = sd.get_stream("rotation").copy()

    # scale waveforms
    for tr in acc:
        tr.data *= tra_scale
    for tr in rot:
        tr.data *= rot_scale

    # rotate waveforms
    if wave_type == "love":
        rot_z = 2*rot.select(channel="*Z")[0].data # times two for velocity scaling (plotting only)
        acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data,
                                   acc.select(channel="*E")[0].data,
                                   velocity_results['parameters']['baz'])
        

    elif wave_type == "rayleigh":
        acc_z = acc.select(channel="*Z")[0].data
        rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data,
                                   rot.select(channel="*E")[0].data,
                                   velocity_results['parameters']['baz'])

    # prepare mask
    if cc_threshold is not None:
        mask = velocity_results['ccoef'] > cc_threshold
    else:
        mask = velocity_results['ccoef'] >= 0

    # Plot waveforms based on wave type
    if  wave_type == 'love':

        # Plot transverse acceleration
        ax_wave.plot(times, acc_t, 'black', 
                    label=f"{sd.tra_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[3][0]}HT", lw=lw)
        
        # Plot vertical rotation on twin axis
        ax_wave2 = ax_wave.twinx()
        ax_wave2.plot(times, rot_z, 'darkred',
                     label=f"2x {sd.rot_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[3][0]}JZ", lw=lw)
        
    elif wave_type == 'rayleigh':

        # Plot vertical acceleration
        ax_wave.plot(times, acc_z, 'black',
                    label=f"{sd.tra_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[3][0]}HZ", lw=lw)
        
        # Plot transverse rotation on twin axis
        ax_wave2 = ax_wave.twinx()
        ax_wave2.plot(times, rot_t, 'darkred',
                     label=f"{sd.rot_seed[0].split('.')[1]}.{sd.tra_seed[0].split('.')[3][0]}JT", lw=lw)

    # Configure waveform axes
    ax_wave.grid(True, which='both', ls='--', alpha=0.3)
    ax_wave.legend(loc=1)
    ax_wave.set_ylabel(f"acceleration ({tra_unit})", fontsize=font)
    ax_wave2.tick_params(axis='y', colors="darkred")
    ax_wave2.set_ylabel(f"rotation rate ({rot_unit})", color="darkred", fontsize=font)
    ax_wave2.legend(loc=4)

    sd.sync_twin_axes(ax_wave, ax_wave2)
    
    # Plot velocities
    cmap = plt.get_cmap("viridis", 10)
    scatter = ax_vel.scatter(velocity_results['time'][mask], 
                           velocity_results['velocity'][mask],
                           c=velocity_results['ccoef'][mask], 
                           cmap=cmap, s=70, alpha=1.0,
                           vmin=0, vmax=1, edgecolors="k", lw=1, zorder=2)
    
    # Add error bars
    ax_vel.errorbar(velocity_results['time'][mask], 
                   velocity_results['velocity'][mask],
                   xerr=velocity_results['terr'][mask],
                   color='black', alpha=0.4, ls='none', zorder=1)
    
    # Configure velocity axis
    ax_vel.set_ylabel("phase velocity (m/s)", fontsize=font)
    ax_vel.set_xlabel("time (s)", fontsize=font)
    ax_vel.set_ylim(bottom=0)
    if vmax is not None:
        ax_vel.set_ylim(top=vmax)
    ax_vel.grid(True, which='both', ls='--', alpha=0.3)
    
    for a in [ax_vel, ax_wave]:
        a.set_xlim(0, times.max())

    if minors:
        ax_wave.minorticks_on()
        ax_vel.minorticks_on()
        ax_wave2.minorticks_on()
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cb = plt.colorbar(scatter, cax=cbar_ax)
    cb.set_label('cross-correlation coefficient', fontsize=font)
    
    # Add title
    title = f"{velocity_results['parameters']['wave_type'].capitalize()} Waves"
    title += (f" | {sd.tbeg.date} {str(sd.tbeg.time).split('.')[0]} UTC"
              f" | {sd.fmin}-{sd.fmax} Hz"
              f" | T = {velocity_results['parameters']['win_time_s']:.1f} s"
              f" | {velocity_results['parameters']['overlap']*100:.0f}% overlap")
    if cc_threshold is not None:
        title += f" | cc > {cc_threshold}"
    fig.suptitle(title, fontsize=font+2, y=0.95)
    
    # plt.tight_layout()
    plt.show()
    return fig