"""
Functions for animating waveforms and particle motion.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from obspy import Stream, UTCDateTime
from obspy.signal.rotate import rotate_ne_rt
from typing import Optional, Union


def animate_waveforms(sd, time_step: float = 0.5, duration: Optional[float] = None,
                     save_path: Optional[str] = None, dpi: int = 150, show_arrivals: bool = False,
                     rotate_zrt: bool = False, tail_duration: float = 50.0, baz: Optional[float] = None,
                     n_frames: Optional[int] = None) -> FuncAnimation:
    """
    Create an animation of waveforms and particle motion.
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        The sixdegrees object containing the waveform data
    time_step : float
        Time step between frames in seconds (default: 0.5)
    duration : float, optional
        Duration of the animation in seconds. If None, uses full stream length
    save_path : str, optional
        Path to save the animation (e.g., 'animation.mp4'). If None, displays animation
    dpi : int
        DPI for the saved animation (default: 150)
    show_arrivals : bool
        Whether to show theoretical P and S wave arrival times (default: False)
    rotate_zrt : bool
        Whether to rotate horizontal components to radial and transverse (default: False)
    tail_duration : float
        Duration of particle motion tail in seconds (default: 50.0)
    baz : float, optional
        Backazimuth in degrees. If provided, overrides event-based backazimuth
    n_frames : int, optional
        Number of frames for the animation. If provided, adjusts duration accordingly
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        The animation object
    """
    
    def normalize_trace(data):
        """Normalize trace data to [-1, 1]."""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    # Get streams
    trans_st = sd.get_stream(stream_type="translation")
    rot_st = sd.get_stream(stream_type="rotation")
    
    # Check if streams are empty
    if not trans_st or not rot_st:
        raise ValueError("Translation or rotation stream is empty")
    
    # define components
    components = ['Z', 'N', 'E']

    # define time delta
    dt = trans_st[0].stats.delta

    #define phases
    phases = ['P', 'S']

    # Get sampling rate and adjust time_step if needed
    sampling_rate = trans_st[0].stats.sampling_rate
    min_time_step = 1.0 / sampling_rate
    
    if time_step < min_time_step:
        print(f"Warning: time_step {time_step}s is smaller than minimum {min_time_step}s. Adjusting.")
        time_step = min_time_step

    # Rotate to ZRT if requested
    if rotate_zrt:
        try:
            # Use provided backazimuth or get from event
            if baz is None:
                event_info = sd.get_event_info(sd.get_stream()[0].stats.starttime, 
                                            base_catalog="USGS",
                                            )
                baz = event_info.get('backazimuth')

            if baz is not None:
                print(f"Using backazimuth: {baz}°")

                HRdata, HTdata = rotate_ne_rt(
                    trans_st.select(component='N')[0].data,
                    trans_st.select(component='E')[0].data,
                    baz
                )
                JRdata, JTdata = rotate_ne_rt(
                    rot_st.select(component='N')[0].data,
                    rot_st.select(component='E')[0].data,
                    baz
                )

                HZ = trans_st.select(component='Z')[0]
                JZ = rot_st.select(component='Z')[0]

                # set R for N
                JR = rot_st.select(component='N')[0].copy()
                JR.data = JRdata
                JR.stats.channel = JR.stats.channel[:-1] + 'R'

                HR = trans_st.select(component='N')[0].copy()
                HR.data = HRdata
                HR.stats.channel = HR.stats.channel[:-1] + 'R'

                # set T for E
                JT = rot_st.select(component='E')[0].copy()
                JT.data = JTdata
                JT.stats.channel = JT.stats.channel[:-1] + 'T'

                HT = trans_st.select(component='E')[0].copy()
                HT.data = HTdata
                HT.stats.channel = HT.stats.channel[:-1] + 'T'

                trans_st = Stream([HZ, HR, HT])
                rot_st = Stream([JZ, JR, JT])

                components = ['Z', 'R', 'T']
            else:
                print("Warning: Could not get backazimuth, using ZNE components")
        except Exception as e:
            print(f"Warning: Error rotating to ZRT: {e}, using ZNE components")

    print(rot_st, trans_st)

    # Verify all components exist in both streams
    for comp in components:
        if not trans_st.select(component=comp) or not rot_st.select(component=comp):
            raise ValueError(f"Component {comp} not found in both streams")
    
    # Get reference time array and calculate duration
    ref_trace = trans_st.select(component=components[0])[0]
    t_trans = ref_trace.times()
    total_duration = t_trans[-1]
    
    # Calculate time limits and frames
    if n_frames is not None:
        duration = n_frames * time_step
    elif duration is None:
        duration = total_duration
    else:
        duration = min(duration, total_duration)
    
    # Calculate number of frames based on duration and time_step
    n_frames = int(duration / time_step)
    
    # Adjust time_step to match exact duration
    time_step = duration / n_frames
    
    print(f"Animation parameters:")
    print(f"Duration: {duration:.2f}s")
    print(f"Time step: {time_step:.3f}s")
    print(f"Number of frames: {n_frames}")
    print(f"Frame rate: {1/time_step:.1f} fps")
    
    # Normalize all traces
    for comp in components:
        trans_st.select(component=comp)[0].data = normalize_trace(trans_st.select(component=comp)[0].data)
        rot_st.select(component=comp)[0].data = normalize_trace(rot_st.select(component=comp)[0].data)
    
    # Set up the figure with two rows
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])  # Adjusted height ratio
    
    # Generate title with time period, frequency range, and station name
    def generate_title():
        # Get time information from the stream
        starttime = sd.get_stream()[0].stats.starttime
        endtime = starttime + duration
        
        # Format time period
        time_str = f"{starttime.strftime('%Y-%m-%d %H:%M:%S')} - {endtime.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        # Get station name from stream
        station_name = f"{sd.get_stream()[0].stats.network}.{sd.get_stream()[0].stats.station}"
        
        # Get frequency range from configuration
        fmin = sd.config.get('fmin', None)
        fmax = sd.config.get('fmax', None)
        
        if fmin is not None and fmax is not None:
            freq_str = f" | f = {fmin}-{fmax} Hz"
        else:
            freq_str = ""
        
        # Get backazimuth if available
        baz_str = ""
        if baz is not None:
            baz_str = f" | BAZ = {baz:.1f}°"
        elif hasattr(sd, 'event_info') and sd.event_info and 'backazimuth' in sd.event_info:
            baz_str = f" | BAZ = {sd.event_info['backazimuth']:.1f}°"
        
        # Combine all parts
        title = f"{station_name} | {time_str}{freq_str}{baz_str}"
        return title
    
    # Set the figure title
    fig.suptitle(generate_title(), fontsize=14, y=0.95)
    
    # First row: single panel for all waveforms
    ax_waves = fig.add_subplot(gs[0, :])
    
    # Second row: two panels for particle motion
    ax_love = fig.add_subplot(gs[1, 0])
    ax_rayleigh = fig.add_subplot(gs[1, 1])
    
    # Setup waveform plot with closer vertical offsets
    offsets = np.arange(6) * 1.5  # Reduced spacing from 2.5 to 1.5
    trace_pairs = list(zip(components * 2, ['Translation'] * 3 + ['Rotation'] * 3, offsets))
    
    # Initialize lines for each trace
    wave_lines_past = []
    wave_lines_future = []
    
    for comp, trace_type, offset in trace_pairs:
        tr = trans_st.select(component=comp)[0] if trace_type == 'Translation' else rot_st.select(component=comp)[0]
        color = 'k' if trace_type == 'Translation' else 'darkred'
        
        # Plot future data in grey
        line_future, = ax_waves.plot(t_trans, tr.data + offset, color='lightgrey', alpha=0.5)
        # Initialize past data line
        line_past, = ax_waves.plot([], [], color=color)
        
        # Add channel name as label on the left side
        ax_waves.text(-0.01, offset, tr.stats.channel, 
                     transform=ax_waves.get_yaxis_transform(),
                     verticalalignment='center',
                     horizontalalignment='right')
        
        wave_lines_past.append(line_past)
        wave_lines_future.append(line_future)
    
    ax_waves.set_ylim(-1, 9)  # Adjusted for closer spacing
    ax_waves.set_xlim(0, duration)
    ax_waves.set_xlabel('Time (s)')
    ax_waves.set_yticks([])  # Remove y-axis ticks
    
    # Remove top and right spines
    ax_waves.spines['top'].set_visible(False)
    ax_waves.spines['right'].set_visible(False)
    
    # Initialize particle motion plots
    ax_love.set_aspect('equal')
    ax_rayleigh.set_aspect('equal')
    
    # Set titles and labels
    ax_love.set_title('Love Wave Particle Motion')
    ax_rayleigh.set_title('Rayleigh Wave Particle Motion')
    
    ax_love.set_xlabel(f'HT')
    ax_rayleigh.set_xlabel(f'JT')
    ax_love.set_ylabel(f'JZ')
    ax_rayleigh.set_ylabel(f'HZ')

    # Set equal limits for particle motion plots based on normalized data
    pm_lim = 1.2  # slightly larger than normalized range (-1, 1)
    ax_love.set_xlim(-pm_lim, pm_lim)
    ax_love.set_ylim(-pm_lim, pm_lim)
    ax_rayleigh.set_xlim(-pm_lim, pm_lim)
    ax_rayleigh.set_ylim(-pm_lim, pm_lim)
    
    # Add grid to particle motion plots
    ax_love.grid(True, ls='--', zorder=0)
    ax_rayleigh.grid(True, ls='--', zorder=0)
    
    # Create red fade colormap for particle motion trails
    cmap = plt.cm.Blues

    # amount of trail samples
    tail_samples = int(tail_duration / dt)

    # Initialize particle motion lines and points
    love_trail = ax_love.scatter([], [], c=[], cmap=cmap, s=10, vmin=0, vmax=tail_samples, zorder=10)
    rayleigh_trail = ax_rayleigh.scatter([], [], c=[], cmap=cmap, s=10, vmin=0, vmax=tail_samples, zorder=10)
    love_point = ax_love.scatter([], [], color='darkblue', s=50, zorder=10)
    rayleigh_point = ax_rayleigh.scatter([], [], color='darkblue', s=50, zorder=10)
    
    # Add cursor line and trail region
    cursor_line = ax_waves.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
    
    # Define global trail region variable
    global trail_region
    # Create initial shaded region for trail duration (will be updated in animation)
    trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), 0, 0,
                                       color='lightblue', alpha=0.2, zorder=0)

    # Add P and S wave arrival lines if requested
    starttime = sd.get_stream()[0].stats.starttime
    if show_arrivals:
        try:
            p_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='P')) - starttime
        except:
            # replace P with Pdiff
            p_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='Pdiff')) - starttime
            phases = ['Pdiff', 'S']

        s_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='S')) - starttime
        if p_arrival is not None:
            ax_waves.axvline(x=p_arrival, color='black', linestyle='-', alpha=0.5)
            ax_waves.text(p_arrival, ax_waves.get_ylim()[1], phases[0], 
                         horizontalalignment='right', verticalalignment='bottom')
        if s_arrival is not None:
            ax_waves.axvline(x=s_arrival, color='black', linestyle='-', alpha=0.5)
            ax_waves.text(s_arrival, ax_waves.get_ylim()[1], phases[1], 
                         horizontalalignment='right', verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    def init():
        """Initialize animation"""
        global trail_region
        for line in wave_lines_past:
            line.set_data([], [])
        love_trail.set_offsets(np.c_[[], []])
        rayleigh_trail.set_offsets(np.c_[[], []])
        love_point.set_offsets(np.c_[[], []])
        rayleigh_point.set_offsets(np.c_[[], []])
        trail_region.remove()  # Remove old region
        # Create new empty region
        trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), 0, 0,
                                           color='lightblue', alpha=0.2, zorder=0)
        return wave_lines_past + [love_trail, rayleigh_trail, love_point, rayleigh_point, cursor_line, trail_region]
    
    def animate(frame):
        """Animation function"""
        global trail_region
        current_time = frame * time_step
        
        # Ensure we don't exceed the data length
        current_time = min(current_time, duration)
        
        # Update cursor position
        cursor_line.set_xdata([current_time, current_time])
        
        # Update trail region by removing old and creating new
        trail_region.remove()
        start_time = max(0, current_time - tail_duration)
        trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), start_time, current_time,
                                           color='lightblue', alpha=0.2, zorder=0)
        
        # Update waveform lines
        for i, (comp, trace_type, offset) in enumerate(trace_pairs):
            tr = trans_st.select(component=comp)[0] if trace_type == 'Translation' else rot_st.select(component=comp)[0]
            mask = t_trans <= current_time
            wave_lines_past[i].set_data(t_trans[mask], tr.data[mask] + offset)
        
        try:
            # Update particle motion plots
            tail_samples = int(tail_duration / dt)
            current_idx = min(int(current_time * sampling_rate), len(t_trans) - 1)
            start_idx = max(0, current_idx - tail_samples)
            
            if rotate_zrt:  # Only show particle motion in ZRT coordinates
                # Get relevant components for Love waves (HT and RZ)
                rz = rot_st.select(component='Z')[0].data[start_idx:current_idx+1]
                ht = trans_st.select(component='T')[0].data[start_idx:current_idx+1]
                
                # Get relevant components for Rayleigh waves (RT and HZ)
                hz = trans_st.select(component='Z')[0].data[start_idx:current_idx+1]
                rt = rot_st.select(component='T')[0].data[start_idx:current_idx+1]
                
                # Create fade effect - scale from 0 to tail_samples
                n_points = current_idx + 1 - start_idx
                # Create fade values that increase from oldest to newest points
                fade_values = np.arange(n_points) if n_points > 0 else np.array([])
                
                # Update Love wave plot
                if len(ht) > 0:
                    love_trail.set_offsets(np.c_[ht, rz])
                    love_trail.set_array(fade_values)
                    love_point.set_offsets([[ht[-1], rz[-1]]])  # Current point is at the end
                
                # Update Rayleigh wave plot
                if len(rt) > 0:
                    rayleigh_trail.set_offsets(np.c_[rt, hz])
                    rayleigh_trail.set_array(fade_values)
                    rayleigh_point.set_offsets([[rt[-1], hz[-1]]])  # Current point is at the end
            
        except Exception as e:
            print(f"Warning: Error updating particle motion: {e}")
        
        return wave_lines_past + [love_trail, rayleigh_trail, love_point, rayleigh_point, cursor_line, trail_region]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=time_step*10, blit=True)
    
    # Save or display animation
    if save_path:
        anim.save(save_path, writer='ffmpeg', dpi=dpi)
        plt.close()
    else:
        plt.show()
    
    return anim
