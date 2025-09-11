"""
Functions for animating waveforms, particle motion, and 3D cube visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from obspy import Stream, UTCDateTime
from obspy.signal.rotate import rotate_ne_rt
from typing import Optional, Union


def animate_waveforms_3d(sd, time_step: float = 0.5, duration: Optional[float] = None,
                        save_path: Optional[str] = None, dpi: int = 150, show_arrivals: bool = False,
                        rotate_zrt: bool = False, tail_duration: float = 50.0, baz: Optional[float] = None,
                        n_frames: Optional[int] = None, cube_scale: float = 0.3, 
                        normalize_traces: bool = True) -> FuncAnimation:
    """
    Create an animation of waveforms, particle motion, and 3D cube visualization.
    
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
    cube_scale : float
        Scale factor for cube size (default: 0.3)
    normalize_traces : bool
        Whether to normalize each trace individually to [-1, 1] (default: True)
    
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

    def integrate_twice(data, dt):
        """Double integrate acceleration to get displacement."""
        # First integration: acceleration to velocity
        velocity = np.cumsum(data) * dt
        # Second integration: velocity to displacement
        displacement = np.cumsum(velocity) * dt
        return displacement

    def integrate_once(data, dt):
        """Single integrate rotation rate to get rotation angle."""
        # Integration: rotation rate to rotation angle
        rotation = np.cumsum(data) * dt
        return rotation

    def create_cube(center=(0, 0, 0), size=0.2):
        """Create a cube centered at the given point."""
        # Define the 8 vertices of the cube
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top face
        ]) * size / 2
        
        # Translate to center
        vertices += np.array(center)
        
        # Define the 6 faces of the cube
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        
        return vertices, faces

    def apply_rotation(vertices, rotation_angles):
        """Apply rotation around x, y, z axes."""
        rx, ry, rz = rotation_angles
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        return vertices @ R.T

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

                HT = rot_st.select(component='E')[0].copy()
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
    
    # Normalize all traces if requested
    if normalize_traces:
        print("Normalizing traces to [-1, 1] range...")
        for comp in components:
            trans_st.select(component=comp)[0].data = normalize_trace(trans_st.select(component=comp)[0].data)
            rot_st.select(component=comp)[0].data = normalize_trace(rot_st.select(component=comp)[0].data)
    else:
        print("Skipping trace normalization - using original amplitudes")
    
    # Integrate translational data (acceleration to displacement)
    print("Integrating translational data (acceleration to displacement)...")
    for comp in components:
        trans_data = trans_st.select(component=comp)[0].data
        trans_st.select(component=comp)[0].data = integrate_twice(trans_data, dt)
    
    # Integrate rotational data (rotation rate to rotation angle)
    print("Integrating rotational data (rotation rate to rotation angle)...")
    for comp in components:
        rot_data = rot_st.select(component=comp)[0].data
        rot_st.select(component=comp)[0].data = integrate_once(rot_data, dt)
    
    # Set up the figure with three panels
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5], width_ratios=[1, 1, 1])
    
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
    
    # Second row: three panels - Love, 3D Cube, Rayleigh
    ax_love = fig.add_subplot(gs[1, 0])
    ax_cube = fig.add_subplot(gs[1, 1], projection='3d')
    ax_rayleigh = fig.add_subplot(gs[1, 2])
    
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
    ax_cube.set_title('3D Seismic Motion')
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
    
    # Setup 3D cube plot
    ax_cube.set_xlabel('X (Displacement)')
    ax_cube.set_ylabel('Y (Displacement)')
    ax_cube.set_zlabel('Z (Displacement)')
    
    # Set 3D plot limits to -1, 1
    ax_cube.set_xlim(-1, 1)
    ax_cube.set_ylim(-1, 1)
    ax_cube.set_zlim(-1, 1)
    
    # Create initial cube at origin
    cube_vertices, cube_faces = create_cube(center=(0, 0, 0), size=cube_scale)
    
    # Define colors for each face
    face_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    
    # Create cube collection
    cube_collection = Poly3DCollection([cube_vertices[face] for face in cube_faces], 
                                      facecolors=face_colors, alpha=0.7, edgecolors='black')
    ax_cube.add_collection3d(cube_collection)
    
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
                ht = trans_st.select(component='Z')[0].data[start_idx:current_idx+1]
                
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
            
            # Update 3D cube
            if current_idx < len(t_trans):
                # Get current translational values (X, Y, Z) - now displacement after integration
                if rotate_zrt:
                    # Use ZRT components
                    x_trans = trans_st.select(component='R')[0].data[current_idx] if 'R' in components else 0
                    y_trans = trans_st.select(component='T')[0].data[current_idx] if 'T' in components else 0
                    z_trans = trans_st.select(component='Z')[0].data[current_idx]
                else:
                    # Use ZNE components
                    x_trans = trans_st.select(component='E')[0].data[current_idx] if 'E' in components else 0
                    y_trans = trans_st.select(component='N')[0].data[current_idx] if 'N' in components else 0
                    z_trans = trans_st.select(component='Z')[0].data[current_idx]
                
                # Get current rotational values (X, Y, Z) - now rotation angles after integration
                if rotate_zrt:
                    # Use ZRT components
                    x_rot = rot_st.select(component='R')[0].data[current_idx] if 'R' in components else 0
                    y_rot = rot_st.select(component='T')[0].data[current_idx] if 'T' in components else 0
                    z_rot = rot_st.select(component='Z')[0].data[current_idx]
                else:
                    # Use ZNE components
                    x_rot = rot_st.select(component='E')[0].data[current_idx] if 'E' in components else 0
                    y_rot = rot_st.select(component='N')[0].data[current_idx] if 'N' in components else 0
                    z_rot = rot_st.select(component='Z')[0].data[current_idx]
                
                # Scale the translations and rotations for visualization
                translation_scale = 0.5  # Scale down displacement for better visualization
                rotation_scale = 1.0     # Use rotation angles directly
                
                # Calculate new cube center (origin + displacement)
                new_center = (x_trans * translation_scale, 
                             y_trans * translation_scale, 
                             z_trans * translation_scale)
                
                # Calculate rotation angles (in radians)
                rotation_angles = (x_rot * rotation_scale, 
                                  y_rot * rotation_scale, 
                                  z_rot * rotation_scale)
                
                # Create new cube vertices
                new_vertices, _ = create_cube(center=new_center, size=cube_scale)
                
                # Apply rotation
                rotated_vertices = apply_rotation(new_vertices - np.array(new_center), rotation_angles)
                rotated_vertices += np.array(new_center)
                
                # Update cube collection
                cube_collection.set_verts([rotated_vertices[face] for face in cube_faces])
            
        except Exception as e:
            print(f"Warning: Error updating particle motion or cube: {e}")
        
        return wave_lines_past + [love_trail, rayleigh_trail, love_point, rayleigh_point, cursor_line, trail_region]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=time_step*10, blit=False)  # blit=False for 3D plots
    
    # Save or display animation
    if save_path:
        anim.save(save_path, writer='ffmpeg', dpi=dpi)
        plt.close()
    else:
        plt.show()
    
    return anim
