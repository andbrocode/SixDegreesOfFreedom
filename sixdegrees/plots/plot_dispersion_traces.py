"""
Functions for plotting filtered traces from compute_dispersion_curve output.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from obspy.signal.rotate import rotate_ne_rt


def plot_dispersion_traces(dispersion_results: Dict, unitscale: str = "nano",
                           figsize: Optional[Tuple[float, float]] = None,
                           title: Optional[str] = None, data_type: str = "acceleration") -> plt.Figure:
    """
    Plot filtered traces from compute_dispersion_curve output.
    
    Creates subpanels showing filtered traces for each frequency band, similar to
    plot_trace_dispersion, using the filtered waveforms and velocities from 
    compute_dispersion_curve. No regression or filtering is performed - uses
    the pre-computed filtered traces and velocities directly.
    
    Parameters:
    -----------
    dispersion_results : dict
        Output dictionary from compute_dispersion_curve function
    unitscale : str
        Unit scale: "nano" or "micro"
    figsize : tuple, optional
        Figure size (width, height). If None, uses default based on number of bands.
    title : str, optional
        Custom title for the plot. If None, generates automatic title.
    data_type : str
        Type of data: "acceleration" (rotation rate and acceleration) or "velocity" (rotation and velocity).
        Default is "acceleration". This determines units and labels.
        
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    # Validate input
    if 'frequency_bands' not in dispersion_results:
        raise ValueError("dispersion_results must contain 'frequency_bands' key")
    
    frequency_bands = dispersion_results['frequency_bands']
    wave_type = dispersion_results.get('wave_type', 'love')
    
    if len(frequency_bands) == 0:
        raise ValueError("No frequency bands found in dispersion_results")
    
    n_bands = len(frequency_bands)
    
    # Define scaling factors based on data_type
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
        tra_label_prefix = "v"
        rot_label_prefix = "r"
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
        tra_label_prefix = "a"
        rot_label_prefix = r"$\dot{r}$"
    
    # Set figure size
    if figsize is None:
        figsize = (15, 2 * n_bands)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]
    
    plt.subplots_adjust(hspace=0.0)
    
    # Plot settings
    font = 12
    lw = 0.8
    
    # Process each frequency band
    for i, band in enumerate(frequency_bands):
        ax = axes[i]
        
        # Remove bottom and top spines
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Get filtered streams
        rot_filtered = band['filtered_rot']
        acc_filtered = band['filtered_acc']
        
        # Get times
        times = rot_filtered[0].times()
        
        # Get backazimuth used
        baz_used = band.get('baz_used', None)
        if baz_used is None or np.isnan(baz_used):
            # Try to get from backazimuths array
            backazimuths = band.get('backazimuths', [])
            if len(backazimuths) > 0:
                baz_used = np.nanmedian(backazimuths)
            else:
                baz_used = None
        
        if baz_used is None:
            raise ValueError(f"No valid backazimuth found for frequency band {i+1}")
        
        # Get velocity for scaling from results
        # Use KDE peak velocity or median of velocities array
        velocities = band.get('velocities', [])
        if len(velocities) > 0 and not np.all(np.isnan(velocities)):
            # Use median of valid velocities, or KDE peak if available
            valid_velocities = velocities[~np.isnan(velocities)]
            if len(valid_velocities) > 0:
                velocity_for_scaling = np.nanmedian(valid_velocities)
            else:
                velocity_for_scaling = band.get('kde_peak_velocity', np.nan)
        else:
            velocity_for_scaling = band.get('kde_peak_velocity', np.nan)
        
        if np.isnan(velocity_for_scaling):
            raise ValueError(f"No valid velocity found for frequency band {i+1}")
        
        # Convert velocity from m/s to slope units
        # In original regression: slope (from regression) is in units such that slope*1e3 gives m/s
        # So slope = velocity_for_scaling / 1e3
        slope = velocity_for_scaling / 1e3
        
        # Get velocity deviation for display
        velocity_deviation = band.get('kde_deviation', np.nan)
        
        # Get components based on wave type
        if wave_type.lower() == "rayleigh":
            # Rayleigh: vertical velocity (acc_z) vs horizontal rotation (rot_t)
            acc_z = acc_filtered.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(
                rot_filtered.select(channel="*N")[0].data,
                rot_filtered.select(channel="*E")[0].data,
                baz_used
            )
            
            # Apply scaling
            acc_z_scaled = acc_z * acc_scaling
            rot_t_scaled = rot_t * rot_scaling
            
            # Scale rotation data by the slope (from velocity)
            rot_t_scaled = rot_t_scaled * slope
            
            # Plot velocity/acceleration (black) on left axis
            ax.plot(times, acc_z_scaled, color="black", lw=lw, label=f"{tra_label_prefix}_Z")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_t_scaled, color="red", lw=lw, label=f"{rot_label_prefix}_H")
            
            # Remove bottom and top spines
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            
            # Get max values for symmetric ylim
            acc_max = max([abs(np.min(acc_z_scaled)), abs(np.max(acc_z_scaled))])
            rot_max = max([abs(np.min(rot_t_scaled)), abs(np.max(rot_t_scaled))])
            
            # Set ylims symmetric around zero
            if acc_max > 0:
                ax.set_ylim(-acc_max * 1.05, acc_max * 1.05)
            else:
                ax.set_ylim(-1, 1)
            if rot_max > 0:
                ax2.set_ylim(-rot_max * 1.05, rot_max * 1.05)
            else:
                ax2.set_ylim(-1, 1)
            
            # Remove y-axis labels and ticks
            ax.set_ylabel("")
            ax2.set_ylabel("")
            ax.set_yticks([])
            ax2.set_yticks([])
            
            # Show scale factor and backazimuth as text
            text_y_pos = 0.8
            scale_unit = "m/s" if data_type.lower() == "velocity" else "m/s²"
            # Format scale with velocity deviation if available
            if not np.isnan(velocity_deviation):
                ax.text(0.02, text_y_pos, f"scale: {velocity_for_scaling:.0f}±{velocity_deviation:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            else:
                ax.text(0.02, text_y_pos, f"scale: {velocity_for_scaling:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            ax.text(0.02, abs(1-text_y_pos), f"baz: {baz_used:.1f}°", 
                    transform=ax.transAxes, fontsize=font-2,
                    rotation=0, va='center', ha='left')
            
        elif wave_type.lower() == "love":
            # Love: vertical rotation (rot_z) vs horizontal velocity (acc_t)
            rot_z = rot_filtered.select(channel="*Z")[0].data
            acc_r, acc_t = rotate_ne_rt(
                acc_filtered.select(channel="*N")[0].data,
                acc_filtered.select(channel="*E")[0].data,
                baz_used
            )
            
            # Apply scaling
            rot_z_scaled = 2 * rot_z * rot_scaling  # factor 2 for formula for Love waves
            acc_t_scaled = acc_t * acc_scaling
            
            # Scale rotation data by the slope (from velocity)
            rot_z_scaled = rot_z_scaled * slope
            
            # Plot velocity/acceleration (black) on left axis
            ax.plot(times, acc_t_scaled, color="black", lw=lw, label=f"{tra_label_prefix}_H")
            
            # Plot rotation (red) on right axis
            ax2 = ax.twinx()
            ax2.plot(times, rot_z_scaled, color="red", lw=lw, label=f"{rot_label_prefix}_Z")
            
            # Remove bottom and top spines
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            
            # Get max values for symmetric ylim
            acc_max = max([abs(np.min(acc_t_scaled)), abs(np.max(acc_t_scaled))])
            rot_max = max([abs(np.min(rot_z_scaled)), abs(np.max(rot_z_scaled))])
            
            # Set ylims symmetric around zero
            if acc_max > 0:
                ax.set_ylim(-acc_max * 1.05, acc_max * 1.05)
            else:
                ax.set_ylim(-1, 1)
            if rot_max > 0:
                ax2.set_ylim(-rot_max * 1.05, rot_max * 1.05)
            else:
                ax2.set_ylim(-1, 1)
            
            # Remove y-axis labels and ticks
            ax.set_ylabel("")
            ax2.set_ylabel("")
            ax.set_yticks([])
            ax2.set_yticks([])
            
            # Show scale factor and backazimuth as text
            text_y_pos = 0.8
            scale_unit = "m/s" if data_type.lower() == "velocity" else "m/s²"
            # Format scale with velocity deviation if available
            if not np.isnan(velocity_deviation):
                ax.text(0.02, text_y_pos, f"scale: {velocity_for_scaling:.0f}±{velocity_deviation:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            else:
                ax.text(0.02, text_y_pos, f"scale: {velocity_for_scaling:.0f} {scale_unit}", 
                        transform=ax.transAxes, fontsize=font-2,
                        rotation=0, va='center', ha='left')
            ax.text(0.02, abs(1-text_y_pos), f"baz: {baz_used:.1f}°", 
                    transform=ax.transAxes, fontsize=font-2,
                    rotation=0, va='center', ha='left')
        
        # Set x-axis label only on bottom subplot
        if i == n_bands - 1:
            ax.set_xlabel("Time (s)", fontsize=font)
        
        # Add frequency band label on the right
        fl = band['f_lower']
        fu = band['f_upper']
        # Format frequency based on magnitude
        if fl < 0.1:
            fl_str = f"{fl:.3f}"
        elif fl < 1:
            fl_str = f"{fl:.2f}"
        else:
            fl_str = f"{fl:.1f}"
        
        if fu < 0.1:
            fu_str = f"{fu:.3f}"
        elif fu < 1:
            fu_str = f"{fu:.2f}"
        else:
            fu_str = f"{fu:.1f}"
        
        ax.text(0.99, 0.8, f"{fl_str}-{fu_str} Hz", 
                transform=ax.transAxes, fontsize=font-2,
                rotation=0, va='center', ha='right')
    
    ax.spines['bottom'].set_visible(True)
    
    # Set title
    if title is None:
        # Generate automatic title
        start_time = acc_filtered[0].stats.starttime
        title = f"{wave_type.capitalize()} waves"
        if baz_used is not None:
            title += f" | BAz = {baz_used:.1f}°"
        title += f" | {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    fig.suptitle(title, fontsize=font+2, y=0.93)
    
    return fig
