"""
Functions for plotting backazimuth results on a map.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from matplotlib.gridspec import GridSpec

def plot_backazimuth_map(sd, results, event_info=None, map_projection='orthographic', 
                        bin_step=5, figsize=(15, 8), debug=False):
    """
    Plot backazimuth estimation results from compute_backazimuth_simple
    
    Parameters:
    -----------
    sd : sixdegrees.SixDegrees
        SixDegrees object containing the data
    results : dict
        Results from compute_backazimuth_simple()
    event_info : dict, optional
        Event information for comparison
    map_projection : str
        Map projection type ('orthographic' or 'platecarree')
    bin_step : float
        Bin spacing in degrees for histograms
    figsize : tuple, optional
        Figure size (width, height). If None, auto-determined.
    debug : bool
        Enable debugging output
        
    Returns:
    --------
    matplotlib.figure.Figure
        Backazimuth analysis plot
    """
    if debug:
        print("\n" + "="*60)
        print("BACKAZIMUTH MAP PLOTTING DEBUG")
        print("="*60)

    def _plot_spherical_map_backazimuth(ax, event_info, baz_estimates, station_lat, station_lon, 
                                      projection='orthographic'):
        """Plot 2D spherical map with backazimuth information"""
        if debug:
            print(f"\n--- MAP PLOTTING ---")
            print(f"Station: ({station_lat:.6f}, {station_lon:.6f})")
            print(f"Projection: {projection}")
            print(f"Estimates: {baz_estimates}")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
            if debug:
                print("✓ Using Cartopy")
        except ImportError:
            use_cartopy = False
            if debug:
                print("⚠ Using matplotlib fallback")
        
        # Set up map features
        if use_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
            
            if projection == 'orthographic':
                ax.gridlines(alpha=0.5)
                ax.set_global()
            else:
                gl = ax.gridlines(draw_labels=True, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
            
            transform = ccrs.PlateCarree()
        else:
            if projection == 'orthographic':
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
                ax.axis('off')
            else:
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
                ax.grid(True, alpha=0.5)
            transform = None
        
        # Validate and normalize coordinates
        if not (-90 <= station_lat <= 90):
            print(f"ERROR: Invalid station latitude: {station_lat}")
            return
        station_lon_norm = ((station_lon + 180) % 360) - 180
        
        if debug and abs(station_lon_norm - station_lon) > 1e-6:
            print(f"Normalized station longitude: {station_lon} → {station_lon_norm}")
        
        # Plot station
        if use_cartopy:
            ax.plot(station_lon_norm, station_lat, marker='^', color='red', markersize=15,
                    label='Station', markeredgecolor='black', markeredgewidth=2,
                    transform=transform, zorder=5)
            if debug:
                print(f"✓ Station plotted at ({station_lon_norm}, {station_lat})")
        else:
            if projection == 'orthographic':
                x_st, y_st = _project_to_sphere(station_lat, station_lon_norm, station_lat, station_lon_norm)
                ax.plot(x_st, y_st, marker='^', color='red', markersize=15,
                        label='Station', markeredgecolor='black', markeredgewidth=2, zorder=5)
                if debug:
                    print(f"✓ Station plotted at sphere coords ({x_st:.3f}, {y_st:.3f})")
            else:
                ax.plot(station_lon_norm, station_lat, marker='^', color='red', markersize=15,
                        label='Station', markeredgecolor='black', markeredgewidth=2, zorder=5)
        
        # Plot event if available
        if event_info and 'latitude' in event_info and 'longitude' in event_info:
            event_lat = event_info['latitude']
            event_lon = ((event_info['longitude'] + 180) % 360) - 180
            
            if debug:
                print(f"Event: ({event_lat:.6f}, {event_lon:.6f})")
            
            if use_cartopy:
                ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=20,
                        label='Event', markeredgecolor='black', markeredgewidth=2,
                        transform=transform, zorder=5)
            else:
                if projection == 'orthographic':
                    x_ev, y_ev = _project_to_sphere(event_lat, event_lon, station_lat, station_lon_norm)
                    ax.plot(x_ev, y_ev, marker='*', color='yellow', markersize=20,
                            label='Event', markeredgecolor='black', markeredgewidth=2, zorder=5)
                else:
                    ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=20,
                            label='Event', markeredgecolor='black', markeredgewidth=2, zorder=5)
        
        # Plot great circles
        colors = {'love': 'darkblue', 'rayleigh': 'red', 'tangent': 'purple'}
        
        # Theoretical great circle first
        if event_info and 'backazimuth' in event_info:
            theo_baz = event_info['backazimuth']
            if debug:
                print(f"\nTheoretical BAZ: {theo_baz:.1f}°")
            
            try:
                if use_cartopy:
                    gc_lons, gc_lats = _great_circle_path_2d(
                        station_lat, station_lon_norm, theo_baz)
                    
                    # CRITICAL FIX: Check that great circle starts at station
                    if debug:
                        lat_diff = abs(gc_lats[0] - station_lat)
                        lon_diff = abs(gc_lons[0] - station_lon_norm)
                        print(f"  Theoretical GC start: ({gc_lats[0]:.6f}, {gc_lons[0]:.6f})")
                        print(f"  Station coords:       ({station_lat:.6f}, {station_lon_norm:.6f})")
                        print(f"  Difference:           ({lat_diff:.2e}, {lon_diff:.2e})")
                    
                    ax.plot(gc_lons, gc_lats, color='green', linewidth=4, 
                        linestyle=':', label=f'Theoretical: {theo_baz:.1f}°', alpha=0.9,
                        transform=transform, zorder=3)
                else:
                    _plot_great_circle_basic(ax, station_lat, station_lon_norm, 
                                              theo_baz, 'green', f'Theoretical: {theo_baz:.1f}°', 
                                              projection, linestyle=':')
            except Exception as e:
                print(f"ERROR plotting theoretical great circle: {e}")
        
        # Estimated great circles
        for wave_type, baz_deg in baz_estimates.items():
            if debug:
                print(f"\n{wave_type.upper()} BAZ: {baz_deg:.1f}°")
            
            try:
                color = colors.get(wave_type, 'purple')
                
                if use_cartopy:
                    gc_lons, gc_lats = _great_circle_path_2d(
                        station_lat, station_lon_norm, baz_deg)
                    
                    # CRITICAL FIX: Verify great circle starts at station
                    if debug:
                        lat_diff = abs(gc_lats[0] - station_lat)
                        lon_diff = abs(gc_lons[0] - station_lon_norm)
                        print(f"  {wave_type.upper()} GC start: ({gc_lats[0]:.6f}, {gc_lons[0]:.6f})")
                        print(f"  Station coords:       ({station_lat:.6f}, {station_lon_norm:.6f})")
                        print(f"  Difference:           ({lat_diff:.2e}, {lon_diff:.2e})")
                    
                    ax.plot(gc_lons, gc_lats, color=color, linewidth=3, 
                           label=f'{wave_type.upper()}: {baz_deg:.1f}°', alpha=0.8,
                           transform=transform, zorder=4)
                else:
                    _plot_great_circle_basic(ax, station_lat, station_lon_norm, 
                                              baz_deg, color, f'{wave_type.upper()}: {baz_deg:.1f}°', 
                                              projection)
            except Exception as e:
                print(f"ERROR plotting {wave_type} great circle: {e}")
        
        ax.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left')

    def _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=120, num_points=100):
        """Calculate great circle path points."""
        # Convert to radians
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        azimuth_rad = np.radians(azimuth)
        
        # Calculate points
        distances = np.linspace(0.0, np.radians(max_distance_deg), num_points)
        
        lats_rad = np.arcsin(
            np.sin(lat0_rad) * np.cos(distances) + 
            np.cos(lat0_rad) * np.sin(distances) * np.cos(azimuth_rad)
        )
        
        dlon = np.arctan2(
            np.sin(azimuth_rad) * np.sin(distances) * np.cos(lat0_rad),
            np.cos(distances) - np.sin(lat0_rad) * np.sin(lats_rad)
        )
        
        lons_rad = lon0_rad + dlon
        
        # Convert to degrees
        lats_deg = np.degrees(lats_rad)
        lons_deg = np.degrees(lons_rad)
        
        # Normalize longitude to [-180, 180]
        lons_deg = ((lons_deg + 180) % 360) - 180
        
        return lons_deg, lats_deg

    def _plot_great_circle_basic(ax, lat0, lon0, azimuth, color, label, projection, linestyle='-'):
        """Plot great circle on basic matplotlib plot."""
        if projection == 'orthographic':
            # Calculate points along great circle
            gc_lons, gc_lats = _great_circle_path_2d(lat0, lon0, azimuth)
            
            # Project points to sphere
            x_gc, y_gc = [], []
            for lat, lon in zip(gc_lats, gc_lons):
                x, y = _project_to_sphere(lat, lon, lat0, lon0)
                if not np.isnan(x) and not np.isnan(y):
                    x_gc.append(x)
                    y_gc.append(y)
            
            ax.plot(x_gc, y_gc, color=color, linewidth=2, label=label, 
                   linestyle=linestyle, alpha=0.8)
        else:
            gc_lons, gc_lats = _great_circle_path_2d(lat0, lon0, azimuth)
            ax.plot(gc_lons, gc_lats, color=color, linewidth=2, label=label, 
                   linestyle=linestyle, alpha=0.8)

    def _project_to_sphere(lat, lon, center_lat, center_lon):
        """Project lat/lon coordinates to orthographic projection."""
        lat, lon = np.radians(lat), np.radians(lon)
        center_lat, center_lon = np.radians(center_lat), np.radians(center_lon)
        
        cos_c = np.sin(center_lat) * np.sin(lat) + np.cos(center_lat) * np.cos(lat) * np.cos(lon - center_lon)
        
        # Check if point is visible
        if cos_c < 0:
            return np.nan, np.nan
        
        x = np.cos(lat) * np.sin(lon - center_lon)
        y = np.cos(center_lat) * np.sin(lat) - np.sin(center_lat) * np.cos(lat) * np.cos(lon - center_lon)
        
        return x, y

    # Create figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[3, 0.8, 0.1, 0.3], 
                 hspace=0.0, wspace=0.0)
    
    # Create map projection
    if map_projection == 'orthographic':
        # Calculate optimal center point
        center_lon = results['station_coordinates'].get('longitude', 0)
        center_lat = results['station_coordinates'].get('latitude', 0)
        
        if event_info and 'latitude' in event_info and 'longitude' in event_info:
            # Calculate midpoint
            center_lat = (center_lat + event_info['latitude']) / 2
            center_lon = (center_lon + event_info['longitude']) / 2
        
        try:
            import cartopy.crs as ccrs
            proj = ccrs.Orthographic(center_lon, center_lat)
            ax_map = fig.add_subplot(gs[0, 0], projection=proj)
        except ImportError:
            ax_map = fig.add_subplot(gs[0, 0])
    else:
        try:
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree()
            ax_map = fig.add_subplot(gs[0, 0], projection=proj)
        except ImportError:
            ax_map = fig.add_subplot(gs[0, 0])
    
    # Plot map
    _plot_spherical_map_backazimuth(
        ax_map, event_info, results['estimates'], 
        results['station_coordinates']['latitude'],
        results['station_coordinates']['longitude'],
        map_projection
    )
    
    # Add histograms if results available
    if 'detailed_results' in results:
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_map)
        
        # Plot histograms for each wave type
        colors = {'love': 'darkblue', 'rayleigh': 'red', 'tangent': 'purple'}
        for wave_type, data in results['detailed_results'].items():
            baz = data['baz']
            cc = data['cc']
            
            # Create histogram
            counts, bins = np.histogram(baz, bins=np.arange(0, 361, bin_step), 
                                      weights=cc, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot histogram
            ax_hist.barh(bin_centers, counts, height=bin_step*0.8, 
                        alpha=0.6, color=colors.get(wave_type, 'purple'),
                        label=wave_type.upper())
            
            # Add KDE if enough points
            if len(baz) > 1:
                kde = sts.gaussian_kde(baz, weights=cc)
                y_kde = np.linspace(0, 360, 360)
                kde_values = kde(y_kde)
                ax_hist.plot(kde_values, y_kde, color=colors.get(wave_type, 'purple'),
                           linewidth=2, alpha=0.8)
        
        ax_hist.set_xlabel('Density')
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend()
    
    plt.tight_layout()
    return fig