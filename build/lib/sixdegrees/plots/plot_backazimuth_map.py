def plot_backazimuth_map(results, event_info=None, map_projection='orthographic', 
                        bin_step=5, figsize=(15, 8), debug=False):
    """
    Plot backazimuth estimation results from compute_backazimuth_simple
    
    Parameters:
    -----------
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
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as sts
    from matplotlib.gridspec import GridSpec
    
    if debug:
        print("\n" + "="*60)
        print("BACKAZIMUTH MAP PLOTTING DEBUG")
        print("="*60)

    def _plot_spherical_map_backazimuth(ax, event_info, baz_estimates, station_lat, station_lon, 
                                        projection='orthographic'):
        """Plot 2D spherical map with backazimuth information"""
        import numpy as np
        
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
        colors = {'love': 'blue', 'rayleigh': 'red', 'tangent': 'purple'}
        
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
                        linestyle=':', label=f'Theoretical: {theo_baz:.f}°', alpha=0.9,
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
                        print(f"  {wave_type} GC start: ({gc_lats[0]:.6f}, {gc_lons[0]:.6f})")
                        print(f"  Difference:           ({lat_diff:.2e}, {lon_diff:.2e})")
                        if lat_diff > 1e-6 or lon_diff > 1e-6:
                            print(f"  ⚠ WARNING: Great circle doesn't start at station!")
                    
                    ax.plot(gc_lons, gc_lats, color=color, linewidth=3, 
                        label=f'{wave_type.upper()}: {baz_deg:.0f}°', alpha=0.8,
                        transform=transform, zorder=4)
                else:
                    _plot_great_circle_basic(ax, station_lat, station_lon_norm, 
                                                baz_deg, color, f'{wave_type.upper()}: {baz_deg:.1f}°', 
                                                projection)
            except Exception as e:
                print(f"ERROR plotting {wave_type} great circle: {e}")
        
        # Set title and legend
        # ax.set_title('Geographic View', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left')

    def _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=120, num_points=100):
        """
        FIXED: Calculate great circle path points ensuring start at station
        """
        import numpy as np
        
        if debug:
            print(f"    Computing great circle: lat0={lat0:.6f}, lon0={lon0:.6f}, az={azimuth:.1f}°")
        
        # Validate inputs
        if not (-90 <= lat0 <= 90):
            raise ValueError(f"Invalid latitude: {lat0}")
        
        # Normalize azimuth to [0, 360)
        azimuth = azimuth % 360
        
        # Convert to radians
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        azimuth_rad = np.radians(azimuth)
        
        # CRITICAL FIX: Ensure distances array starts exactly at 0
        distances = np.linspace(0.0, np.radians(max_distance_deg), num_points)
        
        # Calculate great circle points using spherical trigonometry
        # Using the standard great circle formulas
        lats_rad = np.arcsin(
            np.sin(lat0_rad) * np.cos(distances) + 
            np.cos(lat0_rad) * np.sin(distances) * np.cos(azimuth_rad)
        )
        
        # Calculate longitude differences
        dlon = np.arctan2(
            np.sin(azimuth_rad) * np.sin(distances) * np.cos(lat0_rad),
            np.cos(distances) - np.sin(lat0_rad) * np.sin(lats_rad)
        )
        
        lons_rad = lon0_rad + dlon
        
        # Convert back to degrees
        lats_deg = np.degrees(lats_rad)
        lons_deg = np.degrees(lons_rad)
        
        # Normalize longitude to [-180, 180]
        lons_deg = ((lons_deg + 180) % 360) - 180
        
        # CRITICAL VERIFICATION: First point must exactly match input
        if debug:
            lat_error = abs(lats_deg[0] - lat0)
            lon_error = abs(lons_deg[0] - lon0)
            print(f"    First point check: lat_error={lat_error:.2e}, lon_error={lon_error:.2e}")
            if lat_error > 1e-10 or lon_error > 1e-10:
                print(f"    ⚠ WARNING: First point doesn't match input!")
                print(f"    Input:  ({lat0:.10f}, {lon0:.10f})")
                print(f"    Output: ({lats_deg[0]:.10f}, {lons_deg[0]:.10f})")
        
        # FORCE exact match for first point to eliminate numerical errors
        lats_deg[0] = lat0
        lons_deg[0] = lon0
        
        return lons_deg, lats_deg

    def _plot_great_circle_basic(ax, lat0, lon0, azimuth, color, label, projection, linestyle='-'):
        """Plot great circle for basic matplotlib (non-cartopy) plots"""
        import numpy as np
        
        try:
            if projection == 'orthographic':
                gc_lats, gc_lons = _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=90)
                x_coords, y_coords = _project_to_sphere(gc_lats, gc_lons, lat0, lon0)
                
                # Only plot points that are on the visible hemisphere
                visible = (x_coords**2 + y_coords**2 <= 1) & ~np.isnan(x_coords) & ~np.isnan(y_coords)
                if np.any(visible):
                    ax.plot(x_coords[visible], y_coords[visible], color=color, linewidth=3, 
                        label=label, alpha=0.8, linestyle=linestyle)
                    if debug:
                        print(f"    Plotted {np.sum(visible)}/{len(visible)} visible points on sphere")
            else:
                gc_lats, gc_lons = _great_circle_path_2d(lat0, lon0, azimuth)
                ax.plot(gc_lons, gc_lats, color=color, linewidth=3, 
                    label=label, alpha=0.8, linestyle=linestyle)
                if debug:
                    print(f"    Plotted {len(gc_lats)} points on flat projection")
        except Exception as e:
            print(f"ERROR in _plot_great_circle_basic: {e}")

    def _project_to_sphere(lat, lon, center_lat, center_lon):
        """FIXED: Project lat/lon to sphere coordinates for orthographic-like view"""
        import numpy as np
        
        # Convert to arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Calculate angular distance from center
        cos_c = (np.sin(center_lat_rad) * np.sin(lat_rad) + 
                 np.cos(center_lat_rad) * np.cos(lat_rad) * np.cos(lon_rad - center_lon_rad))
        
        # Orthographic projection (only visible hemisphere)
        x = np.full_like(lat, np.nan, dtype=float)
        y = np.full_like(lat, np.nan, dtype=float)
        
        # Only project points on visible hemisphere (cos_c >= 0)
        visible = cos_c >= 0
        
        if np.any(visible):
            x[visible] = np.cos(lat_rad[visible]) * np.sin(lon_rad[visible] - center_lon_rad)
            y[visible] = (np.cos(center_lat_rad) * np.sin(lat_rad[visible]) - 
                          np.sin(center_lat_rad) * np.cos(lat_rad[visible]) * 
                          np.cos(lon_rad[visible] - center_lon_rad))
        
        return x, y

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
                    # Validate event coordinates
                    event_lat = event_info['latitude']
                    event_lon = event_info['longitude']
                    
                    if not (np.isfinite(event_lat) and np.isfinite(event_lon)):
                        # If event coordinates are invalid, center on station
                        proj = ccrs.Orthographic(center_lon, center_lat)
                        ax = fig.add_subplot(gridspec, projection=proj)
                        return ax
                    
                    # Normalize longitudes to [-180, 180]
                    event_lon = ((event_lon + 180) % 360) - 180
                    station_lon = ((center_lon + 180) % 360) - 180
                    
                    # Convert to radians for spherical geometry calculation
                    lat1, lon1 = np.radians(center_lat), np.radians(station_lon)
                    lat2, lon2 = np.radians(event_lat), np.radians(event_lon)
                    
                    try:
                        # Calculate midpoint using spherical geometry
                        Bx = np.cos(lat2) * np.cos(lon2 - lon1)
                        By = np.cos(lat2) * np.sin(lon2 - lon1)
                        
                        # Calculate midpoint
                        center_lat = np.degrees(np.arctan2(np.sin(lat1) + np.sin(lat2),
                                                        np.sqrt((np.cos(lat1) + Bx)**2 + By**2)))
                        
                        # Calculate central meridian that contains both points
                        dlon = lon2 - lon1
                        if abs(dlon) > np.pi:
                            dlon = -(2*np.pi - abs(dlon)) * np.sign(dlon)
                            
                        center_lon = np.degrees(lon1 + dlon/2)
                        center_lon = ((center_lon + 180) % 360) - 180
                        
                        # Calculate angular distance between points
                        angular_dist = np.degrees(np.arccos(np.sin(lat1) * np.sin(lat2) + 
                                                          np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)))
                        
                        # Add rotation to ensure both points are visible
                        # Rotation angle depends on angular distance between points
                        rotation_angle = min(30, max(10, angular_dist / 4))  # Scale rotation with distance
                        center_lon = center_lon + rotation_angle * np.sign(dlon)
                        
                    except (ValueError, RuntimeWarning) as e:
                        # If calculations fail, use simple midpoint
                        center_lat = (center_lat + event_lat) / 2
                        center_lon = (station_lon + event_lon) / 2
                    
                    # Ensure final center coordinates are valid
                    if not (np.isfinite(center_lon) and np.isfinite(center_lat)):
                        center_lon, center_lat = 0, 0
                    
                    # Create the projection with the calculated center
                    proj = ccrs.Orthographic(center_lon, center_lat)
                    ax = fig.add_subplot(gridspec, projection=proj)
                    
                    # Set map bounds to ensure visibility
                    if event_info and 'latitude' in event_info and 'longitude' in event_info:
                        try:
                            # Add padding around the points
                            padding = max(20, min(90, angular_dist / 2))  # Dynamic padding with limits
                            
                            # Ensure all values are finite
                            bounds = [
                                min(station_lon, event_lon) - padding,
                                max(station_lon, event_lon) + padding,
                                min(center_lat, event_info['latitude']) - padding,
                                max(center_lat, event_info['latitude']) + padding
                            ]
                            
                            if all(np.isfinite(b) for b in bounds):
                                ax.set_extent(bounds, crs=ccrs.PlateCarree())
                        except (ValueError, RuntimeWarning):
                            # If setting extent fails, let cartopy handle the bounds
                            pass
                    
                else:
                    ax = fig.add_subplot(gridspec, projection=ccrs.PlateCarree())
                    
                return ax
        except ImportError:
            return fig.add_subplot(gridspec)

    # Rest of the plot_backazimuth_map function remains the same...
    if not results or 'detailed_results' not in results:
        print("No results to plot")
        return None
    detailed_results = results['detailed_results']
    baz_estimates = results.get('estimates', {})
    station_coords = results.get('station_coordinates', {})
    
    num_wave_types = len(detailed_results)
    if num_wave_types == 0:
        print("No wave type results to plot")
        return None
    
    # Auto-determine figure size if not provided
    if figsize is None:
        if num_wave_types == 1:
            figsize = (16, 8)
        else:
            figsize = (16, 10)
    
    # Create figure layout
    fig = plt.figure(figsize=figsize)
    
    if num_wave_types == 1:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[0, 0], map_projection, station_coords, event_info)
        ax_hist = fig.add_subplot(gs[0, 1])
        hist_axes = [ax_hist]
    elif num_wave_types == 2:
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[:, 0], map_projection, station_coords, event_info)
        ax_hist1 = fig.add_subplot(gs[0, 1])
        ax_hist2 = fig.add_subplot(gs[1, 1])
        hist_axes = [ax_hist1, ax_hist2]
    else:  # 3 wave types
        gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[:, 0], map_projection, station_coords, event_info)
        ax_hist1 = fig.add_subplot(gs[0, 1])
        ax_hist2 = fig.add_subplot(gs[1, 1])
        ax_hist3 = fig.add_subplot(gs[2, 1])
        hist_axes = [ax_hist1, ax_hist2, ax_hist3]
    
    # Plot the map
    if event_info:
        _plot_spherical_map_backazimuth(
            ax_map, event_info, baz_estimates, 
            station_coords.get('latitude', 0),
            station_coords.get('longitude', 0),
            map_projection
        )
    
    # Plot histograms
    colors = {'love': 'blue', 'rayleigh': 'red', 'tangent': 'purple'}
    angles = np.arange(0, 361, bin_step)
    angle_fine = np.arange(0, 360, 1)
    
    wave_types_list = list(detailed_results.keys())
    
    for i, wave_type in enumerate(wave_types_list):
        data = detailed_results[wave_type]
        baz = data['baz']
        cc = data['cc']
        
        ax = hist_axes[i]
        color = colors.get(wave_type, f'C{i}')
        
        # Compute statistics
        # baz_mean = np.average(baz, weights=cc)
        # baz_std = np.sqrt(np.average((baz - baz_mean)**2, weights=cc))
        # baz_max = baz_estimates.get(wave_type, baz_mean)

        # Plot histogram
        counts, _ = np.histogram(baz, bins=angles, density=True)
        bin_centers = (angles[:-1] + angles[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_step*0.8, 
            alpha=0.7, color=color, edgecolor='black', linewidth=0.5, 
            label=f'N={len(baz)}')
        
        # Plot KDE overlay
        if len(baz) > 1:
            kde = sts.gaussian_kde(baz, weights=cc)
            kde_values = kde.pdf(angle_fine)
            ax.plot(angle_fine, kde_values, color='black', linewidth=2, label='KDE')
        
        # Compute the maximum of the KDE and index
        kde_max = kde_values.max()
        kde_max_idx = np.argmax(kde_values)
        baz_max = kde_max_idx

        # Mark estimated maximum
        ax.plot([kde_max_idx, kde_max_idx], [0, kde_max], color='black', linestyle='--', linewidth=2, 
                label=f'Est: {kde_max_idx:.0f}°')
        
        # Mark theoretical BAZ if available
        if event_info and 'backazimuth' in event_info:
            ax.axvline(event_info['backazimuth'], color='green', 
                    linestyle=':', linewidth=3, label=f"Theo: {event_info['backazimuth']:.0f}°")
            
            # Calculate deviation
            dev = abs(kde_max_idx - event_info['backazimuth'])
            if dev > 180:
                dev = 360 - dev
            
            # Add statistics text
            stats_text = (f"Max: {kde_max_idx}°\n"
                          f"Deviation: {round(dev, 0)}°")
        else:
            stats_text = (f"Max: {kde_max_idx}°")
        
        # Add statistics text box
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #         verticalalignment='top', fontsize=10,
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Configure histogram axis
        ax.set_title(f'{wave_type.upper()} Wave Backazimuth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Remove 0.00 tick label from density axis
        yticks = ax.get_yticks()
        yticks_filtered = yticks[yticks > 0.001]
        if len(yticks_filtered) > 0:
            ax.set_yticks(yticks_filtered)
    ax.set_xlabel('Backazimuth (°)')
    
    # Overall title
    title = f"Analysis"
    try:
        title += f" | T = {results['parameters']['baz_win_sec']}s ({results['parameters']['baz_win_overlap']*100:.0f}%)"
        title += f" | CC > {results['parameters']['cc_threshold']}"
    except:
        pass
    
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig
