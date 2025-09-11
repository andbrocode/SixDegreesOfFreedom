"""
Plot array geometry showing station positions relative to the reference station.

This module provides functionality to visualize seismic array geometry
with station positions, distances, and status information.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.util import util_geo_km
from typing import Optional


def plot_array_geometry(station_coordinates: dict, reference_station: str, 
                       failed_stations: list = None, show_distances: bool = True, 
                       show_dropped: bool = True, save_path: Optional[str] = None) -> None:
    """
    Plot the array geometry showing station positions relative to the reference station.
    
    Args:
        station_coordinates (dict): Dictionary of station coordinates
        reference_station (str): Reference station name
        failed_stations (list): List of failed/dropped stations
        show_distances (bool): Whether to show distances to reference station
        show_dropped (bool): Whether to show dropped/failed stations
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    if failed_stations is None:
        failed_stations = []
        
    # Get reference station coordinates
    ref_coords = station_coordinates[reference_station]
    ref_lat = ref_coords['latitude']
    ref_lon = ref_coords['longitude']
    
    # Create figure with white background
    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Prepare station categories
    active_stations = []
    failed_stations_list = []
    ref_coords_plot = None
    
    # Process all stations
    for station, coords in station_coordinates.items():
        # Convert to local coordinate system (in km)
        x, y = util_geo_km(
            ref_lon,
            ref_lat,
            coords['longitude'],
            coords['latitude']
        )
        
        # Convert to meters
        x *= 1000
        y *= 1000
        
        station_info = {
            'x': x,
            'y': y,
            'label': station.split('.')[1],
            'full_name': station
        }
        
        if station in failed_stations:
            failed_stations_list.append(station_info)
        elif station == reference_station:
            ref_coords_plot = station_info
        else:
            active_stations.append(station_info)
    
    # Plot grid (behind everything)
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Plot stations by status
    legend_elements = []
    
    # Plot active stations
    if active_stations:
        active_x = [s['x'] for s in active_stations]
        active_y = [s['y'] for s in active_stations]
        active_scatter = plt.scatter(
            active_x, active_y, 
            c='dodgerblue',
            s=100,
            marker='^', 
            label='Active Stations', 
            zorder=2
        )
        legend_elements.append(active_scatter)
        
        # Add labels and distances for active stations
        for station in active_stations:
            # Station label
            plt.annotate(
                station['label'], 
                (station['x'], station['y']),
                xytext=(5, 5), 
                textcoords='offset points',
                color='black',
                zorder=4
            )
            
            # Distance label if requested
            if show_distances:
                distance = np.sqrt(station['x']**2 + station['y']**2)
                plt.annotate(
                    f'{distance:.1f}m',
                    (station['x'], station['y']),
                    xytext=(5, -15),
                    textcoords='offset points',
                    fontsize=8,
                    color='gray',
                    zorder=4
                )
    
    # Plot reference station
    if ref_coords_plot:
        ref_scatter = plt.scatter(
            ref_coords_plot['x'], ref_coords_plot['y'], 
            c='green', 
            s=100,
            marker='o', 
            label='Reference Station', 
            zorder=3
        )
        legend_elements.append(ref_scatter)
        
        # Add reference station label
        plt.annotate(
            ref_coords_plot['label'], 
            (ref_coords_plot['x'], ref_coords_plot['y']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontweight='bold',
            color='red',
            zorder=4
        )
    
    # Plot failed/dropped stations
    if show_dropped and failed_stations_list:
        # Plot markers
        failed_x = [s['x'] for s in failed_stations_list]
        failed_y = [s['y'] for s in failed_stations_list]
        
        # Plot dropped station markers
        dropped_scatter = plt.scatter(
            failed_x, failed_y, 
            c='lightgray',
            s=80,
            marker='d',  # square marker
            label='Dropped Stations', 
            alpha=0.9,
            zorder=1
        )
        legend_elements.append(dropped_scatter)
        
        # Add 'x' overlay on dropped stations
        plt.scatter(
            failed_x,
            failed_y,
            c='red',
            s=50,
            marker='x',
            alpha=0.9,
            zorder=1
        )
        
        # Add labels for dropped stations
        for station in failed_stations_list:
            plt.annotate(
                station['label'],
                (station['x'], station['y']),
                xytext=(5, 5),
                textcoords='offset points',
                color='gray',
                alpha=0.7,
                style='italic',
                zorder=4
            )

    # Calculate plot limits
    all_x = [s['x'] for s in active_stations + failed_stations_list + ([ref_coords_plot] if ref_coords_plot else [])]
    all_y = [s['y'] for s in active_stations + failed_stations_list + ([ref_coords_plot] if ref_coords_plot else [])]
    
    max_range = max(
        abs(max(all_x, default=0)), abs(min(all_x, default=0)),
        abs(max(all_y, default=0)), abs(min(all_y, default=0))
    )
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    margin = max_range * 0.1
    plt.xlim(-max_range - margin, max_range + margin)
    plt.ylim(-max_range - margin, max_range + margin)
    
    # Add labels and title
    plt.xlabel('Easting (m)',fontsize=13)
    plt.ylabel('Northing (m)',fontsize=13)
    plt.title('Array Geometry', pad=15,fontsize=14)
    
    # Adjust legend with collected elements
    if legend_elements:
        plt.legend(
            handles=legend_elements, 
            loc='upper right',
        )
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
