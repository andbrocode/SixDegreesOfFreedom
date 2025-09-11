#!/usr/bin/env python3
"""
Debug script to test the frequency patterns plotting function.
"""

import numpy as np
import matplotlib.pyplot as plt
from sixdegrees.seismicarray import seismicarray

# Create a simple test array
def create_test_array():
    """Create a simple test array with known geometry."""
    # Create a simple square array
    stations = {
        'NET.STA1': {'latitude': 0.0, 'longitude': 0.0},
        'NET.STA2': {'latitude': 0.001, 'longitude': 0.0},      # North
        'NET.STA3': {'latitude': 0.0, 'longitude': 0.001},      # East  
        'NET.STA4': {'latitude': -0.001, 'longitude': 0.0},     # South
        'NET.STA5': {'latitude': 0.0, 'longitude': -0.001},     # West
    }
    
    array = seismicarray()
    array.station_coordinates = stations
    array.reference_station = 'NET.STA1'
    array.failed_stations = []
    
    # Calculate distances
    array.station_distances = {}
    for station, coords in stations.items():
        if station != array.reference_station:
            from obspy.geodetics.base import gps2dist_azimuth
            dist, _, _ = gps2dist_azimuth(
                stations[array.reference_station]['latitude'],
                stations[array.reference_station]['longitude'],
                coords['latitude'],
                coords['longitude']
            )
            array.station_distances[station] = dist
    
    return array

def test_frequency_plotting():
    """Test the frequency plotting function."""
    print("Creating test array...")
    array = create_test_array()
    
    print("Computing azimuth distance range...")
    results = array.compute_azimuth_distance_range(azimuth_step=10.0, plot=False)
    
    print("Azimuth angles:", results['azimuth_angles'])
    print("Min projections:", results['min_projections'])
    print("Max projections:", results['max_projections'])
    
    print("Plotting frequency patterns...")
    velocity_range = [1000, 2000, 3000, 4000, 5000]
    
    # Test the plotting function
    array.plot_frequency_patterns(
        velocity_range=velocity_range,
        optional_amplitude_uncertainty=1e-7,
        log_scale=False
    )

if __name__ == "__main__":
    test_frequency_plotting()
