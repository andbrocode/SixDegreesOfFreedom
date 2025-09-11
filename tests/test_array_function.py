#!/usr/bin/env python3
"""
Test script to verify the compute_azimuth_distance_range function works correctly.
"""

import pytest
import tempfile
import os
import yaml
from sixdegrees.seismicarray import seismicarray
from obspy import UTCDateTime

def test_function():
    try:
        # Initialize array and get station data
        array = seismicarray("./config/array_config.yml")

        # Define time window
        tbeg = UTCDateTime("2025-07-29 23:30:00")
        tend = UTCDateTime("2025-07-30 00:30:00")

        # Get inventories
        array.request_inventories(tbeg, tend)

        # Compute azimuth distance range
        results = array.compute_azimuth_distance_range(
            azimuth_step=5.0,  # 5-degree steps
            plot=True,         # Show plot
            show_station_labels=False,
            # save_path="azimuth_analysis.png"  # Save plot
        )
        
        print("Function executed successfully!")
        print(f"Results keys: {list(results.keys())}")
        print(f"Azimuth angles shape: {results['azimuth_angles'].shape}")
        print(f"Min projections shape: {results['min_projections'].shape}")
        print(f"Max projections shape: {results['max_projections'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_seismicarray_copy_method():
    """Test the copy method for seismicarray objects."""
    # Create a temporary config file for testing
    config_data = {
        'stations': ['STA1', 'STA2', 'STA3'],
        'reference_station': 'STA1',
        'fdsn_client': 'IRIS',
        'channel_prefix': 'B',
        'response_output': 'VEL',
        'output_format': 'file',
        'vp': 6200.0,
        'vs': 3700.0,
        'sigmau': 1e-7
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        # Create original object
        original = seismicarray(config_file)
        
        # Modify some attributes to test copying
        original.station_coordinates = {
            'STA1': {'latitude': 40.0, 'longitude': 15.0},
            'STA2': {'latitude': 40.1, 'longitude': 15.1},
            'STA3': {'latitude': 40.2, 'longitude': 15.2}
        }
        original.station_distances = {
            'STA1': 0.0,
            'STA2': 10.5,
            'STA3': 21.0
        }
        original.failed_stations = ['STA4']
        original.adr_parameters['vp'] = 6500.0
        
        # Create copy
        copied = original.copy()
        
        # Test that copy is a different object
        assert copied is not original
        
        # Test that basic attributes are copied
        assert copied.stations == original.stations
        assert copied.reference_station == original.reference_station
        assert copied.channel_prefix == original.channel_prefix
        assert copied.response_output == original.response_output
        assert copied.output_format == original.output_format
        
        # Test that modified attributes are copied
        assert copied.station_coordinates == original.station_coordinates
        assert copied.station_distances == original.station_distances
        assert copied.failed_stations == original.failed_stations
        assert copied.adr_parameters == original.adr_parameters
        
        # Test that modifying the copy doesn't affect the original
        copied.station_coordinates['STA1']['latitude'] = 41.0
        assert original.station_coordinates['STA1']['latitude'] == 40.0
        
        # Test that modifying the original doesn't affect the copy
        original.failed_stations.append('STA5')
        assert 'STA5' not in copied.failed_stations
        
        # Test that modifying nested data in copy doesn't affect original
        copied.adr_parameters['vs'] = 4000.0
        assert original.adr_parameters['vs'] == 3700.0
        assert copied.adr_parameters['vs'] == 4000.0
        
    finally:
        # Clean up temporary file
        os.unlink(config_file)

def test_seismicarray_copy_with_complex_data():
    """Test the copy method with complex data structures."""
    config_data = {
        'stations': ['STA1', 'STA2'],
        'reference_station': 'STA1',
        'fdsn_client': 'IRIS'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        # Create original object
        original = seismicarray(config_file)
        
        # Add complex data structures
        original.azimuthal_distances = {
            'azimuth_angles': [0, 45, 90, 135, 180, 225, 270, 315],
            'min_projections': [1.0, 1.5, 2.0, 1.8, 1.2, 1.6, 2.1, 1.9],
            'max_projections': [5.0, 5.5, 6.0, 5.8, 5.2, 5.6, 6.1, 5.9],
            'azimuth_step': 45.0
        }
        
        # Create copy
        copied = original.copy()
        
        # Test that complex data is deep copied
        assert copied.azimuthal_distances == original.azimuthal_distances
        
        # Test that modifying nested data in copy doesn't affect original
        copied.azimuthal_distances['azimuth_angles'].append(360)
        assert len(original.azimuthal_distances['azimuth_angles']) == 8
        assert len(copied.azimuthal_distances['azimuth_angles']) == 9
        
        # Test that modifying nested data in original doesn't affect copy
        original.azimuthal_distances['min_projections'][0] = 0.5
        assert copied.azimuthal_distances['min_projections'][0] == 1.0
        assert original.azimuthal_distances['min_projections'][0] == 0.5
        
    finally:
        # Clean up temporary file
        os.unlink(config_file)

if __name__ == "__main__":
    success = test_function()
    if success:
        print("\n✓ Test passed!")
    else:
        print("\n✗ Test failed!")
