#!/usr/bin/env python3
"""
Test script to verify the compute_azimuth_distance_range function works correctly.
"""

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

if __name__ == "__main__":
    success = test_function()
    if success:
        print("\n✓ Test passed!")
    else:
        print("\n✗ Test failed!")
