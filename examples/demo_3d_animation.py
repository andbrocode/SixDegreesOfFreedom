#!/usr/bin/env python3
"""
Demonstration script for the 3D animation functionality.
"""
import sys
import os

# Add the package to the path
sys.path.insert(0, '/home/andbro/cursorfiles/sixdegrees')

def demo_3d_animation():
    """Demonstrate the 3D animation functionality."""
    from sixdegrees.sixdegrees import sixdegrees
    from sixdegrees.plots import animate_waveforms_3d
    
    print("=== SixDegrees 3D Animation Demo ===\n")
    
    # Configuration
    config = {
        'tbeg': "2023-09-08 22:13:00",
        'tend': "2023-09-08 23:00:00",
        'origin_time': "2023-09-08 22:11:00",
        'magnitude': 6.8,
        'station_lon': 11.275476,
        'station_lat': 48.162941,
        'seed': "XX.ROMY..",
        'data_source': "mseed_file",
        'path_to_mseed_file': "./data/romy_eventM6.8.mseed",
        'fmin': 0.02,
        'fmax': 0.2,
        'verbose': True
    }
    
    print("Creating SixDegrees object...")
    sd = sixdegrees(conf=config)
    
    print("Loading data...")
    sd.load_data(config['tbeg'], config['tend'])
    
    print("Processing data...")
    sd.trim_stream(set_common=True, set_interpolate=False)
    sd.filter_data(fmin=config['fmin'], fmax=config['fmax'], output=False)
    sd.polarity_stream(pol_dict={"HZ": -1, "JZ": 1}, raw=True)
    
    print("\nCreating 3D animation...")
    print("This will show:")
    print("- Waveforms (top panel)")
    print("- Love wave particle motion (bottom left)")
    print("- 3D cube visualization (bottom center)")
    print("- Rayleigh wave particle motion (bottom right)")
    
    # Create 3D animation with normalization (default)
    print("\n1. Creating animation with trace normalization (default)...")
    anim1 = animate_waveforms_3d(
        sd=sd,
        time_step=0.5,
        duration=60,  # 1 minute animation
        show_arrivals=True,
        rotate_zrt=True,
        tail_duration=10.0,
        cube_scale=0.3,  # Default cube size (0.3 side length)
        normalize_traces=True  # Normalize each trace to [-1, 1]
    )
    
    # Create 3D animation without normalization
    print("\n2. Creating animation without trace normalization...")
    anim2 = animate_waveforms_3d(
        sd=sd,
        time_step=0.5,
        duration=60,  # 1 minute animation
        show_arrivals=True,
        rotate_zrt=True,
        tail_duration=10.0,
        cube_scale=0.3,  # Default cube size (0.3 side length)
        normalize_traces=False  # Use original trace amplitudes
    )
    
    print("\nAnimation created successfully!")
    print("The 3D cube will:")
    print("- Be centered at (0,0,0) with axes limits -1 to 1")
    print("- Translate based on displacement (double-integrated acceleration)")
    print("- Rotate based on rotation angles (single-integrated rotation rate)")
    print("- Show different colors for each face")
    print("\nNormalization options:")
    print("- normalize_traces=True: Each trace normalized to [-1, 1] (default)")
    print("- normalize_traces=False: Use original trace amplitudes")
    
    return anim1, anim2

if __name__ == "__main__":
    demo_3d_animation()
