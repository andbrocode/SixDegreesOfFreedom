#!/usr/bin/env python
"""
G-ring velocity dispersion computation for single day with parallelized hours.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from obspy import UTCDateTime
from multiprocessing import Pool, cpu_count
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sixdegrees.sixdegrees import sixdegrees
from sixdegrees.utils.get_kde_stats_velocity import get_kde_stats_velocity


def print_status(message, status="INFO"):
    """Simple status printing."""
    status_symbols = {"INFO": "✓", "ERROR": "❌", "WARNING": "⚠️", "PROCESSING": "🔄"}
    print(f"{status_symbols.get(status, '•')} {message}")

def load_config(config_file):
    """Load configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def process_hour(hour, sd, config, day_beg, day_end):
    """Process single hour efficiently."""
    try:
        hour_beg = day_beg + hour * 3600
        hour_end = hour_beg + 3600
        time_center = hour_beg + 1800
        
        # Create hour copy and trim
        sd_hour = sd.copy()
        sd_hour.trim(hour_beg - 100, hour_end + 100)
        
        if not sd_hour or len(sd_hour.st) == 0:
            return None
        
        # Check data streams
        try:
            rot_stream = sd_hour.get_stream("rotation", raw=True)
            tra_stream = sd_hour.get_stream("translation", raw=True)
            if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                return None
        except:
            return None
        
        # Get frequency bands
        flower, fupper, fcenter = sd_hour.get_octave_bands(
            fmin=config.get('freq_min', 0.1),
            fmax=config.get('freq_max', 2.0),
            faction_of_octave=config.get('octave_fraction', 3)
        )
        
        results = []
        
        for fmin, fmax, fcenter in zip(flower, fupper, fcenter):
            try:
                # Filter and trim
                sd_filtered = sd_hour.copy()
                sd_filtered.filter_data(fmin=fmin, fmax=fmax)
                sd_filtered.trim(hour_beg, hour_end)
                
                if not sd_filtered.st or len(sd_filtered.st) == 0:
                    continue
                
                # Check streams
                try:
                    rot_stream = sd_filtered.get_stream("rotation", raw=True)
                    tra_stream = sd_filtered.get_stream("translation", raw=True)
                    if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                        continue
                except:
                    continue
                
                # Compute backazimuths
                win_time_s = int(config.get('t_win_factor', 2.0) / fcenter)
                baz_results = sd_filtered.compute_backazimuth(
                    wave_type=config.get('wave_type', 'love'),
                    baz_step=1,
                    baz_win_sec=win_time_s,
                    baz_win_overlap=config.get('overlap', 0.5),
                    verbose=False,
                    out=True
                )
                
                if baz_results is None:
                    continue
                
                # Compute velocities
                vel_results = sd_filtered.compute_velocities_optimized(
                    rotation_data=sd_filtered.get_stream("rotation", raw=True),
                    translation_data=sd_filtered.get_stream("translation", raw=True),
                    wave_type=config.get('wave_type', 'love'),
                    baz_results=baz_results,
                    baz_mode=config.get('baz_mode', 'mid'),
                    method=config.get('method', 'theilsen'),
                    cc_threshold=config.get('cc_threshold', 0.0),
                    r_squared_threshold=config.get('r_squared_threshold', 0.0),
                    zero_intercept=True
                )
                
                if vel_results is None:
                    continue
                
                # Extract valid measurements
                mask = ~np.isnan(vel_results['velocity'])
                velocities = vel_results['velocity'][mask]
                cc_values = vel_results['cc_value'][mask]
                
                if len(velocities) < 5:
                    continue
                
                # Apply KDE analysis
                try:
                    out = get_kde_stats_velocity(velocities, cc_values)
                    vel_max = out['max']
                    vel_dev = out['dev']
                except:
                    vel_max = np.median(velocities)
                    vel_dev = np.std(velocities)
                
                results.append({
                    'timestamp': time_center.datetime,
                    'day': time_center.datetime.date(),
                    'hour': hour,
                    'fband': fcenter,
                    'velocity': vel_max,
                    'deviation': vel_dev,
                    'n_measurements': len(velocities)
                })
                
            except:
                continue
        
        return pd.DataFrame(results) if results else None
        
    except:
        return None


def save_results(daily_df, output_dir, date):
    """Save results to CSV."""
    data_dir = Path(output_dir) / 'velocity_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    daily_file = data_dir / f"gring_velocity_{date}.csv"
    daily_df.to_csv(daily_file, index=False)
    print_status(f"Results saved: {daily_file}")
    return daily_file


def main():
    """Main processing routine."""
    parser = argparse.ArgumentParser(description='G-ring velocity dispersion computation for single day')
    parser.add_argument('config_file', help='Configuration YAML file')
    parser.add_argument('date', help='Date to process (YYYY-MM-DD)')
    parser.add_argument('--output-dir', '-o', default='./output', help='Output directory')
    parser.add_argument('--n-processes', '-n', type=int, default=None, help='Number of processes')
    parser.add_argument('--hours', type=int, default=24, help='Number of hours to process')
    
    args = parser.parse_args()
    
    # Parse date
    try:
        day_beg = UTCDateTime(args.date)
        date = day_beg.date
        day_end = day_beg + 24 * 3600
    except Exception as e:
        print_status(f"Invalid date format '{args.date}': {e}", "ERROR")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(args.config_file)
    except Exception as e:
        print_status(f"Failed to load config: {e}", "ERROR")
        sys.exit(1)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_processes = args.n_processes or min(cpu_count(), 8)
    
    print_status(f"Processing {date} with {n_processes} processes")
    
    # Initialize and load data
    config['tbeg'] = day_beg
    config['tend'] = day_end
    sd = sixdegrees(config)
    
    start_time = time.time()
    try:
        sd.load_data(day_beg, day_end)
        sd.trim_stream()
        
        if not sd.st or len(sd.st) == 0:
            print_status("No data available", "ERROR")
            sys.exit(1)
        
        print_status(f"Data loaded: {len(sd.st)} traces ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print_status(f"Failed to load data: {e}", "ERROR")
        sys.exit(1)
    
    # Process hours in parallel
    print_status("Starting parallel processing...", "PROCESSING")
    process_start = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(process_hour, 
                              [(hour, sd, config, day_beg, day_end) 
                               for hour in range(args.hours)])
    
    process_time = time.time() - process_start
    
    # Combine results
    successful_results = [r for r in results if r is not None]
    
    if successful_results:
        daily_df = pd.concat(successful_results, ignore_index=True)
        save_results(daily_df, output_dir, date)
        
        # Summary
        print_status(f"Completed: {len(successful_results)}/{args.hours} hours successful")
        print_status(f"Total measurements: {len(daily_df)}")
        print_status(f"Processing time: {process_time:.1f}s")
        
        if not daily_df.empty:
            velocities = daily_df['velocity'].dropna()
            if len(velocities) > 0:
                print_status(f"Velocity range: {velocities.min():.1f} - {velocities.max():.1f} m/s")
    else:
        print_status("No data processed", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
