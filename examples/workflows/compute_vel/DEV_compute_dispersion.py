#!/usr/bin/env python
"""
G-ring velocity dispersion computation for single day using compute_dispersion_curve.
Processes data in time intervals (default 3 hours) and creates daily dispersion curves.
Output format matches compute_gring_velocities_v2.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from obspy import UTCDateTime
from multiprocessing import Pool, cpu_count
import argparse
import time
from tqdm import tqdm

from sixdegrees.sixdegrees import sixdegrees
from sixdegrees.plots.plot_dispersion_curve import plot_dispersion_curve


def print_status(message, status="INFO"):
    """Simple status printing."""
    status_symbols = {"INFO": "✓", "ERROR": "❌", "WARNING": "⚠️", "PROCESSING": "🔄"}
    print(f"{status_symbols.get(status, '•')} {message}")


def load_config(config_file):
    """Load configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def process_time_window(window_idx, time_beg, time_end, sd, config):
    """
    Process a time window (default 3 hours) using compute_dispersion_curve.
    
    Parameters:
    -----------
    window_idx : int
        Index of the time window
    time_beg : UTCDateTime
        Start time of the window
    time_end : UTCDateTime
        End time of the window
    sd : sixdegrees
        sixdegrees object with loaded data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns matching compute_gring_velocities_v2.py format:
        - timestamp: datetime of window center
        - day: date
        - fband: center frequency
        - velocity: KDE peak velocity
        - deviation: KDE deviation
        - n_measurements: number of valid measurements (from regression samples)
        - mean_cc: mean cross-correlation (placeholder, not directly available)
        - mean_r_squared: mean R-squared (placeholder, not directly available)
    """
    try:
        time_center = time_beg + (time_end - time_beg) / 2
        
        # Create window copy and trim with padding
        sd_window = sd.copy()
        sd_window.trim(time_beg - 100, time_end + 100)
        
        if not sd_window or len(sd_window.st) == 0:
            return None
        
        # Check data streams
        try:
            rot_stream = sd_window.get_stream("rotation", raw=True)
            tra_stream = sd_window.get_stream("translation", raw=True)
            if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                return None
            del rot_stream, tra_stream
        except Exception as e:
            return None
        
        # Trim to exact window for processing
        sd_window.trim(time_beg, time_end)
        
        # Compute dispersion curve
        try:
            results = sd_window.compute_dispersion_curve(
                wave_type=config.get('wave_type', 'love'),
                fmin=config.get('freq_min', 0.1),
                fmax=config.get('freq_max', 2.0),
                octave_fraction=config.get('octave_fraction', 3),
                window_factor=config.get('twin_factor', 8.0),
                use_theoretical_baz=config.get('use_theoretical_baz', False),
                cc_threshold=config.get('cc_threshold', 0.2),
                baz_step=config.get('baz_step', 1),
                time_window_overlap=config.get('twin_overlap', config.get('overlap', 0.5)),
                velocity_method=config.get('method', 'odr'),
                zero_intercept=config.get('zero_intercept', True),
                cc_method=config.get('cc_method', 'max'),
                n_jobs=config.get('n_jobs', None),
                verbose=False
            )
        except Exception as e:
            return None
        
        if results is None or len(results.get('frequency_bands', [])) == 0:
            return None
        
        # Extract data from frequency_bands to ensure alignment
        window_results = []
        for band in results.get('frequency_bands', []):
            freq = band.get('f_center', np.nan)
            vel = band.get('kde_peak_velocity', np.nan)
            unc = band.get('kde_deviation', np.nan)
            n_samples = band.get('total_regression_samples', 0)
            
            # Skip if invalid
            if np.isnan(freq) or np.isnan(vel):
                continue
            
            window_results.append({
                'timestamp': time_center.datetime,
                'day': time_center.datetime.date(),
                'fband': freq,
                'velocity': vel,
                'deviation': unc,
                'n_measurements': int(n_samples) if not np.isnan(n_samples) else 0,
                'mean_cc': np.nan,  # Not directly available from compute_dispersion_curve
                'mean_r_squared': np.nan  # Not directly available from compute_dispersion_curve
            })
        
        if window_results:
            return pd.DataFrame(window_results)
        else:
            return None
            
    except Exception as e:
        return None


def save_results(results_df, output_dir, date):
    """Save results to CSV file matching compute_gring_velocities_v2.py format."""
    data_dir = Path(output_dir) / 'velocity_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = data_dir / f"gring_velocity_{date}.csv"
    results_df.to_csv(csv_file, index=False)
    
    print_status(f"Results saved: {csv_file}")
    return csv_file


def create_daily_plots(results_df, output_dir, date, config):
    """Create daily dispersion curve plots."""
    plot_dir = Path(output_dir) / 'dispersion_curves'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if results_df is None or results_df.empty:
        print_status("No results to plot", "WARNING")
        return
    
    wave_type = config.get('wave_type', 'love').capitalize()
    
    # Group by frequency band and compute statistics
    freq_stats = results_df.groupby('fband').agg({
        'velocity': ['mean', 'std', 'count'],
        'deviation': 'mean',
        'n_measurements': 'sum'
    }).reset_index()
    
    freq_stats.columns = ['fband', 'velocity_mean', 'velocity_std', 'count', 
                         'deviation_mean', 'n_measurements']
    
    # Create plot
    print("  Creating dispersion curve plot...", end=' ', flush=True)
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Velocity vs Frequency
        ax1.errorbar(freq_stats['fband'], freq_stats['velocity_mean'], 
                    yerr=freq_stats['velocity_std'], 
                    fmt='o-', capsize=5, capthick=2, markersize=6)
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'{wave_type} Waves - Velocity Dispersion Curve - {date}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Number of measurements
        ax2.bar(freq_stats['fband'], freq_stats['count'], width=freq_stats['fband']*0.1)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Number of Time Windows', fontsize=12)
        ax2.set_title(f'Data Coverage by Frequency Band - {date}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        plot_file = plot_dir / f"dispersion_curve_{date}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print("✓")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def main():
    """Main processing routine."""
    parser = argparse.ArgumentParser(description='G-ring velocity dispersion computation for single day using compute_dispersion_curve')
    parser.add_argument('config_file', help='Configuration YAML file')
    parser.add_argument('date', help='Date to process (YYYY-MM-DD)')
    parser.add_argument('--output-dir', '-o', default='./output', help='Output directory')
    parser.add_argument('--n-processes', '-n', type=int, default=None, help='Number of processes')
    parser.add_argument('--hours-per-window', type=float, default=3.0, help='Hours per time window (default: 3)')
    
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
    n_processes = args.n_processes or config.get('n_processes', 1)
    hours_per_window = args.hours_per_window or config.get('hours_per_window', 3.0)
    window_duration = hours_per_window * 3600
    
    print_status(f"Processing {date} with {n_processes} processes")
    print_status(f"Time window size: {hours_per_window} hours")
    
    # Initialize and load data
    config['tbeg'] = day_beg
    config['tend'] = day_end
    sd = sixdegrees(config)
    
    start_time = time.time()
    try:
        if config.get('resample_rate', None) is not None:
            sd.load_data(day_beg, day_end, resample_rate=config.get('resample_rate'))
        else:
            sd.load_data(day_beg, day_end)
        sd.trim_stream()
        
        if not sd.st or len(sd.st) == 0:
            print_status("No data available", "ERROR")
            sys.exit(1)
        
        print_status(f"Data loaded: {len(sd.st)} traces ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print_status(f"Failed to load data: {e}", "ERROR")
        sys.exit(1)
    
    # Generate time windows
    time_windows = []
    time_current = day_beg
    while time_current < day_end:
        window_end = min(time_current + window_duration, day_end)
        time_windows.append((time_current, window_end))
        time_current = window_end
    
    print_status(f"Subdividing day into {len(time_windows)} time windows ({hours_per_window}h each)")
    
    # Process time windows in parallel
    print_status("Starting parallel processing...", "PROCESSING")
    process_start = time.time()
    
    if n_processes > 1 and len(time_windows) > 1:
        # Parallel processing
        with Pool(processes=min(n_processes, len(time_windows))) as pool:
            tasks = [(i, tw_beg, tw_end, sd, config) for i, (tw_beg, tw_end) in enumerate(time_windows)]
            results = list(tqdm(
                pool.starmap(process_time_window, tasks),
                total=len(time_windows),
                desc=f"Processing {date}",
                unit="window",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ))
    else:
        # Sequential processing
        results = []
        for i, (tw_beg, tw_end) in enumerate(time_windows):
            result = process_time_window(i, tw_beg, tw_end, sd, config)
            results.append(result)
    
    process_time = time.time() - process_start
    
    # Filter successful results and combine
    successful_results = [r for r in results if r is not None and not r.empty]
    
    if successful_results:
        # Combine all window results
        combined_df = pd.concat(successful_results, ignore_index=True)
        
        # Save results
        print_status("Saving results...", "PROCESSING")
        save_results(combined_df, output_dir, date)
        
        # Create daily plots
        if config.get('plot_dispersion', True):
            print_status("Creating daily plots...", "PROCESSING")
            create_daily_plots(combined_df, output_dir, date, config)
        
        # Summary
        total_bands = len(combined_df)
        total_measurements = combined_df['n_measurements'].sum()
        avg_velocity = combined_df['velocity'].mean()
        print_status(f"Completed: {len(successful_results)}/{len(time_windows)} windows successful")
        print_status(f"Total frequency bands: {total_bands}")
        print_status(f"Total measurements: {total_measurements:,}")
        print_status(f"Average velocity: {avg_velocity:.1f} m/s")
        print_status(f"Processing time: {process_time:.1f}s ({process_time/len(time_windows):.1f}s/window avg)")
    else:
        print_status("No data processed", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
