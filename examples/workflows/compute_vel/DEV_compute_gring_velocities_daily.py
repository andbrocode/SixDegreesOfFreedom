#!/usr/bin/env python
"""
G-ring velocity dispersion computation for single day using new plot_trace_dispersion function.
Processes hourly data and creates daily dispersion curves with and without optimization.
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
import pickle
from tqdm import tqdm

from sixdegrees.sixdegrees import sixdegrees
from sixdegrees.plots.plot_trace_dispersion import plot_trace_dispersion
from sixdegrees.plots.plot_dispersion_curve import plot_dispersion_curve


def print_status(message, status="INFO"):
    """Simple status printing."""
    status_symbols = {"INFO": "✓", "ERROR": "❌", "WARNING": "⚠️", "PROCESSING": "🔄"}
    print(f"{status_symbols.get(status, '•')} {message}")


def load_config(config_file):
    """Load configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_bootstrap_config(config):
    """Extract bootstrap configuration from config dict."""
    bootstrap_config = config.get('bootstrap', {})
    if bootstrap_config.get('enabled', False):
        return {
            'n_iterations': bootstrap_config.get('n_iterations', 100),
            'stat': bootstrap_config.get('stat', 'mean'),
            'sample_fraction': bootstrap_config.get('sample_fraction', 0.8),
            'random_seed': bootstrap_config.get('random_seed', 42)
        }
    return None


def process_hour(hour, sd, config, day_beg, day_end):
    """Process single hour using plot_trace_dispersion."""
    try:
        hour_beg = day_beg + hour * 3600
        hour_end = hour_beg + 3600
        time_center = hour_beg + 1800
        
        # Create hour copy and trim with padding
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
            del rot_stream, tra_stream
        except Exception as e:
            return None
        
        # Get frequency bands
        flower, fupper, fcenter = sd_hour.get_octave_bands(
            fmin=config.get('freq_min', 0.1),
            fmax=config.get('freq_max', 2.0),
            faction_of_octave=config.get('octave_fraction', 3)
        )
        
        # Create frequency bands list
        frequency_bands = [(round(fl, 3), round(fu, 3)) for fl, fu in zip(flower, fupper)]

        if not frequency_bands:
            return None
        
        # Get bootstrap configuration
        bootstrap = None
        if config.get('bootstrap', {}).get('enabled', False):
            bootstrap = get_bootstrap_config(config)
        
        # Get backazimuth (optional - will be extracted from sd_object if not provided)
        baz = config.get('baz', None)
        
        # Process with optimization
        results = None
        try:
            # Trim to exact hour for processing
            sd_hour.trim(hour_beg, hour_end)
            
            fig, out = plot_trace_dispersion(
                sd_object=sd_hour,
                wave_type=config.get('wave_type', 'love'),
                frequency_bands=frequency_bands,
                baz=config.get('baz', 180),  # default set to 180 degres and range with +- 180 degrees
                optimized=True,
                baz_range=config.get('baz_range', 180.0),
                baz_step=config.get('baz_step', 1.0),
                regression_method=config.get('method', 'odr'),
                zero_intercept=config.get('zero_intercept', True),
                data_type=config.get('data_type', 'acceleration'),
                bootstrap=bootstrap,
                output=True,
            )
            fig.savefig(f"./output/tests/dispersion_curve_hourly_{day_beg}.png")
            plt.close(fig)  # Close figure to save memory
            
            if out is not None:
                results = {
                    'timestamp': time_center.datetime,
                    'day': time_center.datetime.date(),
                    'hour': hour,
                    'frequencies': out.get('frequencies', np.array([])),
                    'velocity_avg': out.get('velocities', np.array([])),
                    'velocity_dev': out.get('velocity_errors', np.array([])),
                    'backazimuths': out.get('backazimuths', np.array([]))  # Store determined backazimuths
                }
        except Exception as e:
            pass  # Error will be reflected in None result
        
        return results
        
    except Exception as e:
        return None


def save_hourly_results(hourly_results, output_dir, date):
    """Save hourly results to pickle file."""
    data_dir = Path(output_dir) / 'velocity_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    hourly_file = data_dir / f"gring_velocity_hourly_{date}.pkl"
    with open(hourly_file, 'wb') as f:
        pickle.dump(hourly_results, f)
    
    print_status(f"Hourly results saved: {hourly_file}")
    return hourly_file


def create_daily_plots(hourly_results, output_dir, date, config):
    """Create daily dispersion curve plots."""
    plot_dir = Path(output_dir) / 'dispersion_curves'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out None results
    results = [r for r in hourly_results if r is not None]
    
    if not results:
        print_status("No results to plot", "WARNING")
        return
    
    wave_type = config.get('wave_type', 'love').capitalize()
    
    # Create plot with all hourly estimates
    print("  Creating hourly estimates plot...", end=' ', flush=True)
    if results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for result in results:
            freq = result['frequencies']
            vel = result['velocity_avg']
            vel_err = result.get('velocity_dev', None)
            
            # Filter NaN values
            valid_mask = ~(np.isnan(freq) | np.isnan(vel))
            freq = freq[valid_mask]
            vel = vel[valid_mask]
            if vel_err is not None:
                vel_err = vel_err[valid_mask]
            
            if len(freq) > 0:
                if vel_err is not None and len(vel_err) > 0 and not np.all(np.isnan(vel_err)):
                    ax.errorbar(freq, vel, yerr=vel_err, fmt='o-', alpha=0.5, 
                               markersize=3, capsize=2, capthick=0.5)
                else:
                    ax.plot(freq, vel, 'o-', alpha=0.5, markersize=3)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Phase Velocity (m/s)', fontsize=11)
        ax.set_title(f'{wave_type} Waves - Hourly Estimates (Optimized)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        
        plt.suptitle(f'Daily Dispersion Curves - {date}', fontsize=13, y=0.995)
        plt.tight_layout()
        
        plot_file = plot_dir / f"dispersion_curve_hourly_{date}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print("✓")
    
    # Create mean dispersion curve using plot_dispersion_curve
    print("  Creating mean dispersion curve plot...", end=' ', flush=True)
    if results:
        try:
            # Combine all results
            all_freq = []
            all_vel = []
            all_vel_err = []
            all_baz = []  # Store backazimuths
            
            for result in results:
                freq = result['frequencies']
                vel = result['velocity_avg']
                vel_err = result.get('velocity_dev', None)
                baz = result.get('backazimuths', None)
                
                valid_mask = ~(np.isnan(freq) | np.isnan(vel))
                freq = freq[valid_mask]
                vel = vel[valid_mask]
                if vel_err is not None:
                    vel_err = vel_err[valid_mask]
                if baz is not None:
                    baz = baz[valid_mask]
                
                all_freq.extend(freq)
                all_vel.extend(vel)
                if vel_err is not None:
                    all_vel_err.extend(vel_err)
                if baz is not None:
                    all_baz.extend(baz)
            
            if all_freq:
                # Group by frequency and compute mean
                df = pd.DataFrame({
                    'frequencies': all_freq,
                    'velocities': all_vel,
                    'velocity_errors': all_vel_err if all_vel_err else [np.nan] * len(all_freq),
                    'backazimuths': all_baz if all_baz else [np.nan] * len(all_freq)
                })
                
                # Group by frequency band and compute statistics
                freq_groups = df.groupby('frequencies').agg({
                    'velocities': 'mean',
                    'velocity_errors': lambda x: np.sqrt(np.nanmean(x**2)) if not x.isna().all() else np.nan,
                    'backazimuths': 'mean'  # Mean backazimuth per frequency
                }).reset_index()
                
                mean_results = {
                    'frequencies': freq_groups['frequencies'].values,
                    'velocities': freq_groups['velocities'].values,
                    'velocity_errors': freq_groups['velocity_errors'].values,
                    'backazimuths': freq_groups['backazimuths'].values
                }
                
                # Plot mean dispersion curve
                wave_type_lower = config.get('wave_type', 'love').lower()
                fig = plot_dispersion_curve(
                    love_results=mean_results if wave_type_lower == 'love' else None,
                    rayleigh_results=mean_results if wave_type_lower == 'rayleigh' else None,
                    show_errors=True,
                    title=f'{wave_type} Waves - Daily Mean (Optimized)'
                )
                
                plot_file = plot_dir / f"dispersion_curve_mean_{date}.png"
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✓")
        except Exception as e:
            print(f"✗ Error: {str(e)}")


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
    n_processes = args.n_processes or config.get('n_processes', 1)
    
    print_status(f"Processing {date} with {n_processes} processes")
    
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
    
    # Process hours in parallel
    print_status("Starting parallel processing...", "PROCESSING")
    process_start = time.time()
    
    # Process with progress bar
    with Pool(processes=n_processes) as pool:
        # Create tasks
        tasks = [(hour, sd, config, day_beg, day_end) for hour in range(args.hours)]
        
        # Process with tqdm progress bar
        results = list(tqdm(
            pool.starmap(process_hour, tasks),
            total=args.hours,
            desc=f"Processing {date}",
            unit="hour",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ))
    
    process_time = time.time() - process_start
    
    # Filter successful results
    successful_results = [r for r in results if r is not None]
    
    if successful_results:
        # Save hourly results
        print_status("Saving hourly results...", "PROCESSING")
        save_hourly_results(successful_results, output_dir, date)
        
        # Create daily plots
        if config.get('plot_dispersion', True):
            print_status("Creating daily plots...", "PROCESSING")
            create_daily_plots(successful_results, output_dir, date, config)
        
        # Summary
        total_bands = sum(len(r['frequencies']) for r in successful_results)
        print_status(f"Completed: {len(successful_results)}/{args.hours} hours successful")
        print_status(f"Total frequency bands processed: {total_bands}")
        print_status(f"Processing time: {process_time:.1f}s ({process_time/args.hours:.1f}s/hour avg)")
    else:
        print_status("No data processed", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
