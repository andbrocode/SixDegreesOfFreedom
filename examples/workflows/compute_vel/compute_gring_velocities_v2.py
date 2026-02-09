#!/usr/bin/env python
"""
G-ring velocity dispersion curve computation script (v2).

Simplified version that:
1. Loads data once for the entire day
2. Precomputes frequency bands and time window parameters
3. Parallelizes by frequency band
4. For each frequency band: computes backazimuths per time window, then velocity estimates

Author: Generated for sixdegrees package
Date: 2024
"""

import os
import gc
import sys
import copy
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from obspy import UTCDateTime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from sixdegrees.sixdegrees import sixdegrees
from sixdegrees.utils.get_kde_stats_velocity import get_kde_stats_velocity

import warnings
warnings.filterwarnings('ignore')



def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "velocity_processing.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def process_frequency_band(args: Tuple) -> Optional[Dict]:
    """
    Process a single frequency band for the entire day.
    
    compute_backazimuth and compute_velocities_optimized handle time windows internally.
    Uses sd.filter_data() to filter the data.
    
    Parameters:
    -----------
    args : tuple
        (sd_copy, config, day_beg, day_end, fmin, fmax, fcenter, win_time_s)
        
    Returns:
    --------
    dict or None
        Dictionary with frequency band results:
        - fband: center frequency
        - velocity: KDE maximum velocity
        - deviation: KDE deviation
        - n_measurements: number of valid measurements
        - mean_cc: mean cross-correlation
        - mean_r_squared: mean R-squared
    """
    (sd_copy, config, day_beg, day_end, fmin, fmax, fcenter, win_time_s) = args
    
    try:
        # Use filter_data to filter the frequency band
        sd_copy.filter_data(fmin=fmin, fmax=fmax)
        
        # Check if streams are valid
        rot_stream = sd_copy.get_stream("rotation", raw=False)
        tra_stream = sd_copy.get_stream("translation", raw=False)
        
        if len(rot_stream) == 0 or len(tra_stream) == 0:
            return None
        else:
            del rot_stream, tra_stream
        
        # Compute backazimuths (handles time windows internally)
        try:
            baz_results = sd_copy.compute_backazimuth(
                wave_type=config.get('wave_type', 'love'),
                baz_step=1,
                baz_win_sec=win_time_s,
                baz_win_overlap=config.get('overlap', 0.5),
                verbose=False,
                out=True
            )
        except Exception:
            return None
        
        if baz_results is None or len(baz_results.get('backazimuth', [])) == 0:
            return None
        
        # Compute velocities (processes all time windows from baz_results)
        try:
            vel_results = sd_copy.compute_velocities_optimized(
                wave_type=config.get('wave_type', 'love'),
                baz_results=baz_results,
                cc_method=config.get('cc_method', 'max'),
                method=config.get('method', 'odr'),
                cc_threshold=config.get('cc_threshold', 0.5),
                r_squared_threshold=config.get('r_squared_threshold', 0.0),
                zero_intercept=True,
                verbose=False,
                plot=False
            )
        except Exception:
            return None
        
        if vel_results is None:
            return None
        
        # Extract all measurements (from all time windows)
        velocities = vel_results.get('velocity', [])
        cc_values = vel_results.get('ccoef', [])
        r_squared_values = vel_results.get('r_squared', [])
        
        # Filter NaN values
        valid_mask = ~(np.isnan(velocities) | np.isnan(cc_values))
        if not np.any(valid_mask):
            return None
        
        velocities = np.array(velocities)[valid_mask]
        cc_values = np.array(cc_values)[valid_mask]
        r_squared_values = np.array(r_squared_values)[valid_mask] if len(r_squared_values) > 0 else np.array([np.nan] * len(velocities))
        
        # Check if we have enough measurements
        if len(velocities) < 5:
            return None
        
        # Apply KDE analysis
        try:
            kde_result = get_kde_stats_velocity(velocities, cc_values)
            vel_max = kde_result['max']
            vel_dev = kde_result['dev']
        except Exception:
            return None
        
        # Return results
        return {
            'timestamp': day_beg.datetime,
            'day': day_beg.datetime.date(),
            'fband': fcenter,
            'velocity': vel_max,
            'deviation': vel_dev,
            'n_measurements': len(velocities),
            'mean_cc': np.nanmean(cc_values),
            'mean_r_squared': np.nanmean(r_squared_values)
        }
        
    except Exception:
        return None
    finally:
        # Cleanup
        gc.collect()


def process_time_window(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Process a time window (e.g., 6 hours) of data.
    
    1. Load data for the time window
    2. Precompute frequency bands and time window parameters
    3. Process frequency bands in parallel
    4. Return results
    
    Parameters:
    -----------
    args : tuple
        (time_beg, time_end, config_file, verbose)
    """
    time_beg, time_end, config_file, verbose = args
    
    try:
        logger = logging.getLogger(f"window_{time_beg}")
        
        # Load config
        config = sixdegrees.load_from_yaml(config_file)
        config['tbeg'] = time_beg
        config['tend'] = time_end
        
        # Initialize sixdegrees object
        sd = sixdegrees(config)
        
        # Add dummy event_info for noise analysis
        sd.event_info = {
            'origin_time': time_beg,
            'latitude': None,
            'longitude': None,
            'depth_km': None,
            'magnitude': None,
            'magnitude_type': None,
            'distance_km': None,
            'distance_deg': None,
            'azimuth': None,
            'backazimuth': None,
            'catalog': 'noise',
            'event_id': f'noise_{time_beg}'
        }
        
        # Load data for the time window
        if config.get('resample_rate', None) is not None:
            sd.load_data(time_beg, time_end, resample_rate=config.get('resample_rate'))
        else:
            sd.load_data(time_beg, time_end)
        
        sd.trim_stream()
        
        if not sd.st or len(sd.st) == 0:
            return None
        
        # Check streams
        try:
            rot_stream = sd.get_stream("rotation", raw=True)
            tra_stream = sd.get_stream("translation", raw=True)
            if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                return None
            else:
                del rot_stream, tra_stream
        except Exception:
            return None
        
        # Precompute frequency bands
        flower, fupper, fmid = sd.get_octave_bands(
            fmin=config.get('freq_min', 0.01),
            fmax=config.get('freq_max', 1.0),
            faction_of_octave=config.get('octave_fraction', 3)
        )
        
        # Precompute time window parameters for each frequency band
        t_win_factor = config.get('t_win_factor', 8.0)
        
        freq_args = []
        for fmin, fmax, fcenter in zip(flower, fupper, fmid):
            fmin, fmax, fcenter = round(fmin, 3), round(fmax, 3), round(fcenter, 3)
            
            # Calculate window length for this frequency
            win_time_s = max(10, int(t_win_factor / fcenter))
            
            # Create a copy of sd object for this frequency band
            sd_copy = copy.deepcopy(sd)
            
            # Prepare arguments
            freq_args.append((
                sd_copy,
                config,
                time_beg,
                time_end,
                fmin,
                fmax,
                fcenter,
                win_time_s
            ))
        
        # Process frequency bands sequentially (to avoid nested multiprocessing)
        # Note: We can't use Pool here because process_time_window is called from a Pool
        # and daemonic processes cannot spawn children
        window_results = []
        for args in freq_args:
            result = process_frequency_band(args)
            if result is not None:
                window_results.append(result)
        
        # Cleanup
        del sd
        gc.collect()
        
        if window_results:
            return pd.DataFrame(window_results)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to process window {time_beg} to {time_end}: {str(e)}")
        return None
    finally:
        gc.collect()


def process_day(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Process one day by subdividing into smaller time windows.
    
    Parameters:
    -----------
    args : tuple
        (date, config_file, output_dir, verbose)
    """
    date, config_file, output_dir, verbose = args
    
    try:
        logger = logging.getLogger(f"day_{date}")
        
        if verbose:
            print(f"\nStarting processing for {date}")
        
        # Create day start and end times
        day_beg = UTCDateTime(date)
        day_end = day_beg + 24*3600
        
        # Load config
        config = sixdegrees.load_from_yaml(config_file)
        
        # Get time window size (in hours, default 6 hours)
        hours_per_window = config.get('hours_per_window', 6)
        window_duration = hours_per_window * 3600
        
        # Generate time windows for the day
        time_windows = []
        time_current = day_beg
        while time_current < day_end:
            window_end = min(time_current + window_duration, day_end)
            time_windows.append((time_current, window_end))
            time_current = window_end
        
        if verbose:
            print(f"  Subdividing day into {len(time_windows)} time windows ({hours_per_window}h each)")
        
        # Process time windows in parallel
        n_window_processes = config.get('n_window_processes', config.get('n_processes', 4))
        if n_window_processes is None or n_window_processes < 1:
            n_window_processes = 1
        
        window_args = [(tw_beg, tw_end, config_file, verbose) for tw_beg, tw_end in time_windows]
        
        daily_results = []
        if n_window_processes > 1 and len(window_args) > 1:
            if verbose:
                print(f"  Processing {len(window_args)} windows with {min(n_window_processes, len(window_args))} parallel processes")
            with Pool(processes=min(n_window_processes, len(window_args))) as pool:
                window_dfs = list(tqdm(
                    pool.imap(process_time_window, window_args),
                    total=len(window_args),
                    desc=f"Time windows ({date})",
                    disable=verbose,
                    unit="window",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                ))
                daily_results = [df for df in window_dfs if df is not None]
        else:
            # Sequential processing
            if verbose:
                print(f"  Processing {len(window_args)} windows sequentially:")
            for i, args in enumerate(window_args, 1):
                tw_beg, tw_end = args[0], args[1]
                if verbose:
                    print(f"    [{i}/{len(window_args)}] Window {tw_beg.strftime('%H:%M')}-{tw_end.strftime('%H:%M')}...", end=' ', flush=True)
                result = process_time_window(args)
                if result is not None:
                    daily_results.append(result)
                    if verbose:
                        print(f"✓ ({len(result)} bands)")
                else:
                    if verbose:
                        print("✗")
        
        if daily_results:
            # Combine all window results for the day
            daily_df = pd.concat(daily_results, ignore_index=True)
            
            # Save
            data_dir = Path(output_dir) / 'velocity_data'
            data_dir.mkdir(parents=True, exist_ok=True)
            daily_file = data_dir / f"gring_velocity_{date}.csv"
            daily_df.to_csv(daily_file, index=False)
            
            if verbose:
                total_measurements = daily_df['n_measurements'].sum()
                avg_velocity = daily_df['velocity'].mean()
                print(f"  Saved results: {len(daily_df)} bands, {total_measurements} total measurements, avg velocity: {avg_velocity:.1f} m/s")
            
            # Create plot if enabled
            if config.get('plot_dispersion', False):
                if verbose:
                    print(f"  Creating dispersion curve plot...", end=' ', flush=True)
                plot_dispersion_curve(daily_df, output_dir, date)
                if verbose:
                    print("✓")
            
            logger.info(f"Processed {date}: {len(daily_df)} frequency bands, {daily_df['n_measurements'].sum()} measurements")
            if verbose:
                print(f"  ✓ Day {date} completed successfully")
            
            return daily_df
        else:
            logger.warning(f"No data processed for {date}")
            if verbose:
                print(f"  ✗ No data processed for {date}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to process {date}: {str(e)}")
        if verbose:
            print(f"Error processing {date}: {str(e)}")
        return None
    finally:
        gc.collect()


def plot_dispersion_curve(daily_df: pd.DataFrame, output_dir: Path, date: str) -> None:
    """Plot the velocity dispersion curve for a single day."""
    try:
        if daily_df.empty:
            return
        
        # Group by frequency band
        freq_stats = daily_df.groupby('fband').agg({
            'velocity': ['mean', 'std', 'count'],
            'deviation': 'mean',
            'mean_cc': 'mean',
            'mean_r_squared': 'mean'
        }).reset_index()
        
        freq_stats.columns = ['fband', 'velocity_mean', 'velocity_std', 'count', 
                             'deviation_mean', 'mean_cc', 'mean_r_squared']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Velocity vs Frequency
        ax1.errorbar(freq_stats['fband'], freq_stats['velocity_mean'], 
                    yerr=freq_stats['velocity_std'], 
                    fmt='o-', capsize=5, capthick=2, markersize=6)
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'Velocity Dispersion Curve - {date}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Number of measurements
        ax2.bar(freq_stats['fband'], freq_stats['count'], width=freq_stats['fband']*0.1)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Number of Measurements', fontsize=12)
        ax2.set_title(f'Data Coverage by Frequency Band - {date}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = Path(output_dir) / 'dispersion_curves'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_dir / f"dispersion_curve_{date}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot for {date}: {str(e)}")


def main():
    """Main processing routine."""
    if len(sys.argv) != 4:
        print("Usage: compute_gring_velocities_v2.py config.yml start_date end_date")
        print("Example: compute_gring_velocities_v2.py config_gring_velocity.yml 2024-01-01 2024-01-31")
        sys.exit(1)
    
    if not os.path.exists(sys.argv[1]):
        print(f"Config file {sys.argv[1]} does not exist")
        sys.exit(1)
    
    config_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    verbose = config.get('verbose', False)
    
    # Setup output directory and logging
    output_dir = Path(config.get('output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting G-ring velocity processing from {start_date} to {end_date}")
    
    if verbose:
        print(f"Configuration loaded from: {config_file}")
        print(f"Processing period: {start_date} to {end_date}")
        print(f"Output directory: {output_dir}")
        print(f"Frequency band processes: {config.get('n_processes', 4)}")
    
    # Generate date range
    start_dt = UTCDateTime(start_date).datetime.date()
    end_dt = UTCDateTime(end_date).datetime.date()
    dates = pd.date_range(start_dt, end_dt, freq='D')
    
    # Process days sequentially
    process_args = [(date.date(), config_file, output_dir, verbose) for date in dates]
    
    logger.info(f"Processing {len(dates)} days sequentially")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {len(dates)} days: {start_date} to {end_date}")
        print(f"{'='*60}\n")
    
    # Process days with progress bar
    results = []
    successful_days = []
    failed_days = []
    
    for i, args in enumerate(tqdm(process_args, desc="Processing days", disable=verbose, 
                                   unit="day", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        date_str = str(args[0])
        if verbose:
            print(f"\n[{i+1}/{len(dates)}] Day {date_str}")
        result = process_day(args)
        if result is not None:
            results.append(result)
            successful_days.append(date_str)
        else:
            failed_days.append(date_str)
    
    # Summary
    successful = len(results)
    failed = len(dates) - successful
    
    logger.info(f"Processing Summary: Total days: {len(dates)}, Successful: {successful}, Failed: {failed}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"  Total days:        {len(dates)}")
        print(f"  Successful:       {successful} ({successful/len(dates)*100:.1f}%)")
        print(f"  Failed:           {failed}")
        
        if successful > 0:
            total_bands = sum(len(df) for df in results)
            total_measurements = sum(df['n_measurements'].sum() for df in results)
            avg_velocity = np.mean([df['velocity'].mean() for df in results])
            print(f"\n  Results:")
            print(f"    Frequency bands:  {total_bands}")
            print(f"    Measurements:     {total_measurements:,}")
            print(f"    Avg velocity:     {avg_velocity:.1f} m/s")
        
        data_dir = Path(output_dir) / 'velocity_data'
        if data_dir.exists():
            daily_files = list(data_dir.glob("gring_velocity_*.csv"))
            print(f"\n  Output files:")
            print(f"    CSV files:        {len(daily_files)}")
        
        if config.get('plot_dispersion', False):
            plot_dir = Path(output_dir) / 'dispersion_curves'
            if plot_dir.exists():
                plot_files = list(plot_dir.glob("dispersion_curve_*.png"))
                print(f"    Plot files:       {len(plot_files)}")
        
        if failed_days:
            print(f"\n  Failed days:       {', '.join(failed_days[:5])}")
            if len(failed_days) > 5:
                print(f"                     ... and {len(failed_days) - 5} more")
        
        print(f"{'='*60}\n")
    
    logger.info(f"Processing completed! Results saved to: {output_dir}")
    if verbose:
        print(f"\nProcessing completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
