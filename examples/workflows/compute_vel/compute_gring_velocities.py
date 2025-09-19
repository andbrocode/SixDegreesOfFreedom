#!/usr/bin/env python
"""
G-ring velocity dispersion curve computation script.
Loads data for entire days and performs frequency-dependent velocity analysis with parallelized daily processing.
"""

import os
import gc
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from obspy import UTCDateTime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from sixdegrees.plots.plot_backazimuth_results import plot_backazimuth_results
from sixdegrees.plots.plot_velocities import plot_velocities
from sixdegrees.utils.get_kde_stats_velocity import get_kde_stats_velocity

import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import sixdegrees
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sixdegrees.sixdegrees import sixdegrees



def setup_logging(output_dir):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"velocity_processing.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def process_hour(sd, config, hour_beg, hour_end):
    """Process one hour of data for velocity dispersion curve estimation."""

    time_center = hour_beg + (hour_end-hour_beg) / 2

    # Initialize DataFrame for this hour
    hourly_df = pd.DataFrame({
        'timestamp': [time_center.datetime],
        'day': [time_center.datetime.date()],
        'fband': [np.nan],
        'velocity': [np.nan],
        'deviation': [np.nan],
        'n_measurements': [np.nan]
    })

    try:
        # make a copy of the sixdegrees object
        sd_hour = sd.copy()

        # trim the stream to the current hour with 100 seconds padding
        sd_hour.trim(hour_beg-100, hour_end+100)

        if not sd_hour or len(sd_hour.st) == 0:
            return None
        
        # Check if we have both rotation and translation data
        try:
            rot_stream = sd_hour.get_stream("rotation", raw=True)
            tra_stream = sd_hour.get_stream("translation", raw=True)
            if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                print(f"  ✗ Missing rotation or translation data for hour {hour_beg}")
                return None
        except Exception as e:
            print(f"  ✗ Error accessing rotation/translation streams: {str(e)}")
            return None

        # Get frequency bands
        flower, fupper, fmid = sd_hour.get_octave_bands(
            fmin=config.get('freq_min', 0.01),
            fmax=config.get('freq_max', 1.0),
            faction_of_octave=config.get('octave_fraction', 3)
        )
                
        # Process each frequency band individually for better KDE analysis
        hourly_results = []
        
        for fmin, fmax, fcenter in zip(flower, fupper, fmid):

            # adjust to 3 decimal places
            fmin, fmax, fcenter = round(fmin, 3), round(fmax, 3), round(fcenter, 3)
            print(f"  Processing frequency band {fcenter} Hz ({fmin} - {fmax} Hz)")

            try:
                # Filter data for this frequency band
                sd_filtered = sd_hour.copy()

                # apply bandpass filter
                sd_filtered.filter_data(fmin=fmin, fmax=fmax)
                
                # trim the stream to the current hour
                sd_filtered.trim(hour_beg, hour_end)

                if not sd_filtered.st or len(sd_filtered.st) == 0:
                    continue
                
                # Check if we have both rotation and translation data after filtering
                try:
                    rot_stream = sd_filtered.get_stream("rotation", raw=True)
                    tra_stream = sd_filtered.get_stream("translation", raw=True)
                    if not rot_stream or len(rot_stream) == 0 or not tra_stream or len(tra_stream) == 0:
                        continue
                except Exception as e:
                    continue
                
                # Calculate window length based on center frequency
                t_win_factor = config.get('t_win_factor', 2.0)
                win_time_s = int(t_win_factor / fcenter)
                
                # Compute backazimuths first
                try:
                    baz_results = sd_filtered.compute_backazimuth(
                        wave_type=config.get('wave_type', 'love'),
                        baz_step=1,
                        baz_win_sec=win_time_s,
                        baz_win_overlap=config.get('overlap', 0.5),
                        verbose=False,
                        out=True
                    )

                    fig = plot_backazimuth_results(
                        sd_filtered, 
                        baz_results,
                        baz_theo=300,
                        cc_threshold=0.5,
                        cc_method='max'
                    )
                except Exception as e:
                    print(f"  ✗ Error computing backazimuths for {fcenter:.3f} Hz: {str(e)}")
                    continue
                
                if baz_results is None:
                    continue
                
                # Compute velocities using the optimized function
                try:
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
                    vel_results['parameters']['baz'] = 300
                    fig = plot_velocities(
                        sd_filtered,
                        velocity_results=vel_results,
                        vmax=6000,  # optional
                        minors=True, # optional
                        cc_threshold=0.7,
                    )
                except Exception as e:
                    print(f"  ✗ Error computing velocities for {fcenter:.3f} Hz: {str(e)}")
                    continue
                
                if vel_results is None:
                    continue
                
                # Extract valid velocity measurements
                mask = ~np.isnan(vel_results['velocity'])
                velocities = vel_results['velocity'][mask]
                cc_values = vel_results['ccoef'][mask]
                
                if len(velocities) < 5:  # Need at least 5 measurements for KDE
                    continue
                
                # Apply KDE analysis to get maximum and deviation
                try:
                    # Use a proper velocity KDE function instead of backazimuth KDE
                    out = get_kde_stats_velocity(velocities, cc_values)
                    vel_max = out['max']
                    vel_dev = out['dev']
                
                except Exception as e:
                    vel_max = np.nan
                    vel_dev = np.nan
                
                # Store results for this frequency band
                freq_result = {
                    'timestamp': time_center.datetime,
                    'day': time_center.datetime.date(),
                    'fband': fcenter,
                    'velocity': vel_max,
                    'deviation': vel_dev,
                    'n_measurements': len(velocities)
                }

                hourly_results.append(freq_result)

                del sd_filtered

            except Exception as e:
                print(f"Error processing frequency band {fcenter:.3f} Hz: {str(e)}")
                continue

        if hourly_results:
            hourly_df = pd.DataFrame(hourly_results)
        else:
            return None

        gc.collect()
        del sd_hour, vel_results, baz_results

    except Exception as e:
        print(f"Error processing hour {hour_beg}: {str(e)}")
        return None
    
    return hourly_df


def process_day(args):
    """Process one day of data in hourly intervals."""
    date, config_file, output_dir, verbose = args
    
    try:
        # Setup day-specific logging
        logger = logging.getLogger(f"day_{date}")
        
        if verbose:
            print(f"Starting processing for {date}")
        
        # Create day start and end times
        day_beg = UTCDateTime(date)
        day_end = day_beg + 24*3600
        
        # Read config file
        config = sixdegrees.load_from_yaml(config_file)

        config['tbeg'] = day_beg
        config['tend'] = day_end

        # Initialize sixdegrees object
        sd = sixdegrees(config)
        
        # Request data for the entire day
        if verbose:
            print(f"Loading data for {date}...")
        sd.load_data(day_beg, day_end)

        # ensure the stream has traces with the same length
        sd.trim_stream()
    
        if not sd.st or len(sd.st) == 0:
            logger.warning(f"No data available for {date}")
            if verbose:
                print(f"No data available for {date}")
            return None
        
        if verbose:
            print(f"Processing 24 hours for {date}...")
        
        # Process each hour of the day
        daily_results = []
        
        for hour in range(24):
            if verbose and hour % 6 == 0:  # Show progress every 6 hours
                print(f"  Processing hour {hour:2d}/23 for {date}")
            
            hour_beg = day_beg + hour * 3600
            hour_end = hour_beg + 3600

            hourly_df = process_hour(sd, config, hour_beg, hour_end)

            if hourly_df is not None:
                daily_results.append(hourly_df)
        
        if daily_results:
            # Combine all hourly results for the day
            daily_df = pd.concat(daily_results, ignore_index=True)
            
            # Save daily DataFrame
            data_dir = Path(output_dir) / 'velocity_data'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            daily_file = data_dir / f"gring_velocity_{date}.csv"
            daily_df.to_csv(daily_file, index=False)
            
            # Create daily dispersion curve plot if enabled
            if config.get('plot_dispersion', False):
                plot_dispersion_curve(daily_df, output_dir, date)
            
            logger.info(f"Processed {date}: {len(daily_df)} frequency bands, saved to {daily_file}")
            if verbose:
                print(f"Completed {date}: {len(daily_df)} frequency bands processed")
            return daily_df
        else:
            logger.warning(f"No data processed for {date}")
            if verbose:
                print(f"No data processed for {date}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to process {date}: {str(e)}")
        if verbose:
            print(f"Error processing {date}: {str(e)}")
        return None


def plot_dispersion_curve(daily_df, output_dir, date):
    """Plot the velocity dispersion curve for a single day."""
    try:
        if daily_df.empty:
            print(f"No data to plot for {date}")
            return
        
        # Group by frequency band and compute statistics
        freq_stats = daily_df.groupby('fband').agg({
            'velocity': ['mean', 'std', 'count'],
            'deviation': 'mean'
        }).reset_index()
        
        # Flatten column names
        freq_stats.columns = ['fband', 'velocity_mean', 'velocity_std', 'count', 'deviation_mean']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Velocity vs Frequency
        ax1.errorbar(freq_stats['fband'], freq_stats['velocity_mean'], 
                    yerr=freq_stats['velocity_std'], 
                    fmt='o-', capsize=5, capthick=2, markersize=6)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title(f'Velocity Dispersion Curve - {date}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Number of measurements vs Frequency
        ax2.bar(freq_stats['fband'], freq_stats['count'], width=freq_stats['fband']*0.1)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Number of Measurements')
        ax2.set_title(f'Data Coverage by Frequency Band - {date}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = Path(output_dir) / 'dispersion_curves'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_dir / f"dispersion_curve_{date}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Daily dispersion curve plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"Error creating daily dispersion curve plot for {date}: {str(e)}")


def main():
    """Main processing routine."""
    if len(sys.argv) != 4:
        print("Usage: compute_gring_velocities.py config.yml start_date end_date")
        print("Example: compute_gring_velocities.py config_gring_velocity.yml 2024-01-01 2024-01-31")
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
    
    # Get verbose setting from configuration
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
        print(f"Daily dispersion curve plotting: {config.get('plot_dispersion', False)}")
    
    # Convert to datetime objects and generate date range
    start_dt = UTCDateTime(start_date).datetime.date()
    end_dt = UTCDateTime(end_date).datetime.date()
    dates = pd.date_range(start_dt, end_dt, freq='D')
    
    # Prepare arguments for parallel processing
    process_args = [(date.date(), config_file, output_dir, verbose) for date in dates]
    
    # Process days in parallel
    if config.get('Nprocesses', False):
        num_processes = min(cpu_count(), len(dates), config.get('Nprocesses', 4))
    else:
        num_processes = min(cpu_count(), len(dates), 4)
    
    logger.info(f"Processing {len(dates)} days using {num_processes} processes")
    
    if verbose:
        print(f"Processing {len(dates)} days using {num_processes} processes")
        print("Starting parallel processing...")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_day, process_args),
            total=len(process_args),
            desc="Processing days",
            disable=verbose  # Disable tqdm if verbose mode is on
        ))
    
    # Filter successful results
    successful_results = [r for r in results if r is not None]
    failed_days = len(dates) - len(successful_results)
    
    logger.info(f"Processing Summary: Total days: {len(dates)}, Successful: {len(successful_results)}, Failed: {failed_days}")
    
    if verbose:
        print(f"\nProcessing Summary:")
        print(f"  Total days: {len(dates)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {failed_days}")
        print(f"  Success rate: {len(successful_results)/len(dates)*100:.1f}%")
        
        # Show output files
        data_dir = Path(output_dir) / 'velocity_data'
        if data_dir.exists():
            daily_files = list(data_dir.glob("gring_velocity_*.csv"))
            print(f"  Daily data files created: {len(daily_files)}")
        
        if config.get('plot_dispersion', False):
            plot_dir = Path(output_dir) / 'dispersion_curves'
            if plot_dir.exists():
                plot_files = list(plot_dir.glob("dispersion_curve_*.png"))
                print(f"  Daily dispersion curve plots created: {len(plot_files)}")

    logger.info(f"Processing completed! Results saved to: {output_dir}")
    if verbose:
        print(f"\nProcessing completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
