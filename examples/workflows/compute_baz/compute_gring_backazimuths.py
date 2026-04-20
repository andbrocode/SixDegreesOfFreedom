#!/usr/bin/env python
"""
Simplified G-ring backazimuth analysis script.
Loads data for entire days and performs hourly analysis with parallelized daily processing.
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

import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import sixdegrees
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sixdegrees.sixdegrees import sixdegrees


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"processing.log"
    
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
    """Process one hour of data for backazimuth estimation."""

    time_center = hour_beg + (hour_end-hour_beg) / 2

    hourly_df = pd.DataFrame({
        'timestamp': [time_center.datetime],
        'baz_max': [np.nan],
        'baz_mid': [np.nan],
        'baz_max_std': [np.nan],
        'baz_mid_std': [np.nan],
        'baz_max_mad': [np.nan],
        'baz_mid_mad': [np.nan],
        'count_max': [np.nan],
        'count_mid': [np.nan],
    })

    try:
        # make a copy of the sixdegrees object
        sd_hour = sd.copy()

        # trim the stream to the current hour
        sd_hour.trim(hour_beg, hour_end)

        if not sd_hour or len(sd_hour.st) == 0:
            return None
        
        # Filter data for secondary microseismic band
        sd_hour.filter_data(
            fmin=config.get('freq_min', None),
            fmax=config.get('freq_max', None),
        )

        # Compute backazimuth for Love waves
        results = sd_hour.compute_backazimuth(
            wave_type='love',
            baz_step=config.get('baz_step', None),
            baz_win_sec=config.get('baz_win_sec', None),
            baz_win_overlap=config.get('baz_win_overlap', None),
            verbose=False,
            out=True
        )

        # prepare mask for cc threshold
        mask_max = results['cc_max'] > config.get('cc_threshold', 0.0)
        mask_mid = results['cc_mid'] > config.get('cc_threshold', 0.0)

        # apply masks
        baz_max_masked = results['baz_max'][mask_max]
        cc_max_masked = results['cc_max'][mask_max]
        baz_mid_masked = results['baz_mid'][mask_mid]
        cc_mid_masked = results['cc_mid'][mask_mid]

        # compute kde stats for max approach
        if len(baz_max_masked) > 5:  # Need at least 5 points for KDE
            try:
                # get kde stats
                kde_stats = sixdegrees.get_kde_stats(
                    baz_max_masked,
                    cc_max_masked,
                    _baz_steps=0.5,
                    Ndegree=60,
                    plot=False
                )

                # add results to dataframe
                hourly_df.loc[0, 'baz_max'] = kde_stats['baz_estimate']
                hourly_df.loc[0, 'baz_max_std'] = kde_stats['kde_dev']
                hourly_df.loc[0, 'baz_max_mad'] = kde_stats['kde_mad']
                hourly_df.loc[0, 'count_max'] = len(baz_max_masked)
            except:
                pass
        if len(baz_mid_masked) > 5:  # Need at least 5 points for KDE
            try:
                # get kde stats
                kde_stats = sixdegrees.get_kde_stats(
                    baz_mid_masked,
                    cc_mid_masked,
                    _baz_steps=0.5,
                    Ndegree=60,
                    plot=False
                )

                # add results to dataframe
                hourly_df.loc[0, 'baz_mid'] = kde_stats['baz_estimate']
                hourly_df.loc[0, 'baz_mid_std'] = kde_stats['kde_dev']
                hourly_df.loc[0, 'baz_mid_mad'] = kde_stats['kde_mad']
                hourly_df.loc[0, 'count_mid'] = len(baz_mid_masked)
            except:
                pass

        gc.collect()
        del sd_hour, results

    except Exception as e:
        print(f"Error processing hour {hour_beg}: {str(e)}")
    
    return hourly_df


def process_day(args):
    """Process one day of data in hourly intervals."""
    date, config_file, output_dir = args
    
    try:
        # Setup day-specific logging
        logger = logging.getLogger(f"day_{date}")
        
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
        sd.load_data(day_beg, day_end)

        sd.trim_stream()
    
        if not sd.st or len(sd.st) == 0:
            logger.warning(f"No data available for {date}")
            return None
        
        # Process each hour of the day
        daily_results = []
        
        for hour in range(24):
            hour_beg = day_beg + hour * 3600
            hour_end = hour_beg + 3600

            hourly_df = process_hour(sd, config, hour_beg, hour_end)

            if hourly_df is not None:
                daily_results.append(hourly_df)
        
        if daily_results:
            # Combine all hourly results for the day
            daily_df = pd.concat(daily_results, ignore_index=True)
            
            # Save daily DataFrame
            data_dir = Path(output_dir) / 'backazimuth_data'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            daily_file = data_dir / f"gring_backazimuth_{date}.csv"
            daily_df.to_csv(daily_file, index=False)
            
            logger.info(f"Processed {date}: {len(daily_df)} windows, saved to {daily_file}")
            return daily_df
        else:
            logger.warning(f"No data processed for {date}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to process {date}: {str(e)}")
        return None


def aggregate_yearly_data(output_dir, year):
    """Aggregate daily dataframes into yearly dataframe."""
    try:
        data_dir = Path(output_dir) / 'backazimuth_data'
        year_data = []
        
        # Find all daily files for the year
        daily_files = list(data_dir.glob(f"gring_backazimuth_{year}-*.csv"))
        
        if not daily_files:
            print(f"No daily files found for year {year}")
            return None
        
        # Load and combine all daily data
        for daily_file in sorted(daily_files):
            try:
                daily_df = pd.read_csv(daily_file)
                year_data.append(daily_df)
            except Exception as e:
                print(f"Failed to load {daily_file}: {str(e)}")
        
        if year_data:
            yearly_df = pd.concat(year_data, ignore_index=True)
            yearly_df['year'] = year
            
            # Save yearly DataFrame
            yearly_file = data_dir / f"gring_backazimuth_{year}.csv"
            yearly_df.to_csv(yearly_file, index=False)
            
            print(f"Saved yearly results: {yearly_file} ({len(yearly_df)} records)")
            return yearly_df
        else:
            print(f"No data aggregated for year {year}")
            return None
            
    except Exception as e:
        print(f"Error aggregating yearly data: {str(e)}")
        return None


def main():
    """Main processing routine."""
    if len(sys.argv) != 4:
        print("Usage: gring_backazimuth_analysis.py config.yml start_date end_date")
        print("Example: gring_backazimuth_analysis.py config_gring_backazimuth.yml 2024-01-01 2024-01-31")
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
    
    # Setup output directory and logging
    output_dir = Path(config.get('output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting G-ring backazimuth processing from {start_date} to {end_date}")
    
    # Convert to datetime objects and generate date range
    start_dt = UTCDateTime(start_date).datetime.date()
    end_dt = UTCDateTime(end_date).datetime.date()
    dates = pd.date_range(start_dt, end_dt, freq='D')
    
    # Prepare arguments for parallel processing
    process_args = [(date.date(), config_file, output_dir) for date in dates]
    
    # Process days in parallel
    if config.get('Nprocesses', False):
        num_processes = min(cpu_count(), len(dates), config.get('Nprocesses', 4))
    else:
        num_processes = min(cpu_count(), len(dates), 4)
    
    logger.info(f"Processing {len(dates)} days using {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_day, process_args),
            total=len(process_args),
            desc="Processing days"
        ))
    
    # Filter successful results
    successful_results = [r for r in results if r is not None]
    failed_days = len(dates) - len(successful_results)
    
    logger.info(f"Processing Summary: Total days: {len(dates)}, Successful: {len(successful_results)}, Failed: {failed_days}")
        
    # Aggregate yearly data
    years = sorted(set([d.year for d in dates]))
    for year in years:
        yearly_df = aggregate_yearly_data(output_dir, year)

    logger.info(f"Processing completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
