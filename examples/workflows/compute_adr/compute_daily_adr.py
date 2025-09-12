#!/usr/bin/env python
"""
Script to create daily continuous 6-DoF data from array data and store it in SDS format.
"""

import os
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
from sixdegrees.seismicarray import seismicarray


class ProcessingStats:
    """Class to track processing statistics."""
    def __init__(self):
        self.dates = []
        self.station_counts = []
        self.success = []
        self.processing_times = []
        
    def add_day(self, date, station_count, success, proc_time):
        self.dates.append(date)
        self.station_counts.append(station_count)
        self.success.append(success)
        self.processing_times.append(proc_time)
        
    def to_dataframe(self):
        return pd.DataFrame({
            'date': self.dates,
            'station_count': self.station_counts,
            'success': self.success,
            'processing_time': self.processing_times
        })
        
    def plot_summary(self, output_dir):
        """Create summary plot of processing statistics."""
        df = self.to_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot station counts
        ax1.plot(df['date'], df['station_count'], 'b-', label='Stations Used')
        ax1.set_ylabel('Number of Stations')
        ax1.grid(True)
        ax1.legend()
        
        # Plot processing time
        ax2.plot(df['date'], df['processing_time'], 'g-', label='Processing Time')
        ax2.set_ylabel('Processing Time [s]')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'processing_summary.png')
        plt.close()

def setup_logging(log_dir):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"adr_processing_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

def process_day(array, date, sds_root, config):
    """Process one day of data."""
    try:
        starttime = UTCDateTime(date)
        endtime = starttime + 24*3600  # Add 24 hours
        proc_start = datetime.now()
        
        # Get inventories
        array.request_inventories(starttime, endtime)

        # Get and process waveforms
        array.request_waveforms(
            starttime=starttime,
            endtime=endtime,
            remove_response=True,
            detrend=True,
            taper=True,
            filter_params=config.get('filter_params', {
                'type': 'bandpass',
                'freqmin': 0.001,  # 1000s period
                'freqmax': 1.0,     # 0.1s period
                'corners': 4,
                'zerophase': True
            }),
        )
        
        # Count valid stations after waveform processing
        valid_stations = len(set(tr.stats.station for tr in array.stream))
        
        # Compute ADR
        array.compute_adr()
        
        # Save as 6-DoF data in SDS format
        array.save_6dof_data(
            output_format='sds',
            output_path=sds_root
        )
        
        proc_time = (datetime.now() - proc_start).total_seconds()
        logging.info(f"Successfully processed {date} with {valid_stations} stations in {proc_time:.1f}s")
        
        return True, valid_stations, proc_time
        
    except Exception as e:
        logging.error(f"Failed to process {date}: {str(e)}")
        return False, 0, 0

def main():
    """Main processing routine."""
    # Load command line arguments
    if len(sys.argv) != 2:
        print("Usage: create_daily_adr.py config.yml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration
    config = load_config(config_file)
    
    # Setup paths
    base_dir = Path(config.get('base_dir', '.'))
    sds_root = base_dir / config.get('sds_dir', 'SDS')
    log_dir = base_dir / config.get('log_dir', 'logs')
    
    # Create directories
    for directory in [sds_root, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir)
    
    # Initialize array
    try:
        array = seismicarray(config_file)
        logging.info("Array initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize array: {str(e)}")
        sys.exit(1)
    
    # Pre-compute all dates
    start_date = UTCDateTime(config['start_date']).datetime.date()
    end_date = UTCDateTime(config['end_date']).datetime.date()
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Initialize statistics
    stats = ProcessingStats()
    successful_days = 0
    failed_days = 0
    
    # Process each day with progress bar
    logging.info(f"Processing {len(dates)} days from {start_date} to {end_date}")
    
    for date in tqdm(dates, desc="Processing days", unit="day"):
        success, station_count, proc_time = process_day(array, date.date(), sds_root, config)
        
        # Update statistics
        stats.add_day(date.date(), station_count, success, proc_time)
        
        if success:
            successful_days += 1
        else:
            failed_days += 1
    
    # Create summary plot
    stats.plot_summary(log_dir)
    
    # Save statistics to CSV
    stats_file = log_dir / 'processing_stats.csv'
    stats.to_dataframe().to_csv(stats_file, index=False)
    
    # Log summary
    logging.info(f"Processing Summary:")
    logging.info(f"Total days processed: {len(dates)}")
    logging.info(f"Successful: {successful_days}")
    logging.info(f"Failed: {failed_days}")
    logging.info(f"Average stations per day: {np.mean(stats.station_counts):.1f}")
    logging.info(f"Average processing time: {np.mean(stats.processing_times):.1f}s")
    logging.info(f"Summary plot saved to: {log_dir}/processing_summary.png")
    logging.info(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
