#!/usr/bin/env python3
"""
Merge backazimuth data from CSV files for a given date range.
Based on the aggregate function from gring_backazimuth_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def merge_backazimuth_data(start_date, end_date, data_path, output_file=None, verbose=False):
    """
    Merge backazimuth data from CSV files for a given date range.
    
    Parameters:
    -----------
    start_date : str or datetime
        Start date in format 'YYYY-MM-DD' or datetime object
    end_date : str or datetime  
        End date in format 'YYYY-MM-DD' or datetime object
    data_path : str or Path
        Path to directory containing backazimuth CSV files
    output_file : str or Path, optional
        Output file path. If None, will be auto-generated based on date range
    verbose : bool, optional
        Enable verbose logging
        
    Returns:
    --------
    pd.DataFrame or None
        Merged DataFrame if successful, None if no data found
    """
    
    logger = setup_logging(verbose)
    
    try:
        # Convert dates to datetime objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        # Ensure data_path is a Path object
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return None
            
        if not data_path.is_dir():
            logger.error(f"Data path is not a directory: {data_path}")
            return None
        
        logger.info(f"Merging backazimuth data from {start_date.date()} to {end_date.date()}")
        logger.info(f"Searching in directory: {data_path}")
        
        # Generate date range
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        logger.info(f"Date range: {len(date_range)} days")
        
        # Find all daily files for the date range
        daily_files = []
        for date in date_range:
            # Look for files with pattern: gring_backazimuth_YYYY-MM-DD.csv
            daily_file = data_path / f"gring_backazimuth_{date.strftime('%Y-%m-%d')}.csv"
            if daily_file.exists():
                daily_files.append(daily_file)
            else:
                logger.debug(f"File not found: {daily_file}")
        
        if not daily_files:
            logger.warning(f"No daily files found in {data_path}")
            logger.info("Looking for files with pattern: gring_backazimuth_YYYY-MM-DD.csv")
            return None
        
        logger.info(f"Found {len(daily_files)} daily files to merge")
        
        # Load and combine all daily data
        merged_data = []
        successful_files = 0
        failed_files = 0
        
        for daily_file in sorted(daily_files):
            try:
                logger.debug(f"Loading: {daily_file}")
                daily_df = pd.read_csv(daily_file)
                
                # Check if DataFrame has data
                if len(daily_df) > 0:
                    merged_data.append(daily_df)
                    successful_files += 1
                    logger.debug(f"Loaded {len(daily_df)} records from {daily_file.name}")
                else:
                    logger.debug(f"Empty file: {daily_file.name}")
                    
            except Exception as e:
                logger.error(f"Failed to load {daily_file}: {str(e)}")
                failed_files += 1
        
        if not merged_data:
            logger.warning("No data loaded from any files")
            return None
        
        # Combine all data
        logger.info(f"Combining data from {successful_files} files...")
        merged_df = pd.concat(merged_data, ignore_index=True)
        
        # Add year column if not present
        if 'year' not in merged_df.columns:
            merged_df['year'] = merged_df['timestamp'].apply(
                lambda x: pd.to_datetime(x).year if pd.notna(x) else None
            )
        
        # Sort by timestamp
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully merged {len(merged_df)} records")
        
        # Generate output filename if not provided
        if output_file is None:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            if start_str == end_str:
                output_file = data_path / f"gring_backazimuth_{start_str}_merged.csv"
            else:
                output_file = data_path / f"gring_backazimuth_{start_str}_to_{end_str}_merged.csv"
        
        # Ensure output_file is a Path object
        output_file = Path(output_file)
        
        # Save merged DataFrame
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved merged data to: {output_file}")
        
        # Print summary statistics
        print_summary(merged_df, logger)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging backazimuth data: {str(e)}")
        return None


def print_summary(df, logger):
    """Print summary statistics of the merged data."""
    
    logger.info("=" * 50)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"Total records: {len(df)}")
    
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) > 0:
            logger.info(f"Date range: {valid_timestamps.min()} to {valid_timestamps.max()}")
        else:
            logger.warning("No valid timestamps found")
    
    # Count non-null values for each backazimuth column
    baz_columns = ['baz_max', 'baz_mid']
    for col in baz_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            logger.info(f"{col}: {non_null_count} non-null values")
    
    # Count files processed
    if 'year' in df.columns:
        years = df['year'].unique()
        logger.info(f"Years covered: {sorted(years)}")
    
    logger.info("=" * 50)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge backazimuth data from CSV files for a given date range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 merge_backazimuth_data.py 2024-01-01 2024-01-31 /path/to/data
  python3 merge_backazimuth_data.py 2024-01-01 2024-01-31 /path/to/data -o merged_data.csv
  python3 merge_backazimuth_data.py 2024-01-01 2024-01-31 /path/to/data --verbose
        """
    )
    
    parser.add_argument('start_date', 
                       help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('end_date', 
                       help='End date in YYYY-MM-DD format')
    
    parser.add_argument('data_path', 
                       help='Path to directory containing backazimuth CSV files')
    
    parser.add_argument('-o', '--output', 
                       help='Output file path (optional, auto-generated if not provided)')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        result = merge_backazimuth_data(
            start_date=args.start_date,
            end_date=args.end_date,
            data_path=args.data_path,
            output_file=args.output,
            verbose=args.verbose
        )
        
        if result is not None:
            print(f"\nMerge completed successfully!")
            print(f"Output file: {args.output or 'auto-generated'}")
            return 0
        else:
            print(f"\nMerge failed - no data found or error occurred")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
