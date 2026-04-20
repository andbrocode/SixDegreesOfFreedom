#!/usr/bin/env python3
"""
Very simple script to plot backazimuth results.
Usage: python3 simple_plot_backazimuth.py [csv_file] [options]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from datetime import datetime

def align_data_by_time(x_data, y_data, x_times, y_times):
    """
    Align x and y data using time as reference, filling missing values with NaN.
    
    Parameters:
    -----------
    x_data : array-like
        X-axis data (time values)
    y_data : array-like  
        Y-axis data (backazimuth values)
    x_times : array-like
        Time stamps corresponding to x_data
    y_times : array-like
        Time stamps corresponding to y_data
        
    Returns:
    --------
    aligned_x : array
        Aligned x data
    aligned_y : array
        Aligned y data with NaN for missing time points
    """
    try:
        # Convert inputs to numpy arrays for easier handling
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        x_times = np.asarray(x_times)
        y_times = np.asarray(y_times)
        
        # If the arrays are already the same length, return them as-is
        if len(x_data) == len(y_data):
            return x_data, y_data
        
        # Convert times to pandas datetime for easier alignment
        x_times_pd = pd.to_datetime(x_times)
        y_times_pd = pd.to_datetime(y_times)
        
        # Create pandas Series
        x_series = pd.Series(x_data, index=x_times_pd)
        y_series = pd.Series(y_data, index=y_times_pd)
        
        # Get all unique time points from both series
        all_times = pd.Index(x_series.index).union(pd.Index(y_series.index)).sort_values()
        
        # Reindex both series to the common time index, filling missing values with NaN
        x_aligned = x_series.reindex(all_times)
        y_aligned = y_series.reindex(all_times)
        
        return x_aligned.values, y_aligned.values
        
    except Exception as e:
        # If alignment fails, return original data
        print(f"Warning: Data alignment failed: {e}. Using original data.")
        return np.asarray(x_data), np.asarray(y_data)

def plot_backazimuth_simple(csv_file, error_type='mad', plot_type='both', date1=None, date2=None):
    """Simple backazimuth plotting function."""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # sort by timestamp
    df = df.sort_values('timestamp')

    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by date range if specified
    if date1 and date2:
        date1 = pd.to_datetime(date1)
        date2 = pd.to_datetime(date2)
        df = df[(df['timestamp'] >= date1) & (df['timestamp'] <= date2)]
        print(f"Filtered data from {date1} to {date2}: {len(df)} records")
    
    # Calculate relative times from first sample
    if 'timestamp' in df.columns and len(df) > 0:
        first_time = df['timestamp'].min()
        time_diff = df['timestamp'] - first_time
        
        # Convert to appropriate units
        total_duration = time_diff.max()
        
        if total_duration.total_seconds() < 3600:  # Less than 1 hour
            time_relative = time_diff.dt.total_seconds() / 60  # minutes
            time_unit = 'minutes'
        elif total_duration.total_seconds() < 86400:  # Less than 1 day
            time_relative = time_diff.dt.total_seconds() / 3600  # hours
            time_unit = 'hours'
        elif total_duration.days < 7:  # Less than 1 week
            time_relative = time_diff.dt.total_seconds() / 86400  # days
            time_unit = 'days'
        else:  # More than 1 week
            time_relative = time_diff.dt.total_seconds() / (86400 * 7)  # weeks
            time_unit = 'weeks'
        
        print(f"Time range: {total_duration.total_seconds():.1f} seconds ({time_unit})")
        print(f"First sample: {first_time}")
        print(f"Last sample: {df['timestamp'].max()}")
    else:
        time_relative = None
        time_unit = 'time'
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time series
    if plot_type in ['max', 'both'] and 'baz_max' in df.columns:
        max_data = df.dropna(subset=['baz_max'])
        if len(max_data) > 0:
            # Determine error column based on error_type
            error_col = f'baz_max_{error_type}' if f'baz_max_{error_type}' in df.columns else 'baz_max_std'
            
            # Use relative time if available, otherwise use timestamp
            if time_relative is not None:
                # Get the indices of non-null baz_max values
                valid_indices = df['baz_max'].notna()
                x_data = time_relative[valid_indices]
                y_data = max_data['baz_max']
            else:
                x_data = max_data['timestamp']
                y_data = max_data['baz_max']
            
            if error_col in df.columns and error_col in max_data.columns:
                error_data = max_data[error_col]
                ax1.errorbar(x_data, y_data, 
                           yerr=error_data, 
                           fmt='ro', alpha=0.3, markersize=1, capsize=0)
            
            ax1.scatter(x_data, y_data, 
                        color='red', alpha=0.8, label='baz_max', s=10)

    if plot_type in ['mid', 'both'] and 'baz_mid' in df.columns:
        mid_data = df.dropna(subset=['baz_mid'])
        if len(mid_data) > 0:
            # Determine error column based on error_type
            error_col = f'baz_mid_{error_type}' if f'baz_mid_{error_type}' in df.columns else 'baz_mid_std'
            
            # Use relative time if available, otherwise use timestamp
            if time_relative is not None:
                # Get the indices of non-null baz_mid values
                valid_indices = df['baz_mid'].notna()
                x_data = time_relative[valid_indices]
                y_data = mid_data['baz_mid']
            else:
                x_data = mid_data['timestamp']
                y_data = mid_data['baz_mid']
            
            if error_col in df.columns and error_col in mid_data.columns:
                error_data = mid_data[error_col]
                ax1.errorbar(x_data, y_data, 
                           yerr=error_data, 
                           fmt='bs', alpha=0.3, markersize=4, capsize=0)
            ax1.scatter(x_data, y_data, 
                        color='blue', alpha=0.8, label='baz_mid', s=20)

    ax1.set_ylabel('Backazimuth (degrees)')
    ax1.set_xlabel(f'Time ({time_unit})')
    ax1.set_title('Backazimuth Time Series')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 360)
    
    # Format x-axis for relative time
    if time_relative is not None:
        ax1.ticklabel_format(style='plain', axis='x')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    else:
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Histogram
    bins = np.arange(0, 360+5, 5)
    if plot_type in ['max', 'both'] and 'baz_max' in df.columns:
        max_data = df['baz_max'].dropna()
        if len(max_data) > 0:
            ax2.hist(max_data, bins=bins, alpha=0.7, color='red', 
                   label='baz_max', edgecolor='black')
    
    if plot_type in ['mid', 'both'] and 'baz_mid' in df.columns:
        mid_data = df['baz_mid'].dropna()
        if len(mid_data) > 0:
            ax2.hist(mid_data, bins=bins, alpha=0.7, color='blue', 
                   label='baz_mid', edgecolor='black')
    
    ax2.set_xlabel('Backazimuth (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Backazimuth Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 360)
    
    plt.tight_layout()
    plt.show()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simple script to plot backazimuth results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python3 simple_plot_backazimuth.py data.csv
            python3 simple_plot_backazimuth.py data.csv --error mad --type max
            python3 simple_plot_backazimuth.py data.csv --error dev --type mid --date1 2024-01-01 --date2 2024-01-31
        """
    )
    
    parser.add_argument('csv_file', help='CSV file containing backazimuth data')
    
    parser.add_argument('--error', choices=['mad', 'dev'], default='mad',
                       help='Error type to use for error bars: mad (median absolute deviation) or dev (standard deviation)')
    
    parser.add_argument('--type', choices=['mid', 'max', 'both'], default='both',
                       help='Type of backazimuth to plot: mid (middle), max (maximum), or both (default)')
    
    parser.add_argument('--date1', help='Start date for filtering data (YYYY-MM-DD format)')
    
    parser.add_argument('--date2', help='End date for filtering data (YYYY-MM-DD format)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        plot_backazimuth_simple(
            csv_file=args.csv_file,
            error_type=args.error,
            plot_type=args.type,
            date1=args.date1,
            date2=args.date2
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
