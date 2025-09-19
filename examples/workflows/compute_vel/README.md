# G-ring Velocity Dispersion Curve Computation

This workflow computes velocity dispersion curves for G-ring array data using frequency-dependent analysis.

## Overview

The script performs the following operations:

1. **Data Loading**: Reads seismic data for specified time intervals (typically 1 day)
2. **Frequency Analysis**: Uses `compute_frequency_dependent_parameters_parallel` to compute velocities for each frequency band
3. **Data Storage**: Efficiently stores results in DataFrames
4. **Statistical Analysis**: Uses `get_kde_stats` to determine the maximum of the velocity distribution and deviation for each frequency band
5. **Visualization**: Creates dispersion curve plots showing velocity vs frequency

## Files

- `compute_gring_velocities.py`: Main processing script
- `config_gring_velocity.yml`: Configuration template
- `README.md`: This documentation

## Usage

```bash
python compute_gring_velocities.py config_gring_velocity.yml start_date end_date
```

### Examples

```bash
# Basic usage
python compute_gring_velocities.py config_gring_velocity.yml 2024-01-01 2024-01-31

# With verbose output (set verbose: true in config file)
python compute_gring_velocities.py config_gring_velocity.yml 2024-01-01 2024-01-31

# Using the improved version
python compute_gring_velocities_improved.py config_gring_velocity.yml 2024-01-01 2024-01-31
```

## Configuration

Edit `config_gring_velocity.yml` to specify:

- **Data source**: Network, station, channels
- **Time range**: Start and end times
- **Frequency bands**: Min/max frequency and octave fraction
- **Analysis parameters**: Wave type, method, window settings
- **Quality control**: Cross-correlation threshold
- **Processing**: Number of parallel processes, verbose output
- **Output**: Output directory, daily dispersion curve plotting

## Output

The script generates:

1. **Daily CSV files**: `gring_velocity_YYYY-MM-DD.csv` with columns:
   - `timestamp`: Time of measurement
   - `day`: Date
   - `fband`: Frequency band (Hz)
   - `velocity`: Velocity estimate (m/s)
   - `deviation`: Velocity deviation (m/s)
   - `n_measurements`: Number of measurements used

2. **Daily dispersion curve plots** (optional): `dispersion_curve_YYYY-MM-DD.png` showing:
   - Velocity vs frequency for each day
   - Data coverage by frequency band for each day
   - Enable with `plot_daily_dispersion: true` in config

3. **Log files**: Processing logs in `logs/` directory

## Key Features

- **Parallel Processing**: Uses multiprocessing for efficient computation
- **Frequency Bands**: Automatic octave-based frequency band division
- **Statistical Analysis**: KDE-based velocity distribution analysis
- **Quality Control**: Cross-correlation threshold filtering
- **Error Handling**: Robust error handling and logging
- **Daily Visualization**: Optional daily dispersion curve plots
- **Verbose Mode**: Detailed progress information controlled by config parameter
- **No Yearly Aggregation**: Focus on daily results for better granularity

## Dependencies

- sixdegrees
- obspy
- pandas
- numpy
- matplotlib
- scipy
- tqdm
- pyyaml

## Notes

- The script processes data in hourly intervals for each day
- Results are stored efficiently in CSV format for further analysis
- The KDE analysis provides robust estimates of velocity distributions
- Parallel processing significantly speeds up computation for large datasets
