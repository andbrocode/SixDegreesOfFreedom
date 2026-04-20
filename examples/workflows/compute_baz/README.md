# Backazimuth Analysis Workflow

This workflow provides automated backazimuth estimation and analysis for 6-DoF seismic data, with comprehensive data processing, merging, and visualization capabilities.

## Overview

The Backazimuth Analysis workflow processes 6-DoF seismic data to estimate backazimuth (direction of wave arrival) using array-derived rotation and translation data. It's particularly useful for analyzing secondary microseisms and other coherent seismic signals.

## Files

- `gring_backazimuth_analysis.py` - Main analysis workflow
- `merge_backazimuth_data.py` - Data merging and aggregation utility
- `simple_plot_backazimuth.py` - Visualization and plotting tool
- `config_gring_backazimuth.yml` - Configuration file
- `README.md` - This documentation

## Features

- **Automated Processing**: Batch processing of multiple days with parallel execution
- **Data Merging**: Tools for combining and aggregating backazimuth results
- **Visualization**: Comprehensive plotting and analysis tools
- **Quality Control**: Correlation threshold filtering and quality metrics
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Comprehensive Logging**: Detailed processing logs and statistics

## Quick Start

### 1. Configuration

Edit `config_gring_backazimuth.yml` to set your parameters:

```yaml
# Data source
data_source: fdsn

# FDSN clients
fdsn_client_rot: LMU  # For rotation data
fdsn_client_tra: BGR  # For translation data

# SEED identifiers
seed: XX.RLAS..
rot_seed:
  - BW.RLAS..BJZ  # Rotation sensor
tra_seed:
  - GR.WET..BHZ   # Translation sensors
  - GR.WET..BHN
  - GR.WET..BHE

# Processing parameters
baz_step: 1          # Backazimuth step in degrees
baz_win_sec: 50     # Window length in seconds
baz_win_overlap: 0.5 # Window overlap fraction
freq_min: 0.1        # Minimum frequency (Hz)
freq_max: 0.15      # Maximum frequency (Hz)
wave_type: love      # Wave type for analysis
cc_threshold: 0.8    # Correlation threshold
```

### 2. Run the Analysis

```bash
# Process data for a date range
python gring_backazimuth_analysis.py config_gring_backazimuth.yml 2024-01-01 2024-01-31
```

### 3. Merge Results

```bash
# Merge daily results into a single file
python merge_backazimuth_data.py 2024-01-01 2024-01-31 ./output/backazimuth_data/
```

### 4. Visualize Results

```bash
# Plot backazimuth data
python simple_plot_backazimuth.py ./output/backazimuth_data/gring_backazimuth_2024-01-01_to_2024-01-31_merged.csv
```

## Configuration Parameters

### Data Source
- `data_source`: Data source type ("fdsn" for online, "sds" for local)
- `fdsn_client_rot`: FDSN client for rotation data
- `fdsn_client_tra`: FDSN client for translation data

### SEED Identifiers
- `seed`: Output stream identifier
- `rot_seed`: List of rotation sensor channels
- `tra_seed`: List of translation sensor channels

### Processing Parameters
- `baz_step`: Backazimuth step size in degrees
- `baz_win_sec`: Analysis window length in seconds
- `baz_win_overlap`: Window overlap fraction (0.0 to 1.0)
- `freq_min`: Minimum frequency for analysis (Hz)
- `freq_max`: Maximum frequency for analysis (Hz)
- `wave_type`: Wave type ("love" or "rayleigh")
- `cc_threshold`: Minimum correlation coefficient threshold

### Station Information
- `station_name`: Station name
- `station_latitude`: Station latitude
- `station_longitude`: Station longitude

### Performance
- `Nprocesses`: Number of parallel processes to use

## Output Structure

The workflow generates the following output structure:

```
output/
├── backazimuth_data/
│   ├── gring_backazimuth_2024-01-01.csv
│   ├── gring_backazimuth_2024-01-02.csv
│   ├── ...
│   └── gring_backazimuth_2024-01-01_to_2024-01-31_merged.csv
└── logs/
    └── processing.log
```

### Output Data Format

Each CSV file contains the following columns:
- `timestamp`: Time of analysis window
- `baz_max`: Maximum backazimuth estimate
- `baz_mid`: Middle backazimuth estimate
- `baz_max_std`: Standard deviation of maximum estimate
- `baz_mid_std`: Standard deviation of middle estimate
- `baz_max_mad`: Median absolute deviation of maximum estimate
- `baz_mid_mad`: Median absolute deviation of middle estimate
- `count_max`: Number of points used for maximum estimate
- `count_mid`: Number of points used for middle estimate

## Usage Examples

### Basic Analysis
```bash
# Process a single day
python gring_backazimuth_analysis.py config_gring_backazimuth.yml 2024-01-01 2024-01-01

# Process a month
python gring_backazimuth_analysis.py config_gring_backazimuth.yml 2024-01-01 2024-01-31
```

### Data Merging
```bash
# Merge daily files
python merge_backazimuth_data.py 2024-01-01 2024-01-31 ./output/backazimuth_data/

# Merge with custom output file
python merge_backazimuth_data.py 2024-01-01 2024-01-31 ./output/backazimuth_data/ -o merged_results.csv
```

### Visualization
```bash
# Basic plotting
python simple_plot_backazimuth.py data.csv

# Plot with specific parameters
python simple_plot_backazimuth.py data.csv --error mad --type max --date1 2024-01-01 --date2 2024-01-31
```

## Advanced Usage

### Custom Frequency Bands

Modify the frequency parameters for different analysis:

```yaml
# Primary microseisms (20s period)
freq_min: 0.05
freq_max: 0.1

# Secondary microseisms (10s to 7s period)
freq_min: 0.1
freq_max: 0.15

# Short period (1s period)
freq_min: 0.8
freq_max: 1.2
```

### Quality Control

Adjust quality control parameters:

```yaml
# Stricter quality control
cc_threshold: 0.9

# More lenient quality control
cc_threshold: 0.7
```

### Parallel Processing

Optimize performance for your system:

```yaml
# Use all available cores
Nprocesses: 8

# Use fewer cores for memory-constrained systems
Nprocesses: 2
```

## Data Merging Utility

The `merge_backazimuth_data.py` script provides:

- **Date Range Merging**: Combine data from multiple days
- **Quality Filtering**: Remove low-quality data points
- **Statistics Generation**: Compute summary statistics
- **Flexible Output**: Custom output file naming

### Merge Options

```bash
# Basic merging
python merge_backazimuth_data.py start_date end_date data_path

# With custom output
python merge_backazimuth_data.py start_date end_date data_path -o custom_output.csv

# With verbose logging
python merge_backazimuth_data.py start_date end_date data_path --verbose
```

## Visualization Tool

The `simple_plot_backazimuth.py` script provides:

- **Time Series Plots**: Backazimuth over time
- **Histogram Analysis**: Distribution of backazimuth values
- **Error Visualization**: Error bars and uncertainty estimates
- **Date Filtering**: Plot specific time ranges

### Plot Options

```bash
# Plot both max and mid estimates
python simple_plot_backazimuth.py data.csv --type both

# Plot only maximum estimates
python simple_plot_backazimuth.py data.csv --type max

# Plot with MAD error bars
python simple_plot_backazimuth.py data.csv --error mad

# Plot specific date range
python simple_plot_backazimuth.py data.csv --date1 2024-01-01 --date2 2024-01-31
```

## Troubleshooting

### Common Issues

1. **No Data Retrieved**
   - Check FDSN client connectivity
   - Verify SEED identifiers
   - Check data availability for specified time range

2. **Processing Failures**
   - Check log files for detailed error messages
   - Verify configuration parameters
   - Ensure sufficient disk space

3. **Poor Quality Results**
   - Adjust correlation threshold
   - Check frequency band selection
   - Verify station configuration

4. **Memory Issues**
   - Reduce number of parallel processes
   - Process shorter time ranges
   - Check available system memory

### Log Files

Check the following for detailed information:
- `logs/processing.log` - Detailed processing log
- Console output - Real-time progress and warnings
- Individual CSV files - Check for data quality

## Performance Optimization

### Memory Usage
- Use appropriate number of processes
- Process data in smaller chunks
- Monitor memory usage during processing

### Processing Speed
- Use parallel processing
- Optimize frequency band selection
- Use appropriate window sizes

### Storage
- Use efficient data formats
- Compress output files when possible
- Clean up temporary files regularly

## Dependencies

- `sixdegrees` package
- `obspy` for seismic data handling
- `numpy` and `pandas` for data processing
- `matplotlib` for visualization
- `pyyaml` for configuration handling
- `tqdm` for progress bars
- `multiprocessing` for parallel processing

## Support

For issues and questions:
1. Check the log files for error messages
2. Verify configuration parameters
3. Consult the main sixdegrees documentation
4. Check the GitHub repository for known issues
