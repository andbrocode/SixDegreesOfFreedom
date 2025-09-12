# Array-Derived Rotation (ADR) Computation Workflow

This workflow computes array-derived rotation data from seismic array observations and stores the results in SDS (SeisComP Data Structure) format for continuous processing.

## Overview

The Array-Derived Rotation workflow processes seismic array data to compute rotational components (roll, pitch, yaw) from translational array observations. This is essential for creating 6-DoF seismic datasets when direct rotation measurements are not available.

## Files

- `compute_daily_adr.py` - Main processing script
- `config_daily_adr.yml` - Configuration file
- `README.md` - This documentation

## Features

- **Daily Processing**: Processes data day by day for efficient memory management
- **SDS Storage**: Stores results in standard SDS format for easy integration
- **Array Configuration**: Flexible array geometry and station configuration
- **Quality Control**: Processing statistics and error monitoring
- **Filtering**: Configurable bandpass filtering for noise reduction
- **Logging**: Comprehensive logging and progress tracking

## Quick Start

### 1. Configuration

Edit `config_daily_adr.yml` to set your parameters:

```yaml
# Processing time range
start_date: 2025-08-01
end_date: 2025-08-03

# Directory structure
base_dir: ./
sds_dir: /path/to/your/sds/directory
log_dir: ./logs

# Array configuration
fdsn_client: IRIS
stations:
  - PY.PFOIX
  - PY.BPH01
  - PY.BPH02
  - PY.BPH03
reference_station: PY.PFOIX
channel_prefix: B

# Processing parameters
response_output: VEL
output_format: sds
```

### 2. Run the Workflow

```bash
python compute_daily_adr.py
```

## Configuration Parameters

### Time Range
- `start_date`: Start date for processing (YYYY-MM-DD format)
- `end_date`: End date for processing (YYYY-MM-DD format)

### Directory Structure
- `base_dir`: Base directory for processing
- `sds_dir`: Directory for SDS format output
- `log_dir`: Directory for log files

### Array Configuration
- `fdsn_client`: FDSN client for data retrieval (e.g., "IRIS")
- `stations`: List of station codes to include in the array
- `reference_station`: Reference station for array processing
- `channel_prefix`: Channel prefix for data retrieval

### Processing Parameters
- `response_output`: Output response type ("VEL" for velocity, "DIS" for displacement)
- `output_format`: Output format ("sds" for SDS structure)

### Filter Parameters
- `type`: Filter type ("bandpass", "highpass", "lowpass")
- `freqmin`: Minimum frequency (Hz)
- `freqmax`: Maximum frequency (Hz)
- `corners`: Number of filter corners
- `zerophase`: Use zero-phase filtering

### ADR Parameters
- `vp`: P-wave velocity (m/s)
- `vs`: S-wave velocity (m/s)
- `sigmau`: Uncertainty in displacement (m)

## Output Structure

The workflow generates the following output structure:

```
sds_dir/
├── YYYY/
│   └── PY/
│       └── PFOIX/
│           └── BXZ/
│               ├── PY.PFOIX..BXZ.D.YYYY.DDD
│               ├── PY.PFOIX..BXR.D.YYYY.DDD
│               └── PY.PFOIX..BXT.D.YYYY.DDD
logs/
├── processing.log
└── stats_summary.csv
```

### Output Files
- **BXZ**: Vertical rotation (yaw)
- **BXR**: Radial rotation (pitch)
- **BXT**: Transverse rotation (roll)

## Usage Examples

### Basic Usage
```bash
# Process data for a single day
python compute_daily_adr.py
```

### Custom Configuration
```bash
# Use custom configuration file
python compute_daily_adr.py --config custom_config.yml
```

## Processing Statistics

The workflow generates comprehensive processing statistics:

- **Station Count**: Number of stations used for each day
- **Success Rate**: Percentage of successful processing
- **Processing Time**: Time taken for each day
- **Quality Metrics**: Data quality indicators

Statistics are saved to `logs/stats_summary.csv` and visualized in summary plots.

## Error Handling

The workflow includes robust error handling:

- **Missing Data**: Skips days with insufficient data
- **Network Issues**: Retries failed data requests
- **Processing Errors**: Logs errors and continues processing
- **Quality Checks**: Validates output data quality

## Troubleshooting

### Common Issues

1. **No Data Available**
   - Check station codes and time range
   - Verify FDSN client connectivity
   - Check data availability in specified time range

2. **Processing Failures**
   - Check log files for detailed error messages
   - Verify array geometry and station configuration
   - Ensure sufficient disk space for output

3. **Quality Issues**
   - Adjust filter parameters
   - Check array geometry configuration
   - Verify velocity model parameters

### Log Files

Check the following log files for detailed information:
- `logs/processing.log` - Detailed processing log
- `logs/stats_summary.csv` - Processing statistics
- Console output - Real-time progress and warnings

## Dependencies

- `sixdegrees` package
- `obspy` for seismic data handling
- `numpy` and `pandas` for data processing
- `matplotlib` for visualization
- `pyyaml` for configuration handling

## Advanced Usage

### Custom Array Geometries

Modify the station list in the configuration file to include your array geometry. The workflow automatically computes array parameters from station coordinates.

### Custom Filtering

Adjust filter parameters based on your analysis requirements:
- Lower `freqmin` for longer period analysis
- Higher `freqmax` for shorter period analysis
- Adjust `corners` for filter steepness

### Batch Processing

The workflow is designed for batch processing of multiple days. It automatically handles:
- Memory management for large datasets
- Error recovery and continuation
- Progress tracking and logging

## Support

For issues and questions:
1. Check the log files for error messages
2. Verify configuration parameters
3. Consult the main sixdegrees documentation
4. Check the GitHub repository for known issues
