# SixDegreesOfFreedom

## Basic Codes for 6-DoF Analysis

The codes in this package enable comprehensive seismological processing of 6 degree-of-freedom (DoF) data. A 6-DoF station ideally combines co-located observations of three components of translation and three components of rotation data. The package provides well-documented functions for array-derived rotation computation, backazimuth analysis, velocity estimation, and advanced visualization tools.

## Installation

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/andbrocode/SixDegreesOfFreedom.git
cd SixDegreesOfFreedom/
```

2. Create and activate the conda environment:
```bash
conda env create -f docs/environment.yml
conda activate sixdegrees
```

3. Upgrade pip and setuptools (required for editable installs):
```bash
pip install --upgrade pip setuptools wheel
```

4. Install the package in development mode:
```bash
pip install -e .
```

**Note**: If you encounter an error about missing 'build_editable' hook, ensure you have setuptools >= 64.0.0 installed. The `setup.py` file is provided as a fallback for older setuptools versions.

### Uninstallation

To uninstall the package:

1. **Uninstall the package** (if installed in development mode):
```bash
pip uninstall sixdegrees
```

2. **If you want to completely remove the conda environment**:
```bash
conda deactivate  # Exit the environment first
conda env remove -n sixdegrees
```

3. **To remove the cloned repository** (optional):
```bash
cd ..  # Navigate out of the repository directory
rm -rf SixDegreesOfFreedom/  # Remove the directory
```

**Note**: If you installed the package in development mode (`pip install -e .`), uninstalling will remove the package but keep your local repository files intact. You can reinstall later without re-cloning.

### Dependencies

The package requires Python ≥3.9 and the following main dependencies:
- numpy ≥1.20.0
- scipy ≥1.7.0
- matplotlib ≥3.4.0
- obspy ≥1.3.0
- pandas ≥1.3.0
- scikit-learn ≥0.24.0
- pyyaml ≥6.0
- acoustics ≥0.2.3
- cartopy ≥0.20.0 (optional, for map plotting)

For a complete list of dependencies, see `docs/environment.yml`.

## Documentation

### Project Structure

The project is organized as follows:
- `sixdegrees/`: Core package code
  - `sixdegrees.py`: Main implementation with comprehensive docstrings
  - `seismicarray.py`: Seismic array processing with full documentation
  - `plots/`: Visualization modules for data analysis and plotting
  - `utils/`: Utility functions for data processing
- `examples/`: Jupyter notebooks, example scripts, and workflows
  - `workflows/`: Ready-to-use processing workflows
    - `compute_adr/`: Array-derived rotation computation workflows
    - `compute_baz/`: Backazimuth analysis workflows
  - Various demonstration notebooks for different analysis types
  - Example configuration files and sample data
- `docs/`: Documentation and environment specifications
- `tests/`: Comprehensive test suite

### Quick Start

```python
from sixdegrees import sixdegrees

# Create configuration dictionary
config = {
    'tbeg': "2023-09-08 22:13:00",
    'tend': "2023-09-08 23:00:00",
    'station_lon': 11.275476,
    'station_lat': 48.162941,
    'seed': "XX.ROMY..",
    'rot_seed': ["XX.ROMY..BJZ", "XX.ROMY..BJN", "XX.ROMY..BJE"],
    'tra_seed': ["XX.ROMY..BHZ", "XX.ROMY..BHN", "XX.ROMY..BHE"],
    'data_source': "mseed_file",
    'path_to_mseed_file': "./data/romy_eventM6.8.mseed",
}

# Initialize sixdegrees object
sd = sixdegrees(conf=config)

# Load data
sd.load_data(config['tbeg'], config['tend'])

# Filter data
sd.filter_data(fmin=0.02, fmax=0.2)

# Compute backazimuth
baz_results = sd.compute_backazimuth(
    wave_type='love',
    baz_step=1,
    baz_win_sec=30,
    baz_win_overlap=0.5,
    out=True
)

# Compute velocities
velocities = sd.compute_velocities(
    wave_type='love',
    win_time_s=30.0,
    overlap=0.5,
    cc_threshold=0.75,
    method='odr'
)
```

### Example Usage

The package includes several Jupyter notebooks and workflows in the `examples/` directory:

#### Jupyter Notebooks
- Array-derived rotation computation from the Pinon Flats seismic array
- Data acquisition examples for G-ring laser and BSPF station
- Analysis examples for BSPF station, ROMY ring laser, and G-ring laser
- Backazimuth analysis and visualization demos

#### Processing Workflows
- **Array-Derived Rotation (ADR)**: `examples/workflows/compute_adr/`
  - Daily continuous 6-DoF data creation from array data
  - SDS format storage and management
- **Backazimuth Analysis**: `examples/workflows/compute_baz/`
  - Automated backazimuth estimation and analysis
  - Data merging and visualization tools

See individual workflow README files for detailed usage instructions.

## Recent Updates

### Version 0.1.1
- **Enhanced Documentation**: Added comprehensive docstrings to all core functions in `sixdegrees.py`
- **Improved Code Quality**: All functions now have proper type hints and detailed parameter descriptions
- **Better Developer Experience**: Enhanced code readability and maintainability

## Testing

### Test Structure

The test suite is organized into three main categories:
- `test_core/`: Tests for core functionality
  - Basic initialization
  - Data source handling
  - Configuration validation
  - Time and coordinate validation
- `test_plots/`: Tests for plotting functions
- `test_utils/`: Tests for utility functions

### Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_core/
pytest tests/test_plots/
pytest tests/test_utils/

# Run with coverage report
pytest --cov=sixdegrees tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

The license information is found in the LICENSE file
