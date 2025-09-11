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

3. Install the package in development mode:
```bash
pip install -e .
```

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
- `examples/`: Jupyter notebooks and example scripts
  - Various demonstration notebooks for different analysis types
  - Example configuration files
  - Sample data files
- `docs/`: Documentation and environment specifications
- `tests/`: Comprehensive test suite

### Example Usage

The package includes several Jupyter notebooks in the `examples/` directory that demonstrate various use cases:
- Example to compute array-derived rotations from the Pinon Flats seismic array
- Examples of how to obtain data from G-ring laser and BSPF station
- Analyses examples for BSPF station at Pinon Flats observatory, for the ROMY ring laser and G-ring laser

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

The license information is found in the LICENCE file
