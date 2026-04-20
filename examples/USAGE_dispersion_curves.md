# Usage Guide: Dispersion Curve Functions

This guide explains how to use the new dispersion curve computation and plotting functions.

## Overview

Three new functions have been added:

1. **`compute_dispersion_curve`** - Computes dispersion curves using octave frequency bands with adaptive time windows
2. **`plot_dispersion_traces`** - Plots filtered traces for each frequency band
3. **`plot_dispersion_curves`** - Plots the final dispersion curves (velocity vs frequency)

## Basic Usage

### Step 1: Compute Dispersion Curves

```python
from sixdegrees import sixdegrees
from sixdegrees.plots import plot_dispersion_traces, plot_dispersion_curves

# Initialize sixdegrees object (assuming you have data loaded)
sd = sixdegrees(conf={
    'seed': 'XX.XXXX..XXX',
    'tbeg': '2024-01-01T00:00:00',
    'tend': '2024-01-01T01:00:00',
    # ... other config parameters
})

# Load your data
sd.load_data()

# Compute dispersion curve for Love waves
love_results = sd.compute_dispersion_curve(
    wave_type="love",
    fmin=0.01,              # Minimum frequency in Hz
    fmax=0.5,               # Maximum frequency in Hz
    octave_fraction=3,      # 1/3 octave bands
    window_factor=1.0,      # Time window = window_factor / fc
    use_theoretical_baz=False,  # Compute backazimuth (True to use theoretical)
    cc_threshold=0.2,       # Cross-correlation threshold
    baz_step=1,             # Backazimuth search step (degrees)
    baz_win_overlap=0.5,    # Backazimuth window overlap
    velocity_overlap=0.5,   # Velocity window overlap
    velocity_method='odr',  # Regression method: 'odr', 'ransac', etc.
    verbose=True            # Print progress
)

# Compute dispersion curve for Rayleigh waves
rayleigh_results = sd.compute_dispersion_curve(
    wave_type="rayleigh",
    fmin=0.01,
    fmax=0.5,
    octave_fraction=3,
    window_factor=1.0,
    use_theoretical_baz=False,
    cc_threshold=0.2,
    verbose=True
)
```

### Step 2: Plot Filtered Traces

```python
# Plot filtered traces for Love waves
fig_traces_love = plot_dispersion_traces(
    dispersion_results=love_results,
    unitscale="nano",           # or "micro"
    figsize=(15, 10),           # Optional: (width, height)
    title="Love Wave Traces",   # Optional: custom title
    data_type="acceleration",   # or "velocity"
    regression_method="odr",    # Regression method
    zero_intercept=True,        # Force zero intercept
    bootstrap=None              # Optional: bootstrap dict for uncertainty
)

# Plot filtered traces for Rayleigh waves
fig_traces_rayleigh = plot_dispersion_traces(
    dispersion_results=rayleigh_results,
    unitscale="nano",
    data_type="acceleration"
)
```

### Step 3: Plot Dispersion Curves

```python
# Plot dispersion curves (velocity vs frequency)
fig_curves = plot_dispersion_curves(
    love_results=love_results,
    rayleigh_results=rayleigh_results,
    figsize=(8, 6),
    xlog=False,                 # Logarithmic x-axis
    ylog=False,                 # Logarithmic y-axis
    markersize=7,
    linewidth=1.5,
    title="Dispersion Curves",  # Optional
    show_errors=True            # Show error bars from KDE deviations
)

# Or plot from a single result
fig_curves_love = plot_dispersion_curves(
    dispersion_results=love_results,
    show_errors=True
)
```

## Complete Example

```python
import matplotlib.pyplot as plt
from sixdegrees import sixdegrees
from sixdegrees.plots import plot_dispersion_traces, plot_dispersion_curves

# Setup
sd = sixdegrees(conf={
    'seed': 'XX.XXXX..XXX',
    'tbeg': '2024-01-01T00:00:00',
    'tend': '2024-01-01T01:00:00',
    'fmin': 0.01,
    'fmax': 0.5,
    # ... other parameters
})

# Load data
sd.load_data()

# Compute dispersion curves
print("Computing Love wave dispersion...")
love_results = sd.compute_dispersion_curve(
    wave_type="love",
    fmin=0.01,
    fmax=0.5,
    octave_fraction=3,
    window_factor=1.0,
    use_theoretical_baz=False,
    cc_threshold=0.2,
    verbose=True
)

print("\nComputing Rayleigh wave dispersion...")
rayleigh_results = sd.compute_dispersion_curve(
    wave_type="rayleigh",
    fmin=0.01,
    fmax=0.5,
    octave_fraction=3,
    window_factor=1.0,
    use_theoretical_baz=False,
    cc_threshold=0.2,
    verbose=True
)

# Plot traces
print("\nPlotting filtered traces...")
fig_traces_love = plot_dispersion_traces(love_results, unitscale="nano")
fig_traces_rayleigh = plot_dispersion_traces(rayleigh_results, unitscale="nano")

# Plot dispersion curves
print("\nPlotting dispersion curves...")
fig_curves = plot_dispersion_curves(
    love_results=love_results,
    rayleigh_results=rayleigh_results,
    show_errors=True
)

plt.show()
```

## Accessing Results

The `compute_dispersion_curve` function returns a dictionary with the following structure:

```python
results = {
    'wave_type': 'love',  # or 'rayleigh'
    'frequency_bands': [
        {
            'f_lower': 0.01,           # Lower frequency
            'f_upper': 0.0126,         # Upper frequency
            'f_center': 0.0112,        # Center frequency
            'time_window': 89.3,       # Time window length (seconds)
            'filtered_rot': Stream,    # Filtered rotation stream
            'filtered_acc': Stream,    # Filtered acceleration stream
            'backazimuths': array,     # Backazimuths for each time window
            'velocities': array,       # Velocities for each time window
            'ccoefs': array,           # Cross-correlation coefficients
            'times': array,            # Time points
            'kde_peak_velocity': 3500, # KDE peak velocity (m/s)
            'kde_deviation': 150,      # KDE deviation (m/s)
            'kde_stats': {...},        # Full KDE statistics
            'baz_used': 45.2           # Backazimuth used for velocity computation
        },
        # ... more frequency bands
    ],
    'parameters': {
        'fmin': 0.01,
        'fmax': 0.5,
        'octave_fraction': 3,
        'window_factor': 1.0,
        'cc_threshold': 0.2,
        'use_theoretical_baz': False
    }
}
```

## Advanced Usage

### Using Theoretical Backazimuth

```python
# If you want to use theoretical backazimuth instead of computing it
results = sd.compute_dispersion_curve(
    wave_type="love",
    use_theoretical_baz=True,  # Uses sd.baz_theo or sd.theoretical_baz
    # ... other parameters
)
```

### Custom Frequency Bands

The function automatically generates octave bands, but you can control them:

```python
results = sd.compute_dispersion_curve(
    wave_type="love",
    fmin=0.01,
    fmax=0.5,
    octave_fraction=3,  # 1/3 octave bands (more bands)
    # octave_fraction=1 for full octave bands (fewer bands)
    window_factor=2.0,  # Longer time windows (2 * 1/fc)
    # ... other parameters
)
```

### Bootstrap Uncertainty in Traces

```python
fig = plot_dispersion_traces(
    dispersion_results=love_results,
    bootstrap={
        'n_iterations': 1000,
        'stat': 'mean',
        'random_seed': 42,
        'sample_fraction': 0.8
    }
)
```

## Key Parameters

### `compute_dispersion_curve`

- **`wave_type`**: `"love"` or `"rayleigh"`
- **`fmin`, `fmax`**: Frequency range in Hz
- **`octave_fraction`**: Octave fraction (1=octaves, 3=1/3 octaves, etc.)
- **`window_factor`**: Multiplier for time window length (window = factor / fc)
- **`use_theoretical_baz`**: Use theoretical backazimuth instead of computing
- **`cc_threshold`**: Minimum cross-correlation coefficient threshold
- **`baz_step`**: Backazimuth search step size (degrees)
- **`velocity_method`**: Regression method (`'odr'`, `'ransac'`, etc.)

### `plot_dispersion_traces`

- **`dispersion_results`**: Output from `compute_dispersion_curve`
- **`unitscale`**: `"nano"` or `"micro"` for unit scaling
- **`data_type`**: `"acceleration"` or `"velocity"`
- **`regression_method`**: Method for regression
- **`bootstrap`**: Optional bootstrap dict for uncertainty estimation

### `plot_dispersion_curves`

- **`dispersion_results`**: Single result from `compute_dispersion_curve`
- **`love_results`**: Love wave results
- **`rayleigh_results`**: Rayleigh wave results
- **`show_errors`**: Show error bars from KDE deviations
- **`xlog`, `ylog`**: Logarithmic axes

## Notes

- The function processes each frequency band independently
- Time windows are adaptive: `window_length = window_factor / center_frequency`
- KDE statistics are computed for each frequency band to get peak velocity and deviation
- Filtered waveforms are stored in the results for each frequency band
- The function handles both Love and Rayleigh waves with appropriate component selection
