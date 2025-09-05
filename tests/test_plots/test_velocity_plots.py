"""
Tests for velocity plotting functions.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream, Trace
from sixdegrees.plots.plot_velocities import plot_velocities
from sixdegrees.plots.plot_velocities_win import plot_velocities_win
from sixdegrees.plots.plot_velocity_comparison import plot_velocity_comparison
from sixdegrees.plots.plot_velocity_method_comparison import plot_velocity_method_comparison

@pytest.fixture
def sample_velocity_data():
    """Create sample velocity data for testing."""
    # Create time points
    times = np.linspace(0, 3600, 100)  # 100 points over 1 hour
    
    # Create velocity data
    vel = 3000 + 200 * np.sin(2 * np.pi * times / 3600)  # Varying around 3000 m/s
    cc = 0.7 + 0.2 * np.random.random(len(times))  # Cross-correlation values
    
    # Create velocity results dictionary
    results = {
        'time': times,
        'velocity': vel,
        'ccoef': cc,
        'parameters': {
            'wave_type': 'love',
            'win_time_s': 60,
            'overlap': 0.5
        }
    }
    
    return results

@pytest.fixture
def sample_window_data():
    """Create sample windowed velocity data."""
    # Create windows
    n_windows = 20
    window_times = np.linspace(0, 3600, n_windows)
    
    # Create velocity data for each window
    velocities = np.random.normal(3000, 100, n_windows)
    cc_values = np.random.uniform(0.5, 1.0, n_windows)
    backazimuth = np.random.normal(180, 10, n_windows)
    
    # Create results dictionary
    results = {
        'time': window_times,
        'velocity': velocities,
        'ccoef': cc_values,
        'backazimuth': backazimuth,
        'parameters': {
            'wave_type': 'love',
            'win_time_s': 60,
            'overlap': 0.5
        }
    }
    
    return results

@pytest.fixture
def sample_method_comparison_data():
    """Create sample data for method comparison."""
    # Create methods and their results
    methods = ['Method A', 'Method B', 'Method C']
    velocities = {
        'Method A': {
            'time': np.linspace(0, 3600, 50),
            'velocity': np.random.normal(3000, 100, 50),
            'ccoef': np.random.uniform(0.6, 0.9, 50),
            'parameters': {'win_time_s': 60, 'overlap': 0.5}
        },
        'Method B': {
            'time': np.linspace(0, 3600, 50),
            'velocity': np.random.normal(3100, 150, 50),
            'ccoef': np.random.uniform(0.5, 0.8, 50),
            'parameters': {'win_time_s': 60, 'overlap': 0.5}
        },
        'Method C': {
            'time': np.linspace(0, 3600, 50),
            'velocity': np.random.normal(2900, 120, 50),
            'ccoef': np.random.uniform(0.7, 1.0, 50),
            'parameters': {'win_time_s': 60, 'overlap': 0.5}
        }
    }
    
    return velocities

@pytest.fixture
def sample_stream():
    """Create a sample stream for testing."""
    # Create synthetic data
    t = np.linspace(0, 3600, 3600)  # 1 hour of data at 1 Hz
    data_z = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
    data_n = np.sin(2 * np.pi * 0.1 * t + np.pi/4)  # Phase shifted
    data_e = np.sin(2 * np.pi * 0.1 * t + np.pi/2)  # Phase shifted
    
    # Create traces
    tr_z = Trace(data=data_z)
    tr_n = Trace(data=data_n)
    tr_e = Trace(data=data_e)
    
    # Set trace attributes
    for tr in [tr_z, tr_n, tr_e]:
        tr.stats.sampling_rate = 1.0
        tr.stats.starttime = UTCDateTime('2023-01-01')
        tr.stats.network = 'BW'
        tr.stats.station = 'ROMY'
        tr.stats.coordinates = {
            'latitude': 47.7714,
            'longitude': 11.2752,
            'elevation': 565.0
        }
    
    tr_z.stats.channel = 'BJZ'
    tr_n.stats.channel = 'BJN'
    tr_e.stats.channel = 'BJE'
    
    return Stream([tr_z, tr_n, tr_e])

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

def test_plot_velocities(sample_velocity_data, sample_stream, temp_output_dir):
    """Test velocity plotting."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self.st = stream
            self.sampling_rate = stream[0].stats.sampling_rate
    
    # Plot velocities
    fig = plot_velocities(
        sd=MockSixDegrees(sample_stream),
        velocity_results=sample_velocity_data,
        minors=True
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    # Save figure
    output_file = temp_output_dir / "velocities.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_velocities_win(sample_window_data, sample_stream, temp_output_dir):
    """Test windowed velocity plotting."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self.st = stream
            self.sampling_rate = stream[0].stats.sampling_rate
    
    # Plot windowed velocities
    fig = plot_velocities_win(
        sd=MockSixDegrees(sample_stream),
        results_velocities=sample_window_data,
        cc_threshold=0.5,
        baz_theo=180.0
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    # Save figure
    output_file = temp_output_dir / "velocities_win.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_velocity_comparison(sample_method_comparison_data, temp_output_dir):
    """Test velocity comparison plotting."""
    # Plot velocity comparison
    fig = plot_velocity_comparison(
        results_vel1=sample_method_comparison_data['Method A'],
        results_vel2=sample_method_comparison_data['Method B'],
        labels=('Method A', 'Method B')
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    # Save figure
    output_file = temp_output_dir / "velocity_comparison.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_velocity_method_comparison(sample_method_comparison_data, temp_output_dir):
    """Test velocity method comparison plotting."""
    # Plot method comparison
    fig = plot_velocity_method_comparison(
        sd=['Method A', 'Method B', 'Method C'],
        love_velocities_ransac=sample_method_comparison_data,
        love_velocities_odr=sample_method_comparison_data,
        cc_threshold=0.75
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    
    # Save figure
    output_file = temp_output_dir / "velocity_method_comparison.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_velocities_invalid_input(sample_stream):
    """Test velocity plotting with invalid input."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self.st = stream
            self.sampling_rate = stream[0].stats.sampling_rate
    
    # Create invalid results (missing required data)
    results = {
        'time': np.linspace(0, 3600, 100),
        'velocity': np.zeros(100)
        # Missing parameters
    }
    
    # Should raise KeyError for missing parameters
    with pytest.raises(KeyError):
        plot_velocities(MockSixDegrees(sample_stream), results)

def test_plot_velocities_win_invalid_input(sample_stream):
    """Test windowed velocity plotting with invalid input."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self.st = stream
            self.sampling_rate = stream[0].stats.sampling_rate
    
    # Create invalid results (missing required data)
    results = {
        'time': np.linspace(0, 3600, 100),
        'velocity': np.zeros(100)
        # Missing backazimuth and ccoef
    }
    
    # Should raise KeyError for missing data
    with pytest.raises(KeyError):
        plot_velocities_win(MockSixDegrees(sample_stream), results)

def test_plot_velocity_comparison_invalid_input():
    """Test velocity comparison plotting with invalid input."""
    # Create mismatched data
    results_vel1 = {
        'time': np.linspace(0, 3600, 100),
        'velocity': np.zeros(100),
        'parameters': {'win_time_s': 60, 'overlap': 0.5}
    }
    results_vel2 = {
        'time': np.linspace(0, 3600, 90),  # Different length
        'velocity': np.zeros(90),
        'parameters': {'win_time_s': 60, 'overlap': 0.5}
    }
    
    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        plot_velocity_comparison(results_vel1, results_vel2)

def test_plot_velocity_method_comparison_invalid_input():
    """Test velocity method comparison plotting with invalid input."""
    # Create mismatched data
    methods = ['Method A', 'Method B']
    velocities = {
        'Method A': {
            'time': np.linspace(0, 3600, 100),
            'velocity': np.zeros(100),
            'ccoef': np.zeros(100)
        },
        'Method B': {
            'time': np.linspace(0, 3600, 90),  # Different length
            'velocity': np.zeros(90),
            'ccoef': np.zeros(90)
        }
    }
    
    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        plot_velocity_method_comparison(methods, velocities, velocities)