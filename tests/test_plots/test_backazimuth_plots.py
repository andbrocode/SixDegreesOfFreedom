"""
Tests for backazimuth plotting functions.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream, Trace
from sixdegrees.plots.plot_backazimuth_map import plot_backazimuth_map
from sixdegrees.plots.plot_backazimuth_results import plot_backazimuth_results
from sixdegrees.plots.plot_backazimuth_deviation_analysis import plot_backazimuth_deviation_analysis
from sixdegrees.plots.plot_frequency_backazimuth_analysis import plot_frequency_backazimuth_analysis

@pytest.fixture
def sample_backazimuth_data():
    """Create sample backazimuth data for testing."""
    # Create time points
    times = np.linspace(0, 3600, 100)  # 100 points over 1 hour
    
    # Create backazimuth data
    baz = 180 + 20 * np.sin(2 * np.pi * times / 3600)  # Varying around 180 degrees
    cc = 0.7 + 0.2 * np.random.random(len(times))  # Cross-correlation values
    
    # Create detailed results
    detailed_results = {
        'love': {
            'baz': baz,
            'cc': cc,
            'cc_mid': cc,  # Use same values for cc_mid
            'cc_max': cc,  # Use same values for cc_max
            'times': times,
            'twin_center': times[::2],  # Window centers
            'twin_start': times[:-1:2],  # Window start times
            'twin_end': times[1::2]  # Window end times
        }
    }
    
    # Create estimates
    estimates = {
        'love': np.mean(baz)
    }
    
    # Create station coordinates
    station_coordinates = {
        'latitude': 47.7714,
        'longitude': 11.2752
    }
    
    # Create parameters
    parameters = {
        'baz_win_sec': 60,
        'baz_win_overlap': 0.5,
        'cc_threshold': 0.5,
        'wave_type': 'love'
    }
    
    return {
        'detailed_results': detailed_results,
        'estimates': estimates,
        'station_coordinates': station_coordinates,
        'parameters': parameters
    }

@pytest.fixture
def sample_event_info():
    """Create sample event information."""
    return {
        'latitude': 48.1234,
        'longitude': 11.5678,
        'backazimuth': 180.0
    }

@pytest.fixture
def sample_frequency_data():
    """Create sample frequency-dependent backazimuth data."""
    # Create frequency bands
    freqs = np.logspace(-2, 0, 20)  # 20 frequency points from 0.01 to 1 Hz
    
    # Create backazimuth data for each frequency
    baz = np.random.normal(180, 10, len(freqs))
    cc = np.random.uniform(0.5, 1.0, len(freqs))
    
    # Create wave type results
    wave_types = {
        'love': {
            'backazimuth': baz,
            'coherence': cc,
            'frequencies': freqs,
            'peak_baz': np.array([180.0 + np.random.normal(0, 5) for _ in range(len(freqs))]),
            'peak_cc': np.array([0.8 + np.random.normal(0, 0.1) for _ in range(len(freqs))]),
            'kde_values': np.random.random((len(freqs), 36)),  # 36 backazimuth bins
            'n_estimates': np.random.randint(10, 50, len(freqs))  # Number of estimates per frequency
        }
    }
    
    # Create frequency bands dictionary
    frequency_bands = {
        'center': freqs,
        'lower': freqs * 0.9,
        'upper': freqs * 1.1
    }
    
    # Create backazimuth grid
    baz_grid = np.linspace(0, 360, 36)
    
    # Create parameters
    parameters = {
        'wave_type': 'love',
        'fmin': 0.01,
        'fmax': 1.0,
        'octave_fraction': 3  # Number of bands per octave
    }
    
    return {
        'wave_types': wave_types,
        'frequency_bands': frequency_bands,
        'baz_grid': baz_grid,
        'parameters': parameters
    }

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
    
    tr_z.stats.channel = 'HHZ'  # Use HH channels for acceleration
    tr_n.stats.channel = 'HHN'
    tr_e.stats.channel = 'HHE'
    
    return Stream([tr_z, tr_n, tr_e])

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

def test_plot_backazimuth_map(sample_backazimuth_data, sample_event_info, temp_output_dir):
    """Test backazimuth map plotting."""
    # Plot backazimuth map
    fig = plot_backazimuth_map(
        results=sample_backazimuth_data,
        event_info=sample_event_info,
        map_projection='platecarree',  # Use simpler projection for testing
        bin_step=5
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have map and histogram axes
    
    # Save figure
    output_file = temp_output_dir / "backazimuth_map.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_backazimuth_results(sample_backazimuth_data, sample_stream, temp_output_dir):
    """Test backazimuth results plotting."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self._stream = stream
            self.tra_seed = ['BW.ROMY.00.HHN', 'BW.ROMY.00.HHE', 'BW.ROMY.00.HHZ']
            self.rot_seed = ['BW.ROMY.00.BJN', 'BW.ROMY.00.BJE', 'BW.ROMY.00.BJZ']
        
        def get_stream(self, data_type):
            if data_type == "rotation":
                # Create rotation stream
                t = np.linspace(0, 3600, 3600)
                data = np.sin(2 * np.pi * 0.1 * t)
                tr = Trace(data=data)
                tr.stats.sampling_rate = 1.0
                tr.stats.starttime = UTCDateTime('2023-01-01')
                tr.stats.network = 'BW'
                tr.stats.station = 'ROMY'
                tr.stats.channel = 'BJZ'  # Use BJ channels for rotation
                return Stream([tr])
            else:
                # Return acceleration stream
                return self._stream.copy()
    
    # Plot results
    fig = plot_backazimuth_results(
        sd=MockSixDegrees(sample_stream),
        baz_results=sample_backazimuth_data['detailed_results']['love'],
        wave_type='love',
        baz_theo=180.0,
        unitscale='nano'
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    # Save figure
    output_file = temp_output_dir / "backazimuth_results.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_backazimuth_deviation_analysis(sample_frequency_data, sample_event_info, temp_output_dir):
    """Test backazimuth deviation analysis plotting."""
    # Plot analysis
    fig = plot_backazimuth_deviation_analysis(
        results=sample_frequency_data,
        event_info=sample_event_info,
        figsize=(15, 8)
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    
    # Save figure
    output_file = temp_output_dir / "backazimuth_deviation.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_frequency_backazimuth_analysis(sample_frequency_data, sample_event_info, temp_output_dir):
    """Test frequency-dependent backazimuth analysis plotting."""
    # Plot analysis
    fig = plot_frequency_backazimuth_analysis(
        results=sample_frequency_data,
        event_info=sample_event_info,
        vmax_percentile=95,
        show_peak_line=True
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    
    # Save figure
    output_file = temp_output_dir / "frequency_backazimuth.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_backazimuth_map_invalid_input():
    """Test backazimuth map plotting with invalid input."""
    # Create invalid results (missing required data)
    results = {
        'detailed_results': {},  # Empty results
        'estimates': {},
        'station_coordinates': {}
    }
    
    # Should return None for invalid input
    fig = plot_backazimuth_map(results)
    assert fig is None

def test_plot_backazimuth_results_invalid_input(sample_stream):
    """Test backazimuth results plotting with invalid input."""
    # Create SixDegrees-like object
    class MockSixDegrees:
        def __init__(self, stream):
            self.runit = "rad/s"
            self.tunit = "m/s²"
            self.mu = "μ"
            self._stream = stream
            self.tra_seed = ['BW.ROMY.00.HHN', 'BW.ROMY.00.HHE', 'BW.ROMY.00.HHZ']
            self.rot_seed = ['BW.ROMY.00.BJN', 'BW.ROMY.00.BJE', 'BW.ROMY.00.BJZ']
        
        def get_stream(self, data_type):
            if data_type == "rotation":
                # Create rotation stream
                t = np.linspace(0, 3600, 3600)
                data = np.sin(2 * np.pi * 0.1 * t)
                tr = Trace(data=data)
                tr.stats.sampling_rate = 1.0
                tr.stats.starttime = UTCDateTime('2023-01-01')
                tr.stats.network = 'BW'
                tr.stats.station = 'ROMY'
                tr.stats.channel = 'BJZ'  # Use BJ channels for rotation
                return Stream([tr])
            else:
                # Return acceleration stream
                return self._stream.copy()
    
    # Create mismatched data lengths
    times = [UTCDateTime('2023-01-01') + t for t in range(10)]
    baz = np.zeros(9)  # Different length
    cc = np.zeros(10)
    
    # Create results dictionary
    results = {
        'times': times,
        'baz': baz,
        'cc': cc,
        'cc_mid': cc,  # Use same values for cc_mid
        'cc_max': cc,  # Use same values for cc_max
        'twin_center': times[::2],  # Window centers
        'twin_start': times[:-1:2],  # Window start times
        'twin_end': times[1::2]  # Window end times
    }
    
    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        plot_backazimuth_results(MockSixDegrees(sample_stream), results, wave_type='love')

def test_plot_frequency_backazimuth_analysis_invalid_input():
    """Test frequency backazimuth analysis plotting with invalid input."""
    # Create invalid results (missing required keys)
    results = {
        'frequency_bands': {
            'center': np.array([0.1, 0.2, 0.3])
        }
        # Missing wave_types
    }
    
    # Should raise KeyError for missing data
    with pytest.raises(KeyError):
        plot_frequency_backazimuth_analysis(results)