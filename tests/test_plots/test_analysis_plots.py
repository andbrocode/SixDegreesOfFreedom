"""
Tests for analysis plotting functions.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
from sixdegrees.plots.plot_cwt import plot_cwt
from sixdegrees.plots.plot_cwt_all import plot_cwt_all
from sixdegrees.plots.plot_frequency_time_map_adaptive import plot_frequency_time_map_adaptive

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
def sample_cwt_data():
    """Create sample CWT data for testing."""
    # Create time and frequency points
    times = np.linspace(0, 3600, 100)
    freqs = np.logspace(-2, 0, 50)
    
    # Create CWT data
    cwt = np.random.random((len(freqs), len(times)))
    
    # Create CWT output dictionary
    cwt_output = {
        'BJZ': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]  # Simple cone of influence
        },
        'BJN': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]
        },
        'BJE': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]
        },
        'HHZ': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]
        },
        'HHN': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]
        },
        'HHE': {
            'times': times,
            'frequencies': freqs,
            'cwt_power': cwt,
            'cone': np.ones_like(times) * freqs[0]
        }
    }
    
    return cwt_output

@pytest.fixture
def sample_frequency_time_data():
    """Create sample frequency-time data for testing."""
    # Create frequency bands
    freqs = np.logspace(-2, 0, 20)
    times = np.linspace(0, 3600, 100)
    
    # Create backazimuth and coherence data
    baz = np.random.normal(180, 10, (len(freqs), len(times)))
    coh = np.random.uniform(0.5, 1.0, (len(freqs), len(times)))
    
    # Create results dictionary
    results = {
        'adaptive_windows': True,
        'frequency_bands': freqs,
        'time_windows': [times] * len(freqs),
        'backazimuth_data': [baz[i] for i in range(len(freqs))],
        'correlation_data': [coh[i] for i in range(len(freqs))],
        'window_factor': 10,
        'parameters': {
            'wave_type': 'love',
            'fmin': 0.01,
            'fmax': 1.0
        }
    }
    
    return results

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

def test_plot_cwt(sample_stream, sample_cwt_data, temp_output_dir):
    """Test CWT plotting."""
    # Plot CWT
    fig = plot_cwt(
        st=sample_stream,
        cwt_output=sample_cwt_data,
        clog=False,
        ylim=0.5,
        scale=1e6
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have waveform and CWT subplots
    
    # Check colorbar
    assert len(fig.axes[-1].collections) > 0  # Should have a colorbar
    
    # Save figure
    output_file = temp_output_dir / "cwt.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_cwt_all(sample_stream, sample_cwt_data, temp_output_dir):
    """Test plotting CWT for all components."""
    # Create streams with proper channel names
    rot = Stream([
        Trace(data=np.zeros(100), header={'channel': 'BJZ', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')}),
        Trace(data=np.zeros(100), header={'channel': 'BJN', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')}),
        Trace(data=np.zeros(100), header={'channel': 'BJE', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')})
    ])
    acc = Stream([
        Trace(data=np.zeros(100), header={'channel': 'HHZ', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')}),
        Trace(data=np.zeros(100), header={'channel': 'HHN', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')}),
        Trace(data=np.zeros(100), header={'channel': 'HHE', 'sampling_rate': 1.0, 'starttime': UTCDateTime('2023-01-01')})
    ])
    
    # Plot CWT for all components
    fig = plot_cwt_all(
        rot=rot,
        acc=acc,
        cwt_output=sample_cwt_data,
        clog=False,
        ylim=0.5
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # Should have at least 3 subplots (one per component)
    
    # Save figure
    output_file = temp_output_dir / "cwt_all.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_frequency_time_map_adaptive(sample_frequency_time_data, temp_output_dir):
    """Test frequency-time map plotting."""
    # Plot frequency-time map
    fig = plot_frequency_time_map_adaptive(
        results=sample_frequency_time_data,
        plot_type='backazimuth',
        vmin=0,
        vmax=360
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    # Check colorbars
    for ax in fig.axes[-2:]:  # Last two axes should be colorbars
        assert len(ax.collections) > 0
    
    # Save figure
    output_file = temp_output_dir / "frequency_time_map.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_cwt_invalid_input():
    """Test CWT plotting with invalid input."""
    # Create empty stream
    st = Stream()
    cwt_output = {}
    
    # Should raise ValueError for empty stream
    with pytest.raises(ValueError):
        plot_cwt(st, cwt_output)
    
    # Create stream with invalid frequency range
    st = Stream([Trace(data=np.zeros(100), header={'channel': 'BJZ', 'sampling_rate': 1.0})])
    cwt_output = {
        'BJZ': {
            'times': np.zeros(90),  # Mismatched length
            'frequencies': np.zeros(50),
            'cwt_power': np.zeros((50, 90)),
            'cone': np.zeros(90)
        }
    }
    
    # Should raise ValueError for mismatched dimensions
    with pytest.raises(ValueError):
        plot_cwt(st, cwt_output)

def test_plot_cwt_all_invalid_input():
    """Test CWT all plotting with invalid input."""
    # Create empty streams
    rot = Stream()
    acc = Stream()
    cwt_output = {}
    
    # Should raise ValueError for empty streams
    with pytest.raises(ValueError):
        plot_cwt_all(rot, acc, cwt_output)
    
    # Create streams with invalid data
    rot = Stream([Trace(data=np.zeros(100), header={'channel': 'BJZ', 'sampling_rate': 1.0})])
    acc = Stream([Trace(data=np.zeros(90), header={'channel': 'HHZ', 'sampling_rate': 1.0})])  # Different length
    cwt_output = {
        'BJZ': {
            'times': np.zeros(90),
            'frequencies': np.zeros(50),
            'cwt_power': np.zeros((50, 90)),
            'cone': np.zeros(90)
        }
    }
    
    # Should raise ValueError for mismatched data lengths
    with pytest.raises(ValueError):
        plot_cwt_all(rot, acc, cwt_output)

def test_plot_frequency_time_map_adaptive_invalid_input():
    """Test frequency-time map plotting with invalid input."""
    # Create invalid results (missing required data)
    results = {
        'adaptive_windows': True,
        'frequency_bands': np.array([0.1, 0.2, 0.3]),
        'time_windows': [np.linspace(0, 3600, 100)],
        'backazimuth_data': [np.zeros(90)],  # Mismatched length
        'correlation_data': [np.zeros(100)],
        'window_factor': 10,
        'parameters': {
            'wave_type': 'love',
            'fmin': 0.01,
            'fmax': 1.0
        }
    }
    
    # Should raise ValueError for mismatched dimensions
    with pytest.raises(ValueError):
        plot_frequency_time_map_adaptive(results)