"""
Tests for basic plotting functions.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
from sixdegrees.plots.plot_waveform_cc import plot_waveform_cc
from sixdegrees.plots.plot_spectra import plot_spectra, SpectralMethod

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

def test_plot_waveform_cc(sample_stream, temp_output_dir):
    """Test waveform cross-correlation plotting."""
    # Create test data
    rot = sample_stream.copy()  # Rotation rate stream
    acc = sample_stream.copy()  # Acceleration stream
    
    # Add some time shift to acc
    for tr in acc:
        tr.data = np.roll(tr.data, 10)
    
    # Plot waveforms
    fig = plot_waveform_cc(
        rot0=rot,
        acc0=acc,
        baz=45.0,  # Example backazimuth
        fmin=0.01,
        fmax=0.5,
        wave_type="both",
        unitscale="nano"
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Should have at least 2 subplots for "both" wave types
    
    # Save figure
    output_file = temp_output_dir / "waveform_cc.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_spectra(sample_stream, temp_output_dir):
    """Test spectra plotting."""
    # Create test data
    rot = sample_stream.copy()  # Rotation rate stream
    acc = sample_stream.copy()  # Acceleration stream
    
    # Plot spectra
    fig = plot_spectra(
        rot=rot,
        acc=acc,
        fmin=0.01,
        fmax=0.5,
        method=SpectralMethod.MULTITAPER,
        n_win=5,
        time_bandwidth=4.0
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # Should have at least 3 subplots (one per component)
    
    # Check axis labels
    for ax in fig.axes[::2]:  # Check primary axes (every other axis)
        assert ax.get_xlabel() == 'Frequency (Hz)'
        if ax == fig.axes[0]:  # First subplot
            assert ax.get_ylabel() == r'ASD (rad/s/$\sqrt{Hz}$)'
    
    # Save figure
    output_file = temp_output_dir / "spectra.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_plot_waveform_cc_invalid_input():
    """Test waveform plotting with invalid input."""
    # Create empty streams
    rot = Stream()
    acc = Stream()
    
    # Should raise ValueError for empty streams
    with pytest.raises(ValueError):
        plot_waveform_cc(rot0=rot, acc0=acc, baz=45.0, wave_type="both")
    
    # Create mismatched streams
    rot = Stream([Trace(data=np.zeros(100))])
    acc = Stream([Trace(data=np.zeros(200))])  # Different length
    
    # Should raise ValueError for mismatched data lengths
    with pytest.raises(ValueError):
        plot_waveform_cc(rot0=rot, acc0=acc, baz=45.0, wave_type="both")

def test_plot_spectra_invalid_input():
    """Test spectra plotting with invalid input."""
    # Create empty streams
    rot = Stream()
    acc = Stream()
    
    # Should raise ValueError for empty streams
    with pytest.raises(ValueError):
        plot_spectra(rot=rot, acc=acc, fmin=0.01, fmax=0.5)
    
    # Create streams with invalid frequency range
    rot = Stream([
        Trace(data=np.zeros(100), header={'channel': 'BJZ'}),
        Trace(data=np.zeros(100), header={'channel': 'BJN'}),
        Trace(data=np.zeros(100), header={'channel': 'BJE'})
    ])
    acc = Stream([
        Trace(data=np.zeros(100), header={'channel': 'HHZ'}),
        Trace(data=np.zeros(100), header={'channel': 'HHN'}),
        Trace(data=np.zeros(100), header={'channel': 'HHE'})
    ])
    
    # Should raise ValueError for fmax <= fmin
    with pytest.raises(ValueError):
        plot_spectra(rot=rot, acc=acc, fmin=0.5, fmax=0.1)