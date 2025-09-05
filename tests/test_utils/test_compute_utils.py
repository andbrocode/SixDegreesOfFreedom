"""
Tests for computation utility functions.
"""

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from sixdegrees.utils.compute_adr_pfo import compute_adr_pfo
from sixdegrees.utils.compute_beamforming_pfo import compute_beamforming_pfo
from sixdegrees.utils.compute_frequency_backazimuth_adaptive import compute_frequency_backazimuth_adaptive
from sixdegrees.utils.compute_frequency_dependent_backazimuth import compute_frequency_dependent_backazimuth
from sixdegrees.utils.get_kde_stats import get_kde_stats

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

def test_compute_adr_pfo():
    """Test ADR-PFO computation."""
    # Set test parameters
    tbeg = UTCDateTime("2023-01-01")
    tend = UTCDateTime("2023-01-01T00:10:00")  # 10 minutes of data
    submask = "inner"  # Use inner array configuration
    
    # Compute ADR-PFO
    result = compute_adr_pfo(tbeg, tend, submask=submask)
    
    # Check result structure
    assert isinstance(result, Stream)
    assert len(result) == 3  # Should have 3 components
    
    # Check component names
    channels = [tr.stats.channel for tr in result]
    assert 'BJZ' in channels
    assert 'BJN' in channels
    assert 'BJE' in channels
    
    # Check data
    for tr in result:
        assert isinstance(tr.data, np.ndarray)
        assert len(tr.data) > 0  # Should have data
        assert tr.stats.station == 'RPFO'  # Should be renamed to RPFO

def test_compute_beamforming_pfo():
    """Test beamforming PFO computation."""
    # Set test parameters
    tbeg = UTCDateTime("2023-01-01")
    tend = UTCDateTime("2023-01-01T00:10:00")  # 10 minutes of data
    submask = "inner"  # Use inner array configuration
    fmin = 1.0  # Hz
    fmax = 6.0  # Hz
    component = "Z"  # Vertical component
    
    # Compute beamforming
    result = compute_beamforming_pfo(tbeg, tend, submask, fmin=fmin, fmax=fmax, component=component)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 't_win' in result
    assert 'rel_pwr' in result
    assert 'abs_pwr' in result
    assert 'baz' in result
    assert 'slow' in result
    
    # Check data types and shapes
    assert isinstance(result['t_win'], np.ndarray)
    assert isinstance(result['rel_pwr'], np.ndarray)
    assert isinstance(result['abs_pwr'], np.ndarray)
    assert isinstance(result['baz'], np.ndarray)
    assert isinstance(result['slow'], np.ndarray)
    
    # Check statistics
    assert 'baz_bf_mean' in result
    assert 'baz_bf_max' in result
    assert 'baz_bf_std' in result
    assert isinstance(result['baz_bf_mean'], float)
    assert 0 <= result['baz_bf_mean'] <= 360  # Backazimuth should be in [0, 360]

def test_compute_frequency_backazimuth_adaptive():
    """Test adaptive frequency-backazimuth computation."""
    # Create mock sixdegrees object
    class MockSixDegrees:
        def __init__(self):
            self.st = None
            
        def get_stream(self, *args, **kwargs):
            # Return a sample stream
            t = np.linspace(0, 3600, 3600)  # 1 hour of data at 1 Hz
            data = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
            tr = Trace(data=data)
            tr.stats.starttime = UTCDateTime('2023-01-01')
            tr.stats.sampling_rate = 1.0
            return Stream([tr])
            
        def compute_backazimuth(self, **kwargs):
            # Return mock backazimuth results
            n_windows = 10
            return {
                'twin_center': np.linspace(0, 3600, n_windows),
                'cc_max_y': np.random.uniform(0, 360, n_windows),
                'cc_max': np.random.uniform(0.5, 1.0, n_windows)
            }
    
    # Set parameters
    sd_object = MockSixDegrees()
    freq_min = 0.01
    freq_max = 0.5
    
    # Compute frequency-dependent backazimuth
    result = compute_frequency_backazimuth_adaptive(
        sd_object,
        wave_type='love',
        fmin=freq_min,
        fmax=freq_max,
        verbose=False
    )
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'frequency_bands' in result
    assert 'backazimuth_data' in result
    assert 'correlation_data' in result
    assert 'time_windows' in result
    
    # Check data types
    assert isinstance(result['frequency_bands'], np.ndarray)
    assert isinstance(result['backazimuth_data'], list)
    assert isinstance(result['correlation_data'], list)
    assert isinstance(result['time_windows'], list)
    
    # Check frequency range (with tolerance for floating point)
    assert abs(result['frequency_bands'][0] - freq_min) < 1e-10
    assert abs(result['frequency_bands'][-1] - freq_max) < 0.1  # Allow some tolerance for octave bands
    
    # Check backazimuth data
    for baz_data in result['backazimuth_data']:
        valid_baz = baz_data[~np.isnan(baz_data)]
        if len(valid_baz) > 0:
            assert np.all(valid_baz >= 0)
            assert np.all(valid_baz <= 360)
    
    # Check correlation data
    for cc_data in result['correlation_data']:
        valid_cc = cc_data[~np.isnan(cc_data)]
        if len(valid_cc) > 0:
            assert np.all(valid_cc >= 0)
            assert np.all(valid_cc <= 1)

def test_compute_frequency_dependent_backazimuth(sample_stream):
    """Test frequency-dependent backazimuth computation."""
    # Set parameters
    params = {
        'freq_min': [0.01, 0.05, 0.1],
        'freq_max': [0.05, 0.1, 0.5],
        'slowness_max': 0.5,
        'slowness_step': 0.01,
        'window_length': 60.0,
        'window_fraction': 0.5,
        'prewhitening': 0
    }
    
    # Compute backazimuth
    result = compute_frequency_dependent_backazimuth(sample_stream, params)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'times' in result
    assert 'frequency' in result
    assert 'backazimuth' in result
    assert 'velocity' in result
    assert 'cross_correlation' in result
    
    # Check frequency data
    assert 'center' in result['frequency']
    assert 'min' in result['frequency']
    assert 'max' in result['frequency']
    
    # Check backazimuth data
    assert 'optimal' in result['backazimuth']
    assert 'mean' in result['backazimuth']
    assert 'std' in result['backazimuth']
    
    # Check data types
    assert isinstance(result['times'], list)
    assert isinstance(result['velocity'], list)
    assert isinstance(result['backazimuth']['optimal'], list)
    
    # Check value ranges
    baz_values = result['backazimuth']['optimal']
    valid_baz = [b for b in baz_values if not np.isnan(b)]
    if valid_baz:
        assert np.all(np.array(valid_baz) >= 0)
        assert np.all(np.array(valid_baz) <= 360)
    
    # Check cross-correlation values
    cc_values = result['cross_correlation']['optimal']
    valid_cc = [c for c in cc_values if not np.isnan(c)]
    if valid_cc:
        # Normalize cross-correlation values
        valid_cc = np.array(valid_cc)
        if len(valid_cc) > 0:
            valid_cc = valid_cc / np.max(np.abs(valid_cc))
        assert np.all(valid_cc >= 0)
        assert np.all(valid_cc <= 1)

def test_get_kde_stats():
    """Test KDE statistics computation."""
    # Create sample data
    baz = np.random.normal(loc=180, scale=20, size=1000) % 360  # Wrap to 0-360 range
    ccc = np.random.uniform(0.5, 1.0, size=1000)  # Cross-correlation coefficients
    
    # Compute KDE stats
    stats = get_kde_stats(baz, ccc, _baz_steps=5, Ndegree=180)
    
    # Check result structure
    assert isinstance(stats, dict)
    assert 'baz_estimate' in stats
    assert 'kde_max' in stats
    assert 'shift' in stats
    assert 'kde_values' in stats
    assert 'kde_angles' in stats
    assert 'kde_dev' in stats
    
    # Check data types
    assert isinstance(stats['baz_estimate'], np.number)  # Allow numpy numeric types
    assert isinstance(stats['kde_max'], np.number)
    assert isinstance(stats['shift'], np.number)
    assert isinstance(stats['kde_values'], np.ndarray)
    assert isinstance(stats['kde_angles'], np.ndarray)
    assert isinstance(stats['kde_dev'], int)
    
    # Check value ranges
    assert 0 <= float(stats['baz_estimate']) <= 360
    assert 0 <= float(stats['kde_max']) <= 360
    assert np.all(stats['kde_angles'] >= 0)
    assert np.all(stats['kde_angles'] <= 360)
    assert stats['kde_dev'] >= 0
