"""
Shared test fixtures and configuration for sixdegrees tests.
"""

import os
import pytest
import numpy as np
from obspy import UTCDateTime, Stream, Trace
import yaml

@pytest.fixture
def sample_config():
    """Provide a basic configuration for testing."""
    return {
        'network': 'BW',
        'station': 'ROMY',
        'location': '',
        'channel': 'BJZ',
        'starttime': UTCDateTime('2023-01-01'),
        'endtime': UTCDateTime('2023-01-02'),
        'sampling_rate': 1.0,
        'processing': {
            'filter': {
                'type': 'bandpass',
                'freqmin': 0.01,
                'freqmax': 0.1
            }
        }
    }

@pytest.fixture
def sample_stream():
    """Create a sample Stream object for testing."""
    # Create synthetic data
    npts = 3600  # 1 hour of data at 1 Hz
    t = np.linspace(0, 3600, npts)
    data = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
    
    # Create trace
    stats = {
        'network': 'BW',
        'station': 'ROMY',
        'location': '',
        'channel': 'BJZ',
        'sampling_rate': 1.0,
        'starttime': UTCDateTime('2023-01-01')
    }
    
    trace = Trace(data=data, header=stats)
    return Stream([trace])

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config = {
        'network': 'BW',
        'station': 'ROMY',
        'processing': {
            'filter': {
                'type': 'bandpass',
                'freqmin': 0.01,
                'freqmax': 0.1
            }
        }
    }
    
    config_path = tmp_path / "test_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path
