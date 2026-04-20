"""
Tests for basic utility functions.
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from sixdegrees.utils.read_pickle import read_pickle
from sixdegrees.utils.get_default_config import get_default_config
from sixdegrees.utils.print_dict_tree import print_dict_tree
from sixdegrees.utils.print_deviation_summary import print_deviation_summary

def test_read_pickle(tmp_path):
    """Test pickle reading functionality."""
    # Create test data
    test_data = {
        'array': np.array([1, 2, 3]),
        'dict': {'key': 'value'},
        'time': UTCDateTime('2023-01-01')
    }
    
    # Save test data
    with open(tmp_path / "test.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    # Read and verify data
    loaded_data = read_pickle(str(tmp_path) + "/", "test.pkl")
    assert np.array_equal(loaded_data['array'], test_data['array'])
    assert loaded_data['dict'] == test_data['dict']
    assert loaded_data['time'] == test_data['time']

def test_read_pickle_nonexistent():
    """Test reading nonexistent pickle file."""
    with pytest.raises(FileNotFoundError):
        read_pickle("/tmp/", "nonexistent.pkl")

def test_get_default_config():
    """Test default configuration generation."""
    config = get_default_config()
    
    # Check essential configuration keys
    assert 'minlatitude' in config
    assert 'maxlatitude' in config
    assert 'minlongitude' in config
    assert 'maxlongitude' in config
    assert 'station_lon' in config
    assert 'station_lat' in config
    
    # Check default values
    assert config['minlatitude'] == 31
    assert config['maxlatitude'] == 35
    assert config['minlongitude'] == -119
    assert config['maxlongitude'] == -114
    assert isinstance(config['station_lon'], list)
    assert isinstance(config['station_lat'], list)
    assert len(config['station_lon']) == 1
    assert len(config['station_lat']) == 1

def test_print_dict_tree(capsys):
    """Test dictionary tree printing."""
    test_dict = {
        'level1': {
            'level2a': 'value',
            'level2b': {
                'level3': 123
            }
        }
    }
    
    print_dict_tree(test_dict)
    captured = capsys.readouterr()
    
    # Check output format
    assert 'level1' in captured.out
    assert 'level2a' in captured.out
    assert 'level2b' in captured.out
    assert 'level3' in captured.out
    assert 'value' in captured.out
    assert '123' in captured.out

def test_print_deviation_summary(capsys):
    """Test deviation summary printing."""
    # Create test data
    analysis_results = {
        'deviations': {
            'love': {
                'deviation': np.array([1.2, -0.8, 2.1, -1.5]),
                'frequencies': np.array([0.1, 0.2, 0.3, 0.4]),
                'mean_deviation': 0.25,
                'std_deviation': 1.5,
                'rms_deviation': 1.8
            }
        },
        'theoretical_baz': 180.0,
        'center_frequencies': np.array([0.1, 0.2, 0.3, 0.4])
    }
    
    print_deviation_summary(analysis_results)
    captured = capsys.readouterr()
    
    # Check output format
    assert 'BACKAZIMUTH DEVIATION ANALYSIS SUMMARY' in captured.out
    assert 'Theoretical Backazimuth: 180.0°' in captured.out
    assert 'LOVE WAVES:' in captured.out
    assert 'Mean deviation: 0.25°' in captured.out
    assert 'Std deviation: 1.50°' in captured.out
    assert 'RMS deviation: 1.80°' in captured.out
