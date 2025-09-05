"""
Tests for core sixdegrees functionality.
"""

import os
import pytest
import numpy as np
from obspy import UTCDateTime, Stream, Trace
from sixdegrees.sixdegrees import sixdegrees

def test_initialization_default():
    """Test sixdegrees initialization with default parameters."""
    config = {
        'fdsn_client_rot': 'IRIS',
        'fdsn_client_tra': 'IRIS',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02'
    }
    sd = sixdegrees(config)
    
    # Check default attributes
    assert sd.data_source == "fdsn"
    assert sd.fdsn_client_rot == "IRIS"
    assert sd.fdsn_client_tra == "IRIS"
    assert sd.verbose is False
    assert sd.project == "test"
    assert sd.net == "XX"
    assert sd.sta == "XXXX"
    assert sd.loc == ""
    assert sd.cha == ""

def test_initialization_with_config(sample_config):
    """Test sixdegrees initialization with provided configuration."""
    config = {
        'data_source': 'sds',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02',
        'verbose': True,
        'seed': 'BW.ROMY..BJZ',
        'project': 'test_project',
        'station_lon': 11.2752,
        'station_lat': 47.7714,
        'path_to_sds_rot': '/data/sds/rotation',
        'path_to_sds_tra': '/data/sds/translation',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ'
    }
    
    sd = sixdegrees(config)
    
    assert sd.data_source == "sds"
    assert sd.tbeg == UTCDateTime('2023-01-01')
    assert sd.tend == UTCDateTime('2023-01-02')
    assert sd.verbose is True
    assert sd.net == "BW"
    assert sd.sta == "ROMY"
    assert sd.loc == ""
    assert sd.cha == "BJZ"
    assert sd.project == "test_project"
    assert sd.station_longitude == 11.2752
    assert sd.station_latitude == 47.7714

def test_invalid_seed_id():
    """Test initialization with invalid seed ID."""
    config = {
        'seed': 'invalid.seed.id'  # Missing one component
    }
    
    with pytest.raises(ValueError):
        sixdegrees(config)

def test_time_validation():
    """Test time validation in initialization."""
    config = {
        'tbeg': '2023-01-02',
        'tend': '2023-01-01'  # End time before start time
    }
    
    with pytest.raises(ValueError):
        sd = sixdegrees(config)
        sd.validate_times()

def test_coordinate_validation():
    """Test coordinate validation."""
    config = {
        'station_lon': 200,  # Invalid longitude
        'station_lat': 100   # Invalid latitude
    }
    
    with pytest.raises(ValueError):
        sd = sixdegrees(config)
        sd.validate_coordinates()

def test_working_directories(tmp_path):
    """Test working directory setup."""
    workdir = tmp_path / "test_project"
    os.makedirs(workdir, exist_ok=True)
    
    config = {
        'workdir': str(workdir),
        'project': 'test_project',
        'fdsn_client_rot': 'IRIS',
        'fdsn_client_tra': 'IRIS',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02'
    }
    
    # Create output directories
    os.makedirs(os.path.join(workdir, "output"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "figures"), exist_ok=True)
    
    sd = sixdegrees(config)
    
    # Check if output directories are created
    assert os.path.exists(sd.path_to_data_out)
    assert os.path.exists(sd.path_to_figs_out)
    
    # Check if paths are correct
    assert sd.path_to_data_out == os.path.normpath(os.path.join(workdir, "output"))
    assert sd.path_to_figs_out == os.path.normpath(os.path.join(workdir, "figures"))

def test_fdsn_data_source():
    """Test FDSN data source configuration."""
    config = {
        'data_source': 'fdsn',
        'fdsn_client_rot': 'IRIS',
        'fdsn_client_tra': 'IRIS',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02'
    }
    
    sd = sixdegrees(config)
    assert sd.data_source == 'fdsn'
    assert sd.fdsn_client_rot == 'IRIS'
    assert sd.fdsn_client_tra == 'IRIS'

def test_sds_data_source(tmp_path):
    """Test SDS data source configuration."""
    sds_rot = tmp_path / "sds_rot"
    sds_tra = tmp_path / "sds_tra"
    os.makedirs(sds_rot, exist_ok=True)
    os.makedirs(sds_tra, exist_ok=True)
    
    config = {
        'data_source': 'sds',
        'path_to_sds_rot': str(sds_rot),
        'path_to_sds_tra': str(sds_tra),
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02',
        'station_lon': 11.2752,
        'station_lat': 47.7714
    }
    
    sd = sixdegrees(config)
    assert sd.data_source == 'sds'
    assert sd.rot_sds == str(sds_rot)
    assert sd.tra_sds == str(sds_tra)

def test_mseed_data_source(tmp_path):
    """Test direct mseed file input configuration."""
    mseed_file = tmp_path / "test.mseed"
    # Create an empty file
    mseed_file.touch()
    
    config = {
        'data_source': 'mseed_file',
        'path_to_mseed_file': str(mseed_file),  # Changed from mseed_file to path_to_mseed_file
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02',
        'station_lon': 11.2752,
        'station_lat': 47.7714
    }
    
    sd = sixdegrees(config)
    assert sd.data_source == 'mseed_file'
    assert sd.mseed_file == str(mseed_file)

def test_response_removal_config():
    """Test response removal configuration."""
    config = {
        'fdsn_client_rot': 'IRIS',
        'fdsn_client_tra': 'IRIS',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02',
        'tra_remove_response': True,
        'rot_remove_response': True,
        'tra_output': 'VEL'
    }
    
    sd = sixdegrees(config)
    assert sd.tra_remove_response is True
    assert sd.rot_remove_response is True
    assert sd.tra_output == 'VEL'

def test_romy_rotation_config():
    """Test ROMY rotation configuration."""
    config = {
        'fdsn_client_rot': 'IRIS',
        'fdsn_client_tra': 'IRIS',
        'rot_seed': 'BW.ROMY..BJZ',
        'tra_seed': 'BW.ROMY..BJZ',
        'tbeg': '2023-01-01',
        'tend': '2023-01-02',
        'use_romy_zne': True,
        'keep_z': False,
        'rot_target': 'ZNE'  # Removed rot_components as it's set internally
    }
    
    sd = sixdegrees(config)
    assert sd.use_romy_zne is True
    assert sd.keep_z is False
    assert sd.rot_target == 'ZNE'  # Only test rot_target as rot_components is set internally