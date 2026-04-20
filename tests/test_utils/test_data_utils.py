"""
Tests for data handling utility functions.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from sixdegrees.utils.catalog_to_dataframe import catalog_to_dataframe
from sixdegrees.utils.request_data import request_data
from sixdegrees.utils.add_distances_and_backazimuth import add_distances_and_backazimuth

@pytest.fixture
def sample_catalog():
    """Create a sample earthquake catalog."""
    return {
        'events': [
            {
                'time': '2023-01-01T00:00:00',
                'latitude': 47.7714,
                'longitude': 11.2752,
                'depth': 10.0,
                'magnitude': 3.5,
                'magnitude_type': 'ML'
            },
            {
                'time': '2023-01-02T12:30:00',
                'latitude': 48.1234,
                'longitude': 11.5678,
                'depth': 15.0,
                'magnitude': 4.2,
                'magnitude_type': 'ML'
            }
        ]
    }

def test_catalog_to_dataframe(sample_catalog):
    """Test conversion of catalog to DataFrame."""
    # Convert catalog to DataFrame
    df = catalog_to_dataframe(sample_catalog)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_catalog['events'])
    
    # Check columns
    required_columns = ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'magnitude_type']
    for col in required_columns:
        assert col in df.columns
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['time'])
    assert pd.api.types.is_float_dtype(df['latitude'])
    assert pd.api.types.is_float_dtype(df['longitude'])
    assert pd.api.types.is_float_dtype(df['depth'])
    assert pd.api.types.is_float_dtype(df['magnitude'])

def test_request_data(tmp_path):
    """Test data request functionality."""
    # Test parameters
    network = "BW"
    station = "ROMY"
    location = ""
    channel = "BJZ"
    starttime = UTCDateTime("2023-01-01")
    endtime = UTCDateTime("2023-01-02")
    
    # Create mock FDSN client response
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200
    
    # Mock the request function
    def mock_request(*args, **kwargs):
        return MockResponse(b"Mock waveform data")
    
    # Test data request
    with pytest.raises(Exception):  # Should raise because we're not actually connecting to a server
        request_data(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )

def test_add_distances_and_backazimuth():
    """Test adding distances and backazimuth to catalog."""
    # Create test catalog
    catalog = pd.DataFrame({
        'latitude': [47.7714, 48.1234],
        'longitude': [11.2752, 11.5678],
        'depth': [10.0, 15.0]
    })
    
    # Station coordinates
    station_lat = [47.0]  # List with single value
    station_lon = [11.0]  # List with single value
    
    # Add distances and backazimuth
    result = add_distances_and_backazimuth(
        station_lat,
        station_lon,
        catalog
    )
    
    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert 'distances_deg' in result.columns
    assert 'distances_km' in result.columns
    assert 'backazimuth' in result.columns
    
    # Check data types and ranges
    assert pd.api.types.is_float_dtype(result['distances_deg'])
    assert pd.api.types.is_float_dtype(result['distances_km'])
    assert pd.api.types.is_float_dtype(result['backazimuth'])
    
    assert np.all(result['distances_deg'] >= 0)
    assert np.all(result['distances_km'] >= 0)
    assert np.all((result['backazimuth'] >= 0) & (result['backazimuth'] <= 360))

def test_add_distances_and_backazimuth_invalid_input():
    """Test error handling for invalid inputs in distance calculation."""
    # Create invalid catalog (missing required columns)
    invalid_catalog = pd.DataFrame({
        'latitude': [47.7714, 48.1234]
        # Missing longitude column
    })
    
    # Test with invalid catalog
    with pytest.raises(AttributeError):  # Will raise AttributeError when trying to access missing column
        add_distances_and_backazimuth(
            [47.0],
            [11.0],
            invalid_catalog
        )
    
    # Test with invalid coordinates
    valid_catalog = pd.DataFrame({
        'latitude': [47.7714, 48.1234],
        'longitude': [11.2752, 11.5678],
        'depth': [10.0, 15.0]
    })
    
    with pytest.raises(ValueError):
        add_distances_and_backazimuth(
            [91.0],  # Invalid latitude
            [11.0],
            valid_catalog
        )
