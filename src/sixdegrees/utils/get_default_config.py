"""
Functions for getting default configuration settings.
"""

def get_default_config():
    """
    Returns default configuration settings for BSPF catalog analysis.
    
    Returns:
        dict: Dictionary containing default configuration parameters including:
            - Geographic bounds (minlatitude, maxlatitude, minlongitude, maxlongitude)
            - Station coordinates (station_lon, station_lat)
            - Magnitude threshold (minmagnitude)
            - Time bounds (tbeg, tend)
            - File names (eventfile, triggerfile, gcmt_file)
            - Paths (path_to_data, path_to_catalogs, out_figures, outpath)
    """
    config = {
        # Geographic bounds
        'minlatitude': 31,
        'maxlatitude': 35,
        'minlongitude': -119, 
        'maxlongitude': -114,
        
        # Station coordinates
        'station_lon': [-116.455439],
        'station_lat': [33.610643],
        
        # Magnitude threshold
        'minmagnitude': None,
        
        # Time bounds (will be set later)
        'tbeg': None,
        'tend': None,
        
        # File names
        'eventfile': None,  # Will be set based on dates
        'triggerfile': None,  # Will be set based on dates
        'gcmt_file': None,  # Will be set based on dates
        
        # Paths (will be populated based on environment)
        'path_to_data': None,
        'path_to_catalogs': None,
        'out_figures': None,
        'outpath': None
    }
    
    return config