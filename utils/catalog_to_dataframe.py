"""
Functions for converting ObsPy catalogs to pandas DataFrames.
"""
import pandas as pd

def catalog_to_dataframe(catalog):
    """
    Convert ObsPy catalog to pandas DataFrame.
    
    Args:
        catalog: ObsPy catalog object
        
    Returns:
        pandas.DataFrame: DataFrame containing event information including:
            - timestamp: Event origin time
            - latitude: Event latitude
            - longitude: Event longitude
            - depth: Event depth
            - magnitude: Event magnitude
            - type: Magnitude type
    """
    data = []
    for event in catalog:
        try:
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            
            data.append({
                'timestamp': origin.time,
                'latitude': origin.latitude,
                'longitude': origin.longitude,
                'depth': origin.depth,
                'magnitude': magnitude.mag,
                'type': magnitude.magnitude_type
            })
        except:
            continue
            
    return pd.DataFrame(data)
