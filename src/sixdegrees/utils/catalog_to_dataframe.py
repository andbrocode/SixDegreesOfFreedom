"""
Functions for converting ObsPy catalogs to pandas DataFrames.
"""
import pandas as pd

def catalog_to_dataframe(catalog):
    """
    Convert catalog to pandas DataFrame.
    
    Args:
        catalog: Either an ObsPy catalog object or a dictionary with 'events' list
        
    Returns:
        pandas.DataFrame: DataFrame containing event information including:
            - time: Event origin time
            - latitude: Event latitude
            - longitude: Event longitude
            - depth: Event depth
            - magnitude: Event magnitude
            - magnitude_type: Magnitude type
    """
    data = []
    
    # Handle dictionary input
    if isinstance(catalog, dict) and 'events' in catalog:
        for event in catalog['events']:
            data.append({
                'time': pd.to_datetime(event['time']),
                'latitude': event['latitude'],
                'longitude': event['longitude'],
                'depth': event['depth'],
                'magnitude': event['magnitude'],
                'magnitude_type': event['magnitude_type']
            })
    # Handle ObsPy catalog
    else:
        for event in catalog:
            try:
                origin = event.preferred_origin() or event.origins[0]
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
                
                data.append({
                    'time': pd.to_datetime(origin.time.datetime),
                    'latitude': origin.latitude,
                    'longitude': origin.longitude,
                    'depth': origin.depth,
                    'magnitude': magnitude.mag,
                    'magnitude_type': magnitude.magnitude_type
                })
            except:
                continue
            
    return pd.DataFrame(data)
