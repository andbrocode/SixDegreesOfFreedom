"""
Functions for adding distance and backazimuth calculations to event data.
"""
from obspy.geodetics import locations2degrees, gps2dist_azimuth

def add_distances_and_backazimuth(station_lat, station_lon, events_df):
    """
    Add distance and backazimuth calculations to events DataFrame.
    
    Args:
        station_lat (float): Latitude of the station
        station_lon (float): Longitude of the station
        events_df (pandas.DataFrame): DataFrame containing event information
        
    Returns:
        pandas.DataFrame: DataFrame with added columns:
            - distances_deg: Distance in degrees
            - distances_km: Distance in kilometers
            - backazimuth: Backazimuth in degrees
    """
    distances_deg = []
    distances_km = []
    backazimuths = []
    
    for _, event in events_df.iterrows():
        # Calculate distance in degrees
        dist_deg = locations2degrees(station_lat[0], station_lon[0], 
                                   event.latitude, event.longitude)
        distances_deg.append(dist_deg)
        
        # Calculate distance in km and backazimuth
        dist_m, az, _ = gps2dist_azimuth(station_lat[0], station_lon[0],
                                        event.latitude, event.longitude)
        distances_km.append(dist_m/1000)  # Convert to km
        backazimuths.append(az)
    
    events_df['distances_deg'] = distances_deg
    events_df['distances_km'] = distances_km
    events_df['backazimuth'] = backazimuths
    
    return events_df
