def __add_distances_and_backazimuth(reference_latitude, reference_longitude, df):

    from obspy.geodetics.base import gps2dist_azimuth
    from numpy import zeros
    
    dist = zeros(len(df))
    baz = zeros(len(df))

    
    for ii, ev in enumerate(df.index):
        try:
            ## azimuth = A -> B | backazimuth = B -> A (arg1 = A, arg2=B)
            dist[ii], az, baz[ii] = gps2dist_azimuth(df.latitude[ii], df.longitude[ii],
                                                     reference_latitude, reference_longitude,
                                                     )
        except:
            print(" -> failed to compute!")
            
    df['backazimuth'] = baz
    df['distances_km'] = dist/1000

    return df