def __get_event_info(config, min_mag=5.0):

    from obspy.geodetics.base import gps2dist_azimuth

    event = config['Client'].get_events(starttime=config['tbeg']-60, endtime=config['tend'], minmagnitude=min_mag)

    if len(event) > 1:
        print("-> more than one event\n")
        print(event)
        num = int(input("Select Event number: "))
    else:
        num = 0

    config['event'] = event[num]

    ## Eventtime
    config['eventtime'] = event[num].origins[0].time

    print(event[num])

    dist, az, baz = gps2dist_azimuth(event[num].origins[0].latitude, event[num].origins[0].longitude,
                                     config['sta_lat'], config['sta_lon'],
                                     )
    # to km
    dist /= 1000

    print("Distance ", dist, "km", "Azimuth ", az, "Backazimuth ", baz)

    return config, dist, baz, az